# hbl_dagster/io_managers.py

import json
from contextlib import contextmanager
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from dagster import IOManager, io_manager
from filelock import FileLock
# hbl_dagster/io_managers.py


class DuckDBIOManager(IOManager):
    """
    Stores each asset as a DuckDB table.

    - One table per asset.
    - For partitioned assets: adds a hidden '__partition_key' column and
      rewrites only that partition's slice on each run.
    """

    def __init__(self, db_path: str):
        db_path_obj = Path(db_path).resolve()
        self._db_path = str(db_path_obj)
        # one lock file per DB file
        self._lock_path = db_path_obj.with_suffix(db_path_obj.suffix + ".lock")
        self._lock = FileLock(str(self._lock_path))

    @contextmanager
    def _conn(self):
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        # serialize *all* opens of this DB file across processes
        with self._lock:
            con = duckdb.connect(self._db_path)
            con.execute("SET TimeZone='UTC'")
            try:
                yield con
            finally:
                con.close()

    # --- helpers -------------------------------------------------------------

    def _table_exists(self, con, table_name: str) -> bool:
        try:
            con.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
            return True
        except duckdb.CatalogException:
            return False

    def _jsonify_nested_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if df[col].dtype != "object":
                continue

            # quick sample-based detection
            sample = df[col].dropna().head(20).tolist()
            if not any(isinstance(v, (dict, list)) for v in sample):
                continue

            df[col] = df[col].map(
                lambda v: (
                    json.dumps(v, ensure_ascii=False, default=str)
                    if isinstance(v, (dict, list))
                    else v
                )
            )
        return df

    def _normalize_string_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if not (is_object_dtype(df[col]) or is_string_dtype(df[col])):
                continue

            df[col] = df[col].map(
                lambda v: pd.NA if isinstance(v, str) and v.strip() == "" else v
            )

            sample = df[col].dropna().head(20).tolist()
            if not sample or any(isinstance(v, str) for v in sample):
                df[col] = df[col].astype("string")

        return df

    def _coerce_to_existing_schema(
        self,
        df: pd.DataFrame,
        schema_by_col: dict[str, str],
    ) -> pd.DataFrame:
        df = df.copy()
        for col, duck_type in schema_by_col.items():
            if col not in df.columns:
                continue

            type_upper = duck_type.upper()

            if any(
                token in type_upper
                for token in ["TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT"]
            ):
                numeric = pd.to_numeric(df[col], errors="coerce")
                mask = numeric.isna()
                if mask.any():
                    # pandas 3.x: float64.astype("Int64") raises when NaN present
                    # due to casting="safe" restrictions. Use explicit mask approach.
                    arr = numeric.fillna(0).astype(np.int64)
                    result = pd.array(arr, dtype="Int64")
                    result[mask.values] = pd.NA
                    df[col] = pd.Series(result, index=df.index)
                else:
                    df[col] = numeric.astype("int64")
            elif any(
                token in type_upper for token in ["DOUBLE", "FLOAT", "DECIMAL", "REAL"]
            ):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif "BOOLEAN" in type_upper:
                df[col] = df[col].map(
                    lambda v: (
                        True
                        if isinstance(v, str)
                        and v.strip().lower() in {"true", "t", "1", "yes"}
                        else False
                        if isinstance(v, str)
                        and v.strip().lower() in {"false", "f", "0", "no"}
                        else v
                    )
                )

        return df

    # --- IOManager API -------------------------------------------------------

    def handle_output(self, context, obj):
        if obj is None:
            return

        table_name = context.asset_key.to_user_string().replace("/", "_")

        if context.has_partition_key:
            partition_key = context.partition_key
            # Default to 'fixtureId' if not specified in asset metadata
            partition_col = context.definition_metadata.get(
                "partition_column", "fixture_id"
            )

            with self._conn() as con:
                if isinstance(obj, pd.DataFrame):
                    obj = self._jsonify_nested_columns(obj)
                    obj = self._normalize_string_nulls(obj)

                    if not self._table_exists(con, table_name):
                        con.register("tmp_df", obj)
                        con.execute(
                            f"CREATE TABLE {table_name} AS SELECT * FROM tmp_df"
                        )
                    else:
                        # Incremental update: delete old partition data, then insert
                        con.execute(
                            f"DELETE FROM {table_name} WHERE {partition_col} = ?",
                            [partition_key],
                        )

                        describe_df = con.execute(f"DESCRIBE {table_name}").df()
                        schema_by_col = dict(
                            zip(describe_df["column_name"], describe_df["column_type"])
                        )

                        # Alter table to add missing columns from tmp_df
                        existing_cols = set(schema_by_col)
                        new_cols = set(obj.columns)
                        missing_cols = new_cols - existing_cols

                        con.register("tmp_df", obj)

                        for col in missing_cols:
                            print(
                                f"Adding missing column '{col}' to table '{table_name}'"
                            )
                            # Get the type of the new column from tmp_df
                            col_type = con.execute(
                                f"SELECT typeof({col}) FROM tmp_df LIMIT 1"
                            ).fetchone()[0]
                            con.execute(
                                f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type}"
                            )

                        # Widen existing columns whose type was inferred from
                        # all-null data (e.g. INTEGER) but now receive wider types.
                        # Typical case: first run produces pd.NA → DuckDB stores as
                        # INTEGER; later runs have actual timestamps or strings.
                        _NARROW_INT_TYPES = {"TINYINT", "SMALLINT", "INTEGER"}
                        for col in existing_cols & new_cols:
                            existing_upper = schema_by_col[col].upper()
                            if existing_upper not in _NARROW_INT_TYPES:
                                continue
                            row = con.execute(
                                f"SELECT typeof({col}) FROM tmp_df"
                                f" WHERE {col} IS NOT NULL LIMIT 1"
                            ).fetchone()
                            if row is None:
                                continue  # still all-null; no widening needed
                            new_type = row[0].upper()
                            if new_type != existing_upper:
                                print(
                                    f"Widening column '{col}' in '{table_name}'"
                                    f" from {existing_upper} to {row[0]}"
                                )
                                try:
                                    con.execute(
                                        f"ALTER TABLE {table_name}"
                                        f' ALTER COLUMN "{col}" TYPE {row[0]}'
                                    )
                                    schema_by_col[col] = row[0]
                                except Exception as alter_err:
                                    print(
                                        f"Warning: could not widen '{col}'"
                                        f" from {existing_upper} to {row[0]}: {alter_err}"
                                    )

                        con.unregister("tmp_df")

                        obj = self._coerce_to_existing_schema(obj, schema_by_col)
                        con.register("tmp_df", obj)

                        # Use INSERT BY NAME to handle column order/mismatch gracefully
                        con.execute(
                            f"INSERT INTO {table_name} BY NAME SELECT * FROM tmp_df"
                        )

                    con.unregister("tmp_df")
                else:
                    # Fallback or error for non-DF partitioned output
                    pass
        else:
            # Non-partitioned: overwrite the entire table
            with self._conn() as con:
                if isinstance(obj, pd.DataFrame):
                    obj = self._jsonify_nested_columns(obj)
                    obj = self._normalize_string_nulls(obj)
                    con.register("tmp_df", obj)
                    con.execute(
                        f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM tmp_df"
                    )
                    con.unregister("tmp_df")
                else:
                    con.execute(
                        f"""
                        CREATE OR REPLACE TABLE {table_name} AS
                        SELECT ?::VARCHAR AS value
                        """,
                        [str(obj)],
                    )

    def load_input(self, context):
        upstream_key = context.upstream_output.asset_key
        table_name = upstream_key.to_user_string().replace("/", "_")
        with self._conn() as con:
            try:
                if context.has_asset_partitions:
                    partition_col = context.upstream_output.definition_metadata.get(
                        "partition_column", "fixture_id"
                    )
                    partition_keys = []
                    try:
                        partition_keys = [str(k) for k in context.asset_partition_keys]
                    except Exception:
                        partition_keys = []

                    if len(partition_keys) == 1:
                        return con.execute(
                            f"SELECT * FROM {table_name} WHERE {partition_col} = ?",
                            [partition_keys[0]],
                        ).fetch_df()

                    if len(partition_keys) > 1:
                        placeholders = ", ".join(["?"] * len(partition_keys))
                        return con.execute(
                            f"SELECT * FROM {table_name} WHERE {partition_col} IN ({placeholders})",
                            partition_keys,
                        ).fetch_df()

                    # Partition context exists but no concrete key list is available
                    # (e.g. AllPartitionsSubset). Fall back to full-table load.
                    return con.execute(f"SELECT * FROM {table_name}").fetch_df()
                return con.execute(f"SELECT * FROM {table_name}").fetch_df()
            except duckdb.CatalogException as e:
                raise RuntimeError(
                    f"DuckDB table for asset {upstream_key} not found: {e}"
                ) from e

    def load_full_table(self, table_name: str) -> pd.DataFrame:
        """Load an entire table regardless of partition context. Use for global integrity checks."""
        with self._conn() as con:
            try:
                return con.execute(f"SELECT * FROM {table_name}").fetch_df()
            except duckdb.CatalogException as e:
                raise RuntimeError(f"DuckDB table '{table_name}' not found: {e}") from e


@io_manager
def duckdb_io_manager(init_context):
    """
    Configure with a simple 'db_path' in defs.py (or via defs_debug.py re-export).
    """
    db_path = init_context.resource_config["db_path"]
    return DuckDBIOManager(db_path=db_path)


class InMemoryIOManager(IOManager):
    """
    Keeps asset values only in memory for the lifetime of the run.
    Suitable for models, sklearn Pipelines, torch modules, etc.
    """

    def __init__(self):
        self._store = {}

    def handle_output(self, context, obj):
        self._store[context.asset_key] = obj

    def load_input(self, context):
        key = context.upstream_output.asset_key
        if key not in self._store:
            raise RuntimeError(
                f"In-memory asset {key.to_user_string()} not found. "
                "This asset must be produced in the same run."
            )
        return self._store[key]


@io_manager
def in_memory_io_manager(_):
    return InMemoryIOManager()
