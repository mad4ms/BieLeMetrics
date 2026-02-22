# hbl_dagster/io_managers.py

import json
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd
from dagster import IOManager, io_manager
from filelock import FileLock


class DuckDBIOManager(IOManager):
    """
    Stores each asset as a DuckDB table.

    - One table per asset.
    - For partitioned assets: adds a hidden '__partition_key' column and
      rewrites only that partition's slice on each run.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        db_path_obj = Path(db_path)
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
                    con.register("tmp_df", obj)

                    if not self._table_exists(con, table_name):
                        con.execute(
                            f"CREATE TABLE {table_name} AS SELECT * FROM tmp_df"
                        )
                    else:
                        # Incremental update: delete old partition data, then insert
                        con.execute(
                            f"DELETE FROM {table_name} WHERE {partition_col} = ?",
                            [partition_key],
                        )

                        # Alter table to add missing columns from tmp_df
                        existing_cols = set(
                            con.execute(f"DESCRIBE {table_name}").df()[
                                "column_name"
                            ]
                        )
                        new_cols = set(obj.columns)
                        missing_cols = new_cols - existing_cols

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
                return con.execute(f"SELECT * FROM {table_name}").fetch_df()
            except duckdb.CatalogException as e:
                raise RuntimeError(
                    f"DuckDB table for asset {upstream_key} not found: {e}"
                ) from e

    # positions_kinexon_raw asset can be large, so we add a method to load only a partition
    def load_partitioned_input(
        self, table_name: str, partition_key: str, partition_col: str
    ) -> pd.DataFrame:
        with self._conn() as con:
            try:
                query = f"""
                SELECT *
                FROM {table_name}
                WHERE {partition_col} = ?
                """
                return con.execute(query, [partition_key]).fetch_df()
            except duckdb.CatalogException as e:
                raise RuntimeError(f"DuckDB error: {e}") from e


@io_manager
def duckdb_io_manager(init_context):
    """
    Configure with a simple 'db_path' in defs_debug.py (or via defs.py re-export).
    """
    db_path = init_context.resource_config["db_path"]
    return DuckDBIOManager(db_path=db_path)


# hbl_dagster/io_managers.py
from dagster import IOManager, io_manager


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
