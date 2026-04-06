import json
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd
from dagster import IOManager, io_manager
from filelock import FileLock
from pandas.api.types import is_object_dtype, is_string_dtype


class DuckDBIOManager(IOManager):
    """
    Stores each asset as a DuckDB table (one table per asset).

    Partitioned assets:
        Replace only the current partition with a DELETE + INSERT strategy,
        avoiding full-table reads into pandas.

    Non-partitioned assets:
        CREATE OR REPLACE TABLE from the full DataFrame.
    """

    def __init__(self, db_path: str):
        db_path_obj = Path(db_path).resolve()
        self._db_path = str(db_path_obj)
        self._lock_path = db_path_obj.with_suffix(db_path_obj.suffix + ".lock")
        self._lock = FileLock(str(self._lock_path))

    @contextmanager
    def _conn(self):
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            con = duckdb.connect(self._db_path)
            con.execute("SET TimeZone='UTC'")
            try:
                yield con
            finally:
                con.close()

    def _table_exists(self, con, table_name: str) -> bool:
        try:
            con.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
            return True
        except duckdb.CatalogException:
            return False

    def _jsonify_nested_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """JSON-serialize any column whose values are dicts or lists."""
        df = df.copy()
        for col in df.columns:
            if df[col].dtype != "object":
                continue
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

    def _normalize_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert timezone-aware datetime columns to UTC-naive.

        DuckDB stores these as plain TIMESTAMP (not TIMESTAMPTZ), which avoids
        cast failures when inserting into existing tables created from naive
        pandas Timestamps. All timestamps in this project are UTC, so stripping
        the tz tag loses no information.
        """
        df = df.copy()
        for col in df.columns:
            if (
                pd.api.types.is_datetime64_any_dtype(df[col])
                and df[col].dt.tz is not None
            ):
                df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
        return df

    def _normalize_blank_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace empty or whitespace-only strings with NULL-like values."""
        df = df.copy()
        for col in df.columns:
            if not (is_object_dtype(df[col]) or is_string_dtype(df[col])):
                continue
            df[col] = df[col].map(
                lambda value: (
                    None if isinstance(value, str) and not value.strip() else value
                )
            )
        return df

    def _write(self, con, table_name: str, df: pd.DataFrame) -> None:
        con.register("tmp_df", df)
        try:
            con.execute(
                f"CREATE OR REPLACE TABLE {self._quote_identifier(table_name)} AS "
                "SELECT * FROM tmp_df"
            )
        finally:
            con.unregister("tmp_df")

    def _quote_identifier(self, identifier: str) -> str:
        return f'"{identifier.replace('"', '""')}"'

    def _get_relation_columns(self, con, relation_name: str) -> dict[str, str]:
        rows = con.execute(
            f"DESCRIBE {self._quote_identifier(relation_name)}"
        ).fetchall()
        return {str(name): str(col_type) for name, col_type, *_ in rows}

    # Unified numeric rank for type widening. DOUBLE ranks above all integers so
    # that a DOUBLE source column can widen a narrow integer target (e.g. when
    # pandas represents an int-with-NaN column as float64).
    _NUMERIC_RANK: dict[str, int] = {
        "TINYINT": 0,
        "SMALLINT": 1,
        "INTEGER": 2,
        "INT": 2,
        "BIGINT": 3,
        "HUGEINT": 4,
        "UBIGINT": 3,
        "FLOAT": 5,
        "REAL": 5,
        "DOUBLE": 6,
    }

    def _type_needs_widening(self, target_type: str, source_type: str) -> bool:
        t, s = target_type.upper(), source_type.upper()
        if t == s:
            return False
        t_rank = self._NUMERIC_RANK.get(t)
        s_rank = self._NUMERIC_RANK.get(s)
        if t_rank is not None and s_rank is not None:
            return s_rank > t_rank
        return False

    def _ensure_table_has_source_columns(
        self, con, table_name: str, source_columns: dict[str, str]
    ) -> None:
        target_columns = self._get_relation_columns(con, table_name)
        quoted_table_name = self._quote_identifier(table_name)

        for col_name, col_type in source_columns.items():
            if col_name not in target_columns:
                con.execute(
                    f"ALTER TABLE {quoted_table_name} "
                    f"ADD COLUMN {self._quote_identifier(col_name)} {col_type}"
                )
            elif self._type_needs_widening(target_columns[col_name], col_type):
                con.execute(
                    f"ALTER TABLE {quoted_table_name} "
                    f"ALTER COLUMN {self._quote_identifier(col_name)} TYPE {col_type}"
                )

    def _write_partition(
        self,
        con,
        table_name: str,
        df: pd.DataFrame,
        partition_col: str,
        partition_key: str,
    ) -> None:
        con.register("tmp_df", df)
        try:
            quoted_table_name = self._quote_identifier(table_name)
            quoted_partition_col = self._quote_identifier(partition_col)

            if not self._table_exists(con, table_name):
                con.execute(f"CREATE TABLE {quoted_table_name} AS SELECT * FROM tmp_df")
                return

            self._ensure_table_has_source_columns(
                con,
                table_name,
                self._get_relation_columns(con, "tmp_df"),
            )

            con.execute(
                f"DELETE FROM {quoted_table_name} WHERE {quoted_partition_col} = ?",
                [partition_key],
            )

            if not df.empty:
                con.execute(
                    f"INSERT INTO {quoted_table_name} BY NAME SELECT * FROM tmp_df"
                )
        finally:
            con.unregister("tmp_df")

    # --- IOManager API -------------------------------------------------------

    def handle_output(self, context, obj):
        if not isinstance(obj, pd.DataFrame):
            return

        table_name = context.asset_key.to_user_string().replace("/", "_")
        obj = self._normalize_blank_strings(obj)
        obj = self._normalize_datetimes(obj)
        obj = self._jsonify_nested_columns(obj)

        with self._conn() as con:
            if context.has_partition_key:
                partition_key = context.partition_key
                partition_col = context.definition_metadata.get(
                    "partition_column", "fixture_id"
                )
                self._write_partition(
                    con,
                    table_name,
                    obj,
                    partition_col,
                    str(partition_key),
                )
            else:
                self._write(con, table_name, obj)

    def load_input(self, context):
        upstream_key = context.upstream_output.asset_key
        table_name = upstream_key.to_user_string().replace("/", "_")
        with self._conn() as con:
            try:
                partition_col = context.upstream_output.definition_metadata.get(
                    "partition_column", "fixture_id"
                )

                if context.has_partition_key:
                    return con.execute(
                        f"SELECT * FROM {table_name} WHERE {partition_col} = ?",
                        [str(context.partition_key)],
                    ).fetch_df()

                if context.has_asset_partitions:
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
                            f"SELECT * FROM {table_name}"
                            f" WHERE {partition_col} IN ({placeholders})",
                            partition_keys,
                        ).fetch_df()

                return con.execute(f"SELECT * FROM {table_name}").fetch_df()
            except duckdb.CatalogException as e:
                raise RuntimeError(
                    f"DuckDB table for asset {upstream_key} not found: {e}"
                ) from e

    def load_full_table(self, table_name: str) -> pd.DataFrame:
        """Load an entire table regardless of partition context."""
        with self._conn() as con:
            try:
                return con.execute(f"SELECT * FROM {table_name}").fetch_df()
            except duckdb.CatalogException as e:
                raise RuntimeError(f"DuckDB table '{table_name}' not found: {e}") from e


@io_manager
def duckdb_io_manager(init_context):
    db_path = init_context.resource_config["db_path"]
    return DuckDBIOManager(db_path=db_path)


class InMemoryIOManager(IOManager):
    """
    In-memory store for a single run. Suitable for models and non-serializable objects.
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
                "Must be produced in the same run."
            )
        return self._store[key]


@io_manager
def in_memory_io_manager(_):
    return InMemoryIOManager()
