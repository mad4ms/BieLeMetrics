# hbl_dagster/io_managers.py

from dagster import IOManager, io_manager
import duckdb
import pandas as pd
from pathlib import Path


class DuckDBIOManager(IOManager):
    """
    Stores each asset as a DuckDB table.
    Table name := asset key joined with underscores.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path

    def _conn(self):
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(self._db_path)

    def handle_output(self, context, obj):
        if obj is None:
            return

        table_name = context.asset_key.to_user_string().replace("/", "_")

        with self._conn() as con:
            if isinstance(obj, pd.DataFrame):
                con.register("tmp_df", obj)
                con.execute(
                    f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM tmp_df"
                )
            else:
                # Fall back to a generic table with a single json column
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


@io_manager
def duckdb_io_manager(init_context):
    """
    Configure with a simple 'db_path' in defs.py.
    """
    db_path = init_context.resource_config["db_path"]
    return DuckDBIOManager(db_path=db_path)
