# hbl_dagster/io_managers.py

from contextlib import contextmanager
from dagster import IOManager, io_manager
import duckdb
import pandas as pd
from pathlib import Path
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


@io_manager
def duckdb_io_manager(init_context):
    """
    Configure with a simple 'db_path' in defs.py.
    """
    db_path = init_context.resource_config["db_path"]
    return DuckDBIOManager(db_path=db_path)
