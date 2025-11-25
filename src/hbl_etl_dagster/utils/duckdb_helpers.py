# duckdb_helpers.py
from contextlib import contextmanager
from dagster import IOManager
from filelock import FileLock
from pathlib import Path


@contextmanager
def duckdb_conn(io_manager: IOManager):
    """
    Context manager to centralize access to the DuckDB connection
    from the DuckDBIOManager.
    Assumes IOManager._conn is a contextmanager that already handles locking.
    """
    with io_manager._conn() as con:
        yield con
