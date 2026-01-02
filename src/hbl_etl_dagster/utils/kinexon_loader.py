import logging
import os
from typing import Any, Callable, ContextManager, Dict, Optional

import pandas as pd

from src.fetcher_kinexon.fetch_positions_for_fixture import fetch_positions_for_fixture

ConnFactory = Callable[[], ContextManager]


def _as_string_series(s: pd.Series) -> pd.Series:
    # Robust for parquet + duckdb: bytes/int/str -> pandas string dtype
    def norm(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return pd.NA
        if isinstance(v, (bytes, bytearray)):
            return v.decode("utf-8", "ignore")
        return str(v)

    return s.map(norm).astype("string")


def load_missing_positions(
    conn_factory: ConnFactory,
    df_fixture: pd.DataFrame,
    api_kinexon,
    logger: Optional[logging.Logger] = None,
    skip_if_exists: bool = True,
    cache_dir: str = ".",
) -> Dict[str, Any]:
    """
    Single-fixture/single-session loader.

    Key property:
    - DB connections (and thus the FileLock) are held only briefly for:
      * existence check
      * insert
      * stats
    - Kinexon fetch happens outside any DB lock.
    """

    if df_fixture.empty:
        return {
            "n_rows": 0,
            "n_distinct_sessions": 0,
            "n_existing_sessions_before": 0,
            "n_sessions_in_fixtures": 0,
            "n_sessions_fetched_this_run": 0,
            "n_players_total": 0,
            "n_groups_total": 0,
            "preview_md": "*(empty fixture)*",
        }

    fixture_id = str(df_fixture["fixture_id"].iloc[0])
    session_id = str(df_fixture["session_id"].iloc[0])

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"kinexon_positions_{session_id}.parquet.gzip")

    # -------------------------
    # 1) Short DB existence check (lock held briefly)
    # -------------------------
    already_in_db = False
    if skip_if_exists:
        try:
            with conn_factory() as con:
                already_in_db = bool(
                    con.execute(
                        "SELECT 1 FROM kinexon_positions WHERE session_id = ? LIMIT 1",
                        [session_id],
                    ).fetchone()
                )
        except Exception:
            already_in_db = False

    if already_in_db:
        return {
            "n_rows": 0,
            "n_distinct_sessions": 1,
            "n_existing_sessions_before": 1,
            "n_sessions_in_fixtures": 1,
            "n_sessions_fetched_this_run": 0,
            "n_players_total": 0,
            "n_groups_total": 0,
            "preview_md": "*(skipped; session already present)*",
        }

    # -------------------------
    # 2) Cache read / API fetch (NO DB lock)
    # -------------------------
    df_positions = pd.DataFrame()
    fetched = 0

    if os.path.exists(cache_path):
        try:
            df_positions = pd.read_parquet(cache_path)
            if logger:
                logger.info("Loaded positions from cache: %s", cache_path)
        except Exception as e:
            if logger:
                logger.warning(
                    "Failed reading cache %s: %s (refetching)", cache_path, e
                )
            df_positions = pd.DataFrame()

    if df_positions.empty:
        df_positions = fetch_positions_for_fixture(api_kinexon, session_id)
        if df_positions is None:
            df_positions = pd.DataFrame()
        fetched = 0 if df_positions.empty else 1

    if df_positions.empty:
        return {
            "n_rows": 0,
            "n_distinct_sessions": 0,
            "n_existing_sessions_before": 0,
            "n_sessions_in_fixtures": 1,
            "n_sessions_fetched_this_run": int(fetched),
            "n_players_total": 0,
            "n_groups_total": 0,
            "preview_md": "*(no positions returned)*",
        }

    # -------------------------
    # 3) Normalize + attach context (still NO DB lock)
    # -------------------------
    if "session_id" not in df_positions.columns:
        df_positions["session_id"] = session_id
    df_positions["session_id"] = _as_string_series(df_positions["session_id"])
    df_positions["fixtureId"] = fixture_id

    # Fix your parquet issue: if "league id" can contain strings like "ball"
    # force it to string consistently (prevents pyarrow mixed-type failures).
    if "league id" in df_positions.columns:
        df_positions["league id"] = _as_string_series(df_positions["league id"])

    df_positions = df_positions.drop_duplicates()

    # -------------------------
    # 4) Short DB insert (lock held briefly)
    # -------------------------
    with conn_factory() as con:
        con.register("df_positions", df_positions)
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS kinexon_positions AS
                SELECT * FROM df_positions LIMIT 0
                """
            )
            # Safe enough as long as schema doesn't drift; if it drifts, switch to INSERT BY NAME
            con.execute(
                """
                INSERT INTO kinexon_positions
                SELECT * FROM df_positions
                """
            )
        finally:
            try:
                con.unregister("df_positions")
            except Exception:
                pass

    # -------------------------
    # 5) Cache write (NO DB lock) + never fail the op
    # -------------------------
    try:
        df_positions.to_parquet(cache_path, compression="gzip")
    except Exception as e:
        if logger:
            logger.warning("Failed to write cache %s: %s (continuing)", cache_path, e)

    # -------------------------
    # 6) Stats + preview (short DB lock)
    # -------------------------
    try:
        with conn_factory() as con:
            stats = (
                con.execute(
                    """
                    SELECT
                        COUNT(*) AS n_rows,
                        COUNT(DISTINCT session_id) AS n_sessions,
                        COUNT(DISTINCT "full name") AS n_players,
                        COUNT(DISTINCT "group name") AS n_groups
                    FROM kinexon_positions
                    """
                )
                .df()
                .iloc[0]
            )
            preview_md = (
                con.execute(
                    "SELECT * FROM kinexon_positions WHERE session_id = ? LIMIT 10",
                    [session_id],
                )
                .df()
                .to_markdown(index=False)
            )
    except Exception:
        stats = {"n_rows": 0, "n_sessions": 0, "n_players": 0, "n_groups": 0}
        preview_md = "*(no data)*"

    return {
        "n_rows": int(stats["n_rows"]),
        "n_distinct_sessions": int(stats["n_sessions"]),
        "n_existing_sessions_before": 0,
        "n_sessions_in_fixtures": 1,
        "n_sessions_fetched_this_run": int(fetched),
        "n_players_total": int(stats["n_players"]),
        "n_groups_total": int(stats["n_groups"]),
        "preview_md": preview_md,
    }
