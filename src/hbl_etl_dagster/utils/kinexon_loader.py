import pandas as pd
from typing import Dict, Any, List
import logging
from src.fetcher_kinexon.fetch_positions_for_fixture import (
    fetch_positions_for_fixture,
    fetch_positions_for_fixtures_multithreaded,
)


def load_missing_positions(
    con,
    fixtures: pd.DataFrame,
    api_kinexon,
    max_batch_size: int = 9,
    logger: logging.Logger | None = None,
    skip_if_exists: bool = True,
) -> Dict[str, Any]:
    """
    Load Kinexon position data for sessions referenced in `fixtures` but not yet
    present in `kinexon_positions`.

    Guarantees:
    - No duplicate inserts per batch
    - Table is created exactly once if missing
    - df_events rows are never dropped
    """

    # ------------------------------------------------------------------
    # 1. Determine which session_ids already exist in the DB
    # ------------------------------------------------------------------
    try:
        existing_sessions = con.execute(
            """
            SELECT DISTINCT session_id
            FROM kinexon_positions
            WHERE session_id IS NOT NULL
            """
        ).df()["session_id"]
        session_ids_in_positions = set(existing_sessions)
    except Exception:
        # Table does not exist or is inaccessible
        session_ids_in_positions = set()

    # All session_ids referenced by fixtures
    session_ids_in_fixtures = set(fixtures["session_id"].dropna().unique())

    # Decide what to fetch this run
    session_ids_to_fetch = (
        sorted(session_ids_in_fixtures - session_ids_in_positions)
        if skip_if_exists
        else sorted(session_ids_in_fixtures)
    )

    if logger:
        logger.info(
            f"{len(session_ids_in_positions)} session_ids already in kinexon_positions "
            f"({len(session_ids_in_fixtures)} referenced in fixtures)."
        )

    if not session_ids_to_fetch:
        return {
            "n_rows": 0,
            "n_distinct_sessions": len(session_ids_in_positions),
            "n_existing_sessions_before": len(session_ids_in_positions),
            "n_sessions_in_fixtures": len(session_ids_in_fixtures),
            "n_sessions_fetched_this_run": 0,
            "n_players_total": 0,
            "n_groups_total": 0,
            "preview_md": "*(no new data fetched)*",
        }

    # ------------------------------------------------------------------
    # 2. Prepare batch lists
    # ------------------------------------------------------------------
    session_batches = [
        session_ids_to_fetch[i : i + max_batch_size]
        for i in range(0, len(session_ids_to_fetch), max_batch_size)
    ]

    if logger:
        logger.info(
            f"Fetching positions for {len(session_ids_to_fetch)} sessions "
            f"in {len(session_batches)} batches."
        )

    # session_id -> fixture_id mapping (stable, single-valued)
    session_to_fixture = (
        fixtures.dropna(subset=["session_id"])
        .drop_duplicates("session_id")
        .set_index("session_id")["fixture_id"]
        .to_dict()
    )

    table_initialized = False

    # ------------------------------------------------------------------
    # 3. Fetch + insert batches
    # ------------------------------------------------------------------
    for session_id_batch in session_batches:
        df_positions = fetch_positions_for_fixtures_multithreaded(
            api_kinexon,
            session_id_batch,
            max_workers=max_batch_size,
        )

        if df_positions.empty:
            continue

        if "session_id" not in df_positions.columns:
            if logger:
                logger.warning(
                    "Fetched positions missing session_id column; batch skipped."
                )
            continue

        # Attach fixture_id for downstream joins
        df_positions["fixtureId"] = df_positions["session_id"].map(
            session_to_fixture
        )

        if logger and df_positions["fixtureId"].isna().any():
            logger.warning(
                "Some session_ids have no fixture mapping; fixtureId is NULL."
            )

        con.register("df_positions", df_positions)

        # Create table once; afterwards only INSERT
        if not table_initialized:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS kinexon_positions AS
                SELECT * FROM df_positions
                """
            )
            table_initialized = True
        else:
            con.execute(
                """
                INSERT INTO kinexon_positions
                SELECT * FROM df_positions
                """
            )

        con.unregister("df_positions")

        if logger:
            logger.info(
                f"Inserted {len(df_positions)} rows "
                f"for sessions {session_id_batch}"
            )

    # ------------------------------------------------------------------
    # 4. Collect lightweight stats
    # ------------------------------------------------------------------
    try:
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
            con.execute("SELECT * FROM kinexon_positions LIMIT 10")
            .df()
            .to_markdown(index=False)
        )

    except Exception:
        stats = {"n_rows": 0, "n_sessions": 0, "n_players": 0, "n_groups": 0}
        preview_md = "*(no data)*"

    # ------------------------------------------------------------------
    # 5. Return structured run stats
    # ------------------------------------------------------------------
    return {
        "n_rows": int(stats["n_rows"]),
        "n_distinct_sessions": int(stats["n_sessions"]),
        "n_existing_sessions_before": len(session_ids_in_positions),
        "n_sessions_in_fixtures": len(session_ids_in_fixtures),
        "n_sessions_fetched_this_run": len(session_ids_to_fetch),
        "n_players_total": int(stats["n_players"]),
        "n_groups_total": int(stats["n_groups"]),
        "preview_md": preview_md,
    }
