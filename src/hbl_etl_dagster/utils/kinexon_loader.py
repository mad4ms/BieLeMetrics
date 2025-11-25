import pandas as pd
from typing import Dict, Any, List
import logging
from src.events_kinexon.fetch_positions_for_fixture import (
    fetch_positions_for_fixture,
)


def load_missing_positions(
    con,
    fixtures: pd.DataFrame,
    api_kinexon,
    max_batch_size: int = 9,
    logger: logging.Logger = None,
    skip_if_exists: bool = True,
) -> Dict[str, Any]:
    """
    Load positions for missing session_ids into kinexon_positions.
    Returns stats (n_rows, n_sessions_total, n_fetched, ...).
    """
    # Session IDs already present in positions
    try:
        df_sessions_in_positions = con.execute(
            """
            SELECT DISTINCT session_id
            FROM kinexon_positions
            WHERE session_id IS NOT NULL
            """
        ).df()
        session_ids_in_positions = set(
            df_sessions_in_positions["session_id"].dropna().unique().tolist()
        )
    except Exception:
        session_ids_in_positions = set()

    # Session IDs in fixtures (Kinexon-linked fixtures)
    session_ids_in_fixtures = set(
        fixtures["session_id"].dropna().unique().tolist()
    )

    if skip_if_exists:
        session_ids_to_fetch = sorted(
            session_ids_in_fixtures - session_ids_in_positions
        )
    else:
        session_ids_to_fetch = sorted(session_ids_in_fixtures)

    if logger:
        logger.info(
            f"Already have {len(session_ids_in_positions)} session IDs in "
            "kinexon_positions table."
        )
        percent_present = (
            len(session_ids_in_positions) / len(session_ids_in_fixtures) * 100
            if session_ids_in_fixtures
            else 0.0
        )
        logger.info(
            f"{percent_present:.2f}% of session IDs in fixtures are already present "
            "in kinexon_positions table."
        )

    # Prepare batches
    split_session_id_lists = [
        session_ids_to_fetch[i : i + max_batch_size]
        for i in range(0, len(session_ids_to_fetch), max_batch_size)
    ]

    if logger:
        logger.info(
            f"Fetching positions for {len(session_ids_to_fetch)} sessions "
            f"in {len(split_session_id_lists)} batches."
        )

    # Reverse mapping: session_id -> fixtureId from fixtures
    session_to_fixture = (
        fixtures.dropna(subset=["session_id"])
        .drop_duplicates(subset=["session_id"])
        .set_index("session_id")["fixtureId"]
        .to_dict()
    )

    first_insert_done = False

    for session_id_list in split_session_id_lists:
        if not session_id_list:
            continue

        position_frames = []
        for session_id in session_id_list:
            df_positions_single = fetch_positions_for_fixture(
                api_kinexon, session_id
            )
            if df_positions_single.empty:
                continue
            position_frames.append(df_positions_single)

        if not position_frames:
            continue

        df_positions = pd.concat(position_frames, ignore_index=True)

        # Map fixtureId back
        reverse_batch = {
            sid: session_to_fixture.get(sid)
            for sid in session_id_list
            if sid in session_to_fixture
        }
        df_positions["fixture_id"] = df_positions["session_id"].map(
            reverse_batch
        )

        # Create table on first batch (if it doesn't exist), then append
        # We use CREATE TABLE IF NOT EXISTS for the first time we touch the DB in this run
        # But we also need to handle the case where the table existed before this function ran.
        # The original code had a flag `first_insert_done`.
        # If the table already exists (checked via try-except block earlier), we should just INSERT.

        # Actually, simpler: always try CREATE TABLE IF NOT EXISTS first?
        # Or just check if table exists.
        # The original code logic:
        # if not first_insert_done: CREATE ... AS SELECT ...; first_insert_done = True
        # else: INSERT ...

        # But if the table existed BEFORE the loop, `first_insert_done` should effectively be treated as True?
        # No, `CREATE TABLE IF NOT EXISTS ... AS SELECT` will create and populate if not exists.
        # If it exists, it does nothing. So we can't use it for appending.

        # Better approach:
        # Always try to create the table structure if it doesn't exist (empty).
        # Then insert.

        # However, to preserve original behavior's intent (which was a bit mixed):
        # Let's stick to:
        # 1. Check if table exists.
        # 2. If not, create with first batch.
        # 3. If yes, insert.

        table_exists = True
        try:
            con.execute("SELECT 1 FROM kinexon_positions LIMIT 0")
        except:
            table_exists = False

        if not table_exists:
            con.execute(
                """
                CREATE TABLE kinexon_positions AS
                SELECT * FROM df_positions
                """
            )
            table_exists = True  # Now it exists
        else:
            con.execute(
                """
                INSERT INTO kinexon_positions
                SELECT * FROM df_positions
                """
            )

    # Collect lightweight stats
    try:
        stats_df = con.execute(
            """
            SELECT
                COUNT(*) AS n_rows,
                COUNT(DISTINCT session_id) AS n_sessions,
                COUNT(DISTINCT "full name") AS n_players,
                COUNT(DISTINCT "group name") AS n_groups
            FROM kinexon_positions
            """
        ).df()
        n_rows = int(stats_df.loc[0, "n_rows"])
        n_sessions_total = int(stats_df.loc[0, "n_sessions"])
        n_players_total = int(stats_df.loc[0, "n_players"])
        n_groups_total = int(stats_df.loc[0, "n_groups"])
        preview_df = con.execute(
            "SELECT * FROM kinexon_positions LIMIT 10"
        ).df()
        preview_md = preview_df.to_markdown(index=False)
    except Exception:
        n_rows = 0
        n_sessions_total = 0
        n_players_total = 0
        n_groups_total = 0
        preview_md = "*(no data)*"

    return {
        "n_rows": n_rows,
        "n_distinct_sessions": n_sessions_total,
        "n_existing_sessions_before": len(session_ids_in_positions),
        "n_sessions_in_fixtures": len(session_ids_in_fixtures),
        "n_sessions_fetched_this_run": len(session_ids_to_fetch),
        "n_players_total": n_players_total,
        "n_groups_total": n_groups_total,
        "preview_md": preview_md,
    }
