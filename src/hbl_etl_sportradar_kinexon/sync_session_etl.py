"""
Sync Sportradar fixtures to Kinexon sessions and update DuckDB.

- Reads fixtures and teams from DuckDB
- Uses fuzzy name matching to map home team to Kinexon team id
- Fetches same-day sessions from Kinexon
- Matches sessions by group_names containing the team name
- Updates fixtures.session_id for matched fixtures
- Prints per-fixture ✓/✖ lines and a summary
- Writes a simple run log table: fixtures_session_sync_log
"""

import os
import datetime
import logging
import difflib
from typing import List, Dict, Tuple

import pandas as pd
from .config import get_duckdb_connection, get_api_kinexon

# ---------------------------------------------------------------------
# Config / Logging
# ---------------------------------------------------------------------

LOG_LEVEL = os.getenv("SYNC_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("sync_sessions_etl")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def find_best_team_match(
    target_name: str, team_list: List[Dict], threshold: float = 0.8
) -> Tuple[int, str, float]:
    best_match = None
    best_score = 0.0
    best_id = None
    for team in team_list:
        team_name = team["name"]
        similarity = difflib.SequenceMatcher(
            None, target_name.lower(), team_name.lower()
        ).ratio()
        if similarity > best_score and similarity >= threshold:
            best_score = similarity
            best_match = team_name
            best_id = team["id"]
    return best_id, best_match, best_score


def _normalize_home_name(name_team_home: str) -> str:
    return (
        "HSV Hamburg"
        if name_team_home == "Handball Sport Verein Hamburg"
        else name_team_home
    )


# ---------------------------------------------------------------------
# Core ETL
# ---------------------------------------------------------------------


def sync_fixture_sessions(
    con=None,
    api=None,
    season_label: str = "2024-25",
    similarity_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    For each Sportradar fixture, try to find its Kinexon session_id and update DuckDB.
    Returns a DataFrame with the per-fixture sync result (success/failure).
    """
    # IMPORTANT: default to Kinexon API here
    api = api or get_api_kinexon()
    con = con or get_duckdb_connection()

    # Kinexon team IDs
    ids_team = api.fetch_team_ids(season_label)

    # Load fixtures
    df_fixtures = con.execute("SELECT * FROM fixtures").df()

    logger.info("Processing %d fixtures…", len(df_fixtures))

    results = []
    failures = []

    # Ensure fixtures has session_id
    con.execute(
        "ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS session_id BIGINT"
    )

    for _, row in df_fixtures.iterrows():
        fixture_id = row.get("fixtureId")
        name_local = row.get("nameLocal")
        sr_external_id = row.get("externalId")
        competitors = row.get("competitors")

        if competitors is None:
            competitors = []

        home_list = [comp for comp in competitors if comp.get("isHome")]
        if not home_list:
            logger.info("[%s] ✖ no home competitor", fixture_id)
            failures.append(
                {"fixtureId": fixture_id, "reason": "no_home_competitor"}
            )
            continue

        name_team_home = row.get("name_team_home") or home_list[0].get(
            "nameFullLocal"
        )
        if not name_team_home:
            logger.info("[%s] ✖ missing home team name", fixture_id)
            failures.append(
                {"fixtureId": fixture_id, "reason": "missing_home_team_name"}
            )
            continue

        name_team_home = _normalize_home_name(name_team_home)

        id_team_home, matched_name, similarity_score = find_best_team_match(
            name_team_home, ids_team, threshold=similarity_threshold
        )
        if id_team_home is None:
            logger.info(
                "[%s] %s ✖ team '%s' not found in ids_team (best similarity < %.2f)",
                fixture_id,
                name_local,
                name_team_home,
                similarity_threshold,
            )
            failures.append(
                {
                    "fixtureId": fixture_id,
                    "reason": "team_not_found_fuzzy",
                    "team": name_team_home,
                }
            )
            continue

        if matched_name != name_team_home:
            logger.info(
                "[%s] ≈ fuzzy matched '%s' → '%s' (similarity: %.2f)",
                fixture_id,
                name_team_home,
                matched_name,
                similarity_score,
            )

        start_local = pd.to_datetime(row.get("startTimeLocal"))
        if pd.isna(start_local):
            logger.info("[%s] ✖ startTimeLocal is NaT", fixture_id)
            failures.append(
                {"fixtureId": fixture_id, "reason": "no_start_time"}
            )
            continue

        date_game_start = start_local.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        date_game_end = date_game_start.date() + pd.Timedelta(hours=24)
        dt_start = datetime.datetime.fromisoformat(str(date_game_start))
        dt_end = datetime.datetime.fromisoformat(str(date_game_end))

        sessions = api.get_sessions_for_team(
            id_team_home, start=dt_start, end=dt_end
        )
        if len(sessions) == 0:
            logger.info(
                "[%s] ✖ no sessions on %s for '%s'",
                fixture_id,
                date_game_start.date(),
                name_team_home,
            )
            failures.append(
                {
                    "fixtureId": fixture_id,
                    "reason": "no_sessions",
                    "team": name_team_home,
                }
            )
            continue

        found = False
        for session in sessions:
            df_session = pd.DataFrame([session.to_dict()])

            # group_names may be list or string
            group_names = df_session.get("group_names")
            group_names = (
                group_names.values[0]
                if isinstance(group_names, pd.Series)
                else None
            )
            if isinstance(group_names, str):
                group_names = [group_names]
            if group_names is None:
                group_names = []

            if (name_team_home in group_names) or (
                matched_name in group_names
            ):
                # session id sometimes appears as 'session_id'
                sid_series = df_session.get("session_id")
                sid = (
                    sid_series.values[0]
                    if isinstance(sid_series, pd.Series)
                    else None
                )
                sid = int(sid) if sid is not None else None

                con.execute(
                    "UPDATE fixtures SET session_id = ? WHERE fixtureId = ?",
                    [sid, fixture_id],
                )

                results.append(
                    {
                        "fixtureId": fixture_id,
                        "nameLocal": name_local,
                        "homeTeam": name_team_home,
                        "matchedTeam": matched_name,
                        "teamId": id_team_home,
                        "session_id": sid,
                        "sr_external_id": sr_external_id,
                        "similarity_score": similarity_score,
                        "status": "ok",
                    }
                )
                logger.info(
                    "[%s] ✓ synced → session %s (%s)",
                    fixture_id,
                    sid,
                    matched_name,
                )
                found = True
                break

        if not found:
            logger.info(
                "[%s] ✖ sessions found but none matched group_names for '%s' or '%s'",
                fixture_id,
                name_team_home,
                matched_name,
            )
            failures.append(
                {
                    "fixtureId": fixture_id,
                    "reason": "no_matching_group",
                    "team": name_team_home,
                    "matched_team": matched_name,
                }
            )

    con.execute(
        """
        ALTER TABLE match_events
        ADD COLUMN IF NOT EXISTS session_id BIGINT
    """
    )

    # 2) backfill using fixtureId
    con.execute(
        """
        UPDATE match_events AS me
        SET session_id = f.session_id
        FROM fixtures f
        WHERE me.fixtureId = f.fixtureId
    """
    )

    # Summary stats
    total = len(df_fixtures)
    ok = len(results)
    fail = len(failures)
    pct = (ok / total * 100.0) if total else 0.0

    logger.info("")
    logger.info("=== Sync Summary ===")
    logger.info("Total fixtures:   %d", total)
    logger.info("Synced (matched): %d", ok)
    logger.info("Failed:           %d", fail)
    logger.info("Success rate:     %.1f%%", pct)

    # Build log frame
    df_log_ok = pd.DataFrame(results)
    df_log_fail = pd.DataFrame(failures)
    if not df_log_fail.empty:
        df_log_fail = df_log_fail.assign(status="fail")
    df_log = (
        pd.concat([df_log_ok, df_log_fail], ignore_index=True)
        if not df_log_ok.empty or not df_log_fail.empty
        else pd.DataFrame(columns=["fixtureId", "status"])
    )
    if not df_log.empty:
        df_log = df_log.assign(run_ts=pd.Timestamp.utcnow())

        # --- normalize df_log columns for DuckDB insert ---
    if not df_log.empty:
        # map homeTeam → team for success rows, ensure all expected columns exist
        if "team" not in df_log.columns:
            if "homeTeam" in df_log.columns:
                df_log["team"] = df_log["homeTeam"]
            else:
                df_log["team"] = None
        if "reason" not in df_log.columns:
            df_log["reason"] = None
        if "matched_team" not in df_log.columns:
            if "matchedTeam" in df_log.columns:
                df_log["matched_team"] = df_log["matchedTeam"]
            else:
                df_log["matched_team"] = None
        if "similarity_score" not in df_log.columns:
            df_log["similarity_score"] = None
        if "session_id" not in df_log.columns:
            df_log["session_id"] = None
        if "fixtureId" not in df_log.columns:
            df_log["fixtureId"] = None

    # drop table fixtures_session_sync_log
    con.execute("DROP TABLE IF EXISTS fixtures_session_sync_log")

    # Create log table with the final schema and insert explicitly named columns
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fixtures_session_sync_log (
            run_ts TIMESTAMP,
            fixtureId text,
            status VARCHAR,
            reason VARCHAR,
            session_id BIGINT,
            team VARCHAR,
            matched_team VARCHAR,
            similarity_score DOUBLE
        )
    """
    )

    if not df_log.empty:
        con.execute(
            """
            INSERT INTO fixtures_session_sync_log
                (run_ts, fixtureId, status, reason, session_id, team, matched_team, similarity_score)
            SELECT
                run_ts,
                fixtureId,
                status,
                COALESCE(reason, NULL),
                COALESCE(session_id, NULL),
                COALESCE(team, NULL),
                COALESCE(matched_team, NULL),
                COALESCE(similarity_score, NULL)
            FROM df_log
        """
        )

    return df_log


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    con = get_duckdb_connection("./data/mydb2024-25.duckdb")
    api = get_api_kinexon()
    sync_fixture_sessions(con=con, api=api)
