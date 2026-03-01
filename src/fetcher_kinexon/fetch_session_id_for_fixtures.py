"""fetch_session_id_for_fixtures.py: Module to fetch session IDs for fixtures from Kinexon API."""

import datetime
import difflib
import logging
from typing import Dict, List, Optional, Tuple


import pandas as pd
from kinexon_handball_api.handball import HandballAPI
from fetcher_kinexon import fetch_teams_for_season


def find_best_team_match(
    target_name: str, team_list: List[Dict], threshold: float = 0.8
) -> Tuple[Optional[int], Optional[str], float]:
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


def _rename_teams_on_mismatch(name_team_home: str) -> str:
    if "Minden" in name_team_home:
        return "TSV GWD Minden"
    return (
        "HSV Hamburg"
        if name_team_home == "Handball Sport Verein Hamburg"
        else name_team_home
    )


def fetch_session_ids_for_fixtures(
    api: HandballAPI,
    df_fixtures: pd.DataFrame,
    similarity_threshold: float = 0.8,
    season_year: str = "2024-25",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, int]:
    """
    Fetch session IDs for a list of fixture IDs from the Kinexon Handball API.

    Args:
        api (HandballAPI): An instance of the HandballAPI.
        fixture_ids (List[str]): A list of fixture IDs.

    Returns:
        Dict[str, str]: A dictionary mapping fixture IDs to session IDs.
    """
    session_ids: Dict[str, int] = {}
    if logger:
        logger.info("Fetching session IDs for %d fixtures.", len(df_fixtures))

    teams_in_season = fetch_teams_for_season(api, season_year=season_year)

    for _, row in df_fixtures.iterrows():
        fixture_id = row.get("fixture_id")
        competitors = row.get("competitors")

        if competitors is None:
            competitors = []
        col = "is_home" if "is_home" in competitors[0] else "isHome"
        home_list = [comp for comp in competitors if comp.get(col)]
        if not home_list:
            if logger:
                logger.error(
                    "[%s] ✖ no home competitor found in competitors",
                    fixture_id,
                )
            continue

        name_team_home = row.get("name_team_home") or home_list[0].get("nameFullLocal")
        if not name_team_home:
            if logger:
                logger.error(
                    "[%s] ✖ no name_team_home found for home competitor",
                    fixture_id,
                )
            continue

        name_team_home = _rename_teams_on_mismatch(name_team_home)

        id_team_home, matched_name, similarity_score = find_best_team_match(
            name_team_home, teams_in_season, threshold=similarity_threshold
        )

        if id_team_home is None:
            if logger:
                logger.info(
                    "[%s] ✖ team '%s' not found in ids_team (best similarity < %.2f)",
                    fixture_id,
                    name_team_home,
                    similarity_threshold,
                )

            continue

        if matched_name != name_team_home:
            if logger:
                logger.info(
                    "[%s] ≈ fuzzy matched '%s' → '%s' (similarity: %.2f)",
                    fixture_id,
                    name_team_home,
                    matched_name,
                    similarity_score,
                )
        col = "start_time_local" if "start_time_local" in row else "startTimeLocal"
        start_local = pd.to_datetime(row.get(col))
        if pd.isna(start_local):
            if logger:
                logger.info("[%s] ✖ start_time_local is NaT", fixture_id)
            continue
        # check if in future
        if start_local > pd.Timestamp.now(tz=start_local.tz):
            if logger:
                logger.debug(
                    "[%s] ✖ fixture start_time_local %s is in the future",
                    fixture_id,
                    start_local,
                )
            continue
        date_game_start = start_local.replace(hour=0, minute=0, second=0, microsecond=0)
        date_game_end = date_game_start.date() + pd.Timedelta(hours=24)
        dt_start = datetime.datetime.fromisoformat(str(date_game_start))
        dt_end = datetime.datetime.fromisoformat(str(date_game_end))
        if logger:
            logger.debug(
                "Fetching sessions for team ID %s (Name: %s) at %s to %s",
                id_team_home,
                name_team_home,
                dt_start,
                dt_end,
            )
        sessions = api.get_sessions_for_team(id_team_home, start=dt_start, end=dt_end)
        if len(sessions) == 0:
            if logger:
                logger.info(
                    "[%s] ✖ no sessions on %s for '%s'",
                    fixture_id,
                    date_game_start.date(),
                    name_team_home,
                )

            continue

        found = False
        for session in sessions:
            df_session = pd.DataFrame([session.to_dict()])

            # group_names may be list or string
            group_names = df_session.get("group_names")
            group_names = (
                group_names.values[0] if isinstance(group_names, pd.Series) else None
            )
            if isinstance(group_names, str):
                group_names = [group_names]
            if group_names is None:
                group_names = []

            if (name_team_home in group_names) or (matched_name in group_names):
                # session id sometimes appears as 'session_id'
                sid_series = df_session.get("session_id")
                sid_raw = (
                    sid_series.values[0] if isinstance(sid_series, pd.Series) else None
                )
                sid = int(sid_raw) if sid_raw is not None else None
                found = True
                if sid is not None and isinstance(fixture_id, str):
                    session_ids[fixture_id] = sid

        if not found:
            if logger:
                logger.info(
                    "[%s] ✖ no matching session found for team '%s' on %s",
                    fixture_id,
                    name_team_home,
                    date_game_start.date(),
                )

    return session_ids
