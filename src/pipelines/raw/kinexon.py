"""Raw Kinexon pipeline helpers.

Pure business-logic wrappers around fetcher functions.
No orchestration/persistence concerns should live here.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from kinexon_handball_api.handball import HandballAPI

from src.fetcher_kinexon.fetch_events_for_session import (
    fetch_detected_events_for_session,
)
from src.fetcher_kinexon.fetch_positions_for_fixture import fetch_positions_for_fixture
from src.fetcher_kinexon.fetch_teams import fetch_teams_for_season

logger = logging.getLogger(__name__)


def _normalize_season_year(season_year: str) -> str:
    """Normalize season format to ``YYYY-YY``.

    Examples:
    - "2025" -> "2025-26"
    - "2025-26" -> "2025-26"
    """
    if len(season_year) == 4 and season_year.isdigit():
        year = int(season_year)
        return f"{year}-{str((year + 1) % 100).zfill(2)}"
    return season_year


def get_teams_for_season(
    api: HandballAPI,
    season_year: str = "2024-25",
) -> pd.DataFrame:
    """Fetch Kinexon teams for a season."""
    season = _normalize_season_year(season_year)
    teams_list = fetch_teams_for_season(api=api, season_year=season)
    df_teams = pd.DataFrame(teams_list)
    logger.info("Fetched %d Kinexon teams for season '%s'.", len(df_teams), season)
    return df_teams


def get_sessions_for_team(
    api: HandballAPI,
    team_id: str,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Fetch Kinexon sessions for a team within an optional date range."""
    if start_date is None:
        start_date = pd.Timestamp("1970-01-01", tz="UTC")
    if end_date is None:
        end_date = pd.Timestamp("2100-01-01", tz="UTC")

    sessions = (
        api.get_sessions_for_team(
            team_id=int(team_id),
            start=start_date,
            end=end_date,
        )
        or []
    )

    sessions_dict = [s.to_dict() for s in sessions]
    df_sessions = pd.DataFrame(sessions_dict)
    logger.info("Fetched %d sessions for team_id=%s.", len(df_sessions), team_id)
    return df_sessions


def get_detected_events_for_fixture(
    api: HandballAPI,
    session_id: int,
) -> pd.DataFrame:
    """Fetch detected events for a Kinexon session id."""
    logger.info("Fetching detected events for session_id=%d.", session_id)
    df_events = fetch_detected_events_for_session(api=api, session_id=session_id)
    logger.info(
        "Fetched %d detected events for session_id=%d.", len(df_events), session_id
    )
    return df_events


def get_positions_for_session(
    api: HandballAPI,
    session_id: int,
) -> pd.DataFrame:
    """Fetch positional data for a Kinexon session id."""
    logger.info("Fetching positions for session_id=%d.", session_id)
    df_positions = fetch_positions_for_fixture(api=api, session_id=session_id)
    logger.info(
        "Fetched %d positions for session_id=%d.", len(df_positions), session_id
    )
    return df_positions
