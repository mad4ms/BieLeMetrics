import logging
import os
from typing import Optional

from dotenv import load_dotenv
import pandas as pd

from kinexon_handball_api.handball import HandballAPI
from src.fetcher_kinexon.fetch_session_id_for_fixtures import (
    fetch_session_ids_for_fixtures,
)
from src.fetcher_kinexon.fetch_events_for_session import (
    fetch_detected_events_for_session,
)

from src.fetcher_kinexon.fetch_teams import fetch_teams_for_season
from src.fetcher_kinexon.fetch_positions_for_fixture import (
    fetch_positions_for_fixture,
)


def get_teams_for_season(
    api: HandballAPI,
    season_year: str = "2024-25",
) -> pd.DataFrame:
    """
    Fetch Kinexon teams for a given season.

    Args:
        api (HandballAPI): An instance of the HandballAPI.

    Returns:
        pd.DataFrame: DataFrame with team information.
    """

    # check if season year is provided and in format YYYY-YY,
    # if only YYYY is given, convert to YYYY-YY
    if len(season_year) == 4 and season_year.isdigit():
        season_year = (
            f"{season_year}-{str(int(season_year[-2:]) + 1).zfill(2)}"
        )

    teams_list = fetch_teams_for_season(api=api, season_year=season_year)
    df_teams = pd.DataFrame(teams_list)
    return df_teams


def get_sessions_for_team(
    api: HandballAPI,
    team_id: str,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Fetch Kinexon sessions for a given team ID within an optional date range.
    """
    if start_date is None:
        start_date = pd.Timestamp("1970-01-01", tz="UTC")

    if end_date is None:
        end_date = pd.Timestamp("2100-01-01", tz="UTC")

    sessions = (
        api.get_sessions_for_team(
            team_id=int(team_id), start=start_date, end=end_date
        )
        or []
    )
    sessions_dict = [s.to_dict() for s in sessions]
    df_sessions = pd.DataFrame(sessions_dict)
    return df_sessions


def get_detected_events_for_fixture(
    api: HandballAPI,
    session_id: int,
) -> pd.DataFrame:
    """Fetch detected events for a given Kinexon session ID."""

    logging.info(
        "Fetching detected events for Kinexon session ID %d.", session_id
    )
    df_events = fetch_detected_events_for_session(
        api=api,
        session_id=session_id,
    )
    logging.info(
        "Fetched %d detected events for Kinexon session ID %d.",
        len(df_events),
        session_id,
    )
    return df_events


def get_positions_for_session(
    api: HandballAPI,
    session_id: int,
) -> pd.DataFrame:
    """Fetch positioning data for a given Kinexon session ID."""

    logging.info(
        "Fetching positioning data for Kinexon session ID %d.", session_id
    )
    df_positions = fetch_positions_for_fixture(
        api=api,
        session_id=session_id,
    )
    logging.info(
        "Fetched %d positioning data points for Kinexon session ID %d.",
        len(df_positions),
        session_id,
    )
    return df_positions
