"""Business logic helpers for retrieving competition and season IDs, teams, and fixtures
from the Sportradar API for the '1. Handball-Bundesliga' (or other competitions).

This module provides pure Python functions that wrap lower-level fetchers
from `src.fetcher_sportradar` to obtain unique identifiers and metadata required for
downstream ETL processes. These IDs are essential for querying fixtures,
events, and metadata for a specific competition and season.

These functions are intended to be called from orchestration layers (e.g., Dagster assets)
and should not contain any orchestration or persistence logic.

Typical usage:
    api = HandballAPI(...)
    competition_id = get_competition_id(api, '1. Handball-Bundesliga')
    season_id = get_season_id(api, competition_id, 2025)  # for season 2025-26
    teams_df = get_teams_for_season(api, season_id)
    fixtures_df = get_fixtures_for_season(api, season_id)
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
import pandas as pd

from sportradar_datacore_api.handball import HandballAPI
from src.fetcher_sportradar.fetch_competition_id import fetch_competition_id
from src.fetcher_sportradar.fetch_season_id import fetch_season_id
from src.fetcher_sportradar.fetch_teams import fetch_teams_by_season_id
from src.fetcher_sportradar.fetch_list_fixtures import (
    fetch_list_fixtures,
)
from src.fetcher_sportradar.fetch_fixture_events import (
    fetch_events_for_fixture,
)

from fetcher_sportradar.fetch_players import fetch_players_by_ids

logger = logging.getLogger(__name__)


def get_competition_id(
    api: HandballAPI, competition_name: str = "1. Handball-Bundesliga"
) -> Optional[str]:
    """
    Fetch the competition ID for '1. Handball-Bundesliga'.

    Args:
        api (HandballAPI): An authenticated Sportradar HandballAPI client.
        competition_name (str): The name of the competition to fetch the ID for.
        (default: '1. Handball-Bundesliga')

    Returns:
        Optional[str]: The competition ID if found, else None.

    Raises:
        Exception: If the API call fails or returns an invalid result.
    """
    try:
        competition_id = fetch_competition_id(
            api=api, competition_name=competition_name
        )
        if not competition_id:
            logger.error(
                "Competition ID not found for %s '%s'.",
                competition_name,
                competition_name,
            )
            return None
        logger.info("Fetched competition ID: %s", competition_id)
        return competition_id
    except Exception as exc:
        logger.exception("Failed to fetch competition ID. Error: %s", str(exc))
        raise


def get_season_id(
    api: HandballAPI, competition_id: str, season_year: int
) -> Optional[str]:
    """
    Fetch the season ID for a given competition and year.

    Args:
        api (HandballAPI): An authenticated Sportradar HandballAPI client.
        competition_id (str): The competition ID.
        season_year (int): The year of the season.

    Returns:
        Optional[str]: The season ID if found, else None.

    Raises:
        Exception: If the API call fails or returns an invalid result.
    """
    try:
        season_id = fetch_season_id(
            api=api, competition_id=competition_id, season_year=season_year
        )
        if not season_id:
            logger.error(
                "Season ID not found for competition '%s' and year '%s'.",
                competition_id,
                season_year,
            )
            return None
        logger.info("Fetched season ID: %s", season_id)
        return season_id
    except Exception as exc:
        logger.exception("Failed to fetch season ID. Error: %s", str(exc))
        raise


def get_teams_for_season(api: HandballAPI, season_id: str) -> pd.DataFrame:
    """
    Fetch all teams for a given season from Sportradar.

    Args:
        api (HandballAPI): An authenticated Sportradar HandballAPI client.
        season_id (str): The season ID.

    Returns:
        pd.DataFrame: DataFrame containing team information.

    Raises:
        Exception: If the API call fails or returns an invalid result.
    """
    try:
        df_teams = fetch_teams_by_season_id(api=api, season_id=season_id)
        if df_teams.empty:
            logger.error("No teams found for season ID '%s'.", season_id)
            return pd.DataFrame()
        logger.info(
            "Fetched %d teams for season ID '%s'.", len(df_teams), season_id
        )
        return df_teams
    except Exception as exc:
        logger.exception("Failed to fetch teams. Error: %s", str(exc))
        raise


def get_fixtures_for_season(api: HandballAPI, season_id: str) -> pd.DataFrame:
    """
    Fetch all fixtures for a given season from Sportradar.

    Args:
        api (HandballAPI): An authenticated Sportradar HandballAPI client.
        season_id (str): The season ID.

    Returns:
        pd.DataFrame: DataFrame containing fixture information.

    Raises:
        Exception: If the API call fails or returns an invalid result.
    """
    try:
        df_fixtures = pd.DataFrame(
            fetch_list_fixtures(api=api, season_id=season_id)
        )
        if df_fixtures.empty:
            logger.error("No fixtures found for season ID '%s'.", season_id)
            return pd.DataFrame()
        logger.info(
            "Fetched %d fixtures for season ID '%s'.",
            len(df_fixtures),
            season_id,
        )
        return df_fixtures
    except Exception as exc:
        logger.exception("Failed to fetch fixtures. Error: %s", str(exc))
        raise


def get_fixture_events(api: HandballAPI, fixture_id: str) -> pd.DataFrame:
    """
    Fetch all events for a given fixture from Sportradar.

    Args:
        api (HandballAPI): An authenticated Sportradar HandballAPI client.
        fixture_id (str): The fixture ID.

    Returns:
        pd.DataFrame: DataFrame containing event information.

    Raises:
        Exception: If the API call fails or returns an invalid result.
    """
    try:
        df_events = pd.DataFrame(
            fetch_events_for_fixture(api=api, fixture_id=fixture_id)
        )
        if df_events.empty:
            logger.error("No events found for fixture ID '%s'.", fixture_id)
            return pd.DataFrame()
        logger.info(
            "Fetched %d events for fixture ID '%s'.",
            len(df_events),
            fixture_id,
        )
        return df_events
    except Exception as exc:
        logger.exception("Failed to fetch events. Error: %s", str(exc))
        raise


def get_players_for_fixture(
    api: HandballAPI, df_match_events: pd.DataFrame
) -> pd.DataFrame:
    """
    Fetch all players involved in a given fixture based on match events.

    Args:
        api (HandballAPI): An authenticated Sportradar HandballAPI client.
        df_match_events (pd.DataFrame): DataFrame containing match events.

    Returns:
        pd.DataFrame: DataFrame containing player information.
    Raises:
        Exception: If the API call fails or returns an invalid result.
    try:
        person_ids = extract_person_ids_from_setup(df_match_events)
        df_players = fetch_players_by_ids(api=api, person_ids=person_ids)
        return df_players
    except Exception as exc:
        logger.exception("Failed to fetch players. Error: %s", str(exc))
        raise
    """
    try:
        from src.fetcher_sportradar.fetch_players import (
            extract_person_ids_from_setup,
        )

        person_ids = extract_person_ids_from_setup(df_match_events)
        df_players = fetch_players_by_ids(api=api, person_ids=person_ids)
        # replace "." with "_" in column names
        df_players.columns = [
            col.replace(".", "_") for col in df_players.columns
        ]
        if df_players.empty:
            logger.error("No players found for the given fixture events.")
            return pd.DataFrame()
        logger.info(
            "Fetched %d players for the given fixture events.",
            len(df_players),
        )
        return df_players
    except Exception as exc:
        logger.exception("Failed to fetch players. Error: %s", str(exc))
        raise


if __name__ == "__main__":
    load_dotenv()

    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)

    # Example usage (requires valid API credentials in environment variables)
    api = HandballAPI(
        base_url=os.getenv("BASE_URL", ""),
        auth_url=os.getenv("AUTH_URL", ""),
        client_id=os.getenv("CLIENT_ID", ""),
        client_secret=os.getenv("CLIENT_SECRET", ""),
        org_id=os.getenv("CLIENT_ORGANIZATION_ID"),
        scopes=["read:organization"],
        sport="handball",
    )
    competition_id = get_competition_id(api)
    if competition_id:
        season_id = get_season_id(api, competition_id, season_year=2025)
        if season_id:
            teams_df = get_teams_for_season(api, season_id)
            fixtures_df = get_fixtures_for_season(api, season_id)
            print(teams_df)
            print(fixtures_df)
            # sort by roundNumber
            fixtures_df = fixtures_df.sort_values(by="roundNumber")
            for _, fixture in fixtures_df.iterrows():
                print(fixture)
                fixture_id = fixture.get("fixtureId")
                if fixture_id:
                    events_df = get_fixture_events(api, fixture_id)
                    print(f"Events for fixture {fixture_id}:")
                    print(
                        events_df
                    )  # if empty, check if event is in the future
                    players_df = get_players_for_fixture(api, events_df)
                    print(f"Players for fixture {fixture_id}:")
                    print(players_df)
                    break  # Remove this break to fetch events for all fixtures
