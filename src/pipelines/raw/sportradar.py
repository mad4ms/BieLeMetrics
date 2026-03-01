"""Raw Sportradar pipeline helpers.

Pure business-logic wrappers around fetcher functions.
No orchestration/persistence concerns should live here.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sportradar_datacore_api.handball import HandballAPI

from src.fetcher_sportradar.fetch_competition_id import fetch_competition_id
from src.fetcher_sportradar.fetch_fixture_events import fetch_events_for_fixture
from src.fetcher_sportradar.fetch_list_fixtures import fetch_list_fixtures
from src.fetcher_sportradar.fetch_players import (
    extract_person_ids_from_setup,
    fetch_players_by_ids,
)
from src.fetcher_sportradar.fetch_season_id import fetch_season_id
from src.fetcher_sportradar.fetch_teams import fetch_teams_by_season_id

logger = logging.getLogger(__name__)


def get_competition_id(
    api: HandballAPI, competition_name: str = "1. Handball-Bundesliga"
) -> Optional[str]:
    """Return competition id for a competition name or ``None`` if not found."""
    try:
        competition_id = fetch_competition_id(
            api=api,
            competition_name=competition_name,
        )
        if not competition_id:
            logger.error("Competition ID not found for '%s'.", competition_name)
            return None

        logger.info("Fetched competition ID: %s", competition_id)
        return competition_id
    except Exception:
        logger.exception("Failed to fetch competition ID.")
        raise


def get_season_id(
    api: HandballAPI,
    competition_id: str,
    season_year: int,
) -> Optional[str]:
    """Return season id for ``competition_id``/``season_year`` or ``None``."""
    try:
        season_id = fetch_season_id(
            api=api,
            competition_id=competition_id,
            season_year=season_year,
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
    except Exception:
        logger.exception("Failed to fetch season ID.")
        raise


def get_teams_for_season(api: HandballAPI, season_id: str) -> pd.DataFrame:
    """Fetch teams for a season."""
    try:
        df_teams = fetch_teams_by_season_id(api=api, season_id=season_id)
        if df_teams.empty:
            logger.error("No teams found for season ID '%s'.", season_id)
            return pd.DataFrame()

        logger.info("Fetched %d teams for season ID '%s'.", len(df_teams), season_id)
        return df_teams
    except Exception:
        logger.exception("Failed to fetch teams.")
        raise


def get_fixtures_for_season(api: HandballAPI, season_id: str) -> pd.DataFrame:
    """Fetch fixtures for a season."""
    try:
        df_fixtures = pd.DataFrame(fetch_list_fixtures(api=api, season_id=season_id))
        if df_fixtures.empty:
            logger.error("No fixtures found for season ID '%s'.", season_id)
            return pd.DataFrame()

        logger.info(
            "Fetched %d fixtures for season ID '%s'.",
            len(df_fixtures),
            season_id,
        )
        return df_fixtures
    except Exception:
        logger.exception("Failed to fetch fixtures.")
        raise


def get_fixture_events(api: HandballAPI, fixture_id: str) -> pd.DataFrame:
    """Fetch events for a fixture."""
    try:
        df_events = pd.DataFrame(
            fetch_events_for_fixture(api=api, fixture_id=fixture_id)
        )
        if df_events.empty:
            logger.error("No events found for fixture ID '%s'.", fixture_id)
            return pd.DataFrame()

        logger.info(
            "Fetched %d events for fixture ID '%s'.", len(df_events), fixture_id
        )
        return df_events
    except Exception:
        logger.exception("Failed to fetch fixture events.")
        raise


def get_players_for_fixture(
    api: HandballAPI,
    df_match_events: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch players involved in fixture events using person ids from setup rows."""
    try:
        person_ids = extract_person_ids_from_setup(df_match_events)
        df_players = fetch_players_by_ids(api=api, person_ids=person_ids)
        if df_players.empty:
            logger.error("No players found for the given fixture events.")
            return pd.DataFrame()

        # Normalize column naming for downstream consistency
        df_players.columns = [col.replace(".", "_") for col in df_players.columns]

        logger.info("Fetched %d players for fixture events.", len(df_players))
        return df_players
    except Exception:
        logger.exception("Failed to fetch players for fixture.")
        raise


if __name__ == "__main__":
    from dotenv import load_dotenv
    from src.hbl_etl_dagster.utils.api_helper import get_api_sportradar

    load_dotenv()  # Load environment variables from .env file
    # Example usage (for testing purposes)
    api = get_api_sportradar()
    competition_id = get_competition_id(api)
    if competition_id:
        season_id = get_season_id(api, competition_id, season_year=2023)
        if season_id:
            teams_df = get_teams_for_season(api, season_id)
            fixtures_df = get_fixtures_for_season(api, season_id)
            print(teams_df.head())
            print(fixtures_df.head())
