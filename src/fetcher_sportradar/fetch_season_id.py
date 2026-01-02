"""Module to fetch season ID from Sportradar Handball API."""

import logging

from sportradar_datacore_api.handball import HandballAPI

YEAR_SEASON = "2024"  # saison 24/25


def fetch_season_id(
    api: HandballAPI, competition_id: str, season_year: str = YEAR_SEASON
) -> str:
    """
    Fetch the season ID for a given competition ID and season year
    from the Sportradar Handball API.

    Args:
        api (HandballAPI): An instance of the HandballAPI.
        competition_id (str): The ID of the competition.
        season_year (str): The year of the season.

    Returns:
        str: The season ID.
    """
    logging.debug(
        "Fetching season ID for competition ID %s and season year %s",
        competition_id,
        season_year,
    )
    season_id = api.get_season_id_by_year(
        competition_id=competition_id, season_year=season_year
    )

    if season_id:
        logging.debug(
            "Found season ID for competition ID '%s' and season year '%s': %s",
            competition_id,
            season_year,
            season_id,
        )
        return season_id
    raise ValueError(
        f"Season for competition ID '{competition_id}' and year '{season_year}' not found."
    )
