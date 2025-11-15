"""Fetch competition ID from Sportradar Handball API."""

import logging
from sportradar_datacore_api.handball import HandballAPI

NAME_COMPETITION = "1. Handball-Bundesliga"


def fetch_competition_id(
    api: HandballAPI, competition_name: str = NAME_COMPETITION
) -> str:
    """
    Fetch the competition ID for a given competition name
    from the Sportradar Handball API.

    Args:
        api (HandballAPI): An instance of the HandballAPI.
        competition_name (str): The name of the competition.

    Returns:
        str: The competition ID.
    """
    logging.debug("Fetching competition ID for %s", competition_name)
    competition_id = api.get_competition_id_by_name(competition_name)

    if competition_id:
        logging.debug(
            "Found competition ID for '%s': %s",
            competition_name,
            competition_id,
        )
        return competition_id
    raise ValueError(f"Competition '{competition_name}' not found.")
