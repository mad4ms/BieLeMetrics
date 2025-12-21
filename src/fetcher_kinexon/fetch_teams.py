import logging
from typing import List, Dict, Optional
import pandas as pd
from kinexon_handball_api.handball import HandballAPI
from typing import Tuple
import difflib
import datetime


def fetch_teams_for_season(
    api: HandballAPI, season_year: str = "2024-25"
) -> List[Dict]:
    """
    Fetch team IDs for a specific season from the Kinexon Handball API.

    Args:
        api (HandballAPI): An instance of the HandballAPI.
        season_id (str): The ID of the season. Default is "2024-25".

    Returns:
        List[Dict]: A dictionary of team IDs and names for the specified season.
    """

    logging.info("Fetching team IDs for season year %s", season_year)
    try:
        list_teams = api.fetch_team_ids(season_year)
        logging.info(
            "Fetched %d team IDs for season year %s",
            len(list_teams),
            season_year,
        )
    except Exception as e:
        logging.error(
            "Error fetching team IDs for season year %s: %s",
            season_year,
            str(e),
        )
    return list_teams
