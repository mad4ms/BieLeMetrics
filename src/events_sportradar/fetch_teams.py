import logging
from typing import Any, Dict, List
import pandas as pd

from sportradar_datacore_api.handball import HandballAPI


def fetch_teams_by_season_id(api: HandballAPI, season_id: str) -> pd.DataFrame:
    """
    Fetch the list of teams for a given season ID
    from the Sportradar Handball API.

    Args:
        api (HandballAPI): An instance of the HandballAPI.
        season_id (str): The ID of the season.

    Returns:
        pd.DataFrame: A DataFrame containing the teams.
    """
    logging.debug("Fetching teams for season ID %s", season_id)
    teams = api.get_teams_by_season_id(season_id)

    if teams:
        logging.debug(
            "Found %d teams for season ID '%s'",
            len(teams),
            season_id,
        )
        teams_data = [team.to_dict() for team in teams]

        columns_to_keep = [
            "entityId",
            "organization",
            "nameFullLocal",
            "nameFullLatin",
            "codeLocal",
            "codeLatin",
            "externalId",
        ]
        df_teams = pd.DataFrame()

        for team in teams_data:
            id_team = team["entityId"]
            team_details = api.get_team_by_id(entity_id=id_team)
            df_team_details = pd.json_normalize(team_details[0].to_dict())
            df_team_details = df_team_details[
                [
                    col
                    for col in columns_to_keep
                    if col in df_team_details.columns
                ]
            ]
            df_teams = pd.concat(
                [df_teams, df_team_details], ignore_index=True
            )
        if df_teams.empty:
            raise ValueError(f"No teams found for season ID '{season_id}'.")

        return df_teams
    raise ValueError(f"No teams found for season ID '{season_id}'.")
