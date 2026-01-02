import logging
from typing import Any, Dict, List

import pandas as pd
from sportradar_datacore_api.handball import HandballAPI


def fetch_teams_by_season_id(api: HandballAPI, season_id: str) -> pd.DataFrame:
    """
    Fetch all teams for a given season and return normalized, snake-case DataFrame.
    """
    logging.debug("Fetching teams for season ID %s", season_id)
    teams = api.get_teams_by_season_id(season_id)

    if not teams:
        raise ValueError(f"No teams found for season ID '{season_id}'.")

    logging.debug("Found %d teams for season ID '%s'", len(teams), season_id)

    teams_data = [team.to_dict() for team in teams]

    # mapping original → snake_case
    columns_map = {
        "entityId": "entity_id",
        "organization": "organization",
        "nameFullLocal": "name_full_local",
        "nameFullLatin": "name_full_latin",
        "codeLocal": "code_local",
        "codeLatin": "code_latin",
        "externalId": "external_id",
    }

    df_teams_snake = pd.DataFrame()

    for team in teams_data:
        team_id = team["entityId"]
        team_details = api.get_team_by_id(entity_id=team_id)

        if not team_details:
            continue

        df_details = pd.json_normalize(team_details[0].to_dict())

        # keep only known cols
        cols_present = [c for c in columns_map.keys() if c in df_details.columns]
        df_details = df_details[cols_present]

        # rename to snake_case
        df_details = df_details.rename(columns=columns_map)

        df_teams_snake = pd.concat([df_teams_snake, df_details], ignore_index=True)

    if df_teams_snake.empty:
        raise ValueError(f"No teams found for season ID '{season_id}'.")

    return df_teams_snake
