"""Fetch teams for the season and store into DuckDB.

Usage: import and call fetch_and_store_teams(con, api, season_id) or run directly.
"""

import pandas as pd
from config import (
    get_duckdb_connection,
    get_api_sportradar,
    get_competition_and_season,
)


def fetch_teams(con=None, api=None, season_id=None):
    if api is None or season_id is None:
        api = api or get_api_sportradar()
        _, season_id = get_competition_and_season(api)

    list_entities_season = api.get_teams_by_season_id(season_id=season_id)
    print(f"→ Number of teams in season: {len(list_entities_season)}")

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
    for team in list_entities_season:
        id_team = team.entity_id
        team_details = api.get_team_by_id(entity_id=id_team)
        df_team_details = pd.json_normalize(team_details[0].to_dict())
        df_team_details = df_team_details[
            [col for col in columns_to_keep if col in df_team_details.columns]
        ]
        df_teams = pd.concat([df_teams, df_team_details], ignore_index=True)

    # write to duckdb
    con = con or get_duckdb_connection()
    con.execute("DROP TABLE IF EXISTS teams")
    con.execute("CREATE TABLE IF NOT EXISTS teams AS SELECT * FROM df_teams")
    print("Wrote teams table to DuckDB")
    # print location of duckdb file
    return df_teams


if __name__ == "__main__":
    con = get_duckdb_connection()
    api = get_api_sportradar()
    _, season_id = get_competition_and_season(api)
    fetch_teams(con=con, api=api, season_id=season_id)
