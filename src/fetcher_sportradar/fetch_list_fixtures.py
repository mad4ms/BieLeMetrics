import logging

import pandas as pd
from sportradar_datacore_api.handball import HandballAPI


def fetch_list_fixtures(api: HandballAPI, season_id: str) -> list:
    """
    Fetch the list of fixtures for a given season ID
    from the Sportradar Handball API.

    Args:
        api (HandballAPI): An instance of the HandballAPI.
        season_id (str): The ID of the season.

    Returns:
        list: A list of fixtures.
    """
    logging.debug("Fetching fixtures for season ID %s", season_id)
    fixtures = api.list_matches_by_season(season_id)

    if fixtures:
        logging.debug(
            "Found %d fixtures for season ID '%s'",
            len(fixtures),
            season_id,
        )
        return [fixture.to_dict() for fixture in fixtures]
    raise ValueError(f"No fixtures found for season ID '{season_id}'.")


def refine_fixtures_data(fixtures_data: list) -> pd.DataFrame:
    """
    Refine the raw fixtures data into a structured DataFrame.

    Args:
        fixtures_data (list): Raw list of fixtures data.

    Returns:
        pd.DataFrame: Refined DataFrame with selected columns.
    """
    # map of original keys -> snake_case column names
    columns_to_keep = {
        "fixtureId": "fixture_id",
        "seasonId": "season_id",
        "fixtureNumber": "fixture_number",
        "nameLocal": "name_local",
        "nameLatin": "name_latin",
        "startTimeLocal": "start_time_local",
        "startTimeUTC": "start_time_utc",
        "roundNumber": "round_number",
        "competitors": "competitors",
        "externalId": "external_id",
    }
    df_fixtures = pd.DataFrame()

    # Collect all refined fixture dicts/dataframes
    refined_rows = []

    for fixture in fixtures_data:
        try:
            # Flatten the fixture data
            df_fixture = pd.json_normalize(fixture)
        except NotImplementedError:
            df_fixture = pd.DataFrame([fixture])

        # Select and rename columns
        try:
            available = [
                col for col in columns_to_keep.keys() if col in df_fixture.columns
            ]
        except Exception:
            available = []

        if not available:
            continue

        df_fixture = df_fixture[available].rename(columns=columns_to_keep)
        refined_rows.append(df_fixture)

    if refined_rows:
        df_fixtures = pd.concat(refined_rows, ignore_index=True)

    return df_fixtures


def expand_competitors_in_fixtures(
    df_fixtures: pd.DataFrame,
    df_teams: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expand the competitors information in the fixtures DataFrame,
    merge team names, and add home/away columns.

    Args:
        df_fixtures (pd.DataFrame): DataFrame containing fixtures data.
        df_teams (pd.DataFrame): DataFrame containing teams data.

    Returns:
        pd.DataFrame: DataFrame with expanded competitors information.
    """
    columns_to_keep_competitors = {
        "entityId": "entity_id",
        "isHome": "is_home",
        "draw": "draw",
        "resultPlace": "result_place",
        "score": "score",
    }
    fixtures_expanded = []
    for _, fixture in df_fixtures.iterrows():
        competitors = fixture["competitors"]
        competitors_expanded = pd.json_normalize(competitors)
        available = [
            c
            for c in columns_to_keep_competitors.keys()
            if c in competitors_expanded.columns
        ]
        competitors_expanded = competitors_expanded[available].rename(
            columns=columns_to_keep_competitors
        )

        # Get name_full_local from teams dataframe
        teams_for_merge = df_teams[["entity_id", "name_full_local"]]

        competitors_expanded = competitors_expanded.merge(
            teams_for_merge,
            left_on="entity_id",
            right_on="entity_id",
            how="left",
        )
        # convert competitors_expanded to json and add to fixture
        fixture_dict = fixture.drop("competitors").to_dict()
        fixture_dict["competitors"] = competitors_expanded.to_dict(orient="records")
        # insert entity_id_home and entity_id_away
        fixture_dict["entity_id_home"] = competitors_expanded[
            competitors_expanded["is_home"]
        ]["entity_id"].values[0]
        fixture_dict["entity_id_away"] = competitors_expanded[
            ~competitors_expanded["is_home"]
        ]["entity_id"].values[0]
        # insert name_team_home and name_team_away
        fixture_dict["name_team_home"] = competitors_expanded[
            competitors_expanded["is_home"]
        ]["name_full_local"].values[0]
        fixture_dict["name_team_away"] = competitors_expanded[
            ~competitors_expanded["is_home"]
        ]["name_full_local"].values[0]
        # insert score_home and score_away
        fixture_dict["result_score_home"] = competitors_expanded[
            competitors_expanded["is_home"]
        ]["score"].values[0]
        fixture_dict["score_away"] = competitors_expanded[
            ~competitors_expanded["is_home"]
        ]["score"].values[0]
        # insert resultPlace_home and resultPlace_away
        fixture_dict["result_place_home"] = competitors_expanded[
            competitors_expanded["is_home"]
        ]["result_place"].values[0]
        fixture_dict["result_place_away"] = competitors_expanded[
            ~competitors_expanded["is_home"]
        ]["result_place"].values[0]
        fixtures_expanded.append(fixture_dict)
    return pd.DataFrame(fixtures_expanded)


def calculate_standings(df_fixtures: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and append standings information to the fixtures DataFrame.

    Args:
        df_fixtures (pd.DataFrame): DataFrame containing all fixtures in a season.

    Returns:
        pd.DataFrame: The function returns the DataFrame with standings information.
    """
    import numpy as np

    df_all_fixtures_in_season = df_fixtures.copy()

    # --- prep & sorting ---
    for col in ["score_home", "score_away"]:
        if col in df_all_fixtures_in_season.columns:
            df_all_fixtures_in_season[col] = pd.to_numeric(
                df_all_fixtures_in_season[col], errors="coerce"
            )

    df_all_fixtures_in_season["start_time_utc"] = pd.to_datetime(
        df_all_fixtures_in_season["startTimeUTC"], errors="coerce"
    )

    sort_cols = ["start_time_utc"]
    if "fixture_number" in df_all_fixtures_in_season.columns:
        sort_cols.append("fixture_number")
    sort_cols.append("fixture_id")

    df_all_fixtures_in_season = df_all_fixtures_in_season.sort_values(
        sort_cols
    ).reset_index(drop=True)

    # --- initialize standings state ---
    team_id_cols = ["entity_id_home", "entity_id_away"]
    teams = pd.unique(
        pd.concat(
            [df_all_fixtures_in_season[c] for c in team_id_cols],
            ignore_index=True,
        )
    )

    name_lookup = {}
    if (
        "name_team_home" in df_all_fixtures_in_season.columns
        and "name_team_away" in df_all_fixtures_in_season.columns
    ):
        home_names = df_all_fixtures_in_season.set_index("entity_id_home")[
            "name_team_home"
        ]
        away_names = df_all_fixtures_in_season.set_index("entity_id_away")[
            "name_team_away"
        ]
        name_lookup = (
            pd.concat([home_names, away_names])
            .dropna()
            .groupby(level=0)
            .first()
            .to_dict()
        )

    # Add wins/draws/losses + pts_against to the per-team state
    standings = {
        tid: {
            "pts": 0,  # points earned by the team (2/1/0)
            "pts_against": 0,  # points opponents earned vs this team
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "gf": 0,
            "ga": 0,
            "gd": 0,
            "name": name_lookup.get(tid, ""),
        }
        for tid in teams
    }

    def points_for_pair(h, a):
        if h > a:
            return 2, 0
        if h < a:
            return 0, 2
        return 1, 1  # draw

    def standings_table_df():
        tbl = pd.DataFrame.from_dict(standings, orient="index").assign(
            gd=lambda x: x["gf"] - x["ga"]
        )
        tbl = tbl.sort_values(
            by=["pts", "gd", "gf", "name"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        tbl["rank"] = range(1, len(tbl) + 1)
        return tbl

    # --- iterate fixtures and capture standings after each game ---
    standing_after_home, standing_after_away = [], []

    # New per-fixture cumulative outputs
    (
        wins_home_after,
        draws_home_after,
        losses_home_after,
        pts_against_home_after,
    ) = ([], [], [], [])
    (
        wins_away_after,
        draws_away_after,
        losses_away_after,
        pts_against_away_after,
    ) = ([], [], [], [])

    for _, row in df_all_fixtures_in_season.iterrows():
        home_id = row["entity_id_home"]
        away_id = row["entity_id_away"]
        sh = row["score_home"]
        sa = row["score_away"]

        # if unplayed, just snapshot current ranks & cumulative stats
        if pd.isna(sh) or pd.isna(sa):
            tbl = standings_table_df()
            standing_after_home.append(
                tbl.loc[home_id, "rank"] if home_id in tbl.index else np.nan
            )
            standing_after_away.append(
                tbl.loc[away_id, "rank"] if away_id in tbl.index else np.nan
            )

            # copy current cumulative values
            wins_home_after.append(standings[home_id]["wins"])
            draws_home_after.append(standings[home_id]["draws"])
            losses_home_after.append(standings[home_id]["losses"])
            pts_against_home_after.append(standings[home_id]["pts_against"])

            wins_away_after.append(standings[away_id]["wins"])
            draws_away_after.append(standings[away_id]["draws"])
            losses_away_after.append(standings[away_id]["losses"])
            pts_against_away_after.append(standings[away_id]["pts_against"])
            continue

        sh, sa = int(sh), int(sa)

        # goals
        standings[home_id]["gf"] += sh
        standings[home_id]["ga"] += sa
        standings[away_id]["gf"] += sa
        standings[away_id]["ga"] += sh

        # points
        p_home, p_away = points_for_pair(sh, sa)
        standings[home_id]["pts"] += p_home
        standings[away_id]["pts"] += p_away

        # wins/draws/losses and pts_against (opponents' points vs this team)
        if sh > sa:  # home win
            standings[home_id]["wins"] += 1
            standings[away_id]["losses"] += 1
            standings[home_id]["pts_against"] += 0
            standings[away_id]["pts_against"] += 2
        elif sh < sa:  # away win
            standings[home_id]["losses"] += 1
            standings[away_id]["wins"] += 1
            standings[home_id]["pts_against"] += 2
            standings[away_id]["pts_against"] += 0
        else:  # draw
            standings[home_id]["draws"] += 1
            standings[away_id]["draws"] += 1
            standings[home_id]["pts_against"] += 1
            standings[away_id]["pts_against"] += 1

        # recompute gd explicitly
        standings[home_id]["gd"] = standings[home_id]["gf"] - standings[home_id]["ga"]
        standings[away_id]["gd"] = standings[away_id]["gf"] - standings[away_id]["ga"]

        # standings after THIS game
        tbl = standings_table_df()
        standing_after_home.append(tbl.loc[home_id, "rank"])
        standing_after_away.append(tbl.loc[away_id, "rank"])

        # push cumulative snapshots for both teams
        wins_home_after.append(standings[home_id]["wins"])
        draws_home_after.append(standings[home_id]["draws"])
        losses_home_after.append(standings[home_id]["losses"])
        pts_against_home_after.append(standings[home_id]["pts_against"])

        wins_away_after.append(standings[away_id]["wins"])
        draws_away_after.append(standings[away_id]["draws"])
        losses_away_after.append(standings[away_id]["losses"])
        pts_against_away_after.append(standings[away_id]["pts_against"])

    # attach results to fixtures
    df_all_fixtures_in_season["standing_home"] = standing_after_home
    df_all_fixtures_in_season["standing_away"] = standing_after_away

    df_all_fixtures_in_season["wins_home"] = wins_home_after
    df_all_fixtures_in_season["draws_home"] = draws_home_after
    df_all_fixtures_in_season["losses_home"] = losses_home_after
    df_all_fixtures_in_season["pts_against_home"] = pts_against_home_after

    df_all_fixtures_in_season["wins_away"] = wins_away_after
    df_all_fixtures_in_season["draws_away"] = draws_away_after
    df_all_fixtures_in_season["losses_away"] = losses_away_after
    df_all_fixtures_in_season["pts_against_away"] = pts_against_away_after

    return df_all_fixtures_in_season
