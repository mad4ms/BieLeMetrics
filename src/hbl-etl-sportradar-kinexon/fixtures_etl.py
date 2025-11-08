"""Fetch fixtures, normalize competitors, store fixtures and compute standings.

This module mirrors the notebook's fixtures section. It collects fixtures,
expands the `competitors` field, stores a flat `fixtures` table and an
enhanced `fixtures_enhanced` table with standings snapshots after each match.
"""

import pandas as pd
import numpy as np
from .config import (
    get_duckdb_connection,
    get_api_sportradar,
    get_competition_and_season,
)


columns_to_keep_fixtures = [
    "fixtureId",
    "seasonId",
    "fixtureNumber",
    "nameLocal",
    "nameLatin",
    "startTimeLocal",
    "startTimeUTC",
    "roundNumber",
    "competitors",
    "externalId",
]

columns_to_keep_competitors = [
    "entityId",
    "isHome",
    "draw",
    "resultPlace",
    "score",
]


def fetch_fixtures_and_store(con=None, api=None, season_id=None):
    api = api or get_api_sportradar()
    con = con or get_duckdb_connection()
    if season_id is None:
        _, season_id = get_competition_and_season(api)

    list_fixtures = api.get_list_matches_by_season_id(season_id=season_id)
    print(f"Found {len(list_fixtures)} fixtures.")

    df_list = []
    # fetch teams from DB to map names
    try:
        df_teams = con.execute("SELECT * FROM teams").df()
    except Exception:
        df_teams = pd.DataFrame()

    for match in list_fixtures:
        df_fixture = pd.json_normalize(match.to_dict())
        df_fixture = df_fixture[
            [
                col
                for col in columns_to_keep_fixtures
                if col in df_fixture.columns
            ]
        ]

        competitors_expanded = pd.json_normalize(
            df_fixture["competitors"].explode().to_list()
        )
        competitors_expanded = competitors_expanded[
            [
                col
                for col in columns_to_keep_competitors
                if col in competitors_expanded.columns
            ]
        ]

        if not df_teams.empty:
            competitors_expanded = competitors_expanded.merge(
                df_teams[["entityId", "nameFullLocal"]],
                left_on="entityId",
                right_on="entityId",
                how="left",
            )

        # attach flattened competitors back to fixture as json
        df_fixture = (
            df_fixture.drop(columns=["competitors"])
            if "competitors" in df_fixture.columns
            else df_fixture
        )
        df_fixture = df_fixture.assign(
            competitors=[competitors_expanded.to_dict(orient="records")]
        )

        # insert simple home/away fields if present
        try:
            df_fixture = df_fixture.assign(
                entityId_home=competitors_expanded[
                    competitors_expanded["isHome"] == True
                ]["entityId"].values[0],
                entityId_away=competitors_expanded[
                    competitors_expanded["isHome"] == False
                ]["entityId"].values[0],
                name_team_home=(
                    competitors_expanded[
                        competitors_expanded["isHome"] == True
                    ]["nameFullLocal"].values[0]
                    if "nameFullLocal" in competitors_expanded.columns
                    else None
                ),
                name_team_away=(
                    competitors_expanded[
                        competitors_expanded["isHome"] == False
                    ]["nameFullLocal"].values[0]
                    if "nameFullLocal" in competitors_expanded.columns
                    else None
                ),
                score_home=(
                    competitors_expanded[
                        competitors_expanded["isHome"] == True
                    ]["score"].values[0]
                    if "score" in competitors_expanded.columns
                    else None
                ),
                score_away=(
                    competitors_expanded[
                        competitors_expanded["isHome"] == False
                    ]["score"].values[0]
                    if "score" in competitors_expanded.columns
                    else None
                ),
                resultPlace_home=(
                    competitors_expanded[
                        competitors_expanded["isHome"] == True
                    ]["resultPlace"].values[0]
                    if "resultPlace" in competitors_expanded.columns
                    else None
                ),
                resultPlace_away=(
                    competitors_expanded[
                        competitors_expanded["isHome"] == False
                    ]["resultPlace"].values[0]
                    if "resultPlace" in competitors_expanded.columns
                    else None
                ),
            )
        except Exception:
            # if any of the above fails, keep going and append raw fixture
            pass

        df_list.append(df_fixture)

    if not df_list:
        print("No fixtures fetched.")
        return pd.DataFrame()

    df_all = pd.concat(df_list, ignore_index=True)

    # write to duckdb
    con.execute("DROP TABLE IF EXISTS fixtures")
    con.execute("CREATE TABLE fixtures AS SELECT * FROM df_all")
    con.execute("INSERT INTO fixtures SELECT * FROM df_all")
    print("Wrote fixtures to DuckDB")

    # compute standings/enhancements and save fixtures_enhanced
    try:
        df_all_fixtures_in_season = con.execute("SELECT * FROM fixtures").df()
    except Exception:
        df_all_fixtures_in_season = df_all.copy()

    # cleanup and type conversions
    if "competitors" in df_all_fixtures_in_season.columns:
        df_all_fixtures_in_season = df_all_fixtures_in_season.drop(
            columns=["competitors"]
        )
    for col in ["score_home", "score_away"]:
        if col in df_all_fixtures_in_season.columns:
            df_all_fixtures_in_season[col] = pd.to_numeric(
                df_all_fixtures_in_season[col], errors="coerce"
            )
    df_all_fixtures_in_season["startTimeUTC"] = pd.to_datetime(
        df_all_fixtures_in_season.get("startTimeUTC"), errors="coerce"
    )

    sort_cols = ["startTimeUTC"]
    if "fixtureNumber" in df_all_fixtures_in_season.columns:
        sort_cols.append("fixtureNumber")
    sort_cols.append("fixtureId")
    df_all_fixtures_in_season = df_all_fixtures_in_season.sort_values(
        sort_cols
    ).reset_index(drop=True)

    # initialize standings
    team_id_cols = [
        c
        for c in ["entityId_home", "entityId_away"]
        if c in df_all_fixtures_in_season.columns
    ]
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
        home_names = (
            df_all_fixtures_in_season.set_index("entityId_home")[
                "name_team_home"
            ]
            if "entityId_home" in df_all_fixtures_in_season.columns
            else pd.Series()
        )
        away_names = (
            df_all_fixtures_in_season.set_index("entityId_away")[
                "name_team_away"
            ]
            if "entityId_away" in df_all_fixtures_in_season.columns
            else pd.Series()
        )
        name_lookup = (
            pd.concat([home_names, away_names])
            .dropna()
            .groupby(level=0)
            .first()
            .to_dict()
        )

    standings = {
        tid: {
            "pts": 0,
            "pts_against": 0,
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
        return 1, 1

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

    # iterate and capture
    standing_after_home, standing_after_away = [], []
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
        home_id = row.get("entityId_home")
        away_id = row.get("entityId_away")
        sh = row.get("score_home")
        sa = row.get("score_away")

        if pd.isna(sh) or pd.isna(sa):
            tbl = standings_table_df()
            standing_after_home.append(
                tbl.loc[home_id, "rank"] if home_id in tbl.index else np.nan
            )
            standing_after_away.append(
                tbl.loc[away_id, "rank"] if away_id in tbl.index else np.nan
            )
            wins_home_after.append(standings.get(home_id, {}).get("wins", 0))
            draws_home_after.append(standings.get(home_id, {}).get("draws", 0))
            losses_home_after.append(
                standings.get(home_id, {}).get("losses", 0)
            )
            pts_against_home_after.append(
                standings.get(home_id, {}).get("pts_against", 0)
            )
            wins_away_after.append(standings.get(away_id, {}).get("wins", 0))
            draws_away_after.append(standings.get(away_id, {}).get("draws", 0))
            losses_away_after.append(
                standings.get(away_id, {}).get("losses", 0)
            )
            pts_against_away_after.append(
                standings.get(away_id, {}).get("pts_against", 0)
            )
            continue

        sh, sa = int(sh), int(sa)
        standings[home_id]["gf"] += sh
        standings[home_id]["ga"] += sa
        standings[away_id]["gf"] += sa
        standings[away_id]["ga"] += sh
        p_home, p_away = points_for_pair(sh, sa)
        standings[home_id]["pts"] += p_home
        standings[away_id]["pts"] += p_away

        if sh > sa:
            standings[home_id]["wins"] += 1
            standings[away_id]["losses"] += 1
            standings[home_id]["pts_against"] += 0
            standings[away_id]["pts_against"] += 2
        elif sh < sa:
            standings[home_id]["losses"] += 1
            standings[away_id]["wins"] += 1
            standings[home_id]["pts_against"] += 2
            standings[away_id]["pts_against"] += 0
        else:
            standings[home_id]["draws"] += 1
            standings[away_id]["draws"] += 1
            standings[home_id]["pts_against"] += 1
            standings[away_id]["pts_against"] += 1

        standings[home_id]["gd"] = (
            standings[home_id]["gf"] - standings[home_id]["ga"]
        )
        standings[away_id]["gd"] = (
            standings[away_id]["gf"] - standings[away_id]["ga"]
        )

        tbl = standings_table_df()
        standing_after_home.append(tbl.loc[home_id, "rank"])
        standing_after_away.append(tbl.loc[away_id, "rank"])
        wins_home_after.append(standings[home_id]["wins"])
        draws_home_after.append(standings[home_id]["draws"])
        losses_home_after.append(standings[home_id]["losses"])
        pts_against_home_after.append(standings[home_id]["pts_against"])
        wins_away_after.append(standings[away_id]["wins"])
        draws_away_after.append(standings[away_id]["draws"])
        losses_away_after.append(standings[away_id]["losses"])
        pts_against_away_after.append(standings[away_id]["pts_against"])

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

    # save enhanced fixtures
    con.execute("DROP TABLE IF EXISTS fixtures_enhanced")
    con.execute(
        "CREATE TABLE fixtures_enhanced AS SELECT * FROM df_all_fixtures_in_season"
    )
    print("Wrote fixtures_enhanced to DuckDB")

    return df_all, df_all_fixtures_in_season


if __name__ == "__main__":
    con = get_duckdb_connection()
    api = get_api_sportradar()
    _, season_id = get_competition_and_season(api)
    fetch_fixtures_and_store(con=con, api=api, season_id=season_id)
