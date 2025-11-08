"""Fetch match events and players, normalize, and store to DuckDB.

This follows the notebook's event downloading pipeline. It creates two
DuckDB tables: `match_events` and `players`.
"""

import pandas as pd
import tqdm
from config import (
    get_duckdb_connection,
    get_api_sportradar,
    get_competition_and_season,
)


columns_to_keep_event_list = [
    "fixtureId",
    "class",
    "eventId",
    "eventTime",
    "eventType",
    "subType",
    "attendance",
    "entityId",
    "personId",
    "bib",
    "name",
    "position",
    "scores",
    "periodId",
    "playId",
    "clock",
    "success",
    "x",
    "y",
    "attackType",
    "goalKeeperId",
    "location",
    "failureReason",
    "emptyNet",
]


def fetch_events_and_players(con=None, api=None):
    con = con or get_duckdb_connection()
    api = api or get_api_sportradar()

    list_fixture_ids = con.execute("SELECT fixtureId FROM fixtures").fetchall()
    list_fixture_ids = [fid[0] for fid in list_fixture_ids]
    print(f"Found {len(list_fixture_ids)} fixtures.")

    df_all_players = pd.DataFrame()
    df_all_match_events = pd.DataFrame()

    for fid in tqdm.tqdm(list_fixture_ids):
        match_events = api.get_fixture_events_by_id(
            fid, setup_only=False, with_scores=True
        )

        # flatten data/options into top-level dict
        for event in match_events:
            if "data" in event and event["data"]:
                event.update(event["data"])
                del event["data"]
            if "options" in event and event["options"]:
                event.update(event["options"])
                del event["options"]

        df_match_events = (
            pd.DataFrame(
                match_events,
                columns=[
                    c
                    for c in columns_to_keep_event_list
                    if c in match_events[0].keys()
                ],
            )
            if match_events
            else pd.DataFrame()
        )

        # map team names using teams table
        try:
            df_teams = con.execute("SELECT * FROM teams").df()
        except Exception:
            df_teams = pd.DataFrame()

        if not df_match_events.empty and not df_teams.empty:
            df_match_events = df_match_events.merge(
                df_teams[["entityId", "nameFullLocal"]],
                left_on="entityId",
                right_on="entityId",
                how="left",
            )
            df_match_events = df_match_events.rename(
                columns={"nameFullLocal": "teamName"}
            )

        # extract setup events to fetch player details
        if not df_match_events.empty:
            df_setup_events = df_match_events[
                (df_match_events.get("class") == "setup")
                & (df_match_events.get("eventType") == "person")
            ]
            df_setup_events = (
                df_setup_events.drop_duplicates(subset=["personId"])
                if not df_setup_events.empty
                else pd.DataFrame()
            )
        else:
            df_setup_events = pd.DataFrame()

        if not df_setup_events.empty:
            unique_person_ids = (
                df_setup_events["personId"].dropna().astype(str).unique()
            )
            if len(unique_person_ids) > 0:
                str_person_ids = ",".join(unique_person_ids)
                players = api.get_players_by_ids(person_ids=str_person_ids)
                df_players = pd.json_normalize(players)

                people_map = (
                    df_setup_events[["personId", "entityId", "teamName"]]
                    .dropna(subset=["personId"])
                    .drop_duplicates(subset=["personId"])
                )
                df_players = df_players.merge(
                    people_map,
                    on="personId",
                    how="left",
                    validate="one_to_one",
                )
                df_all_players = pd.concat(
                    [df_all_players, df_players], ignore_index=True
                )

                # merge player names into match_events
                df_match_events = df_match_events.merge(
                    df_players[["nameFullLocal", "personId"]],
                    left_on="personId",
                    right_on="personId",
                    how="left",
                )
                df_match_events = df_match_events.rename(
                    columns={"nameFullLocal": "personName"}
                )

                # goalkeeper name
                if "goalKeeperId" in df_match_events.columns:
                    df_match_events = df_match_events.merge(
                        df_players[["nameFullLocal", "personId"]].rename(
                            columns={
                                "personId": "personId_goalkeeper",
                                "nameFullLocal": "goalkeeperName",
                            }
                        ),
                        left_on="goalKeeperId",
                        right_on="personId_goalkeeper",
                        how="left",
                    )
                    if "personId_goalkeeper" in df_match_events.columns:
                        df_match_events = df_match_events.drop(
                            columns=["personId_goalkeeper"]
                        )

        df_all_match_events = (
            pd.concat(
                [df_all_match_events, df_match_events], ignore_index=True
            )
            if not df_match_events.empty
            else df_all_match_events
        )

    # deduplicate players
    if not df_all_players.empty:
        df_all_players = df_all_players.drop_duplicates(
            subset=["entityId", "personId"], keep="first"
        )

    # write to duckdb
    con.execute("DROP TABLE IF EXISTS match_events")
    con.execute(
        "CREATE TABLE match_events AS SELECT * FROM df_all_match_events"
    )
    con.execute("DROP TABLE IF EXISTS players")
    con.execute("CREATE TABLE players AS SELECT * FROM df_all_players")
    print("Wrote match_events and players tables to DuckDB")

    return df_all_match_events, df_all_players


if __name__ == "__main__":
    con = get_duckdb_connection()
    api = get_api_sportradar()
    fetch_events_and_players(con=con, api=api)
