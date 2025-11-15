from __future__ import annotations

import os
import io
import gzip
import zipfile
import logging
import concurrent.futures
from typing import Any, Dict, Iterable, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

from src.hbl_etl_sportradar_kinexon.config import (
    get_duckdb_connection,
    get_api_sportradar,
    get_api_kinexon,
)

from src.events_sportradar.fetch_competition_id import fetch_competition_id
from src.events_sportradar.fetch_saison_id import fetch_season_id
from src.events_sportradar.fetch_teams import fetch_teams_by_season_id
from src.events_sportradar.fetch_list_fixtures import (
    fetch_list_fixtures,
    refine_fixtures_data,
    expand_competitors_in_fixtures,
    calculate_standings,
)

from src.events_sportradar.fetch_list_fixture_events import (
    fetch_and_process_fixture_events_multithreaded,
)
from src.events_kinexon.fetch_session_id_for_fixtures import (
    fetch_session_ids_for_fixtures,
)
from src.events_kinexon.fetch_positions_for_fixture import (
    fetch_positions_for_fixtures_multithreaded,
)
from src.events_kinexon.fetch_events_for_session import (
    fetch_detected_events_for_sessions_multithreaded,
)

# ---------------------------------------------------------------------
# Config / Logging
# ---------------------------------------------------------------------

LOG_LEVEL = os.getenv("SPORTRADAR_ETL_LOG_LEVEL", "INFO").upper()
MAX_WORKERS = int(os.getenv("SPORTRADAR_ETL_MAX_WORKERS", "16"))

logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("sportradar_etl")


api = get_api_sportradar()


def download_sportradar_etl(
    con: Optional[duckdb.DuckDBPyConnection] = None,
    api_sr: Optional[Any] = None,
    api_kinexon: Optional[Any] = None,
    competition_id: Optional[str] = None,
    season_id: Optional[str] = None,
    max_workers: int = MAX_WORKERS,
) -> None:
    """
    Download and process Sportradar data, storing results in DuckDB.

    Args:
        con (duckdb.DuckDBPyConnection, optional): DuckDB connection.
        api (Any, optional): Sportradar API instance.
        competition_id (str, optional): Competition ID.
        season_id (str, optional): Season ID.
        max_workers (int): Number of threads for parallel processing.

    Returns:
        None
    """
    con = con or get_duckdb_connection()
    api_sr = api_sr or get_api_sportradar()
    api_kinexon = api_kinexon or get_api_kinexon()

    NAME_SEASON = "DAIKIN HBL 2024/25"
    YEAR_SEASON = int(NAME_SEASON.split()[-1].split("/")[0])

    # Fetch competition and season IDs if not provided
    if competition_id is None or season_id is None:
        competition_id = fetch_competition_id(api_sr)
        season_id = fetch_season_id(api_sr, competition_id, YEAR_SEASON)

    log.info(
        "Starting ETL for competition ID: %s, season ID: %s",
        competition_id,
        season_id,
    )

    # Fetch and store teams
    df_teams = fetch_teams_by_season_id(api_sr, season_id)
    # con.execute("DROP TABLE IF EXISTS teams")
    # con.execute("CREATE TABLE teams AS SELECT * FROM df_teams")
    log.info("Stored teams data in DuckDB.")

    # Fixtures
    fixtures_raw = fetch_list_fixtures(api_sr, season_id)
    df_fixtures_refined = refine_fixtures_data(fixtures_raw)
    df_fixtures_expanded = expand_competitors_in_fixtures(
        df_fixtures_refined, df_teams
    )
    # Insert session_ids from Kinexon into fixtures
    dict_session_ids = fetch_session_ids_for_fixtures(
        api_kinexon, df_fixtures_expanded
    )  # will return fixture_id -> session_id mapping
    log.info("Fetched session IDs from Kinexon for fixtures.")
    # this will add a new column 'session_id' to df_fixtures_expanded
    # and map the session IDs accordingly
    df_fixtures_expanded["session_id"] = df_fixtures_expanded["fixtureId"].map(
        dict_session_ids
    )
    # in case we're interested in storing the standings as well
    # calculate_standings
    df_standings = calculate_standings(df_fixtures_expanded)
    # con.execute("DROP TABLE IF EXISTS fixtures")
    # con.execute("CREATE TABLE fixtures AS SELECT * FROM df_fixtures_expanded")
    log.info("Stored fixtures data in DuckDB.")

    list_fixture_ids = df_fixtures_expanded["fixtureId"].tolist()

    # Fetch and process fixture events, players.
    # Returns players and match events dataframes as players are
    # extracted during event processing from setup events.
    df_all_match_events, df_all_players = (
        fetch_and_process_fixture_events_multithreaded(
            api_sr,
            list_fixture_ids,
            df_teams,
            max_workers=max_workers,  # used 16 no problem
        )
    )
    # add session_id to match events
    df_all_match_events["session_id"] = df_all_match_events["fixtureId"].map(
        dict_session_ids
    )
    # save as csv for inspection
    df_all_match_events.to_csv(
        "df_all_match_events_with_session_ids.csv", index=False
    )
    # con.execute("DROP TABLE IF EXISTS match_events")
    # con.execute(
    #     "CREATE TABLE match_events AS SELECT * FROM df_all_match_events"
    # )
    log.info("Stored match events data in DuckDB.")

    # con.execute("DROP TABLE IF EXISTS players")
    # con.execute("CREATE TABLE players AS SELECT * FROM df_all_players")
    log.info("Stored players data in DuckDB.")

    # fetch list of session_ids from fixtures that are not present in kinexon_positions
    df_sessions_in_positions = con.execute(
        """
        SELECT DISTINCT session_id
        FROM kinexon_positions
        WHERE session_id IS NOT NULL
        """
    ).df()
    session_ids_in_positions = set(
        df_sessions_in_positions["session_id"].dropna().unique().tolist()
    )
    # session_ids in fixtures
    session_ids_in_fixtures = set(
        df_fixtures_expanded["session_id"].dropna().unique().tolist()
    )
    session_ids_to_fetch = list(
        session_ids_in_fixtures - session_ids_in_positions
    )
    # Feed  a list of max of 18 session_ids to fetch_positions_for_fixtures_multithreaded
    split_session_id_lists = [
        session_ids_to_fetch[i : i + 9]
        for i in range(0, len(session_ids_to_fetch), 9)
    ]
    all_positions_dfs = []

    # print statistics about whats already in the kinexon_positions table
    log.info(
        "Already have %d session IDs in kinexon_positions table.",
        len(session_ids_in_positions),
    )
    # print how many percent of total session ids in fixtures are already present
    percent_present = (
        len(session_ids_in_positions) / len(session_ids_in_fixtures) * 100
        if session_ids_in_fixtures
        else 0
    )
    log.info(
        "%.2f%% of session IDs in fixtures are already present in kinexon_positions table.",
        percent_present,
    )

    # print statistics
    log.info(
        "Fetching positions for %d sessions in %d batches.",
        len(session_ids_to_fetch),
        len(split_session_id_lists),
    )

    for session_id_list in split_session_id_lists:
        df_positions = fetch_positions_for_fixtures_multithreaded(
            api_kinexon,
            session_id_list,
            max_workers=9,
        )
        df_positions["fixtureId"] = df_positions["session_id"].map(
            {v: k for k, v in dict_session_ids.items() if v in session_id_list}
        )
        all_positions_dfs.append(df_positions)
        # insert if not exists, else create
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS kinexon_positions AS
            SELECT * FROM df_positions
            """
        )
        con.execute(
            """
            INSERT INTO kinexon_positions
            SELECT * FROM df_positions
            """
        )

        # session_id_list_str = ", ".join(session_id_list)

        # save as gzip csv for inspection
        # df_positions.to_csv(
        #     f"kinexon_positions_sessions_{session_id_list_str}.csv.gz",
        #     index=False,
        #     compression="gzip",
        # )
        # break

    # list all session ids already present in kinexon_positions as list of int
    list_session_ids_in_positions = con.execute(
        """
        SELECT DISTINCT session_id
        FROM kinexon_positions
        WHERE session_id IS NOT NULL
        """
    ).df()
    list_session_ids_in_positions = set(
        list_session_ids_in_positions["session_id"].dropna().unique().tolist()
    )

    # fetch_detected_events_for_sessions_multithreaded
    df_all_detected_events = fetch_detected_events_for_sessions_multithreaded(
        api_kinexon,
        list_session_ids_in_positions,
        max_workers=9,
    )

    # to csv for inspection
    df_all_detected_events.to_csv(
        "df_all_detected_events_kinexon.csv", index=False
    )


if __name__ == "__main__":
    con = get_duckdb_connection(db_path="./data/mydb2024-25.duckdb")
    api = get_api_sportradar()
    api_kinexon = get_api_kinexon()
    download_sportradar_etl(con=con, api_sr=api, api_kinexon=api_kinexon)
