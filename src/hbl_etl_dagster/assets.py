# src/hbl_etl_dagster/assets.py

from dagster import (
    asset,
    AssetExecutionContext,
    StaticPartitionsDefinition,
    MetadataValue,
)
import pandas as pd
from typing import Dict
import numpy as np
from src.events_sportradar.fetch_competition_id import fetch_competition_id
from src.events_sportradar.fetch_saison_id import fetch_season_id
from src.events_sportradar.fetch_teams import fetch_teams_by_season_id
from src.events_sportradar.fetch_list_fixtures import (
    fetch_list_fixtures,
    refine_fixtures_data,
    expand_competitors_in_fixtures,
)
from src.events_kinexon.fetch_session_id_for_fixtures import (
    fetch_session_ids_for_fixtures,
)
from src.events_sportradar.fetch_list_fixture_events import (
    fetch_and_process_fixture_events_multithreaded,
)
from src.events_kinexon.fetch_positions_for_fixture import (
    fetch_positions_for_fixtures_multithreaded,
)
from src.events_kinexon.fetch_events_for_session import (
    fetch_detected_events_for_sessions_multithreaded,
)


from .utils.player_league_mapper import PlayerLeagueMapper
from .utils.sportradar_kinexon_event_mapper import refine_throw_time_for_event

# --------------------------------------------------------------------
# Partitions
# --------------------------------------------------------------------

season_partitions = StaticPartitionsDefinition(
    partition_keys=[
        "2024",  # extend or generate dynamically later
    ]
)


# --------------------------------------------------------------------
# In-Memory ID Assets
# --------------------------------------------------------------------


@asset(
    required_resource_keys={"sportradar_api"},
    io_manager_key="file_io_manager",  # This ensures the asset is kept in memory
    compute_kind="api",
    group_name="sportradar_ids",
    description="Fetches the Sportradar competition ID for '1. Handball-Bundesliga'.",
)
def competition_id(context: AssetExecutionContext) -> str:
    """An in-memory asset representing the competition ID."""
    api = context.resources.sportradar_api
    comp_id = fetch_competition_id(api=api)
    context.log.info(f"Resolved competition_id: {comp_id}")
    return str(comp_id)


@asset(
    partitions_def=season_partitions,
    required_resource_keys={"sportradar_api"},
    io_manager_key="file_io_manager",  # This ensures the asset is kept in memory
    compute_kind="api",
    group_name="sportradar_ids",
    description="Fetches the Sportradar season ID for a given year.",
)
def season_id(context: AssetExecutionContext, competition_id: str) -> str:
    """An in-memory asset representing the season ID for a given partition (year)."""
    api = context.resources.sportradar_api
    season_year = int(context.partition_key)

    s_id = fetch_season_id(
        api=api,
        competition_id=competition_id,
        season_year=season_year,
    )
    context.log.info(f"Resolved season_id: {s_id} for year {season_year}")
    return str(s_id)


# --------------------------------------------------------------------
# Persisted Data Assets
# --------------------------------------------------------------------


@asset(
    partitions_def=season_partitions,
    required_resource_keys={"sportradar_api"},
    group_name="sportradar_data",
    compute_kind="duckdb",
)
def teams(context: AssetExecutionContext, season_id: str) -> pd.DataFrame:
    """
    All teams for a given season. Depends on the in-memory season_id asset.
    Persisted to DuckDB via IOManager.
    """
    api = context.resources.sportradar_api
    df = fetch_teams_by_season_id(api=api, season_id=season_id)
    context.log.info(f"Fetched {len(df)} teams for season_id={season_id}")
    context.add_output_metadata(
        {
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "preview": MetadataValue.md(df.head(10).to_markdown(index=False)),
        }
    )
    return df


@asset(
    partitions_def=season_partitions,
    required_resource_keys={"sportradar_api", "kinexon_api"},
    group_name="sportradar_data",
    compute_kind="duckdb",
)
def fixtures(
    context: AssetExecutionContext, season_id: str, teams: pd.DataFrame
) -> pd.DataFrame:
    """
    Full fixtures list for the season, including refined and expanded competitor info.
    Depends on season_id (in-memory) and teams (persisted).
    """
    api = context.resources.sportradar_api

    raw_fixtures = fetch_list_fixtures(api=api, season_id=season_id)
    refined_fixtures = refine_fixtures_data(raw_fixtures)

    # Correctly use the upstream 'teams' asset to expand competitor data
    expanded_fixtures = expand_competitors_in_fixtures(refined_fixtures, teams)

    api = context.resources.kinexon_api

    dict_session_ids = fetch_session_ids_for_fixtures(
        api=api,
        df_fixtures=expanded_fixtures,
    )
    # contains mapping fixture_id -> session_id

    # merge session IDs into expanded_fixtures
    expanded_fixtures["session_id"] = expanded_fixtures["fixtureId"].map(
        dict_session_ids
    )

    context.log.info(
        f"Fetched session IDs for {len(dict_session_ids)} fixtures from Kinexon."
    )

    context.log.info(
        f"Fetched and expanded {len(expanded_fixtures)} fixtures for season_id={season_id}"
    )
    return expanded_fixtures


# partitions_def=season_partitions,
#     required_resource_keys={"sportradar_api"},
#     io_manager_key="file_io_manager",  # This ensures the asset is kept in memory
#     compute_kind="api",
#     group_name="sportradar_ids",
#     description="Fetches the Sportradar season ID for a given year.",


# @asset(
#     partitions_def=season_partitions,
#     required_resource_keys={"kinexon_api"},
#     group_name="kinexon_data",
#     io_manager_key="file_io_manager",  # This ensures the asset is kept in memory
#     compute_kind="api",
# )
# def session_ids(
#     context: AssetExecutionContext, fixtures: pd.DataFrame
# ) -> Dict[str, int]:
#     """
#     Fetch session IDs for fixtures from Kinexon API.
#     Depends on the persisted fixtures asset.
#     """
#     api = context.resources.kinexon_api

#     dict_session_ids = fetch_session_ids_for_fixtures(
#         api=api,
#         df_fixtures=fixtures,
#     )
#     # contains mapping fixture_id -> session_id

#     context.log.info(
#         f"Fetched session IDs for {len(dict_session_ids)} fixtures from Kinexon."
#     )
#     return dict_session_ids


@asset(
    partitions_def=season_partitions,
    required_resource_keys={"sportradar_api", "io_manager"},
    group_name="sportradar_data",
    compute_kind="duckdb",
    description="All match events for the season's fixtures, enriched with session IDs. "
    "Also persists players and match_events tables for downstream processing.",
)
def fixture_events(
    context: AssetExecutionContext,
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fetch and process all fixture events for the current season partition.

    Mirrors the legacy ETL:
      - fetch_and_process_fixture_events_multithreaded(...) to obtain
        match events and players.
      - Adds session_id to each event via the session_ids mapping.

    Additionally:
      - Persists df_all_players as DuckDB table `players`.
      - Persists df_all_match_events as DuckDB table `match_events`
        (in addition to the asset table `fixture_events` created by the IO manager).
    """
    api_sr = context.resources.sportradar_api
    duckdb_io_manager = context.resources.io_manager

    # list of fixture IDs for the season
    list_fixture_ids = fixtures["fixtureId"].unique().tolist()

    # legacy-style multi-threaded fetch + processing
    df_all_match_events, df_all_players = (
        fetch_and_process_fixture_events_multithreaded(
            api=api_sr,
            fixture_ids=list_fixture_ids,
            df_teams=teams,
            max_workers=16,
        )
    )

    # enrich match events with session_id from fixtures
    session_ids = fixtures.set_index("fixtureId")["session_id"].to_dict()
    df_all_match_events["session_id"] = df_all_match_events["fixtureId"].map(
        session_ids
    )
    df_all_match_events["session_id"] = df_all_match_events[
        "session_id"
    ].astype("Int64")

    # Persist helper tables expected by the legacy notebook-based logic
    with duckdb_io_manager._conn() as con:
        # players table
        con.register("df_all_players", df_all_players)
        con.execute(
            "CREATE OR REPLACE TABLE players AS SELECT * FROM df_all_players"
        )
        con.unregister("df_all_players")

        # match_events table (for compatibility with SQL in PlayerLeagueMapper)
        con.register("df_all_match_events", df_all_match_events)
        con.execute(
            "CREATE OR REPLACE TABLE match_events AS SELECT * FROM df_all_match_events"
        )
        con.unregister("df_all_match_events")

    context.log.info(
        "Fetched %d match events for %d fixtures (unique fixtures: %d). "
        "Persisted DuckDB tables 'players' and 'match_events'.",
        len(df_all_match_events),
        len(list_fixture_ids),
        df_all_match_events["fixtureId"].nunique(),
    )

    context.add_output_metadata(
        {
            "n_rows": len(df_all_match_events),
            "n_columns": df_all_match_events.shape[1],
            "preview": MetadataValue.md(
                df_all_match_events.head(10).to_markdown(index=False)
            ),
        }
    )

    # IOManager will persist this as table `fixture_events`
    return df_all_match_events


@asset(
    partitions_def=season_partitions,
    required_resource_keys={"io_manager", "kinexon_api"},
    group_name="kinexon_data",
    compute_kind="duckdb",
    description=(
        "Kinexon position data for all sessions referenced in fixtures. "
        "Fetches only missing sessions compared to the existing kinexon_positions table, "
        "streaming batches directly into DuckDB to avoid high RAM usage."
    ),
    deps=[
        "fixture_events",
    ],  # ensure kinexon_positions runs first
)
def kinexon_positions(
    context: AssetExecutionContext,
    fixtures: pd.DataFrame,
) -> None:
    """
    Memory-efficient incremental loading of Kinexon positions:

      - Reads only the distinct session_ids from the existing kinexon_positions table.
      - Determines which sessions from fixtures are missing.
      - Fetches positions for missing sessions in small batches and inserts each batch
        directly into DuckDB.
      - Never materializes the full kinexon_positions table in Pandas.
    """
    duckdb_io_manager = context.resources.io_manager
    api_kinexon = context.resources.kinexon_api

    SKIP_STEP = True

    with duckdb_io_manager._conn() as con:

        # Session IDs already present in positions (only the key column, small footprint)
        try:
            df_sessions_in_positions = con.execute(
                """
                SELECT DISTINCT session_id
                FROM kinexon_positions
                WHERE session_id IS NOT NULL
                """
            ).df()
            session_ids_in_positions = set(
                df_sessions_in_positions["session_id"]
                .dropna()
                .unique()
                .tolist()
            )
        except Exception:
            session_ids_in_positions = set()

        # Session IDs in fixtures (Kinexon-linked fixtures)
        session_ids_in_fixtures = set(
            fixtures["session_id"].dropna().unique().tolist()
        )

        session_ids_to_fetch = sorted(
            session_ids_in_fixtures - session_ids_in_positions
        )

        context.log.info(
            f"Already have {len(session_ids_in_positions)} session IDs in "
            "kinexon_positions table."
        )
        percent_present = (
            len(session_ids_in_positions) / len(session_ids_in_fixtures) * 100
            if session_ids_in_fixtures
            else 0.0
        )
        context.log.info(
            f"{percent_present:.2f}% of session IDs in fixtures are already present "
            "in kinexon_positions table."
        )

        # Prepare batches of up to 9 session IDs each
        split_session_id_lists = [
            session_ids_to_fetch[i : i + 9]
            for i in range(0, len(session_ids_to_fetch), 9)
        ]
        context.log.info(
            f"Fetching positions for {len(session_ids_to_fetch)} sessions "
            f"in {len(split_session_id_lists)} batches."
        )

        # Reverse mapping: session_id -> fixtureId from fixtures
        session_to_fixture = {
            v: k
            for k, v in fixtures.set_index("fixtureId")["session_id"]
            .to_dict()
            .items()
        }

        first_insert_done = False

        for session_id_list in split_session_id_lists:
            if not session_id_list:
                continue

            if SKIP_STEP:
                context.log.info(
                    f"SKIPPING fetch for session IDs: {session_id_list}"
                )
                continue

            df_positions = fetch_positions_for_fixtures_multithreaded(
                api_kinexon,
                session_id_list,
                max_workers=9,
            )

            if df_positions.empty:
                continue

            reverse_batch = {
                sid: session_to_fixture.get(sid)
                for sid in session_id_list
                if sid in session_to_fixture
            }
            df_positions["fixtureId"] = df_positions["session_id"].map(
                reverse_batch
            )

            # Create table on first batch (if it doesn't exist), then append
            if not first_insert_done:
                con.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kinexon_positions AS
                    SELECT * FROM df_positions
                    """
                )
                first_insert_done = True
            else:
                con.execute(
                    """
                    INSERT INTO kinexon_positions
                    SELECT * FROM df_positions
                    """
                )

        # Collect lightweight stats / preview from DuckDB without loading full table
        try:
            stats_df = con.execute(
                """
                SELECT
                    COUNT(*) AS n_rows,
                    COUNT(DISTINCT session_id) AS n_sessions
                FROM kinexon_positions
                """
            ).df()
            n_rows = int(stats_df.loc[0, "n_rows"])
            n_sessions_total = int(stats_df.loc[0, "n_sessions"])
            preview_df = con.execute(
                "SELECT * FROM kinexon_positions LIMIT 10"
            ).df()
            preview_md = preview_df.to_markdown(index=False)
        except Exception:
            n_rows = 0
            n_sessions_total = 0
            preview_md = "*(no data)*"

    context.log.info(
        "kinexon_positions table now has %d rows for %d distinct sessions.",
        n_rows,
        n_sessions_total,
    )

    context.add_output_metadata(
        {
            "n_rows": n_rows,
            "n_distinct_sessions": n_sessions_total,
            "n_existing_sessions_before": len(session_ids_in_positions),
            "n_sessions_in_fixtures": len(session_ids_in_fixtures),
            "n_sessions_fetched_this_run": len(session_ids_to_fetch),
            "preview": MetadataValue.md(preview_md),
        }
    )

    # No large DataFrame is returned; data lives in DuckDB.
    return None


@asset(
    partitions_def=season_partitions,
    required_resource_keys={"io_manager"},
    group_name="synced_data",
    compute_kind="duckdb",
    description=(
        "Backfills players.league_id by matching Sportradar players to Kinexon "
        "positions on a per-fixture basis."
    ),
    deps=[
        "kinexon_positions",
        "fixture_events",
    ],  # ensure kinexon_positions runs first
)
def players_merged(
    context: AssetExecutionContext,
    fixture_events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Asset wrapping the legacy notebook logic for player ↔ league mapping.
        - Uses the DuckDB IO manager connection.
        - Mutates the `players` table in-place (league_id column).
        - Returns a DataFrame of all players that were updated in this run.
    """

    duckdb_io_manager = context.resources.io_manager
    players_merged = []

    with duckdb_io_manager._conn() as con:
        # Fetch existing players table that was created in fixture_events asset
        df_players_all = con.execute("SELECT * FROM players").df()

        context.log.info(
            f"Starting player-league_id backfill for {len(df_players_all)} total players."
        )

        # Iterate over each fixture from the events data
        for fixture_id, df_events_fixture in fixture_events.groupby(
            "fixtureId"
        ):
            session_id = df_events_fixture["session_id"].dropna().unique()
            if len(session_id) == 0 or pd.isna(session_id[0]):
                context.log.debug(
                    f"Fixture {fixture_id}: Skipping, no session_id found."
                )
                continue
            session_id = session_id[0]

            # Get unique personIds from events for this fixture
            person_ids = (
                df_events_fixture["personId"].dropna().unique().tolist()
            )
            df_players_fixture = df_players_all[
                df_players_all["personId"].isin(person_ids)
            ].copy()

            context.log.info(
                f"Fixture {fixture_id}: Found {len(df_players_fixture)} unique players in events."
            )

            # Skip if no players to process
            if df_players_fixture.empty:
                context.log.warning(
                    f"Fixture {fixture_id}: No players found in events."
                )
                continue

            # Fetch unique player info from Kinexon positions for this session
            try:
                df_positions = con.execute(
                    f"""
                    SELECT DISTINCT
                        "full name"  AS full_name_kinexon,
                        "group name" AS group_name_kinexon,
                        "league id"  AS league_id
                    FROM kinexon_positions
                    WHERE "full name" IS NOT NULL
                    AND "league id" IS NOT NULL
                    AND "group name" IS NOT NULL
                    AND session_id = {session_id}
                    """
                ).df()
            except Exception as e:
                context.log.warning(
                    f"Fixture {fixture_id}: Could not query positions for session {session_id}. Error: {e}"
                )
                continue

            if df_positions.empty:
                # context.log.warning(
                #     f"Fixture {fixture_id}: No Kinexon player data found for session {session_id}."
                # )
                continue

            def fuzzy_match_players(
                df_players: pd.DataFrame,
                df_positions: pd.DataFrame,
            ) -> pd.DataFrame:
                import pandas as pd
                from thefuzz import process

                matched_rows = []
                position_names = (
                    df_positions["full_name_kinexon"].unique().tolist()
                )
                context.log.debug(
                    f"Fixture {fixture_id}: Starting fuzzy matching for {len(df_players)} players against {len(position_names)} Kinexon positions."
                )
                position_groups = (
                    df_positions["group_name_kinexon"].unique().tolist()
                )
                for _, player_row in df_players.iterrows():
                    player_name = player_row["nameFullLocal"]
                    player_team = player_row["teamName"]
                    match, score = process.extractOne(
                        player_name, position_names
                    )
                    if score >= 80:  # threshold for a good match
                        kinexon_row = df_positions[
                            df_positions["full_name_kinexon"] == match
                        ].iloc[0]
                        player_row["kin_league_id"] = kinexon_row["league_id"]
                        matched_rows.append(player_row)
                    else:
                        context.log.debug(
                            f"Player '{player_name}' did not find a good match (score: {score}, {match})."
                            "Might just not be in the kinexon data (e.g. did not play)"
                        )
                        # print teams
                        context.log.debug(
                            f"Available Kinexon groups: {position_groups}, Player team: {player_team}"
                        )

                # log how many kinexon players are not matched
                context.log.info(
                    f"Fixture {fixture_id}: Matched {len(matched_rows)} out of {len(df_players)} players."
                )
                return pd.DataFrame(matched_rows)

            context.log.info(
                f"Fixture {fixture_id}: Performing fuzzy matching of players to Kinexon positions."
            )

            df_players_matched = fuzzy_match_players(
                df_players_fixture, df_positions
            )
            if df_players_matched.empty:
                continue

            players_merged.append(df_players_matched)
    if players_merged:
        players_merged = pd.concat(players_merged).drop_duplicates(
            subset=["personId"]
        )
        context.log.info(
            f"Total players matched across all fixtures: {len(players_merged)}"
        )
    else:
        players_merged = pd.DataFrame()
    context.log.info(
        f"Total unique players updated with league_id: {len(players_merged)}"
    )
    return players_merged


@asset(
    required_resource_keys={"io_manager"},
    group_name="maintenance",
    compute_kind="duckdb",
    description=(
        "Backfills players.league_id by matching Sportradar players to Kinexon "
        "positions (name + team, exact + fuzzy, team-aligned)."
    ),
)
def backfill_player_league_ids(context: AssetExecutionContext) -> pd.DataFrame:
    """
    Asset wrapping the legacy notebook logic for player ↔ league mapping.

    - Uses the DuckDB IO manager connection.
    - Mutates the `players` table in-place (league_id column).
    - Returns the updated players subset (league_id IS NOT NULL) as an asset table.
    """
    duckdb_io_manager = context.resources.io_manager

    with duckdb_io_manager._conn() as con:
        mapper = PlayerLeagueMapper(con=con, logger=context.log)
        df_match_all, df_map_safe, df_players_updated = mapper.run()

    num_unique_players_matched = (
        df_map_safe["personId"].nunique() if not df_map_safe.empty else 0
    )

    context.add_output_metadata(
        {
            "n_match_all": len(df_match_all),
            "n_safe_mappings": len(df_map_safe),
            "n_unique_players_matched": num_unique_players_matched,
            "n_players_updated": len(df_players_updated),
            "preview_matches": MetadataValue.md(
                df_match_all.sort_values("similarity", ascending=False)
                .head(30)
                .to_markdown(index=False)
                if not df_match_all.empty
                else "*(no matches)*"
            ),
            "preview_players_updated": MetadataValue.md(
                df_players_updated.head(20).to_markdown(index=False)
                if not df_players_updated.empty
                else "*(no players updated)*"
            ),
        }
    )

    return df_players_updated


@asset(
    partitions_def=season_partitions,
    required_resource_keys={"io_manager", "kinexon_api"},
    group_name="kinexon_data",
    compute_kind="duckdb",
    description=(""),
)
def kinexon_events(
    context: AssetExecutionContext,
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """
    Memory-efficient incremental loading of Kinexon positions:

      - Reads only the distinct session_ids from the existing kinexon_positions table.
      - Determines which sessions from fixtures are missing.
      - Fetches positions for missing sessions in small batches and inserts each batch
        directly into DuckDB.
      - Never materializes the full kinexon_positions table in Pandas.
    """
    api_kinexon = context.resources.kinexon_api

    # session_ids in kinexon_positions
    list_session_ids_in_positions = (
        fixtures["session_id"].dropna().unique().tolist()
    )

    df_all_detected_events = fetch_detected_events_for_sessions_multithreaded(
        api_kinexon,
        list_session_ids_in_positions,
        max_workers=9,
    )
    context.log.info(
        f"Fetched {len(df_all_detected_events)} detected events for "
        f"{len(list_session_ids_in_positions)} sessions."
    )
    context.add_output_metadata(
        {
            "n_rows": len(df_all_detected_events),
            "n_columns": df_all_detected_events.shape[1],
            "preview": MetadataValue.md(
                df_all_detected_events.head(10).to_markdown(index=False)
            ),
        }
    )
    return df_all_detected_events


@asset(
    partitions_def=season_partitions,
    required_resource_keys={"io_manager"},
    group_name="synced_data",
    compute_kind="duckdb",
    description="Sportradar goal events synchronized with the nearest Kinexon event.",
)
def sportradar_goals_synced(
    context: AssetExecutionContext,
    fixture_events: pd.DataFrame,
    kinexon_events: pd.DataFrame,
    players_merged: pd.DataFrame,
) -> pd.DataFrame:
    """
    Finds the closest Kinexon event in time for each Sportradar goal event
    using the robust, fixture-based nearest-neighbor logic from the reference notebook.
    This asset only syncs event data, not position data.
    """
    context.log.info(
        f"Starting goal sync. Received {len(fixture_events)} fixture events and {len(kinexon_events)} Kinexon events."
    )

    if fixture_events.empty or kinexon_events.empty:
        context.log.warning(
            "One or both input DataFrames are empty. Skipping sync."
        )
        return pd.DataFrame()

    # 1. Prepare Sportradar goals
    df_goals = fixture_events[fixture_events["eventType"] == "goal"].copy()
    if df_goals.empty:
        context.log.warning(
            "No events of type 'goal' found in fixture_events."
        )
        return pd.DataFrame()

    df_goals["eventTime"] = pd.to_datetime(
        df_goals["eventTime"], utc=True, errors="coerce"
    )
    df_goals["eventTime_ms"] = (
        df_goals["eventTime"].astype("int64") // 1_000_000
    )
    df_goals = df_goals.dropna(subset=["eventTime_ms", "fixtureId"])
    df_goals["fixtureId"] = df_goals["fixtureId"].astype(str)
    context.log.info(f"Prepared {len(df_goals)} goal events for matching.")

    # insert goalkeeper_league_id via players_merged
    player_league_map = (
        players_merged[["personId", "kin_league_id"]]
        .drop_duplicates()
        .set_index("personId")["kin_league_id"]
        .to_dict()
    )
    df_goals["goalkeeper_league_id"] = df_goals["goalKeeperId"].map(
        player_league_map
    )

    # 2. Prepare Kinexon events
    df_kinexon_events = kinexon_events.copy()

    # insert fixture_id into df_kinexon from df_goals via session_id
    fixture_id_map = (
        df_goals[["fixtureId", "session_id"]]
        .drop_duplicates()
        .set_index("session_id")["fixtureId"]
        .to_dict()
    )
    df_kinexon_events["fixture_id"] = df_kinexon_events["session_id"].map(
        fixture_id_map
    )
    df_kinexon_events["timestamp_ms"] = pd.to_numeric(
        df_kinexon_events["timestamp_ms"], errors="coerce"
    )
    df_kinexon_events = df_kinexon_events.dropna(
        subset=["timestamp_ms", "fixture_id"]
    )
    df_kinexon_events["fixture_id"] = df_kinexon_events["fixture_id"].astype(
        str
    )
    context.log.info(
        f"Prepared {len(df_kinexon_events)} Kinexon events for matching."
    )

    # 3. Match goals to nearest Kinexon event within each fixture
    TOLERANCE_MS = 30_000  # 30 seconds
    all_synced_fixtures = []
    processed_fixtures = 0

    for fixture_id, goals_grp in df_goals.groupby("fixtureId"):
        processed_fixtures += 1
        kinexon_grp = df_kinexon_events[
            df_kinexon_events["fixture_id"] == fixture_id
        ]
        if kinexon_grp.empty:
            context.log.debug(
                f"No Kinexon events found for fixture_id {fixture_id}. Skipping."
            )
            continue

        # Sort by time for search
        kinexon_grp = kinexon_grp.sort_values("timestamp_ms").reset_index(
            drop=True
        )
        kinexon_times = kinexon_grp["timestamp_ms"].to_numpy(dtype="int64")
        goal_times = goals_grp["eventTime_ms"].to_numpy(dtype="int64")

        # Find insertion points for each goal time in the kinexon times array
        idx_right = np.searchsorted(kinexon_times, goal_times, side="left")
        idx_left = (idx_right - 1).clip(min=0)

        # For each goal, find the best match (left or right)
        best_indices = []
        best_deltas = []
        for i, goal_time in enumerate(goal_times):
            best_idx, best_delta = -1, float("inf")

            # Check right neighbor
            if idx_right[i] < len(kinexon_times):
                delta = kinexon_times[idx_right[i]] - goal_time
                if abs(delta) < abs(best_delta):
                    best_delta = delta
                    best_idx = idx_right[i]

            # Check left neighbor
            if idx_left[i] < len(kinexon_times):
                delta = kinexon_times[idx_left[i]] - goal_time
                if abs(delta) < abs(best_delta):
                    best_delta = delta
                    best_idx = idx_left[i]

            if abs(best_delta) <= TOLERANCE_MS:
                best_indices.append(best_idx)
                best_deltas.append(int(best_delta))
            else:
                best_indices.append(None)
                best_deltas.append(None)

        # Join the matches back to the goals group
        matched_goals_grp = goals_grp.copy()
        matched_goals_grp["kinexon_match_index"] = best_indices
        matched_goals_grp["time_diff_ms"] = best_deltas

        # Filter out goals that didn't get a match
        matched_goals_grp = matched_goals_grp.dropna(
            subset=["kinexon_match_index"]
        ).copy()
        if matched_goals_grp.empty:
            continue

        matched_goals_grp["kinexon_match_index"] = matched_goals_grp[
            "kinexon_match_index"
        ].astype(int)

        # Get the corresponding Kinexon event data
        kinexon_matches = kinexon_grp.iloc[
            matched_goals_grp["kinexon_match_index"]
        ].add_prefix("kin_")

        # Combine goal data with the matched kinexon data
        final_grp = pd.concat(
            [
                matched_goals_grp.reset_index(drop=True),
                kinexon_matches.reset_index(drop=True),
            ],
            axis=1,
        )
        all_synced_fixtures.append(final_grp)

    if not all_synced_fixtures:
        context.log.warning(
            "No goals were successfully synced with Kinexon events across all fixtures."
        )
        return pd.DataFrame()

    df_final_synced = pd.concat(all_synced_fixtures, ignore_index=True)
    total_goals = len(df_goals)
    matched_goals = len(df_final_synced)
    match_rate = (matched_goals / total_goals * 100) if total_goals > 0 else 0

    context.log.info(
        f"Sync process complete. Matched {matched_goals} of {total_goals} goals ({match_rate:.2f}%)."
    )
    context.add_output_metadata(
        {
            "n_rows": matched_goals,
            "n_columns": df_final_synced.shape[1],
            "total_goals_processed": total_goals,
            "match_rate": f"{match_rate:.2f}%",
            "time_tolerance_ms": TOLERANCE_MS,
            "preview": MetadataValue.md(
                df_final_synced.head(10).to_markdown(index=False)
            ),
        }
    )

    return df_final_synced


@asset(
    partitions_def=season_partitions,
    required_resource_keys={"io_manager"},
    group_name="synced_data",
    compute_kind="duckdb",
    description="Refines throw timestamps for synced Sportradar goals using Kinexon position data.",
    deps=["kinexon_positions"],
)
def sportradar_goals_refined(
    context: AssetExecutionContext,
    sportradar_goals_synced: pd.DataFrame,
) -> pd.DataFrame:
    """
    Iterates over synced goals, fetches Kinexon positions per fixture,
    and applies the throw-time refinement heuristic.
    """
    duckdb_io_manager = context.resources.io_manager

    if sportradar_goals_synced.empty:
        context.log.warning("Input sportradar_goals_synced is empty.")
        return pd.DataFrame()

    # We will collect the results here
    refined_rows = []

    # Group by fixture to process batchwise
    for fixture_id, df_fixture_goals in sportradar_goals_synced.groupby(
        "fixtureId"
    ):
        # cols class	eventId	eventTime	eventType	subType	attendance	entityId	personId	bib	name	position	scores	periodId	playId	clock	success	x	y	attackType	goalKeeperId	location	failureReason	emptyNet	fixtureId	teamName	personName	goalkeeperName	session_id	eventTime_ms	goalkeeper_league_id	kinexon_match_index	time_diff_ms	kin_timestamp	kin_timestamp_ms	kin_timezone_id	kin_game_clock	kin_period	kin_player_id	kin_distance	kin_speed_ball	kin_trajectory	kin_shot_position_x	kin_shot_position_y	kin_hit_position_y	kin_hit_position_z	kin_success	kin_shot_category	kin_goalkeeper_id	kin_shot_type	kin_assisting_player_id	kin_validated	kin_id	kin_event_type	kin_league_id	kin_session_id	kin_fixture_id
        date_game = df_fixture_goals["eventTime"].iloc[0]
        date_only = date_game.date() if pd.notna(date_game) else "unknown"
        # uniques of teamName as string: xx_vs_yy
        competitors = "_vs_".join(df_fixture_goals["teamName"].unique())
        context.log.info(
            f"Refining {len(df_fixture_goals)} goals for fixture {fixture_id} ({competitors}) on {date_only}"
        )

        # Fetch positions for this fixture
        try:
            with duckdb_io_manager._conn() as con:
                df_positions = con.execute(
                    f"SELECT * FROM kinexon_positions WHERE fixtureId = '{fixture_id}'"
                ).df()
        except Exception as e:
            context.log.error(
                f"Error fetching positions for fixture {fixture_id}: {e}"
            )
            df_positions = pd.DataFrame()

        if df_positions.empty:
            context.log.warning(
                f"No positions found for fixture {fixture_id}. Skipping refinement."
            )
            continue
            # context.log.warning(
            #     f"No positions found for fixture {fixture_id}. Skipping refinement."
            # )
            # Append original rows without refinement
            for _, row in df_fixture_goals.iterrows():
                refined_rows.append(row)
            continue

        # Pre-process positions
        if "ts" not in df_positions.columns:
            if "ts in ms" in df_positions.columns:
                df_positions["ts"] = pd.to_datetime(
                    df_positions["ts in ms"],
                    unit="ms",
                    utc=True,
                    errors="coerce",
                )

        # Build fixture_to_session map
        fixture_to_session = {}
        if "session_id" in df_positions.columns:
            s_ids = df_positions["session_id"].dropna().unique()
            if len(s_ids) > 0:
                fixture_to_session[fixture_id] = s_ids[0]
        else:
            context.log.warning(
                f"No session_id column in positions for fixture {fixture_id}."
            )

        # Apply refinement
        for _, row in df_fixture_goals.iterrows():
            try:
                diag_series = refine_throw_time_for_event(
                    row=row,
                    df_positions_all=df_positions,
                    fixture_to_session=fixture_to_session,
                    plot=False,
                )
                # Merge diag_series into row
                combined = pd.concat([row, diag_series])
                # Remove duplicate index labels (keep last/refined)
                combined = combined.loc[
                    ~combined.index.duplicated(keep="last")
                ]
                refined_rows.append(combined)
            except Exception as e:
                context.log.error(
                    f"Error refining event {row.get('eventId')}: {e}"
                )
                refined_rows.append(row)

    if not refined_rows:
        return pd.DataFrame()

    df_refined = pd.DataFrame(refined_rows)

    context.add_output_metadata(
        {
            "n_rows": len(df_refined),
            "n_refined": (
                int(df_refined["refined_throw_ts"].notna().sum())
                if "refined_throw_ts" in df_refined.columns
                else 0
            ),
            "preview": MetadataValue.md(
                df_refined.head().to_markdown(index=False)
            ),
        }
    )

    return df_refined
