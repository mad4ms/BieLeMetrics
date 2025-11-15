# src/hbl_etl_dagster/assets.py

from dagster import (
    asset,
    AssetExecutionContext,
    StaticPartitionsDefinition,
    MetadataValue,
)
import pandas as pd
from typing import Dict
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
    required_resource_keys={"sportradar_api"},
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


@asset(
    partitions_def=season_partitions,
    required_resource_keys={"kinexon_api"},
    group_name="kinexon_data",
    io_manager_key="file_io_manager",  # This ensures the asset is kept in memory
    compute_kind="api",
)
def session_ids(
    context: AssetExecutionContext, fixtures: pd.DataFrame
) -> Dict[str, int]:
    """
    Fetch session IDs for fixtures from Kinexon API.
    Depends on the persisted fixtures asset.
    """
    api = context.resources.kinexon_api

    dict_session_ids = fetch_session_ids_for_fixtures(
        api=api,
        df_fixtures=fixtures,
    )
    # contains mapping fixture_id -> session_id

    context.log.info(
        f"Fetched session IDs for {len(dict_session_ids)} fixtures from Kinexon."
    )
    return dict_session_ids


@asset(
    partitions_def=season_partitions,
    group_name="sportradar_data",
    compute_kind="duckdb",
    description="Fixtures data merged with Kinexon session IDs.",
)
def fixtures_with_sessions(
    context: AssetExecutionContext,
    fixtures: pd.DataFrame,
    session_ids: Dict[str, int],
) -> pd.DataFrame:
    """
    Merges the Kinexon session_id map into the fixtures table.
    This creates the final, enriched fixtures asset.
    """
    # Create a DataFrame from the session_ids dictionary
    df_sessions = pd.DataFrame(
        list(session_ids.items()), columns=["fixtureId", "session_id"]
    )

    # Merge the session_id into the fixtures DataFrame
    df_merged = pd.merge(fixtures, df_sessions, on="fixtureId", how="left")

    # Convert session_id to integer, allowing for NaNs where no match was found
    df_merged["session_id"] = df_merged["session_id"].astype("Int64")

    context.log.info(f"Merged session IDs into {len(df_merged)} fixtures.")
    context.add_output_metadata(
        {
            "n_rows": len(df_merged),
            "n_columns": df_merged.shape[1],
            "preview": MetadataValue.md(
                df_merged.head(10).to_markdown(index=False)
            ),
        }
    )

    return df_merged


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
    session_ids: Dict[str, int],
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

    # enrich match events with session_id from Kinexon mapping
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
        "Kinexon position data for all sessions referenced in fixtures_with_sessions. "
        "Fetches only missing sessions compared to the existing kinexon_positions table, "
        "streaming batches directly into DuckDB to avoid high RAM usage."
    ),
)
def kinexon_positions(
    context: AssetExecutionContext,
    fixtures_with_sessions: pd.DataFrame,
    session_ids: Dict[str, int],
) -> None:
    """
    Memory-efficient incremental loading of Kinexon positions:

      - Reads only the distinct session_ids from the existing kinexon_positions table.
      - Determines which sessions from fixtures_with_sessions are missing.
      - Fetches positions for missing sessions in small batches and inserts each batch
        directly into DuckDB.
      - Never materializes the full kinexon_positions table in Pandas.
    """
    duckdb_io_manager = context.resources.io_manager
    api_kinexon = context.resources.kinexon_api

    with duckdb_io_manager._conn() as con:

        return None
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

        # Session IDs in fixtures_with_sessions (Kinexon-linked fixtures)
        session_ids_in_fixtures = set(
            fixtures_with_sessions["session_id"].dropna().unique().tolist()
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

        # Reverse mapping: session_id -> fixtureId
        session_to_fixture = {
            v: k
            for k, v in session_ids.items()
            if v in session_ids_in_fixtures
        }

        first_insert_done = False

        for session_id_list in split_session_id_lists:
            if not session_id_list:
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
    fixtures_with_sessions: pd.DataFrame,
    session_ids: Dict[str, int],
) -> pd.DataFrame:
    """
    Memory-efficient incremental loading of Kinexon positions:

      - Reads only the distinct session_ids from the existing kinexon_positions table.
      - Determines which sessions from fixtures_with_sessions are missing.
      - Fetches positions for missing sessions in small batches and inserts each batch
        directly into DuckDB.
      - Never materializes the full kinexon_positions table in Pandas.
    """
    api_kinexon = context.resources.kinexon_api

    # session_ids in kinexon_positions
    list_session_ids_in_positions = (
        fixtures_with_sessions["session_id"].dropna().unique().tolist()
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
    group_name="kinexon_data",
    compute_kind="duckdb",
    description="Synchronizes Kinexon detected events with the nearest Kinexon position data.",
)
def kinexon_events_synced(
    context: AssetExecutionContext,
    kinexon_events: pd.DataFrame,
    fixture_events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges Kinexon events with their closest position data point in time.
    This mirrors the logic from the synchronization notebook.
    """
    df_kinexon_events = kinexon_events.copy()

    context.log.info(f"Synchronizing {len(fixture_events)} Sportradar events.")
    context.log.info(f"with {len(df_kinexon_events)} Kinexon events.")

    # 1. Prepare Sportradar goals
    df_sportradar_goals = fixture_events[
        fixture_events["eventType"] == "goal"
    ].copy()
    df_sportradar_goals["eventTime"] = pd.to_datetime(
        df_sportradar_goals["eventTime"], format="ISO8601", errors="coerce"
    )
    df_sportradar_goals["eventTime_ms"] = (
        df_sportradar_goals["eventTime"].astype("int64") // 1_000_000
    )
    df_sportradar_goals = df_sportradar_goals.sort_values(
        "eventTime_ms"
    ).dropna(subset=["eventTime_ms", "personId"])
    df_sportradar_goals["personId"] = df_sportradar_goals["personId"].astype(
        str
    )

    # Ensure correct types and sort for merge_asof
    df_kinexon_events["timestamp_ms"] = pd.to_numeric(
        df_kinexon_events["timestamp_ms"], errors="coerce"
    )

    df_kinexon_synced = df_kinexon_events.sort_values("timestamp_ms").copy()
    # The 'player_id' from Kinexon events corresponds to 'personId' from Sportradar
    df_kinexon_synced = df_kinexon_synced.rename(
        columns={"player_id": "personId"}
    )
    df_kinexon_synced["personId"] = df_kinexon_synced["personId"].astype(str)

    # 3. Perform the time-based merge
    # This finds the nearest Kinexon event for each goal, matching on player and time.
    df_synced_goals = pd.merge_asof(
        left=df_sportradar_goals,
        right=df_kinexon_synced,
        left_on="eventTime_ms",
        right_on="timestamp_ms",
        by="personId",  # Match goals to events from the same player
        direction="nearest",
        tolerance=30000,  # 30-second tolerance, as in the notebook
    )

    # Calculate the time difference for diagnostics
    df_synced_goals["time_diff_ms"] = (
        df_synced_goals["timestamp_ms"] - df_synced_goals["eventTime_ms"]
    )

    context.log.info(
        f"Synced {len(df_synced_goals)} goal events. "
        f"{df_synced_goals['timestamp_ms'].notna().sum()} goals had a Kinexon event within tolerance."
    )
    df_synced_goals_not_na = df_synced_goals[
        df_synced_goals["time_diff_ms"].notna()
    ]
    context.add_output_metadata(
        {
            "n_rows": len(df_synced_goals),
            "n_columns": df_synced_goals.shape[1],
            # "n_matched_goals": df_synced_goals["timestamp_ms"].notna().sum(),
            "preview": MetadataValue.md(
                df_synced_goals_not_na.head(10).to_markdown(index=False)
            ),
        }
    )

    return df_synced_goals


@asset(
    partitions_def=season_partitions,
    required_resource_keys={"io_manager"},
    group_name="synced_data",
    compute_kind="duckdb",
    description="Sportradar goal events synchronized with the nearest Kinexon event and position data.",
)
def sportradar_goals_synced(
    context: AssetExecutionContext,
    fixture_events: pd.DataFrame,
    kinexon_events_synced: pd.DataFrame,
) -> pd.DataFrame:
    """
    Finds the closest Kinexon event in time for each Sportradar goal event.

    This asset replicates the core logic of the `nb_4_sync_with_kinexon_event_api.ipynb`
    notebook, using an efficient `merge_asof` operation.
    """
    # 1. Prepare Sportradar goals
    df_goals = fixture_events[fixture_events["eventType"] == "goal"].copy()
    df_goals["eventTime"] = pd.to_datetime(
        df_goals["eventTime"], format="ISO8601", errors="coerce"
    )
    df_goals["eventTime_ms"] = (
        df_goals["eventTime"].astype("int64") // 1_000_000
    )
    df_goals = df_goals.sort_values("eventTime_ms").dropna(
        subset=["eventTime_ms", "personId"]
    )
    df_goals["personId"] = df_goals["personId"].astype(str)

    # 2. Prepare Kinexon synced events
    df_kinexon_synced = kinexon_events_synced.sort_values(
        "timestamp_ms"
    ).copy()
    # The 'player_id' from Kinexon events corresponds to 'personId' from Sportradar
    df_kinexon_synced = df_kinexon_synced.rename(
        columns={"player_id": "personId"}
    )
    df_kinexon_synced["personId"] = df_kinexon_synced["personId"].astype(str)

    # 3. Perform the time-based merge
    # This finds the nearest Kinexon event for each goal, matching on player and time.
    df_synced_goals = pd.merge_asof(
        left=df_goals,
        right=df_kinexon_synced,
        left_on="eventTime_ms",
        right_on="timestamp_ms",
        by="personId",  # Match goals to events from the same player
        direction="nearest",
        tolerance=30000,  # 30-second tolerance, as in the notebook
    )

    # Calculate the time difference for diagnostics
    df_synced_goals["time_diff_ms"] = (
        df_synced_goals["timestamp_ms"] - df_synced_goals["eventTime_ms"]
    )

    context.log.info(
        f"Synced {len(df_synced_goals)} goal events. "
        f"{df_synced_goals['timestamp_ms'].notna().sum()} goals had a Kinexon event within tolerance."
    )
    context.add_output_metadata(
        {
            "n_rows": len(df_synced_goals),
            "n_columns": df_synced_goals.shape[1],
            # "n_matched_goals": df_synced_goals["timestamp_ms"].notna().sum(),
            "preview": MetadataValue.md(
                df_synced_goals.head(10).to_markdown(index=False)
            ),
        }
    )

    return df_synced_goals
