from dagster import (
    asset,
    AssetExecutionContext,
    MetadataValue,
)
import pandas as pd
from events_sportradar.fetch_players_from_fixture_events import (
    enrich_and_filter_players,
)
from src.events_sportradar.fetch_teams import fetch_teams_by_season_id
from src.events_sportradar.fetch_list_fixtures import (
    fetch_list_fixtures,
    refine_fixtures_data,
    expand_competitors_in_fixtures,
)
from src.events_kinexon.fetch_session_id_for_fixtures import (
    fetch_session_ids_for_fixtures,
)
from events_sportradar.fetch_fixture_events import (
    fetch_events_for_fixture,
    process_fixture_events,
)

from .assets_ids import (
    fixtures_partition_def,
)  # pylint: disable=relative-beyond-top-level
from .utils.duckdb_helpers import (
    duckdb_conn,
)  # pylint: disable=relative-beyond-top-level
from .utils.metadata import (
    preview_metadata,
)  # pylint: disable=relative-beyond-top-level


@asset(
    partitions_def=fixtures_partition_def,
    required_resource_keys={"sportradar_api", "io_manager"},
    group_name="sportradar_data",
    compute_kind="duckdb",
    description="Match events for a single fixture, with incremental DuckDB persistence.",
    metadata={"partition_column": "fixture_id"},
)
def fixture_events_raw(
    context: AssetExecutionContext,
    list_fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch and persist events for a single fixture partition."""

    # Get API SR
    api_sr = context.resources.sportradar_api

    # Filter for current partition
    fixture_id = context.partition_key
    list_fixtures["fixture_id"] = list_fixtures["fixture_id"].astype(str)
    df_fixture = list_fixtures[list_fixtures["fixture_id"] == fixture_id]
    session_id = df_fixture["session_id"].iloc[0]

    if df_fixture.empty:
        context.log.warning(
            f"Fixture {fixture_id} not found in fixtures asset output."
        )
        return pd.DataFrame()

    match_events_raw = fetch_events_for_fixture(api_sr, fixture_id)
    if not match_events_raw:
        context.log.warning(
            f"No Sportradar events returned for fixture {fixture_id}."
        )
        return pd.DataFrame()

    match_events_raw = pd.DataFrame(match_events_raw)
    # rename fixtureId to fixture_id
    match_events_raw = match_events_raw.rename(
        columns={"fixtureId": "fixture_id"}
    )

    # Insert session_id if available
    if pd.notna(session_id):
        match_events_raw["session_id"] = int(session_id)
        match_events_raw["session_id"] = match_events_raw["session_id"].astype(
            "Int64"
        )

    context.log.info(
        "Fetched %d match events and for fixture %s.",
        len(match_events_raw),
        fixture_id,
    )
    # convert scores to db friendly format
    if "scores" in match_events_raw.columns:
        match_events_raw["scores"] = match_events_raw["scores"].apply(
            lambda x: str(x) if pd.notna(x) else x
        )

    metadata = {
        "fixture_id": MetadataValue.text(str(fixture_id)),
        "date_game_start": MetadataValue.text(
            str(df_fixture["start_time_local"].iloc[0])
        ),
        "name_team_home": MetadataValue.text(
            str(df_fixture["name_team_home"].iloc[0])
        ),
        "name_team_away": MetadataValue.text(
            str(df_fixture["name_team_away"].iloc[0])
        ),
        "round_number": MetadataValue.text(
            str(df_fixture["round_number"].iloc[0])
        ),
        "n_events": len(match_events_raw),
        "n_columns": match_events_raw.shape[1],
    }

    context.add_output_metadata(metadata)
    context.add_output_metadata(preview_metadata(match_events_raw))
    return match_events_raw


@asset(
    partitions_def=fixtures_partition_def,
    required_resource_keys={"sportradar_api", "io_manager"},
    group_name="sportradar_data",
    compute_kind="duckdb",
    description="Match events for a single fixture, with incremental DuckDB persistence.",
    metadata={"partition_column": "fixture_id"},
)
def fixture_events_match(
    context: AssetExecutionContext,
    fixture_events_raw: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch and persist events for a single fixture partition."""

    # Get API SR
    api_sr = context.resources.sportradar_api

    fixture_id = context.partition_key
    df_fixture_raw = fixture_events_raw.copy()
    df_fixture_raw["fixture_id"] = df_fixture_raw["fixture_id"].astype(str)
    df_fixture_raw = df_fixture_raw[df_fixture_raw["fixture_id"] == fixture_id]

    if df_fixture_raw.empty:
        context.log.warning(
            f"Fixture {fixture_id} not found in fixtures asset output."
        )
        return pd.DataFrame()

    # df_raw to list
    match_events_list = df_fixture_raw.to_dict(orient="records")

    df_match_events, _ = process_fixture_events(
        fixture_id=fixture_id,
        match_events=match_events_list,
        df_teams=teams,
        api=api_sr,
    )

    context.log.info(
        "Fetched %d match events for fixture %s.",
        len(df_match_events),
        # len(df_players),
        fixture_id,
    )

    metadata = {
        "fixture_id": MetadataValue.text(str(fixture_id)),
        "n_unique_players": MetadataValue.text(
            str(df_match_events["person_id"].nunique())
        ),
        # "name_team_home": MetadataValue.text(
        #     str(df_match_events["name_team_home"].iloc[0])
        # ),
        # "name_team_away": MetadataValue.text(
        #     str(df_match_events["name_team_away"].iloc[0])
        # ),
        # "round_number": MetadataValue.text(
        #     str(df_match_events["round_number"].iloc[0])
        # ),
        # "n_events": len(df_match_events),
        # "n_columns": df_match_events.shape[1],
    }

    context.add_output_metadata(metadata)

    context.add_output_metadata(preview_metadata(df_match_events))
    return df_match_events


@asset(
    partitions_def=fixtures_partition_def,
    required_resource_keys={"sportradar_api", "io_manager"},
    group_name="sportradar_data",
    compute_kind="duckdb",
    description="Match events for a single fixture, with incremental DuckDB persistence.",
    metadata={"partition_column": "fixture_id"},
)
def fixture_players(
    context: AssetExecutionContext,
    fixture_events_raw: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch and persist events for a single fixture partition."""

    # Get API SR
    api_sr = context.resources.sportradar_api

    fixture_id = context.partition_key
    df_fixture_raw = fixture_events_raw.copy()
    df_fixture_raw["fixture_id"] = df_fixture_raw["fixture_id"].astype(str)
    df_fixture_raw = df_fixture_raw[df_fixture_raw["fixture_id"] == fixture_id]

    if df_fixture_raw.empty:
        context.log.warning(
            f"Fixture {fixture_id} not found in fixtures asset output."
        )
        return pd.DataFrame()

    _, df_players = process_fixture_events(
        fixture_id=fixture_id,
        match_events=df_fixture_raw.to_dict(orient="records"),
        df_teams=teams,
        api=api_sr,
    )

    if not df_players.empty and "person_id" in df_players.columns:
        df_players = df_players.drop_duplicates(subset=["person_id"])

    # Set partition column
    df_players["fixture_id"] = fixture_id

    # with duckdb_conn(duckdb_io_manager) as con:
    #     # Persist players incrementally
    #     if not df_players.empty:
    #         con.register("df_fixture_players", df_players)
    #         if not _table_exists(con, "players"):
    #             con.execute(
    #                 "CREATE TABLE players AS SELECT * FROM df_fixture_players"
    #             )
    #         else:
    #             con.execute(
    #                 "DELETE FROM players WHERE personId IN ("
    #                 "SELECT personId FROM df_fixture_players WHERE personId IS NOT NULL"
    #                 ")"
    #             )
    #             con.execute(
    #                 "INSERT INTO players SELECT * FROM df_fixture_players"
    #             )
    #         con.unregister("df_fixture_players")

    context.log.info(
        "Fetched %d match players for fixture %s.",
        len(df_players),
        fixture_id,
    )

    context.add_output_metadata(preview_metadata(df_players))
    return df_players
