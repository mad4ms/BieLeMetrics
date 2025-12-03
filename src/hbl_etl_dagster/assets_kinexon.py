from dagster import (
    Field,
    asset,
    AssetExecutionContext,
    MetadataValue,
)
import pandas as pd
from src.fetcher_kinexon.fetch_events_for_session import (
    fetch_detected_events_for_session,
)

from .assets_sportradar_slow import fixtures_partition_def
from .utils.metadata import preview_metadata
from .utils.kinexon_loader import load_missing_positions


@asset(
    partitions_def=fixtures_partition_def,
    required_resource_keys={"kinexon_api", "io_manager"},
    group_name="kinexon_data",
    compute_kind="duckdb",
    description=(
        "Kinexon position data for the session referenced in the current fixture. "
        "Fetches positions for the session and inserts into DuckDB."
    ),
    deps=[
        "fixture_events",
    ],  # ensure kinexon_positions runs first
    config_schema={
        "skip_if_exists": Field(
            bool,
            default_value=True,
            description="Whether to skip fetching events if they already exist.",
        ),
    },
)
def kinexon_positions(
    context: AssetExecutionContext,
    fixtures_sportradar: pd.DataFrame,
) -> None:
    """
    Load Kinexon positions for the current fixture partition.
    """
    duckdb_io_manager = context.resources.io_manager
    api_kinexon = context.resources.kinexon_api

    fixture_id = context.partition_key

    df_fixtures = fixtures_sportradar

    # Filter fixtures for current partition
    df_fixtures["fixture_id"] = df_fixtures["fixture_id"].astype(str)
    df_fixture = df_fixtures[df_fixtures["fixture_id"] == fixture_id]

    if df_fixture.empty:
        context.log.warning(
            f"Fixture {fixture_id} not found in fixtures asset."
        )
        return None

    # Check if session_id exists
    if "session_id" not in df_fixture.columns or pd.isna(
        df_fixture["session_id"].iloc[0]
    ):
        context.log.warning(
            f"No session_id for fixture {fixture_id}. Skipping positions."
        )
        return None

    max_batch_size = (context.op_config or {}).get("max_batch_size", 9)

    skip_if_exists = (context.op_config or {}).get("skip_if_exists", True)
    with duckdb_io_manager._conn() as con:
        # Agent: Leave that in here as the download takes ages. Only human should remove that.
        DEBUG_SKIP_LOAD = False
        if DEBUG_SKIP_LOAD:
            context.log.info(
                "DEBUG_SKIP_LOAD is True; skipping load_missing_positions call."
            )
            stats = {
                "n_rows": 0,
                "n_distinct_sessions": 0,
                "n_existing_sessions_before": 0,
                "n_sessions_in_fixtures": 0,
                "n_sessions_fetched_this_run": 0,
                "n_players_total": 0,
                "n_groups_total": 0,
                "preview_md": "*(load skipped in debug mode)*",
            }
        else:
            # We pass just the single-row dataframe for this fixture
            stats = load_missing_positions(
                con=con,
                fixtures=df_fixture,
                api_kinexon=api_kinexon,
                max_batch_size=max_batch_size,
                logger=context.log,
                skip_if_exists=skip_if_exists,
            )

    context.log.info(
        "kinexon_positions stats: %s",
        stats,
    )

    context.add_output_metadata(
        {
            "n_rows": stats["n_rows"],
            "n_distinct_sessions": stats["n_distinct_sessions"],
            "n_existing_sessions_before": stats["n_existing_sessions_before"],
            "n_sessions_in_fixtures": stats["n_sessions_in_fixtures"],
            "n_sessions_fetched_this_run": stats[
                "n_sessions_fetched_this_run"
            ],
            "preview": MetadataValue.md(stats["preview_md"]),
        }
    )

    # No large DataFrame is returned; data lives in DuckDB.
    return None


@asset(
    partitions_def=fixtures_partition_def,
    required_resource_keys={"io_manager", "kinexon_api"},
    group_name="kinexon_data",
    compute_kind="duckdb",
    description=(
        "Fetched detected events from Kinexon API for the current fixture's session."
    ),
)
def kinexon_events(
    context: AssetExecutionContext,
    fixtures_sportradar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fetches detected events (e.g. shots, passes) from Kinexon API for the current fixture.
    """
    api_kinexon = context.resources.kinexon_api

    fixture_id = context.partition_key

    df_fixtures = fixtures_sportradar

    # Filter fixtures for current partition
    df_fixtures["fixture_id"] = df_fixtures["fixture_id"].astype(str)
    df_fixture = df_fixtures[df_fixtures["fixture_id"] == fixture_id]

    if df_fixture.empty:
        context.log.warning(
            f"Fixture {fixture_id} not found in fixtures asset."
        )
        return pd.DataFrame()

    # session_ids in kinexon_positions
    list_session_ids_in_positions = (
        df_fixture["session_id"].dropna().unique().tolist()
    )

    if not list_session_ids_in_positions:
        context.log.warning(
            f"No session_id for fixture {fixture_id}. Skipping events."
        )
        return pd.DataFrame()

    event_frames = []
    for session_id in list_session_ids_in_positions:
        df_events = fetch_detected_events_for_session(
            api_kinexon,
            session_id,
        )
        if df_events.empty:
            context.log.debug(
                f"No detected events returned for session_id {session_id}."
            )
            continue
        # append fixture_id for context
        df_events["fixture_id"] = str(fixture_id)
        event_frames.append(df_events)

    if not event_frames:
        context.log.warning(
            f"No detected events retrieved for fixture {fixture_id}."
        )
        return pd.DataFrame()

    df_all_detected_events = pd.concat(event_frames, ignore_index=True)
    context.log.info(
        f"Fetched {len(df_all_detected_events)} detected events for "
        f"{len(event_frames)} sessions."
    )
    context.add_output_metadata(preview_metadata(df_all_detected_events))
    return df_all_detected_events
