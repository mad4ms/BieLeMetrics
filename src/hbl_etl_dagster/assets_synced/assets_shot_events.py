# assets_sportradar_raw.py
from dagster import (
    AssetExecutionContext,
    AssetCheckResult,
    DynamicPartitionsDefinition,
    Failure,
    Field,
    MetadataValue,
    asset,
    asset_check,
)
import pandas as pd

from src.pipelines.synced.shot_events import (
    sync_shot_events as sync_shot_events_fn,
)

fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")


@asset(
    group_name="synced",
    compute_kind="duckdb",
    partitions_def=fixtures_partition_def,
    description="Synced shot events for a single fixture (partitioned by fixture_id).",
    deps=[
        "match_positions_normalized",
    ],
)
def shot_events(
    context: AssetExecutionContext,
    matches_normalized: pd.DataFrame,
    match_events_normalized_goals: pd.DataFrame,
    match_detected_shots_normalized: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract synced shot events for a single fixture.

    :param context: Description
    :type context: AssetExecutionContext

    :param matches_normalized: Description
    :type matches_normalized: pd.DataFrame
    :param match_events_normalized_goals: Description
    :type match_events_normalized_goals: pd.DataFrame
    :param match_detected_shots_normalized: Description
    :type match_detected_shots_normalized: pd.DataFrame
    :param players: Description
    :type players: pd.DataFrame
    :return: Description
    :rtype: pd.DataFrame
    """
    fixture_id = context.partition_key

    df_match_detected_shots = match_detected_shots_normalized[
        match_detected_shots_normalized["fixture_id"] == str(fixture_id)
    ].copy()

    df_match_events_goals = match_events_normalized_goals[
        match_events_normalized_goals["fixture_id"] == str(fixture_id)
    ].copy()

    df_players_match = players[players["fixture_id"] == str(fixture_id)].copy()

    df_positions = context.resources.io_manager.load_partitioned_input(
        table_name="match_positions_normalized",
        partition_key=fixture_id,
        partition_col="fixture_id",
    )

    df_match_normalized = matches_normalized[
        matches_normalized["fixture_id"] == str(fixture_id)
    ].copy()

    df_shot_events = sync_shot_events_fn(
        df_match_normalized=df_match_normalized,
        df_match_events_normalized_goals=df_match_events_goals,
        df_match_detected_shots_normalized=df_match_detected_shots,
        df_positions_normalized=df_positions,
        df_players=df_players_match,
    )

    context.log.info("Extracted %d synced shot events", len(df_shot_events))
    context.add_output_metadata(
        {
            "n_rows": len(df_shot_events),
            "n_unique_shot_events": df_shot_events["event_id"].nunique(),
            "n_unique_throw_timestamps": df_shot_events[
                "throw_timestamp_ms"
            ].nunique(),
            "n_columns": df_shot_events.shape[1],
            "preview": MetadataValue.md(
                df_shot_events.head(100).to_markdown(index=False)
            ),
        }
    )
    return df_shot_events
