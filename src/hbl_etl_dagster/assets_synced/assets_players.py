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

from src.pipelines.synced.players import (
    extract_players_for_match as extract_players_for_match_fn,
)

fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")


@asset(
    group_name="synced",
    compute_kind="duckdb",
    partitions_def=fixtures_partition_def,
    description="Synced players for a single fixture (partitioned by fixture_id).",
    deps=[
        "match_positions_normalized",
    ],
)
def players(
    context: AssetExecutionContext,
    matches_normalized: pd.DataFrame,
    match_events_normalized_setup: pd.DataFrame,
    match_detected_shots_normalized: pd.DataFrame,
    match_players_normalized: pd.DataFrame,
    # match_positions_normalized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract synced players for a single fixture.

    :param context: Description
    :type context: AssetExecutionContext
    :param players_sportradar_raw: Description
    :type players_sportradar_raw: pd.DataFrame
    :param match_events_normalized_setup: Description
    :type match_events_normalized_setup: pd.DataFrame
    :param match_positions_normalized: Description
    :type match_positions_normalized: pd.DataFrame
    :param match_players_normalized: Description
    :type match_players_normalized: pd.DataFrame
    :return: Description
    :rtype: pd.DataFrame
    """
    fixture_id = context.partition_key

    df_match = matches_normalized[
        matches_normalized["fixture_id"] == str(fixture_id)
    ].copy()

    df_match_events_setup = match_events_normalized_setup[
        match_events_normalized_setup["fixture_id"] == str(fixture_id)
    ].copy()

    df_match_detected_shots = match_detected_shots_normalized[
        match_detected_shots_normalized["fixture_id"] == str(fixture_id)
    ].copy()

    df_match_players = match_players_normalized[
        match_players_normalized["fixture_id"] == str(fixture_id)
    ].copy()

    df_match_positions = context.resources.io_manager.load_partitioned_input(
        table_name="match_positions_normalized",
        partition_key=fixture_id,
        partition_col="fixture_id",
    )

    df_players_in_events = extract_players_for_match_fn(
        df_match_normalized=df_match,
        df_match_events_normalized_setup=df_match_events_setup,
        df_match_detected_shots_normalized=df_match_detected_shots,
        df_match_positions_normalized=df_match_positions,
        df_match_players_normalized=df_match_players,
    )
    context.log.info("Extracted %d synced players", len(df_players_in_events))
    context.add_output_metadata(
        {
            "n_rows": len(df_players_in_events),
            "n_unique_players": df_players_in_events["person_id"].nunique(),
            "n_original_players": len(df_match_players),
            "coverage_league_id_percent": (
                df_players_in_events["league_id"].nunique()
                / len(df_players_in_events)
            )
            * 100,
            "n_columns": df_players_in_events.shape[1],
            "preview": MetadataValue.md(
                df_players_in_events.head(100).to_markdown(index=False)
            ),
        }
    )

    return df_players_in_events
