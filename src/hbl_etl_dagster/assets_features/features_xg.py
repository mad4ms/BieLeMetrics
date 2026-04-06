import pandas as pd
from dagster import (
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    MetadataValue,
    asset,
)

from src.hbl_etl_dagster.utils.metadata import markdown_table
from src.pipelines.features.calc_xg_features import (
    calculate_xg_features as calculate_xg_features_fn,
)

fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")


@asset(
    group_name="synced",
    compute_kind="duckdb",
    partitions_def=fixtures_partition_def,
    description="Extracted xG features for a single fixture (partitioned by fixture_id).",
)
def features_xg(
    context: AssetExecutionContext,
    matches_normalized: pd.DataFrame,
    shot_events: pd.DataFrame,
    match_positions_normalized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract xG features for a single fixture.

    :param context: AssetExecutionContext
    :param matches_normalized: Non-partitioned match lookup (filtered by fixture_id below).
    :param shot_events: Shot events for the current partition.
    :param match_positions_normalized: Positions for the current partition.
    :return: xG features DataFrame.
    """
    fixture_id = context.partition_key

    # matches_normalized is non-partitioned — filter manually
    df_match_normalized = matches_normalized[
        matches_normalized["fixture_id"] == str(fixture_id)
    ].copy()
    if df_match_normalized.empty:
        raise ValueError(
            f"No matches_normalized rows found for fixture_id={fixture_id} required for xG features"
        )

    df_xg_features = calculate_xg_features_fn(
        df_match_normalized=df_match_normalized,
        df_shot_events=shot_events,
        df_positions_normalized=match_positions_normalized,
    )

    context.log.info(
        "Calculated xG features for %d shots in fixture %s",
        len(df_xg_features),
        fixture_id,
    )
    metadata = {
        "n_rows": len(df_xg_features),
        "n_columns": df_xg_features.shape[1],
        "feature_columns": ", ".join(df_xg_features.columns),
        "preview": MetadataValue.md(markdown_table(df_xg_features, n=100)),
    }
    for metadata_key, column_name in {
        "n_notna_shooter_goal": "shooter_distance_to_goal",
        "n_notna_shooter_goalkeeper": "shooter_distance_to_goalkeeper",
        "n_notna_ball_goal": "ball_distance_to_goal",
    }.items():
        metadata[metadata_key] = (
            int(df_xg_features[column_name].notna().sum())
            if column_name in df_xg_features.columns
            else 0
        )

    context.add_output_metadata(metadata)

    return df_xg_features
