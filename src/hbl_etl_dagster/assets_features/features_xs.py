# assets_sportradar_raw.py
import pandas as pd
from dagster import (
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    MetadataValue,
    asset,
)

from src.pipelines.features.calc_xs_features import (
    calculate_xs_features as calculate_xs_features_fn,
)

fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")


@asset(
    group_name="synced",
    compute_kind="duckdb",
    partitions_def=fixtures_partition_def,
    description="Extracted xS features for a single fixture (partitioned by fixture_id).",
    deps=["match_positions_normalized"],
)
def features_xs(
    context: AssetExecutionContext,
    matches_normalized: pd.DataFrame,
    shot_events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract xS features for a single fixture.

    :param context: AssetExecutionContext
    :param matches_normalized: Normalized match-level data
    :param shot_events: Shot event data
    :return: DataFrame with xS features
    """
    fixture_id = context.partition_key

    df_match_normalized = matches_normalized[
        matches_normalized["fixture_id"] == str(fixture_id)
    ].copy()

    df_match_positions = context.resources.io_manager.load_partitioned_input(
        table_name="match_positions_normalized",
        partition_key=fixture_id,
        partition_col="fixture_id",
    )

    df_match_detected_shots = shot_events[
        shot_events["fixture_id"] == str(fixture_id)
    ].copy()

    df_match_positions = df_match_positions[
        df_match_positions["fixture_id"] == str(fixture_id)
    ].copy()

    df_xs_features = calculate_xs_features_fn(
        df_match_normalized=df_match_normalized,
        df_shot_events=df_match_detected_shots,
        df_positions_normalized=df_match_positions,
    )

    context.log.info(
        "Calculated xS features for %d shots in fixture %s",
        len(df_xs_features),
        fixture_id,
    )

    context.add_output_metadata(
        {
            "n_rows": len(df_xs_features),
            "n_columns": df_xs_features.shape[1],
            "preview": MetadataValue.md(
                df_xs_features.head(100).to_markdown(index=False)
            ),
        }
    )

    return df_xs_features
