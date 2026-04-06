import pandas as pd
from dagster import (
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    MetadataValue,
    asset,
)

from src.hbl_etl_dagster.utils.metadata import markdown_table
from src.pipelines.features.calc_xs_features import (
    calculate_xs_features as calculate_xs_features_fn,
)

fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")


@asset(
    group_name="synced",
    compute_kind="duckdb",
    partitions_def=fixtures_partition_def,
    description="Extracted xS features for a single fixture (partitioned by fixture_id).",
)
def features_xs(
    context: AssetExecutionContext,
    matches_normalized: pd.DataFrame,
    shot_events: pd.DataFrame,
    match_positions_normalized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract xS features for a single fixture.

    :param context: AssetExecutionContext
    :param matches_normalized: Non-partitioned match lookup (filtered by fixture_id below).
    :param shot_events: Shot events for the current partition.
    :param match_positions_normalized: Positions for the current partition.
    :return: xS features DataFrame.
    """
    fixture_id = context.partition_key

    # matches_normalized is non-partitioned — filter manually
    df_match_normalized = matches_normalized[
        matches_normalized["fixture_id"] == str(fixture_id)
    ].copy()

    df_xs_features = calculate_xs_features_fn(
        df_match_normalized=df_match_normalized,
        df_shot_events=shot_events,
        df_positions_normalized=match_positions_normalized,
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
            "preview": MetadataValue.md(markdown_table(df_xs_features, n=100)),
        }
    )

    return df_xs_features
