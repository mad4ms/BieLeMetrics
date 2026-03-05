import pandas as pd
from dagster import (
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    MetadataValue,
    asset,
)

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
    context.add_output_metadata(
        {
            "n_rows": len(df_xg_features),
            "n_columns": df_xg_features.shape[1],
            "n_notna_shooter_goal": int(
                df_xg_features["shooter_distance_to_goal"].notna().sum()
            ),
            "n_notna_shooter_goalkeeper": int(
                df_xg_features["shooter_distance_to_goalkeeper"].notna().sum()
            ),
            "n_notna_ball_goal": int(
                df_xg_features["ball_distance_to_goal"].notna().sum()
            ),
            "preview": MetadataValue.md(
                df_xg_features.head(100).to_markdown(index=False)
            ),
        }
    )

    return df_xg_features
