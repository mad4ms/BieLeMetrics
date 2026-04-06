import pandas as pd
from dagster import (
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    MetadataValue,
    asset,
)

from src.pipelines.normalized.match_positions import (
    normalize_match_positions as normalize_match_positions_fn,
)
from src.hbl_etl_dagster.utils.metadata import markdown_table

fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")


@asset(
    group_name="normalized",
    compute_kind="duckdb",
    partitions_def=fixtures_partition_def,
    description="Normalized match positions for a single fixture (partitioned by fixture_id).",
)
def match_positions_normalized(
    context: AssetExecutionContext,
    positions_kinexon_raw: pd.DataFrame,
    matches_normalized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize match positions.

    :param context: AssetExecutionContext
    :param positions_kinexon_raw: Raw Kinexon positioning data for the current partition.
    :param matches_normalized: Non-partitioned match lookup (filtered by fixture_id below).
    :return: Normalized positions DataFrame.
    """
    fixture_id = context.partition_key

    fixture_meta = matches_normalized[
        matches_normalized["fixture_id"] == str(fixture_id)
    ]
    home_team = None
    if not fixture_meta.empty:
        home_team = fixture_meta["team_name_home"].iloc[0]

    df_normalized = normalize_match_positions_fn(
        df_positions_kinexon_raw=positions_kinexon_raw,
        home_team_name=home_team,
    )

    context.log.info("Normalized %d match positions", len(df_normalized))
    context.add_output_metadata(
        {
            "n_rows": len(df_normalized),
            "n_columns": df_normalized.shape[1],
            "preview": MetadataValue.md(markdown_table(df_normalized, n=5)),
        }
    )

    return df_normalized
