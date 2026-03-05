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
) -> pd.DataFrame:
    """
    Normalize match positions.

    :param context: AssetExecutionContext
    :param positions_kinexon_raw: Raw Kinexon positioning data for the current partition.
    :return: Normalized positions DataFrame.
    """
    df_normalized = normalize_match_positions_fn(
        df_positions_kinexon_raw=positions_kinexon_raw,
    )

    context.log.info("Normalized %d match positions", len(df_normalized))
    context.add_output_metadata(
        {
            "n_rows": len(df_normalized),
            "n_columns": df_normalized.shape[1],
            "preview": MetadataValue.md(df_normalized.head().to_markdown(index=False)),
        }
    )

    return df_normalized
