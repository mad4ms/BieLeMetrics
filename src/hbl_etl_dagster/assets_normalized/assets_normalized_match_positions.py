# assets_sportradar_raw.py
import pandas as pd
from dagster import (
    AssetCheckResult,
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    Failure,
    Field,
    MetadataValue,
    asset,
    asset_check,
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
    deps=["positions_kinexon_raw"],
)
def match_positions_normalized(
    context: AssetExecutionContext,
    # positions_kinexon_raw: pd.DataFrame, # cannot load here as will blow up RAM
) -> pd.DataFrame:
    """
    Normalize match positions.

    :param context: Description
    :type context: AssetExecutionContext
    :param positions_kinexon_raw: Description
    :type positions_kinexon_raw: pd.DataFrame
    :return: Description
    :rtype: pd.DataFrame
    """
    fixture_id = context.partition_key
    duckdb_io_manager = context.resources.io_manager
    positions_kinexon_raw = duckdb_io_manager.load_partitioned_input(
        table_name="positions_kinexon_raw",
        partition_key=fixture_id,
        partition_col="fixture_id",
    )

    df_positions_kinexon = positions_kinexon_raw[
        positions_kinexon_raw["fixture_id"] == str(fixture_id)
    ].copy()

    df_normalized = normalize_match_positions_fn(
        df_positions_kinexon_raw=df_positions_kinexon,
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
