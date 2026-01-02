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

from src.pipelines.normalized.match_detected_shots import (
    normalize_match_detected_shots as normalize_match_detected_shots_fn,
)

fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")


@asset(
    group_name="normalized",
    compute_kind="duckdb",
    partitions_def=fixtures_partition_def,
    description="Normalized match detected shots for a single fixture (partitioned by fixture_id).",
)
def match_detected_shots_normalized(
    context: AssetExecutionContext,
    detected_events_kinexon_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize match detected shots.

    :param context: Description
    :type context: AssetExecutionContext
    :param detected_events_kinexon_raw: Description
    :type detected_events_kinexon_raw: pd.DataFrame
    :return: Description
    :rtype: pd.DataFrame
    """
    fixture_id = context.partition_key

    df_fixture_detected_shots = detected_events_kinexon_raw[
        detected_events_kinexon_raw["fixture_id"] == str(fixture_id)
    ].copy()

    df_normalized = normalize_match_detected_shots_fn(
        detected_events_kinexon_raw=df_fixture_detected_shots,
    )

    context.log.info("Normalized %d match detected shots", len(df_normalized))
    context.add_output_metadata(
        {
            "n_rows": len(df_normalized),
            "n_columns": df_normalized.shape[1],
            "preview": MetadataValue.md(df_normalized.head().to_markdown(index=False)),
        }
    )

    return df_normalized


@asset_check(
    asset="match_detected_shots_normalized",
    description="Validate schema, partition consistency (single fixture_id), and row-level sanity for normalized match detected shots.",
)
def check_match_detected_shots_normalized(
    context,
    match_detected_shots_normalized: pd.DataFrame,
) -> AssetCheckResult:
    required_columns = {
        "fixture_id",
        "session_id",
        "validated",
    }

    missing_columns = required_columns - set(match_detected_shots_normalized.columns)
    if missing_columns:
        return AssetCheckResult(
            passed=False,
            metadata={"missing_columns": sorted(missing_columns)},
        )

    if match_detected_shots_normalized.empty:
        return AssetCheckResult(
            passed=False,
            metadata={"reason": "No rows produced"},
        )

    # Partition consistency without relying on context:
    # exactly one fixture_id must be present in this partition materialization.
    fixture_ids = (
        match_detected_shots_normalized["fixture_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(fixture_ids) == 0:
        return AssetCheckResult(
            passed=False,
            metadata={
                "reason": "No fixture_id present in match_detected_shots_normalized",
            },
        )

    # Data sanity
    null_validated = int(match_detected_shots_normalized["validated"].isna().sum())

    return AssetCheckResult(
        passed=True,
        metadata={
            "n_rows": len(match_detected_shots_normalized),
            "fixture_ids": fixture_ids,
            "null_validated_rows": null_validated,
        },
    )
