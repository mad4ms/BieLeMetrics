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

from src.pipelines.normalized.match_events import (
    normalize_match_events as normalize_match_events_fn,
)
from src.pipelines.normalized.match_events import (
    normalize_match_events_goals as normalize_match_events_goals_fn,
)
from src.pipelines.normalized.match_events import (
    normalize_match_events_setup as normalize_match_events_setup_fn,
)

fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")


@asset(
    group_name="normalized",
    compute_kind="duckdb",
    partitions_def=fixtures_partition_def,
    description="Normalized match events for a single fixture (partitioned by fixture_id).",
)
def match_events_normalized(
    context: AssetExecutionContext,
    fixture_events_sportradar_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize match events.

    :param context: Description
    :type context: AssetExecutionContext
    :param fixture_events_sportradar_raw: Description
    :type fixture_events_sportradar_raw: pd.DataFrame
    :return: Description
    :rtype: pd.DataFrame
    """
    fixture_id = context.partition_key

    df_fixture_events_sr = fixture_events_sportradar_raw[
        fixture_events_sportradar_raw["fixture_id"] == str(fixture_id)
    ].copy()

    df_normalized = normalize_match_events_fn(
        df_fixture_events_sportradar_raw=df_fixture_events_sr,
    )

    context.log.info("Normalized %d match events", len(df_normalized))
    context.add_output_metadata(
        {
            "n_rows": len(df_normalized),
            "n_columns": df_normalized.shape[1],
            "preview_setup": MetadataValue.md(
                df_normalized.head().to_markdown(index=False)
            ),
            "preview_goals": MetadataValue.md(
                df_normalized[df_normalized["event_type"] == "goal"]
                .head()
                .to_markdown(index=False)
            ),
        }
    )

    return df_normalized


@asset(
    group_name="normalized",
    compute_kind="duckdb",
    partitions_def=fixtures_partition_def,
    description="Normalized match events for a single fixture (partitioned by fixture_id).",
)
def match_events_normalized_setup(
    context: AssetExecutionContext,
    match_events_normalized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize match events setup.

    :param context: Description
    :type context: AssetExecutionContext
    :param match_events_normalized: Description
    :type match_events_normalized: pd.DataFrame
    :return: Description
    :rtype: pd.DataFrame
    """
    fixture_id = context.partition_key

    df_match_events_normalized = match_events_normalized[
        match_events_normalized["fixture_id"] == str(fixture_id)
    ].copy()

    df_setup = normalize_match_events_setup_fn(
        df_match_events_normalized=df_match_events_normalized,
    )

    context.log.info("Normalized %d match events setup", len(df_setup))
    context.add_output_metadata(
        {
            "n_rows": len(df_setup),
            "n_columns": df_setup.shape[1],
            "preview": MetadataValue.md(df_setup.head(100).to_markdown(index=False)),
        }
    )

    return df_setup


@asset(
    group_name="normalized",
    compute_kind="duckdb",
    partitions_def=fixtures_partition_def,
    description="Normalized match events for a single fixture (partitioned by fixture_id).",
)
def match_events_normalized_goals(
    context: AssetExecutionContext,
    match_events_normalized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize match events goals.

    :param context: Description
    :type context: AssetExecutionContext
    :param match_events_normalized: Description
    :type match_events_normalized: pd.DataFrame
    :return: Description
    :rtype: pd.DataFrame
    """
    fixture_id = context.partition_key

    df_match_events_normalized = match_events_normalized[
        match_events_normalized["fixture_id"] == str(fixture_id)
    ].copy()

    df_goals = normalize_match_events_goals_fn(
        df_match_events_normalized=df_match_events_normalized,
    )

    context.log.info("Normalized %d match events goals", len(df_goals))
    context.add_output_metadata(
        {
            "n_rows": len(df_goals),
            "n_columns": df_goals.shape[1],
            "preview": MetadataValue.md(df_goals.head(100).to_markdown(index=False)),
        }
    )

    return df_goals
