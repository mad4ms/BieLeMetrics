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

from src.pipelines.normalized.matches import normalize_matches as normalize_matches_fn


@asset(
    group_name="normalized",
    compute_kind="duckdb",
)
def matches_normalized(
    context: AssetExecutionContext,
    fixtures_sportradar_raw: pd.DataFrame,
    sessions_kinexon_raw: pd.DataFrame,
    # teams_sportradar_raw: pd.DataFrame,
    # teams_kinexon_raw: pd.DataFrame,
) -> pd.DataFrame:
    df_normalized = normalize_matches_fn(
        df_fixtures_sportradar_raw=fixtures_sportradar_raw,
        df_sessions_kinexon_raw=sessions_kinexon_raw,
        # df_teams_sportradar_raw=teams_sportradar_raw,
        # df_teams_kinexon_raw=teams_kinexon_raw,
    )

    context.log.info("Normalized %d matches", len(df_normalized))
    context.add_output_metadata(
        {
            "n_rows": len(df_normalized),
            "n_columns": df_normalized.shape[1],
            "preview": MetadataValue.md(df_normalized.head().to_markdown(index=False)),
        }
    )

    return df_normalized
