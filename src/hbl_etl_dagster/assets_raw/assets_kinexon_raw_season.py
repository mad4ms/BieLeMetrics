# assets_sportradar_raw.py
import os

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

from src.pipelines.raw.kinexon import (
    get_sessions_for_team as kinexon_get_sessions_for_team,
)
from src.pipelines.raw.kinexon import (
    get_teams_for_season as kinexon_get_teams_for_season,
)


@asset(
    required_resource_keys={"kinexon_api"},
    group_name="kinexon_raw_season",
    compute_kind="duckdb",
    description="Raw Kinexon teams for a season.",
    config_schema={
        "season_year": Field(
            int,
            default_value=2025,
            description="Season start year (e.g. 2025 -> 2025/26).",
        ),
    },
)
def teams_kinexon_raw(
    context: AssetExecutionContext,
) -> pd.DataFrame:
    api = context.resources.kinexon_api
    season_year = context.op_config["season_year"]
    # build season string e.g. "2025-26"
    season_year = f"{season_year}-{str(season_year +1)[-2:]}"

    df_teams = kinexon_get_teams_for_season(api=api, season_year=season_year)

    context.log.info("Fetched %d teams from Kinexon", len(df_teams))
    context.add_output_metadata(
        {
            "n_rows": len(df_teams),
            "n_columns": df_teams.shape[1],
            "preview": MetadataValue.md(df_teams.head().to_markdown(index=False)),
        }
    )

    return df_teams


@asset(
    required_resource_keys={"kinexon_api"},
    group_name="kinexon_raw_season",
    compute_kind="duckdb",
    description="Raw Kinexon detected events for a single fixture (partitioned by fixture_id).",
)
def sessions_kinexon_raw(
    context: AssetExecutionContext,
    fixtures_sportradar_raw: pd.DataFrame,
    teams_kinexon_raw: pd.DataFrame,
) -> pd.DataFrame:
    api = context.resources.kinexon_api

    date_start = fixtures_sportradar_raw["startTimeLocal"].min()
    date_end = fixtures_sportradar_raw["startTimeLocal"].max()

    df_sessions = pd.DataFrame()
    for team_id in teams_kinexon_raw["id"]:
        df_team_sessions = kinexon_get_sessions_for_team(
            api=api,
            team_id=team_id,
            start_date=date_start,
            end_date=date_end,
        )
        df_sessions = pd.concat([df_sessions, df_team_sessions], ignore_index=True)
        # deduplicate
        df_sessions = df_sessions.drop_duplicates(subset=["id"])

    context.log.info("Fetched %d sessions from Kinexon", len(df_sessions))
    context.add_output_metadata(
        {
            "n_rows": len(df_sessions),
            "n_columns": df_sessions.shape[1],
            "preview": MetadataValue.md(df_sessions.head().to_markdown(index=False)),
        }
    )

    return df_sessions
