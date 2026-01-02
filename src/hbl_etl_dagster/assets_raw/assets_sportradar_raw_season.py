# assets_sportradar_raw.py

# from __future__ import annotations

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

from src.hbl_etl_dagster.utils.metadata import preview_metadata
from src.pipelines.raw.sportradar import get_competition_id as sr_get_competition_id
from src.pipelines.raw.sportradar import (
    get_fixtures_for_season as sr_get_fixtures_for_season,
)
from src.pipelines.raw.sportradar import get_season_id as sr_get_season_id
from src.pipelines.raw.sportradar import get_teams_for_season as sr_get_teams_for_season

fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")


@asset(
    required_resource_keys={"sportradar_api"},
    io_manager_key="file_io_manager",
    compute_kind="api",
    group_name="sportradar_raw_season",
    description="Resolves the Sportradar competition ID.",
    config_schema={
        "competition_name": Field(
            str,
            default_value="1. Handball-Bundesliga",
            description="Competition name to resolve via the Sportradar API.",
        ),
    },
)
def competition_id(context: AssetExecutionContext) -> str:
    api = context.resources.sportradar_api
    competition_name: str = context.op_config["competition_name"]

    comp_id = sr_get_competition_id(api=api, competition_name=competition_name)
    if not comp_id:
        raise Failure(
            description=f"Could not resolve competition_id for competition_name={competition_name!r}"
        )

    context.log.info("Resolved competition_id=%s", comp_id)
    return str(comp_id)


@asset(
    required_resource_keys={"sportradar_api"},
    io_manager_key="file_io_manager",
    compute_kind="api",
    group_name="sportradar_raw_season",
    description="Resolves the Sportradar season ID.",
    config_schema={
        "season_year": Field(
            int,
            default_value=2025,
            description="Season start year (e.g. 2025 -> 2025/26).",
        ),
    },
)
def season_id(context: AssetExecutionContext, competition_id: str) -> str:
    api = context.resources.sportradar_api
    season_year: int = context.op_config["season_year"]

    s_id = sr_get_season_id(
        api=api, competition_id=competition_id, season_year=season_year
    )
    if not s_id:
        raise Failure(
            description=f"Could not resolve season_id for competition_id={competition_id!r}, season_year={season_year!r}"
        )

    context.log.info("Resolved season_id=%s (season_year=%s)", s_id, season_year)
    return str(s_id)


@asset(
    required_resource_keys={"sportradar_api"},
    group_name="sportradar_raw_season",
    compute_kind="duckdb",
    description="Raw teams table as returned by Sportradar for the season.",
)
def teams_sportradar_raw(
    context: AssetExecutionContext, season_id: str
) -> pd.DataFrame:
    api = context.resources.sportradar_api
    df = sr_get_teams_for_season(api=api, season_id=season_id)

    context.log.info("Fetched %d teams (raw) for season_id=%s", len(df), season_id)
    context.add_output_metadata(preview_metadata(df))
    return df


@asset(
    required_resource_keys={"sportradar_api"},
    group_name="sportradar_raw_season",
    compute_kind="duckdb",
    description="Raw fixtures table as returned by Sportradar for the season (no enrichment).",
)
def fixtures_sportradar_raw(
    context: AssetExecutionContext, season_id: str
) -> pd.DataFrame:
    api = context.resources.sportradar_api
    df = sr_get_fixtures_for_season(api=api, season_id=season_id)

    # Defensive normalization: ensure DataFrame output, but do not transform content.
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    context.log.info("Fetched %d fixtures (raw) for season_id=%s", len(df), season_id)

    # Register dynamic partitions if fixtureId (in raw data) exists; no other processing.
    # we need to rename it to standardize the column name
    if "fixtureId" in df.columns:
        # rename to fixture_id for partitioning
        df = df.rename(columns={"fixtureId": "fixture_id"})
        # check if startTimeLocal exists and remove row if in the future
        if "startTimeLocal" in df.columns:
            df["startTimeLocal"] = pd.to_datetime(df["startTimeLocal"])
            now = pd.Timestamp.now(tz=df["startTimeLocal"].dt.tz)
            df = df[df["startTimeLocal"] <= now]

        fixture_ids = df["fixture_id"].dropna().astype(str).tolist()
        context.instance.add_dynamic_partitions(
            fixtures_partition_def.name, fixture_ids
        )
        context.add_output_metadata({"n_partitions_registered": len(fixture_ids)})
    else:
        context.log.warning(
            "Column 'fixtureId' missing; skipping dynamic partition registration."
        )

    context.add_output_metadata(
        {
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "preview": MetadataValue.md(df.head().to_markdown(index=False)),
        }
    )
    return df


@asset_check(asset=fixtures_sportradar_raw)
def check_fixtures_have_fixture_id_column(
    fixtures_sportradar_raw: pd.DataFrame,
) -> AssetCheckResult:
    ok = "fixture_id" in fixtures_sportradar_raw.columns
    return AssetCheckResult(passed=ok, metadata={"has_fixture_id": ok})


@asset_check(asset=fixtures_sportradar_raw)
def check_fixtures_have_unique_ids_when_present(
    fixtures_sportradar_raw: pd.DataFrame,
) -> AssetCheckResult:
    if "fixture_id" not in fixtures_sportradar_raw.columns:
        return AssetCheckResult(
            passed=True, metadata={"skipped": "fixture_id column missing"}
        )

    ok = bool(fixtures_sportradar_raw["fixture_id"].is_unique)
    n_dup = int(
        len(fixtures_sportradar_raw) - fixtures_sportradar_raw["fixture_id"].nunique()
    )
    return AssetCheckResult(passed=ok, metadata={"n_duplicates": n_dup})
