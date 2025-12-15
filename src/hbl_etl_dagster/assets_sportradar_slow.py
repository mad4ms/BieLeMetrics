from dagster import (
    asset,
    AssetExecutionContext,
    MetadataValue,
    DynamicPartitionsDefinition,
    Field,
)
import pandas as pd

from src.fetcher_sportradar.fetch_competition_id import fetch_competition_id
from src.fetcher_sportradar.fetch_saison_id import fetch_season_id
from src.fetcher_sportradar.fetch_teams import fetch_teams_by_season_id
from src.fetcher_sportradar.fetch_list_fixtures import (
    fetch_list_fixtures,
    refine_fixtures_data,
    expand_competitors_in_fixtures,
)
from src.fetcher_kinexon.fetch_session_id_for_fixtures import (
    fetch_session_ids_for_fixtures,
)

from src.hbl_etl_dagster.utils.metadata import (
    preview_metadata,
)

# Define dynamic partitions for fixtures
fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")


@asset(
    required_resource_keys={"sportradar_api"},
    io_manager_key="file_io_manager",  # stored as a small file
    compute_kind="api",
    group_name="sportradar_data",
    description="Fetches the Sportradar competition ID for '1. Handball-Bundesliga'.",
)
def competition_id(context: AssetExecutionContext) -> str:
    """Asset representing the competition ID used in downstream Sportradar assets."""
    api = context.resources.sportradar_api
    comp_id = fetch_competition_id(api=api)
    context.log.info(f"Resolved competition_id: {comp_id}")
    return str(comp_id)  # ID is normalized to string for consistency


@asset(
    required_resource_keys={"sportradar_api"},
    io_manager_key="file_io_manager",  # or "in_memory_io_manager" if you switch
    compute_kind="api",
    group_name="sportradar_data",
    description="Fetches the Sportradar season ID for a configured year.",
    config_schema={
        "season_year": Field(
            int,
            default_value=2025,
            description="Season year to resolve via the Sportradar API.",
        ),
    },
)
def season_id(context: AssetExecutionContext, competition_id: str) -> str:
    """Asset representing the Sportradar season ID for the configured year."""
    api = context.resources.sportradar_api
    season_year: int = context.op_config["season_year"]

    s_id = fetch_season_id(
        api=api,
        competition_id=competition_id,
        season_year=season_year,
    )
    context.log.info(f"Resolved season_id: {s_id} for year {season_year}")
    return str(s_id)


@asset(
    required_resource_keys={"sportradar_api"},
    group_name="sportradar_data",
    compute_kind="duckdb",
)
def teams_sportradar(
    context: AssetExecutionContext, season_id: str
) -> pd.DataFrame:
    """
    All teams for a given season. Depends on the in-memory season_id asset.
    Persisted to DuckDB via IOManager.
    """
    api = context.resources.sportradar_api
    df = fetch_teams_by_season_id(api=api, season_id=season_id)
    if df.empty:
        context.log.warning("No teams fetched for season_id=%s", season_id)
    context.log.info("Fetched %d teams for season_id=%s", len(df), season_id)
    context.add_output_metadata(preview_metadata(df))
    return df


@asset(
    required_resource_keys={"sportradar_api", "kinexon_api"},
    group_name="sportradar_data",
    compute_kind="duckdb",
    description="Raw fixture list for the configured season.",
    config_schema={
        "season_year": Field(
            int,
            default_value=2025,
            description="Season year used in fixture processing.",
        )
    },
)
def fixtures_sportradar(
    context: AssetExecutionContext,
    season_id: str,
    teams_sportradar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Raw fixtures list for the season.
    Depends on season_id (in-memory).
    """
    # Get API SR
    api_sportradar = context.resources.sportradar_api
    # Get API Kinexon
    api_kinexon = context.resources.kinexon_api

    # Fetch raw fixtures from API
    raw_fixtures = fetch_list_fixtures(api=api_sportradar, season_id=season_id)
    context.log.info(
        f"Fetched {len(raw_fixtures)} raw fixtures for season_id={season_id}"
    )
    # Refine fixtures data (e.g. parse dates, normalize fields)
    refined_fixtures = refine_fixtures_data(raw_fixtures)

    # Expand competitors in fixtures
    expanded_fixtures = expand_competitors_in_fixtures(
        refined_fixtures, teams_sportradar
    )
    season_year = context.op_config["season_year"]

    # Fetch Kinexon session IDs for fixtures from Kinexon API
    dict_session_ids = fetch_session_ids_for_fixtures(
        api=api_kinexon,
        df_fixtures=expanded_fixtures,
        season_year=f"{season_year}-{str(season_year +1)[-2:]}",
        logger=context.log,
    )

    # merge session IDs into expanded_fixtures
    expanded_fixtures["session_id"] = expanded_fixtures["fixture_id"].map(
        dict_session_ids
    )
    # ensure fixture_id is str
    expanded_fixtures["fixture_id"] = expanded_fixtures["fixture_id"].astype(
        str
    )
    # sort by start_time_local and round_number
    expanded_fixtures = expanded_fixtures.sort_values(
        by=["start_time_local", "round_number"]
    ).reset_index(drop=True)

    # remove entries from future fixtures
    expanded_fixtures["start_time_local"] = pd.to_datetime(
        expanded_fixtures.get("start_time_local"), utc=True
    )
    now_utc = pd.Timestamp.now(tz="UTC")
    expanded_fixtures = expanded_fixtures[
        expanded_fixtures["start_time_local"] <= now_utc
    ].reset_index(drop=True)

    # Register dynamic partitions for fixtures
    context.instance.add_dynamic_partitions(
        fixtures_partition_def.name,
        expanded_fixtures["fixture_id"].tolist(),
    )
    # Log metadata
    context.log.info(
        f"Fetched session IDs for {len(dict_session_ids)} fixtures from Kinexon."
    )
    context.log.info(
        f"Fetched and expanded {len(expanded_fixtures)} fixtures."
    )
    # Add output metadata
    metadata = {
        "n_games": len(expanded_fixtures),
        "n_columns": expanded_fixtures.shape[1],
        "n_fixtures_with_session_id": expanded_fixtures[
            expanded_fixtures["session_id"].notna()
        ].shape[0],
        "team_names_unique": expanded_fixtures["name_team_home"].nunique(),
        "preview": MetadataValue.md(
            expanded_fixtures.head().to_markdown(index=False)
        ),
    }
    context.add_output_metadata(metadata)

    return expanded_fixtures
