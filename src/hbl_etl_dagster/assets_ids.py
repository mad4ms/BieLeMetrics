from dagster import (
    asset,
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    Field,
)
from src.events_sportradar.fetch_competition_id import fetch_competition_id
from src.events_sportradar.fetch_saison_id import fetch_season_id

# --------------------------------------------------------------------
# Partitions
# --------------------------------------------------------------------

fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")


# --------------------------------------------------------------------
# In-Memory ID Assets
# --------------------------------------------------------------------


@asset(
    required_resource_keys={"sportradar_api"},
    io_manager_key="file_io_manager",  # This ensures the asset is kept in memory
    compute_kind="api",
    group_name="sportradar_ids",
    description="Fetches the Sportradar competition ID for '1. Handball-Bundesliga'.",
)
def competition_id(context: AssetExecutionContext) -> str:
    """An in-memory asset representing the competition ID."""
    api = context.resources.sportradar_api
    comp_id = fetch_competition_id(api=api)
    context.log.info(f"Resolved competition_id: {comp_id}")
    return str(comp_id)


@asset(
    required_resource_keys={"sportradar_api"},
    io_manager_key="file_io_manager",
    compute_kind="api",
    group_name="sportradar_ids",
    description="Fetches the Sportradar season ID for a configured year.",
    config_schema={
        "season_year": Field(
            int,
            default_value=2024,
            description="Season year to resolve via the Sportradar API.",
        ),
    },
)
def season_id(context: AssetExecutionContext, competition_id: str) -> str:
    """In-memory asset representing the chosen season ID."""

    api = context.resources.sportradar_api
    season_year = int((context.op_config or {}).get("season_year", 2024))

    s_id = fetch_season_id(
        api=api,
        competition_id=competition_id,
        season_year=season_year,
    )
    context.log.info(f"Resolved season_id: {s_id} for year {season_year}")
    return str(s_id)
