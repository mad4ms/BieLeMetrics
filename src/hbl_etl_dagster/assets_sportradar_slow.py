from dagster import (
    asset,
    AssetExecutionContext,
    MetadataValue,
)
import pandas as pd

from src.events_sportradar.fetch_teams import fetch_teams_by_season_id
from src.events_sportradar.fetch_list_fixtures import (
    fetch_list_fixtures,
    refine_fixtures_data,
    expand_competitors_in_fixtures,
)
from src.events_kinexon.fetch_session_id_for_fixtures import (
    fetch_session_ids_for_fixtures,
)


from .assets_ids import (
    fixtures_partition_def,
)  # pylint: disable=relative-beyond-top-level
from .utils.duckdb_helpers import (
    duckdb_conn,
)  # pylint: disable=relative-beyond-top-level
from .utils.metadata import (
    preview_metadata,
)  # pylint: disable=relative-beyond-top-level


@asset(
    required_resource_keys={"sportradar_api"},
    group_name="sportradar_data",
    compute_kind="duckdb",
)
def teams(context: AssetExecutionContext, season_id: str) -> pd.DataFrame:
    """
    All teams for a given season. Depends on the in-memory season_id asset.
    Persisted to DuckDB via IOManager.
    """
    api = context.resources.sportradar_api
    df = fetch_teams_by_season_id(api=api, season_id=season_id)
    context.log.info(f"Fetched {len(df)} teams for season_id={season_id}")
    context.add_output_metadata(preview_metadata(df))
    return df


@asset(
    required_resource_keys={"sportradar_api"},
    group_name="sportradar_data",
    compute_kind="duckdb",
    description="Raw fixture list for the configured season.",
)
def list_fixtures_raw(
    context: AssetExecutionContext, season_id: str
) -> pd.DataFrame:
    """
    Raw fixtures list for the season.
    Depends on season_id (in-memory).
    """
    api = context.resources.sportradar_api

    raw_fixtures = fetch_list_fixtures(api=api, season_id=season_id)
    context.log.info(
        f"Fetched {len(raw_fixtures)} raw fixtures for season_id={season_id}"
    )
    df_fixtures = pd.DataFrame(raw_fixtures)
    context.add_output_metadata(preview_metadata(df_fixtures))
    return df_fixtures


@asset(
    required_resource_keys={"kinexon_api"},
    group_name="sportradar_data",
    compute_kind="duckdb",
    description="Full fixture list for the configured season, enriched with Kinexon session IDs.",
)
def list_fixtures(
    context: AssetExecutionContext,
    list_fixtures_raw: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full fixtures list for the season, including refined and expanded competitor info.
    Depends on list_fixtures_raw and teams (persisted).
    """
    # list_fixtures_raw is actually a dataframe, must be passed as a list of dicts to refine_fixtures_data
    refined_fixtures = refine_fixtures_data(
        list_fixtures_raw.to_dict("records")
    )

    # Correctly use the upstream 'teams' asset to expand competitor data
    expanded_fixtures = expand_competitors_in_fixtures(refined_fixtures, teams)

    api = context.resources.kinexon_api

    dict_session_ids = fetch_session_ids_for_fixtures(
        api=api,
        df_fixtures=expanded_fixtures,
    )
    # contains mapping fixture_id -> session_id

    # merge session IDs into expanded_fixtures
    expanded_fixtures["session_id"] = expanded_fixtures["fixture_id"].map(
        dict_session_ids
    )

    expanded_fixtures["fixture_id"] = expanded_fixtures["fixture_id"].astype(
        str
    )
    # sort by start_time_local and round_number
    expanded_fixtures = expanded_fixtures.sort_values(
        by=["start_time_local", "round_number"]
    ).reset_index(drop=True)
    context.instance.add_dynamic_partitions(
        fixtures_partition_def.name,
        expanded_fixtures["fixture_id"].tolist(),
    )

    context.log.info(
        f"Fetched session IDs for {len(dict_session_ids)} fixtures from Kinexon."
    )
    context.log.info(
        f"Fetched and expanded {len(expanded_fixtures)} fixtures."
    )

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
