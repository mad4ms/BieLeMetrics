import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset

from fetcher_sportradar.fetch_fixture_events import (
    fetch_events_for_fixture,
    process_fixture_events,
)

from .assets_sportradar_slow import (  # pylint: disable=relative-beyond-top-level
    fixtures_partition_def,
)
from .utils.metadata import (  # pylint: disable=relative-beyond-top-level
    preview_metadata,
)


@asset(
    partitions_def=fixtures_partition_def,
    required_resource_keys={"sportradar_api", "io_manager"},
    group_name="sportradar_data",
    compute_kind="duckdb",
    description="Fetch, normalize, and process Sportradar events for a single fixture.",
    metadata={"partition_column": "fixture_id"},
)
def fixture_events_sportradar(
    context: AssetExecutionContext,
    fixtures_sportradar: pd.DataFrame,
    teams_sportradar: pd.DataFrame,
) -> pd.DataFrame:
    """Unified raw + processed fixture events pipeline."""

    api_sr = context.resources.sportradar_api
    fixture_id = context.partition_key

    # ---- Retrieve fixture record ----
    fixtures_sportradar["fixture_id"] = fixtures_sportradar["fixture_id"].astype(str)
    df_fixture = fixtures_sportradar[fixtures_sportradar["fixture_id"] == fixture_id]

    if df_fixture.empty:
        context.log.warning(f"Fixture {fixture_id} not found in fixture list.")
        return pd.DataFrame()

    session_id = df_fixture["session_id"].iloc[0]

    # ---- Fetch raw match events ----
    match_events_raw = fetch_events_for_fixture(api_sr, fixture_id)
    if not match_events_raw:
        context.log.warning(f"No Sportradar events returned for fixture {fixture_id}.")
        return pd.DataFrame()

    df_raw = pd.DataFrame(match_events_raw).rename(columns={"fixtureId": "fixture_id"})
    df_raw["fixture_id"] = df_raw["fixture_id"].astype(str)

    if pd.notna(session_id):
        df_raw["session_id"] = int(session_id)
        df_raw["session_id"] = df_raw["session_id"].astype("Int64")

    # Normalize JSON-like scores to strings
    if "scores" in df_raw.columns:
        df_raw["scores"] = df_raw["scores"].apply(
            lambda x: str(x) if pd.notna(x) else x
        )

    context.log.info(
        "Fetched %d raw match events for fixture %s.",
        len(df_raw),
        fixture_id,
    )

    # ---- Process match events ----
    match_events_list = df_raw.to_dict(orient="records")

    df_match_events, _ = process_fixture_events(
        fixture_id=fixture_id,
        match_events=match_events_list,
        df_teams=teams_sportradar,
        api=api_sr,
    )
    # unpack scores into separate columns score_home and score_away
    if "scores" in df_match_events.columns:
        valid_ids = df_match_events["entity_id"].notna()
        unique_entity_ids = df_match_events.loc[valid_ids, "entity_id"].unique()
        unique_entity_ids = [uid for uid in unique_entity_ids if uid != "nan"]
        if len(unique_entity_ids) == 2:
            for id in unique_entity_ids:
                # check if id is df_fixture["entity_id_home"].iloc[0] or away
                side = "home" if id == df_fixture["entity_id_home"].iloc[0] else "away"
                score_col = f"scores_{side}"
                df_match_events[score_col] = df_match_events["scores"].apply(
                    lambda x: x.get(side) if isinstance(x, dict) else None
                )

    context.log.info(
        "Processed %d match events for fixture %s.",
        len(df_match_events),
        fixture_id,
    )

    # ---- Metadata ----
    metadata = {
        "fixture_id": MetadataValue.text(str(fixture_id)),
        "date_game_start": MetadataValue.text(
            str(df_fixture["start_time_local"].iloc[0])
        ),
        "name_team_home": MetadataValue.text(str(df_fixture["name_team_home"].iloc[0])),
        "name_team_away": MetadataValue.text(str(df_fixture["name_team_away"].iloc[0])),
        "round_number": MetadataValue.text(str(df_fixture["round_number"].iloc[0])),
        "n_raw_events": len(df_raw),
        "n_processed_events": len(df_match_events),
        "n_unique_players": df_match_events["person_id"].nunique(),
    }

    context.add_output_metadata(metadata)
    context.add_output_metadata(preview_metadata(df_match_events))

    return df_match_events


@asset(
    partitions_def=fixtures_partition_def,
    required_resource_keys={"sportradar_api", "io_manager"},
    group_name="sportradar_data",
    compute_kind="duckdb",
    description="Match events for a single fixture, with incremental DuckDB persistence.",
    metadata={"partition_column": "fixture_id"},
)
def fixture_players_sportradar(
    context: AssetExecutionContext,
    fixture_events_sportradar: pd.DataFrame,
    teams_sportradar: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch and persist events for a single fixture partition."""

    # Get API SR
    api_sr = context.resources.sportradar_api

    fixture_id = context.partition_key
    df_fixture_events_sportradar = fixture_events_sportradar.copy()
    df_fixture_events_sportradar["fixture_id"] = df_fixture_events_sportradar[
        "fixture_id"
    ].astype(str)
    df_fixture_events_sportradar = df_fixture_events_sportradar[
        df_fixture_events_sportradar["fixture_id"] == fixture_id
    ]

    if df_fixture_events_sportradar.empty:
        context.log.warning(f"Fixture {fixture_id} not found in fixtures asset output.")
        return pd.DataFrame()

    _, df_fixture_players = process_fixture_events(
        fixture_id=fixture_id,
        match_events=df_fixture_events_sportradar.to_dict(orient="records"),
        df_teams=teams_sportradar,
        api=api_sr,
    )

    if not df_fixture_players.empty and "person_id" in df_fixture_players.columns:
        df_fixture_players = df_fixture_players.drop_duplicates(subset=["person_id"])

    # Set partition column (Will insert players per fixture
    # mind that duplicates across fixtures might exist)
    df_fixture_players["fixture_id"] = fixture_id

    df_fixture_players = df_fixture_players.loc[
        :, ~df_fixture_players.columns.duplicated()
    ]

    # -----------------------------------------
    #             Summary Stats
    # -----------------------------------------
    n_players = len(df_fixture_players)
    n_unique_teams = df_fixture_players["team_name"].nunique(dropna=True)
    unique_teams = (
        df_fixture_players["team_name"].unique().tolist()
        if "team_name" in df_fixture_players.columns
        else []
    )

    # players per team
    players_per_team = (
        df_fixture_players["team_name"].value_counts(dropna=True).to_dict()
        if "team_name" in df_fixture_players.columns
        else {}
    )

    # -----------------------------------------
    #                Metadata
    # -----------------------------------------

    metadata = {
        "fixture_id": MetadataValue.text(str(fixture_id)),
        "n_players": n_players,
        "n_unique_teams": n_unique_teams,
        "unique_teams": MetadataValue.json(unique_teams),
        "players_per_team": MetadataValue.json(players_per_team),
    }

    context.add_output_metadata(metadata)
    context.add_output_metadata(preview_metadata(df_fixture_players))

    context.log.info(
        "Extracted %d players for fixture %s.",
        n_players,
        fixture_id,
    )

    return df_fixture_players
