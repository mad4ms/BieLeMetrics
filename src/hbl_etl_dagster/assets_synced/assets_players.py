import pandas as pd
from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    Failure,
    Field,
    MetadataValue,
    asset,
    asset_check,
)

from src.pipelines.synced.players import (
    extract_players_for_match as extract_players_for_match_fn,
)

fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")


@asset(
    group_name="synced",
    compute_kind="duckdb",
    partitions_def=fixtures_partition_def,
    description="Synced players for a single fixture (partitioned by fixture_id).",
    deps=[
        "match_positions_normalized",
    ],
)
def players(
    context: AssetExecutionContext,
    matches_normalized: pd.DataFrame,
    match_events_normalized_setup: pd.DataFrame,
    match_detected_shots_normalized: pd.DataFrame,
    match_players_normalized: pd.DataFrame,
    # match_positions_normalized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract synced players for a single fixture.

    :param context: Description
    :type context: AssetExecutionContext
    :param players_sportradar_raw: Description
    :type players_sportradar_raw: pd.DataFrame
    :param match_events_normalized_setup: Description
    :type match_events_normalized_setup: pd.DataFrame
    :param match_positions_normalized: Description
    :type match_positions_normalized: pd.DataFrame
    :param match_players_normalized: Description
    :type match_players_normalized: pd.DataFrame
    :return: Description
    :rtype: pd.DataFrame
    """
    fixture_id = context.partition_key

    df_match = matches_normalized[
        matches_normalized["fixture_id"] == str(fixture_id)
    ].copy()

    df_match_events_setup = match_events_normalized_setup[
        match_events_normalized_setup["fixture_id"] == str(fixture_id)
    ].copy()

    df_match_detected_shots = match_detected_shots_normalized[
        match_detected_shots_normalized["fixture_id"] == str(fixture_id)
    ].copy()

    df_match_players = match_players_normalized[
        match_players_normalized["fixture_id"] == str(fixture_id)
    ].copy()

    df_match_positions = context.resources.io_manager.load_partitioned_input(
        table_name="match_positions_normalized",
        partition_key=fixture_id,
        partition_col="fixture_id",
    )

    df_players_in_events = extract_players_for_match_fn(
        df_match_normalized=df_match,
        df_match_events_normalized_setup=df_match_events_setup,
        df_match_detected_shots_normalized=df_match_detected_shots,
        df_match_positions_normalized=df_match_positions,
        df_match_players_normalized=df_match_players,
    )
    context.log.info("Extracted %d synced players", len(df_players_in_events))
    context.add_output_metadata(
        {
            "n_rows": len(df_players_in_events),
            "n_unique_players": df_players_in_events["person_id"].nunique(),
            "n_original_players": len(df_match_players),
            "coverage_league_id_percent": (
                df_players_in_events["league_id"].nunique() / len(df_players_in_events)
            )
            * 100,
            "n_columns": df_players_in_events.shape[1],
            "preview": MetadataValue.md(
                df_players_in_events.head(100).to_markdown(index=False)
            ),
        }
    )

    return df_players_in_events


@asset_check(
    asset="players",
    name="check_players_integrity",
)
def check_players_integrity(
    context: AssetCheckExecutionContext,
    players: pd.DataFrame,
) -> AssetCheckResult:
    # ------------------------------------------------------------------
    # 1. Column contract
    # ------------------------------------------------------------------
    required_columns = {
        "fixture_id",
        "entity_id",
        "person_id",
        "name",
        "bib",
        "position",
        "team_name",
        "team_side",
        "date_of_birth",
        "name_family_latin",
        "name_family_local",
        "name_full_latin",
        "name_full_local",
        "name_given_latin",
        "name_given_local",
        "nationality",
        "height",
        "weight",
        "mapped_id",
        "league_id",
        "session_id",
    }

    missing_columns = required_columns - set(players.columns)
    if missing_columns:
        return AssetCheckResult(
            passed=False,
            description=f"Missing required columns: {sorted(missing_columns)}",
        )

    # First execution / nothing materialized yet
    if players.empty:
        return AssetCheckResult(passed=True)

    # ------------------------------------------------------------------
    # 2. Global identity integrity
    #    (same identity must not drift across fixtures)
    # ------------------------------------------------------------------
    identity_key = ["person_id", "entity_id", "league_id"]

    identity_payload = [
        "name",
        "date_of_birth",
        "nationality",
        "height",
        "weight",
        "mapped_id",
        "name_family_latin",
        "name_family_local",
        "name_full_latin",
        "name_full_local",
        "name_given_latin",
        "name_given_local",
    ]

    inconsistencies = (
        players.drop(columns=["fixture_id", "session_id"])
        .groupby(identity_key, dropna=False)[identity_payload]
        .nunique()
        .max(axis=1)
        .loc[lambda s: s > 1]
    )

    if not inconsistencies.empty:
        return AssetCheckResult(
            passed=False,
            description=(
                "Conflicting player identity records detected for the same "
                "(person_id, entity_id, league_id)"
            ),
            metadata={
                "n_conflicting_identity_keys": int(len(inconsistencies)),
            },
        )

    # ------------------------------------------------------------------
    # 3. Partition-wise checks (current fixture only)
    # ------------------------------------------------------------------
    partition_key = context.run.tags.get("dagster/partition")

    if partition_key is None:
        return AssetCheckResult(
            passed=True,
            metadata={"skipped": "no partition context"},
        )

    players_part = players[players["fixture_id"] == str(partition_key)]

    if players_part.empty:
        return AssetCheckResult(
            passed=False,
            metadata={
                "failed": "no rows for partition",
                "fixture_id": partition_key,
            },
        )

    # Uniqueness within fixture
    dup_mask = players_part.duplicated(subset=identity_key)
    if dup_mask.any():
        return AssetCheckResult(
            passed=False,
            metadata={
                "failed": "duplicate (person_id, entity_id, league_id) within fixture",
                "n_duplicates": int(dup_mask.sum()),
                "fixture_id": partition_key,
            },
        )

    # Minimum player sanity check
    THRESHOLD_MIN_PLAYERS = 14
    if len(players_part) < THRESHOLD_MIN_PLAYERS:
        return AssetCheckResult(
            passed=False,
            metadata={
                "failed": (
                    f"too few players: {len(players_part)} "
                    f"(expected ≥ {THRESHOLD_MIN_PLAYERS})"
                ),
                "fixture_id": partition_key,
            },
        )

    # ------------------------------------------------------------------
    # 4. Success
    # ------------------------------------------------------------------
    return AssetCheckResult(
        passed=True,
        metadata={
            "fixture_id": partition_key,
            "n_rows_fixture": len(players_part),
            "n_unique_players_fixture": players_part["person_id"].nunique(),
            "n_unique_identity_keys_global": players[identity_key]
            .drop_duplicates()
            .shape[0],
        },
    )
