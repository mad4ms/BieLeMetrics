import pandas as pd
from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    MetadataValue,
    asset,
    asset_check,
)

from src.hbl_etl_dagster.utils.metadata import markdown_table
from src.pipelines.synced.players import (
    extract_players_for_match as extract_players_for_match_fn,
)

fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")


@asset(
    group_name="synced",
    compute_kind="duckdb",
    partitions_def=fixtures_partition_def,
    description="Synced players for a single fixture (partitioned by fixture_id).",
)
def players(
    context: AssetExecutionContext,
    matches_normalized: pd.DataFrame,
    match_events_normalized_setup: pd.DataFrame,
    match_detected_shots_normalized: pd.DataFrame,
    match_players_normalized: pd.DataFrame,
    match_positions_normalized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract synced players for a single fixture.

    :param context: AssetExecutionContext
    :param matches_normalized: Non-partitioned match lookup (filtered by fixture_id below).
    :param match_events_normalized_setup: Setup events for the current partition.
    :param match_detected_shots_normalized: Detected shots for the current partition.
    :param match_players_normalized: Players for the current partition.
    :param match_positions_normalized: Positions for the current partition.
    :return: Synced players DataFrame.
    """
    fixture_id = context.partition_key

    # matches_normalized is non-partitioned — filter manually
    df_match = matches_normalized[
        matches_normalized["fixture_id"] == str(fixture_id)
    ].copy()

    df_players_in_events = extract_players_for_match_fn(
        df_match_normalized=df_match,
        df_match_events_normalized_setup=match_events_normalized_setup,
        df_match_detected_shots_normalized=match_detected_shots_normalized,
        df_match_positions_normalized=match_positions_normalized,
        df_match_players_normalized=match_players_normalized,
    )
    context.log.info("Extracted %d synced players", len(df_players_in_events))
    context.add_output_metadata(
        {
            "n_rows": len(df_players_in_events),
            "n_unique_players": df_players_in_events["person_id"].nunique(),
            "n_original_players": len(match_players_normalized),
            "coverage_league_id_percent": (
                df_players_in_events["league_id"].nunique() / len(df_players_in_events)
            )
            * 100,
            "n_columns": df_players_in_events.shape[1],
            "preview": MetadataValue.md(markdown_table(df_players_in_events, n=100)),
        }
    )

    return df_players_in_events


@asset_check(
    asset="players",
    name="check_players_integrity",
)
def check_players_integrity(
    context: AssetCheckExecutionContext,
    players: pd.DataFrame,  # current partition only (partition-aware IO manager)
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

    if players.empty:
        return AssetCheckResult(passed=True)

    # ------------------------------------------------------------------
    # 2. Global identity integrity — load full historical table
    #    (same identity must not drift across fixtures)
    # ------------------------------------------------------------------
    all_players = context.resources.io_manager.load_full_table("players")

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
        all_players.drop(columns=["fixture_id", "session_id"], errors="ignore")
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
    # 3. Partition-wise checks — use the already-filtered `players` param
    # ------------------------------------------------------------------
    if players.empty:
        return AssetCheckResult(
            passed=False,
            metadata={
                "failed": "no rows for partition",
                "fixture_id": context.partition_key,
            },
        )

    dup_mask = players.duplicated(subset=identity_key)
    if dup_mask.any():
        return AssetCheckResult(
            passed=False,
            metadata={
                "failed": "duplicate (person_id, entity_id, league_id) within fixture",
                "n_duplicates": int(dup_mask.sum()),
                "fixture_id": context.partition_key,
            },
        )

    THRESHOLD_MIN_PLAYERS = 14
    if len(players) < THRESHOLD_MIN_PLAYERS:
        return AssetCheckResult(
            passed=False,
            metadata={
                "failed": (
                    f"too few players: {len(players)} "
                    f"(expected ≥ {THRESHOLD_MIN_PLAYERS})"
                ),
                "fixture_id": context.partition_key,
            },
        )

    # ------------------------------------------------------------------
    # 4. Success
    # ------------------------------------------------------------------
    return AssetCheckResult(
        passed=True,
        metadata={
            "fixture_id": context.partition_key,
            "n_rows_fixture": len(players),
            "n_unique_players_fixture": players["person_id"].nunique(),
            "n_unique_identity_keys_global": all_players[identity_key]
            .drop_duplicates()
            .shape[0],
        },
    )
