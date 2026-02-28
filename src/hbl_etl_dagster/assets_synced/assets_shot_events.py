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

from src.pipelines.synced.shot_events import sync_shot_events as sync_shot_events_fn

fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")


@asset(
    group_name="synced",
    compute_kind="duckdb",
    partitions_def=fixtures_partition_def,
    description="Synced shot events for a single fixture (partitioned by fixture_id).",
    deps=[
        "match_positions_normalized",
    ],
)
def shot_events(
    context: AssetExecutionContext,
    matches_normalized: pd.DataFrame,
    match_events_normalized_goals: pd.DataFrame,
    match_detected_shots_normalized: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract synced shot events for a single fixture.

    :param context: Description
    :type context: AssetExecutionContext

    :param matches_normalized: Description
    :type matches_normalized: pd.DataFrame
    :param match_events_normalized_goals: Description
    :type match_events_normalized_goals: pd.DataFrame
    :param match_detected_shots_normalized: Description
    :type match_detected_shots_normalized: pd.DataFrame
    :param players: Description
    :type players: pd.DataFrame
    :return: Description
    :rtype: pd.DataFrame
    """
    fixture_id = context.partition_key

    df_match_detected_shots = match_detected_shots_normalized[
        match_detected_shots_normalized["fixture_id"] == str(fixture_id)
    ].copy()

    df_match_events_goals = match_events_normalized_goals[
        match_events_normalized_goals["fixture_id"] == str(fixture_id)
    ].copy()

    df_players_match = players[players["fixture_id"] == str(fixture_id)].copy()

    df_positions = context.resources.io_manager.load_partitioned_input(
        table_name="match_positions_normalized",
        partition_key=fixture_id,
        partition_col="fixture_id",
    )

    df_match_normalized = matches_normalized[
        matches_normalized["fixture_id"] == str(fixture_id)
    ].copy()

    df_shot_events = sync_shot_events_fn(
        df_match_normalized=df_match_normalized,
        df_match_events_normalized_goals=df_match_events_goals,
        df_match_detected_shots_normalized=df_match_detected_shots,
        df_positions_normalized=df_positions,
        df_players=df_players_match,
    )

    context.log.info("Extracted %d synced shot events", len(df_shot_events))
    context.add_output_metadata(
        {
            "n_rows": len(df_shot_events),
            "n_unique_shot_events": df_shot_events["event_id"].nunique(),
            "n_unique_throw_timestamps": df_shot_events["throw_timestamp_ms"].nunique(),
            "n_columns": df_shot_events.shape[1],
            "preview": MetadataValue.md(
                df_shot_events.head(100).to_markdown(index=False)
            ),
        }
    )
    return df_shot_events


@asset_check(
    asset="shot_events",
    name="check_shot_events_integrity",
)
def check_shot_events_integrity(
    context: AssetCheckExecutionContext,
    shot_events: pd.DataFrame,
) -> AssetCheckResult:
    # ------------------------------------------------------------------
    # 1. Column contract
    # ------------------------------------------------------------------
    required_columns = {
        "fixture_id",
        "entity_id",
        "event_id",
        "event_time",
        "event_type",
        "person_id",
        "bib",
        "position",
        "period_id",
        "x",
        "y",
        "throw_timestamp_ms",
        "detected_shot_id",
        "shot_type",
        "shot_category",
        "validated",
    }

    missing_columns = required_columns - set(shot_events.columns)
    if missing_columns:
        return AssetCheckResult(
            passed=False,
            description=f"Missing required columns: {sorted(missing_columns)}",
        )

    # First execution / nothing materialized yet
    if shot_events.empty:
        return AssetCheckResult(passed=True)

    # ------------------------------------------------------------------
    # 2. Global identity integrity
    #    (same event_id must not drift across fixtures)
    # ------------------------------------------------------------------
    identity_key = ["event_id"]

    identity_payload = [
        "fixture_id",
        "entity_id",
        "event_time",
        "event_type",
        "person_id",
        "bib",
        "position",
        "period_id",
        "x",
        "y",
        "shot_type",
        "shot_category",
        "validated",
        "throw_timestamp_ms",
        "detected_shot_id",
    ]

    inconsistencies = (
        shot_events.groupby(identity_key, dropna=False)[identity_payload]
        .nunique()
        .max(axis=1)
        .loc[lambda s: s > 1]
    )

    if not inconsistencies.empty:
        return AssetCheckResult(
            passed=False,
            description=(
                "Conflicting shot event records detected for the same event_id "
                "(identity drift)"
            ),
            metadata={
                "n_conflicting_event_ids": int(len(inconsistencies)),
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

    shot_events_part = shot_events[shot_events["fixture_id"] == str(partition_key)]

    if shot_events_part.empty:
        return AssetCheckResult(
            passed=False,
            metadata={
                "failed": "no rows for partition",
                "fixture_id": partition_key,
            },
        )

    # event_id must be unique within a fixture
    if not shot_events_part["event_id"].astype(str).is_unique:
        n_dup = int(
            shot_events_part["event_id"].astype(str).shape[0]
            - shot_events_part["event_id"].astype(str).nunique()
        )
        return AssetCheckResult(
            passed=False,
            metadata={
                "failed": "duplicate event_id within fixture",
                "n_duplicates": n_dup,
                "fixture_id": partition_key,
            },
        )

    # event_id must be non-null within a fixture
    if shot_events_part["event_id"].isna().any():
        return AssetCheckResult(
            passed=False,
            metadata={
                "failed": "null event_id values",
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
            "n_rows_fixture": len(shot_events_part),
            "n_unique_events_fixture": shot_events_part["event_id"].nunique(),
            "n_unique_event_ids_global": shot_events["event_id"].nunique(),
            "n_unique_players_global": shot_events["person_id"].nunique(),
        },
    )


@asset_check(
    asset="shot_events",
    name="check_shot_events_sync_result",
)
def check_shot_events_sync_result(
    context: AssetCheckExecutionContext,
    shot_events: pd.DataFrame,
) -> AssetCheckResult:
    partition_key = context.run.tags.get("dagster/partition")

    if partition_key is None:
        return AssetCheckResult(
            passed=True,
            metadata={"skipped": "no partition context"},
        )

    df = shot_events[shot_events["fixture_id"] == str(partition_key)]

    if df.empty:
        return AssetCheckResult(
            passed=False,
            metadata={
                "failed": "no rows for fixture",
                "fixture_id": partition_key,
            },
        )

    n_rows = int(len(df))

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------
    n_detected = int(df["detected_shot_id"].notna().sum())
    pct_detected = float(n_detected / n_rows)

    n_throw = int(df["throw_timestamp_ms"].notna().sum())
    pct_throw = float(n_throw / n_rows)

    MIN_DETECTED_PCT = 0.30
    MIN_THROW_PCT = 0.20

    if pct_detected < MIN_DETECTED_PCT:
        return AssetCheckResult(
            passed=False,
            metadata={
                "failed": "low detected-shot match coverage",
                "fixture_id": partition_key,
                "detected_pct": float(round(pct_detected * 100, 2)),
            },
        )

    if pct_throw < MIN_THROW_PCT:
        return AssetCheckResult(
            passed=False,
            metadata={
                "failed": "low throw-timestamp coverage",
                "fixture_id": partition_key,
                "throw_pct": float(round(pct_throw * 100, 2)),
            },
        )

    # ------------------------------------------------------------------
    # Time-delta stats (SAFE)
    # ------------------------------------------------------------------
    delta_stats = {}

    for col in [
        "time_diff_event_throw_ms",
        "time_diff_detected_shot_throw_ms",
    ]:
        if col in df.columns:
            vals = df[col].dropna().astype(float)
            if not vals.empty:
                delta_stats[f"{col}_mean_ms"] = float(round(vals.mean(), 1))
                delta_stats[f"{col}_median_ms"] = float(round(vals.median(), 1))
                delta_stats[f"{col}_std_ms"] = float(round(vals.std(), 1))

    return AssetCheckResult(
        passed=True,
        metadata={
            "fixture_id": partition_key,
            "n_rows": n_rows,
            "detected_shot_coverage_pct": float(round(pct_detected * 100, 2)),
            "throw_timestamp_coverage_pct": float(round(pct_throw * 100, 2)),
            **delta_stats,
        },
    )
