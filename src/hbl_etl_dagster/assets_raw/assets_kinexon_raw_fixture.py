# assets_sportradar_raw.py
import os
from dagster import (
    AssetExecutionContext,
    AssetCheckResult,
    DynamicPartitionsDefinition,
    Failure,
    Field,
    MetadataValue,
    asset,
    asset_check,
)
import pandas as pd

from src.pipelines.raw.kinexon import (
    get_detected_events_for_fixture as kinexon_get_detected_events_for_fixture,
    get_positions_for_session as kinexon_get_positions_for_session,
)


# kinexon_get_positions_for_session
@asset(
    required_resource_keys={
        "kinexon_api",
        "io_manager",
    },
    group_name="kinexon_raw",
    compute_kind="duckdb",
    description="Raw Kinexon positioning data for a single fixture (partitioned by fixture_id).",
    partitions_def=DynamicPartitionsDefinition(name="fixture_partitions"),
)
def positions_kinexon_raw(
    context: AssetExecutionContext,
    matches_normalized: pd.DataFrame,
) -> pd.DataFrame:
    api = context.resources.kinexon_api
    api.connect()
    fixture_id = context.partition_key

    if not fixture_id:
        raise Failure(description="Missing partition_key (fixture_id).")

    df_match = matches_normalized[
        matches_normalized["fixture_id"] == str(fixture_id)
    ].copy()

    if df_match.empty:
        context.log.warning(
            "No session IDs found for fixture_id=%s", fixture_id
        )
        return pd.DataFrame()

    # There can only be one session_id in df_match
    if "session_id" not in df_match.columns:
        df_match["session_id"] = df_match["id"]

    session_id = df_match.iloc[0]["session_id"]
    # to int
    session_id = int(session_id)

    duckdb_io_manager = context.resources.io_manager

    # check if session_id already present in table
    with duckdb_io_manager._conn() as con:
        # does table exist?
        table_exists = (
            con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'positions_kinexon_raw'"
            ).fetchone()[0]
            == 1
        )
        if table_exists:
            existing_sessions = con.execute(
                "SELECT DISTINCT session_id FROM positions_kinexon_raw"
            ).fetchall()
            existing_session_ids = {row[0] for row in existing_sessions}
        else:
            existing_session_ids = set()
    if session_id in existing_session_ids:
        # read existing data from DuckDB
        with duckdb_io_manager._conn() as con:
            df_positions = con.execute(
                f"""
                SELECT *
                FROM positions_kinexon_raw
                WHERE session_id = {session_id}
                """
            ).df()

    else:
        # check if file in format kinexon_positions_{session_id}.0.parquet.gzip exists
        if os.path.exists(
            f"data/positions/kinexon_positions_{session_id}.0.parquet.gzip"
        ):
            df_positions = pd.read_parquet(
                f"data/positions/kinexon_positions_{session_id}.0.parquet.gzip"
            )
        else:
            df_positions = kinexon_get_positions_for_session(
                api=api, session_id=session_id
            )
    # insert fixture_id column for partitioning
    df_positions["fixture_id"] = str(fixture_id)
    # and remove "fixtureId" column if exists
    if "fixtureId" in df_positions.columns:
        df_positions = df_positions.drop(columns=["fixtureId"])

    context.log.info(
        "Fetched %d positioning data points (raw) for fixture_id=%s",
        len(df_positions),
        fixture_id,
    )
    context.add_output_metadata(
        {
            "fixture_id": str(fixture_id),
            "n_rows": len(df_positions),
            "n_columns": df_positions.shape[1],
            "preview": MetadataValue.md(
                df_positions.head().to_markdown(index=False)
            ),
        }
    )
    return df_positions


# kinexon_get_detected_events_for_fixture
@asset(
    required_resource_keys={"kinexon_api"},
    group_name="kinexon_raw",
    compute_kind="duckdb",
    description="Raw Kinexon detected events for a single fixture (partitioned by fixture_id).",
    partitions_def=DynamicPartitionsDefinition(name="fixture_partitions"),
)
def detected_events_kinexon_raw(
    context: AssetExecutionContext,
    matches_normalized: pd.DataFrame,
) -> pd.DataFrame:
    api = context.resources.kinexon_api
    fixture_id = context.partition_key

    if not fixture_id:
        raise Failure(description="Missing partition_key (fixture_id).")

    df_match = matches_normalized[
        matches_normalized["fixture_id"] == str(fixture_id)
    ].copy()

    if df_match.empty:
        context.log.warning(
            "No session IDs found for fixture_id=%s", fixture_id
        )
        return pd.DataFrame()

    # There can only be one session_id in df_match
    if "session_id" not in df_match.columns:
        df_match["session_id"] = df_match["id"]

    session_id = df_match.iloc[0]["session_id"]

    df_events = kinexon_get_detected_events_for_fixture(
        api=api, session_id=session_id
    )
    # insert fixture_id column for partitioning
    df_events["fixture_id"] = str(fixture_id)

    context.log.info(
        "Fetched %d detected events (raw) for fixture_id=%s",
        len(df_events),
        fixture_id,
    )
    context.add_output_metadata(
        {
            "fixture_id": str(fixture_id),
            "n_rows": len(df_events),
            "n_columns": df_events.shape[1],
            "preview": MetadataValue.md(
                df_events.head().to_markdown(index=False)
            ),
        }
    )

    return df_events


@asset_check(asset=detected_events_kinexon_raw)
def check_events_have_unique_event_ids_when_present(
    detected_events_kinexon_raw: pd.DataFrame,
) -> AssetCheckResult:
    if detected_events_kinexon_raw.empty:
        return AssetCheckResult(passed=True, metadata={"skipped": "no rows"})

    ok = True
    n_dup = 0

    col = "Id"

    if col in detected_events_kinexon_raw.columns:
        ok = bool(
            detected_events_kinexon_raw["Id"].dropna().astype(str).is_unique
        )
        n_dup = int(
            detected_events_kinexon_raw["Id"].dropna().astype(str).shape[0]
            - detected_events_kinexon_raw["Id"].dropna().astype(str).nunique()
        )
    else:
        return AssetCheckResult(
            passed=True, metadata={"skipped": f"{col} column missing"}
        )
