import os
from multiprocessing import context

import pandas as pd
from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    Failure,
    Field,
    MetadataValue,
    TableColumn,
    TableSchema,
    asset,
    asset_check,
)

from src.pipelines.raw.kinexon import (
    get_detected_events_for_fixture as kinexon_get_detected_events_for_fixture,
)
from src.pipelines.raw.kinexon import (
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
        context.log.warning("No session IDs found for fixture_id=%s", fixture_id)
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
            "dagster/column_schema": TableSchema(
                columns=[
                    TableColumn(name=col, type=str(dtype))
                    for col, dtype in df_positions.dtypes.items()
                ]
            ),
            "dagster/row_count": len(df_positions),
            "fixture_id": str(fixture_id),
            "n_columns": df_positions.shape[1],
            "preview": MetadataValue.md(df_positions.head().to_markdown(index=False)),
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
        context.log.warning("No session IDs found for fixture_id=%s", fixture_id)
        return pd.DataFrame()

    # There can only be one session_id in df_match
    if "session_id" not in df_match.columns:
        df_match["session_id"] = df_match["id"]

    session_id = df_match.iloc[0]["session_id"]

    df_events = kinexon_get_detected_events_for_fixture(api=api, session_id=session_id)
    # insert fixture_id column for partitioning
    df_events["fixture_id"] = str(fixture_id)

    context.log.info(
        "Fetched %d detected events (raw) for fixture_id=%s",
        len(df_events),
        fixture_id,
    )
    context.add_output_metadata(
        {
            "dagster/column_schema": TableSchema(
                columns=[
                    TableColumn(name=col, type=str(dtype))
                    for col, dtype in df_events.dtypes.items()
                ]
            ),
            "dagster/row_count": len(df_events),
            "fixture_id": str(fixture_id),
            "n_columns": df_events.shape[1],
            "preview": MetadataValue.md(df_events.head().to_markdown(index=False)),
        }
    )
    return df_events


@asset_check(asset=positions_kinexon_raw)
def check_positions_kinexon_raw_notna(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    partition_key = context.run.tags["dagster/partition"]
    df_match_position = context.resources.io_manager.load_partitioned_input(
        table_name="positions_kinexon_raw",
        partition_col="fixture_id",
        partition_key=partition_key,
    )

    # check if length is not zero, fail check if zero
    if df_match_position.empty:
        return AssetCheckResult(passed=False, metadata={"failed": "no rows"})

    # check if > 90% of x in m and y in m columns are not null
    required_columns = ["x in m", "y in m"]
    for col in required_columns:
        if col not in df_match_position.columns:
            return AssetCheckResult(
                passed=False, metadata={"failed": f"{col} column missing"}
            )
        n_notna = df_match_position[col].notna().sum()
        n_total = len(df_match_position)
        if n_total == 0 or (n_notna / n_total) < 0.9:
            return AssetCheckResult(
                passed=False,
                metadata={
                    "failed": f"{col} has less than 90% non-null values ({n_notna}/{n_total})"
                },
            )

    return AssetCheckResult(
        passed=True,
        metadata={
            "checked_columns": ["x in m", "y in m"],
            "n_rows": len(df_match_position),
        },
    )


@asset_check(asset=detected_events_kinexon_raw)
def check_detected_events_have_unique_event_ids_when_present(
    detected_events_kinexon_raw: pd.DataFrame,
) -> AssetCheckResult:
    if detected_events_kinexon_raw.empty:
        return AssetCheckResult(passed=True, metadata={"skipped": "no rows"})

    ok = True
    n_dup = 0

    col = "Id"

    if col in detected_events_kinexon_raw.columns:
        ok = bool(detected_events_kinexon_raw["Id"].dropna().astype(str).is_unique)
        n_dup = int(
            detected_events_kinexon_raw["Id"].dropna().astype(str).shape[0]
            - detected_events_kinexon_raw["Id"].dropna().astype(str).nunique()
        )
    else:
        return AssetCheckResult(
            passed=True, metadata={"skipped": f"{col} column missing"}
        )
