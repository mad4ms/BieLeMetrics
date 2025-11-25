import json
import os
import duckdb
from dagster import (
    sensor,
    SensorEvaluationContext,
    RunRequest,
    AssetSelection,
    DefaultSensorStatus,
    RunsFilter,
    DagsterRunStatus,
)
from .assets_sportradar import (
    fixture_events_raw,
    fixture_events_match,
    fixture_players,
)
from .assets_kinexon import kinexon_positions, kinexon_events
from .assets_sync import (
    players_merged,
    sportradar_goals_synced,
    sportradar_goals_refined,
)
from .assets_ids import fixtures_partition_def


FIXTURE_RUN_CONCURRENCY_LIMIT = int(
    os.getenv("FIXTURE_RUN_CONCURRENCY_LIMIT", "1")
)


@sensor(
    asset_selection=AssetSelection.assets(
        fixture_events_raw,
        fixture_events_match,
        fixture_players,
        kinexon_positions,
        kinexon_events,
        players_merged,
        sportradar_goals_synced,
        sportradar_goals_refined,
    ),
    default_status=DefaultSensorStatus.STOPPED,
)
def fixture_sensor(context: SensorEvaluationContext):
    """Poll DuckDB for new fixtures, update dynamic partitions, and request runs."""

    db_path = "data/hbl.duckdb"
    if not os.path.exists(db_path):
        return

    try:
        con = duckdb.connect(db_path, read_only=True)
        tables = con.execute("SHOW TABLES").fetchdf()
        if "fixtures" not in tables["name"].values:
            con.close()
            return

        df = con.execute("SELECT DISTINCT fixture_id FROM fixtures").fetchdf()
        con.close()

    except Exception as e:
        context.log.error(f"Error updating fixture partitions: {e}")
        return

    current_ids = sorted({str(x) for x in df["fixture_id"].tolist()})
    if not current_ids:
        return

    context.instance.add_dynamic_partitions(
        fixtures_partition_def.name,
        current_ids,
    )

    try:
        seen_ids = set(json.loads(context.cursor)) if context.cursor else set()
    except json.JSONDecodeError:
        seen_ids = set()

    new_ids = [fid for fid in current_ids if fid not in seen_ids]
    if not new_ids:
        return

    run_requests = []
    active_runs = context.instance.get_runs(
        filters=RunsFilter(
            tags={"dagster/concurrency_key": ["fixture_runs"]},
            statuses=[
                DagsterRunStatus.NOT_STARTED,
                DagsterRunStatus.QUEUED,
                DagsterRunStatus.STARTING,
                DagsterRunStatus.STARTED,
                DagsterRunStatus.CANCELING,
            ],
        )
    )

    available_slots = max(0, FIXTURE_RUN_CONCURRENCY_LIMIT - len(active_runs))
    if available_slots == 0:
        context.log.info(
            "Fixture run concurrency limit reached; skipping new requests for now."
        )
        return

    scheduled_ids = []
    for fixture_id in new_ids:
        if available_slots <= 0:
            break
        run_requests.append(
            RunRequest(
                run_key=f"fixture:{fixture_id}",
                partition_key=fixture_id,
                tags={"dagster/concurrency_key": "fixture_runs"},
            )
        )
        scheduled_ids.append(fixture_id)
        available_slots -= 1

    if not run_requests:
        return

    context.update_cursor(json.dumps(sorted(seen_ids.union(scheduled_ids))))
    if len(new_ids) > len(scheduled_ids):
        context.log.info(
            "Deferred %d fixtures due to concurrency cap.",
            len(new_ids) - len(scheduled_ids),
        )

    return run_requests
