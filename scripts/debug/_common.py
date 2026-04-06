"""Shared constants and helpers for the debug scripts."""

import random
import sys
from dataclasses import dataclass
from pathlib import Path

# Repo root — scripts/debug/ → scripts/ → repo root
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

# Ensure repo root is on sys.path so `src.*` imports work regardless of cwd
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import duckdb  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_CONFIG_INFO_ONLY = {
    "loggers": {"console": {"config": {"log_level": "INFO"}}},
}

DB_PATH = "data/hbl_raw.duckdb"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FixtureRunSummary:
    fixture_id: str
    success: bool
    step_successes: int | None
    exit_code: int
    output: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_known_fixture_ids() -> list[str]:
    with duckdb.connect(DB_PATH, read_only=True) as con:
        rows = con.execute(
            "SELECT DISTINCT fixture_id FROM fixtures_sportradar_raw ORDER BY fixture_id"
        ).fetchall()
    return [r[0] for r in rows]


def pick_fixture(fixture_id: str | None, seed: int) -> str:
    if fixture_id:
        return fixture_id

    ids = get_known_fixture_ids()
    if not ids:
        print("No fixture_ids found in fixtures_sportradar_raw. Run 'season' first.")
        sys.exit(1)

    rng = random.Random(seed)
    chosen = rng.choice(ids)
    print(f"Selected fixture_id={chosen} (seed={seed}, {len(ids)} total)")
    return chosen


def get_job(name: str):
    from hbl_etl_dagster.defs import defs

    return defs.resolve_job_def(name)


def count_step_successes(result) -> int:
    return sum(1 for e in result.all_events if e.event_type_value == "STEP_SUCCESS")


def print_fixture_failure_output(output: str, max_lines: int = 30) -> None:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if not lines:
        print("    (no output captured)")
        return

    if len(lines) > max_lines:
        print(f"    … showing last {max_lines} lines of subprocess output …")
        lines = lines[-max_lines:]

    for line in lines:
        print(f"    {line}")


def report(result) -> None:
    if result.success:
        successes = count_step_successes(result)
        print(f"\n✓ Job succeeded  ({successes} steps)")
    else:
        print("\n✗ Job FAILED")
        for e in result.all_events:
            if e.event_type_value in (
                "STEP_FAILURE",
                "RUN_FAILURE",
                "PIPELINE_FAILURE",
            ):
                print(f"  [{e.event_type_value}] {e.message}")
        sys.exit(1)
