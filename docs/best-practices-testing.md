# Best Practices: Testing

This document covers how to write, organize, and run tests for `src/pipelines/` functions.

---

## What to test and where

| Test type | Target | Location |
|-----------|--------|----------|
| Unit | Pure pipeline functions in `src/pipelines/` | `test/test_pipeline_*.py` |
| Integration | API fetchers with real credentials | `test/integration/` |
| Dagster asset | Avoid unless explicitly requested | — |

Tests target pipeline functions, not Dagster asset wrappers. If you add logic to
`src/pipelines/`, add a corresponding test in `test/`.

---

## Running tests

```bash
uv run pytest test/                                              # all tests
uv run pytest test/test_pipeline_synced.py                       # one file
uv run pytest test/test_pipeline_synced.py::test_sync_shot_events  # one test
uv run pytest test/ -x                                           # stop on first failure
uv run pytest test/test_pipeline_synced.py -s                    # show print/log output
uv run pytest test/integration/                                  # integration tests only
```

---

## Writing a unit test

```python
import pandas as pd
import pytest
from pipelines.synced.shot_events import sync_shot_events


def test_sync_shot_events_empty_input():
    df_goals = pd.DataFrame(columns=["fixture_id", "event_id", "event_time_ms", "person_league_id"])
    df_shots = pd.DataFrame(columns=["fixture_id", "detected_shot_id", "timestamp_ms", "league_id"])
    result = sync_shot_events(df_goals, df_shots)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_sync_shot_events_basic_match():
    df_goals = pd.DataFrame({
        "fixture_id": ["fix1"],
        "event_id": ["evt1"],
        "event_time_ms": [1_000_000],
        "person_league_id": ["p1"],
    })
    df_shots = pd.DataFrame({
        "fixture_id": ["fix1"],
        "detected_shot_id": ["shot1"],
        "timestamp_ms": [998_000],   # 2s before SR goal
        "league_id": ["p1"],
    })
    result = sync_shot_events(df_goals, df_shots)
    assert len(result) == 1
    assert result.iloc[0]["detected_shot_id"] == "shot1"
    assert result.iloc[0]["match_method"] == "player_time"
```

---

## Using sample CSV data

Small reference CSVs live in `assets/data_samples/`. Use these for lightweight
data-driven checks that don't require API credentials.

```python
import pandas as pd
from pathlib import Path

DATA = Path("assets/data_samples")

def test_normalize_match_events_with_sample():
    df_raw = pd.read_csv(DATA / "main.sportradar_goals_refined.csv")
    from pipelines.normalized.match_events import normalize_match_events
    result = normalize_match_events(df_raw)
    assert "event_time_ms" in result.columns
    assert result["fixture_id"].notna().all()
```

---

## Integration tests

Integration tests hit real APIs and require credentials from `.env`.
They auto-skip when credentials are absent:

```python
import pytest
import os

@pytest.fixture(autouse=True)
def require_sportradar_creds():
    if not os.getenv("CLIENT_ID"):
        pytest.skip("Sportradar credentials not configured")
```

Keep integration tests in `test/integration/` and do not mix them with unit tests.

---

## conftest.py

`test/conftest.py` adds the project root to `sys.path`. Do not repeat this in individual test files.
Import from `pipelines.*` (absolute, matching the `src/` layout) consistently.

---

## What makes a good test

- **Test behavior, not implementation**: test what the function produces, not how it does it.
- **One assertion cluster per test**: keep tests focused and readable.
- **Descriptive names**: `test_sync_shot_events_falls_back_to_time_only_when_player_missing`.
- **Fast**: avoid sleeping, retrying, or waiting on external I/O in unit tests.
- **Deterministic**: no random seeds unless testing randomized behavior explicitly.

---

## Test coverage priorities

1. Edge cases: empty input, missing required columns, single-row input.
2. Known pitfalls documented in [shot-detection-analysis.md](shot-detection-analysis.md) — regression tests.
3. Core transforms: normalization, sync, feature calculation.
4. ML smoke tests: one tiny training pass, grouped split behavior.

Do not add tests for trivial getters/setters or pure configuration code.

---

## Checking a fix with a regression test

When fixing a bug (e.g. the `detected_shot_id` deduplication bug), add a test that would have
caught it:

```python
def test_sync_does_not_assign_same_detected_shot_to_multiple_goals():
    """Each detected_shot_id should appear at most once in the output."""
    # ... setup two goals that could match the same shot ...
    result = sync_shot_events(df_goals, df_shots)
    assert result["detected_shot_id"].dropna().nunique() == result["detected_shot_id"].dropna().count()
```

---

## Related

- [best-practices-pipeline.md](best-practices-pipeline.md) — writing pipeline functions
- [AGENTS.md](../AGENTS.md) — full test command reference and policy
