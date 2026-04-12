# Pipeline Development Guide

Use this guide when writing or changing business logic in `src/pipelines/`.

It captures repository-specific expectations for function boundaries, DataFrame handling,
partition awareness, and failure behavior.

---

## Scope

This guide is for:

- normalization, sync, feature, and ML helpers in `src/pipelines/`
- function signatures and return contracts
- schema validation, logging, and empty-result handling

If you are only wiring an existing function into Dagster, read `dagster-development-guide.md`.

---

## Where business logic lives

All transforms, computations, and data processing belong in `src/pipelines/`.
Dagster assets in `src/hbl_etl_dagster/` are thin wrappers that call pipeline functions.
Never put business logic in asset files.

```
src/pipelines/
  raw/            # parse raw API responses into DataFrames
  normalized/     # rename, type-cast, clean, validate
  synced/         # cross-source joins (shot_events.py, players.py)
  features/       # feature engineering (calc_xg_features.py)
  ml/             # train, infer, evaluate (train_xg.py)
```

---

## Adding a new pipeline function

1. Create or extend a module in the appropriate subfolder.
2. Write a pure function — takes DataFrames/scalars in, returns a DataFrame out.
3. Add a thin Dagster asset in the matching `assets_*` subfolder that calls it.
4. Register the asset in `defs.py`.
5. Add a unit test in `test/` targeting the pipeline function.

### Minimal pipeline function template

```python
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def transform_foo(df_input: pd.DataFrame) -> pd.DataFrame:
    """One-line description of what this produces.

    Args:
        df_input: DataFrame with columns [...].

    Returns:
        DataFrame with columns [...].
    """
    if df_input.empty:
        logger.warning("transform_foo received empty input")
        return pd.DataFrame()

    df = df_input.copy()
    # ... transform logic ...
    logger.info("transform_foo: produced %d rows", len(df))
    return df
```

### Minimal Dagster asset wrapper template

```python
from dagster import asset, AssetExecutionContext, Output, MetadataValue
import pandas as pd

from pipelines.your_module import transform_foo


@asset(
    group_name="your_group",
    partitions_def=fixtures_partition_def,
)
def your_asset(context: AssetExecutionContext, upstream_asset: pd.DataFrame) -> Output[pd.DataFrame]:
    df = transform_foo(upstream_asset)
    return Output(
        df,
        metadata={"row_count": MetadataValue.int(len(df))},
    )
```

---

## DataFrame conventions

- Use `df_*` prefix for DataFrame variables: `df_shots`, `df_events`.
- Copy before mutating: `df = df_input.copy()`.
- Preserve key identifiers across transforms: `fixture_id`, `event_id`, `person_id`.
- Log row counts at the start and end of significant transforms.
- Return an empty `pd.DataFrame()` only when that is the correct empty-result semantics;
  never silently swallow schema errors.

---

## Partition awareness

- Pipeline functions are partition-unaware — they operate on the slice passed in.
- The `DuckDBIOManager` handles partition reads/writes; do not filter by `fixture_id` manually
  unless you are explicitly reading from a non-partitioned table.
- If a function must know the current partition key, accept it as an explicit argument;
  do not reach into Dagster context from within pipeline functions.

---

## Error handling

- Fail fast: raise `ValueError` or `RuntimeError` on invalid required inputs.
- Use `logger.exception(...)` + re-raise when catching around API calls.
- Do not silently swallow schema problems or missing columns.

```python
required = {"fixture_id", "event_id", "timestamp_ms"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")
```

---

## Nested / JSON columns in DuckDB

The `DuckDBIOManager` auto-JSON-serializes `dict` and `list` columns on write.
- Avoid storing list/dict data in feature tables intended for ML — DuckDB round-trips
  these as strings, which adds fragile deserialization steps.
- For sequence data, store as normalized relational tables (one row per frame-agent),
  not as nested blobs in a single column.

---

## Logging style

```python
import logging
logger = logging.getLogger(__name__)

logger.info("sync_shot_events: fixture=%s, goals=%d, matched=%d", fixture_id, n_goals, n_matched)
logger.warning("sync_shot_events: large time_diff for goal %s: %.1f s", event_id, diff_s)
logger.exception("fetch_session failed for session_id=%s", session_id)  # inside except block
```

---

## Naming

| Thing | Convention | Example |
|-------|-----------|---------|
| Module | `snake_case` | `calc_xg_features.py` |
| Function | `snake_case` verb | `normalize_match_events`, `calculate_xg_features` |
| DataFrame variable | `df_*` | `df_shots`, `df_events` |
| Constant | `UPPER_SNAKE_CASE` | `_MAX_RESIDUAL_STD_MS = 5_000` |
| Private helper | leading underscore | `_sync_goals_to_detected_shots` |

---

## Related

- [dagster-development-guide.md](dagster-development-guide.md) — asset wrappers, IO managers, defs.py
- [testing-guide.md](testing-guide.md) — how to test pipeline functions
