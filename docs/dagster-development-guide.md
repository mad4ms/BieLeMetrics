# Dagster Development Guide

Use this guide when adding or changing Dagster assets, IO-manager usage, job definitions,
or `defs.py` wiring in this repository.

It documents repository-specific conventions, not generic Dagster advice.

---

## Scope

This guide is for:

- thin asset wrappers in `src/hbl_etl_dagster/`
- IO-manager selection and partition handling
- asset registration in `defs.py`
- deciding whether work belongs in a partitioned or global job

If you are changing business logic in `src/pipelines/`, read `pipeline-development-guide.md` first.

---

## Asset structure principles

- Assets are **thin wrappers** — call a pipeline function, log metadata, return the result.
- No business logic inside asset files.
- One asset per file (or closely related assets per file) in the relevant `assets_*` subfolder.
- Register every new asset explicitly in `defs.py`; no wildcard imports.

---

## IO manager selection

| Situation | IO manager key | How to use |
|-----------|---------------|------------|
| DataFrame — default | `"io_manager"` (DuckDB) | Implicit — no `io_manager_key` needed |
| DataFrame with non-default partition column | `"io_manager"` | Add `metadata={"partition_column": "other_col"}` to `@asset` |
| Non-serializable object within a run | `"in_memory_io_manager"` | `@asset(io_manager_key="in_memory_io_manager")` |
| sklearn / XGBoost model artifact | `"file_io_manager"` | `@asset(io_manager_key="file_io_manager")` |

### Overriding partition column

```python
@asset(
    metadata={"partition_column": "session_id"},
    partitions_def=fixtures_partition_def,
)
def my_asset(...) -> pd.DataFrame:
    ...
```

### Non-partitioned DataFrame

If an asset produces a table that is not partition-scoped (e.g., a global lookup), use
`metadata={"partition_column": None}` or skip the `partitions_def`. Do not add a fake
`fixture_id` column just to satisfy the IO manager.

---

## Partition-aware patterns

The default partition column is `fixture_id`. The IO manager:
- **On write**: upserts rows where `fixture_id = <current_partition_key>`, removing old rows for that key first.
- **On read**: filters the table to `fixture_id = <current_partition_key>`.

Do not filter by `fixture_id` manually inside the asset or pipeline function — the IO manager handles it.

```python
@asset(partitions_def=fixtures_partition_def)
def shot_events(
    context: AssetExecutionContext,
    match_events_normalized: pd.DataFrame,
    match_detected_shots_normalized: pd.DataFrame,
) -> Output[pd.DataFrame]:
    # match_events_normalized is already filtered to this partition by the IO manager
    df = sync_shot_events(match_events_normalized, match_detected_shots_normalized)
    return Output(df, metadata={"row_count": MetadataValue.int(len(df))})
```

---

## Output metadata

Always emit at least a row count. Add previews for diagnostic value:

```python
return Output(
    df,
    metadata={
        "row_count": MetadataValue.int(len(df)),
        "preview": MetadataValue.md(df.head(5).to_markdown(index=False)),
    },
)
```

---

## Registering assets in defs.py

Add every new asset to the explicit import list in `src/hbl_etl_dagster/defs.py`.
Do not use `load_assets_from_modules` or wildcard imports.

```python
# defs.py
from hbl_etl_dagster.assets_features.features_xg import features_xg
from hbl_etl_dagster.assets_features.features_xg_sequence import features_xg_sequence  # new

defs = Definitions(
    assets=[
        ...,
        features_xg,
        features_xg_sequence,  # add here
    ],
    ...
)
```

---

## Defining jobs

A job selects a subset of assets to run together. Keep jobs in `defs.py`.

### Partitioned fixture job

```python
from dagster import define_asset_job, AssetSelection

fixture_raw_backfill_job = define_asset_job(
    name="fixture_raw_backfill_job",
    selection=AssetSelection.assets(
        sportradar_raw_fixture,
        kinexon_raw_fixture,
        # ... all fixture-partitioned assets ...
    ),
    partitions_def=fixtures_partition_def,
)
```

### Global (non-partitioned) job

For model training or any asset that consumes data from all fixtures:

```python
xg_global_training_job = define_asset_job(
    name="xg_global_training_job",
    selection=AssetSelection.assets(ml_xg_model_global),
    # no partitions_def
)
```

---

## When to use `in_memory_io_manager`

Use for intermediate objects that are produced and consumed within the same run and cannot be
serialized to disk efficiently (e.g., a fitted scaler passed to an inference asset in the same run).

```python
@asset(io_manager_key="in_memory_io_manager", partitions_def=fixtures_partition_def)
def xg_scaler(features_xg: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(features_xg[FEATURE_COLS])
    return scaler
```

---

## Checking the active entrypoint

The active definitions entrypoint is `src/hbl_etl_dagster/defs.py`.
`defs_debug.py` (if present) is a compatibility re-export only. Always wire new assets into `defs.py`.

To verify your asset is registered:
```bash
uv run dagster asset list -m hbl_etl_dagster.defs | grep your_asset_name
```

---

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Business logic inside an asset function | Move to `src/pipelines/` and call from the asset |
| Manually filtering by `fixture_id` inside pipeline function | Remove — IO manager does this |
| Wildcard import in `defs.py` | Use explicit named imports |
| Model training asset inside partitioned job | Move to a global (non-partitioned) job |
| Storing nested list/dict columns in DuckDB tables | Use normalized relational tables instead |

---

## Related

- [pipeline-development-guide.md](pipeline-development-guide.md) — pipeline function conventions
- [testing-guide.md](testing-guide.md) — testing pipeline functions
- [CLAUDE.md](../CLAUDE.md) — full IO manager reference and job definitions
