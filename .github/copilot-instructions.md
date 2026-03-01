# BieLeMetrics – Copilot / Agent Instructions

## Current Dagster entrypoint (important)
- Active entrypoint: `src/hbl_etl_dagster/defs.py`
- `src/hbl_etl_dagster/defs_debug.py` is a compatibility re-export (`from .defs import defs`).
- Sensors are currently **not used** in the active setup.

## What is active vs. legacy in `src/hbl_etl_dagster/`

### Active asset modules (use these)
- `assets_raw/` (Sportradar + Kinexon raw ingestion)
- `assets_normalized/` (normalized match-level tables)
- `assets_synced/` (cross-source synced entities)
- `assets_features/` (feature engineering)
- `assets_ml/` (model training / feature importance / inference)

### Legacy top-level `assets_*.py` status
- Legacy top-level asset modules were removed during cleanup.
- If you see references in old branches/commits (e.g. `assets_sportradar.py`, `assets_sync.py`), treat them as historical only.

Rule of thumb: for production DAG behavior, implement in the active `assets_*` **subfolders** listed above and wire through `defs.py`.

## Code organization rules
1. Keep API/business logic in `src/events_*` when possible.
2. Keep Dagster orchestration in `src/hbl_etl_dagster/*`.
3. Do not introduce new dependencies from active assets to legacy top-level asset modules.
4. Prefer explicit imports in `defs.py`; avoid wildcard imports.

## Job model
- `season_raw_refresh_job`: refreshes season-level/raw foundation assets.
- `fixture_raw_backfill_job`: partitioned by fixture (`fixtures_partition_def`) and runs fixture-level raw → normalized → synced → feature → ml assets.

## Resources / persistence
- IO manager: DuckDB-backed (`duckdb_io_manager`) with DB path currently configured in `defs.py`.
- External APIs come from `resources.py` (`sportradar_api`, `kinexon_api`).

## Change guidelines
- For new data products, add assets in active subfolders and register in `defs.py`.
- If removing legacy code, verify it is not imported by `defs.py` first.
- Keep changes minimal and avoid touching unrelated notebooks/tests unless requested.

## Testing policy
- Prefer tests for pure pipeline modules in `src/pipelines/*` (business logic).
- Avoid coupling new tests to Dagster asset wrappers unless explicitly requested.
- Use `assets/data_samples/*.csv` for lightweight integration-style checks of pipeline transforms.
