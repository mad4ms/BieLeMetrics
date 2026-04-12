# Architecture

## Overview

BieLeMetrics is a multi-source sports analytics pipeline. It fuses two independent data streams —
Sportradar (structured event data) and Kinexon (raw positional tracking) — into ML-ready features
and trained xG/xS models for professional handball.

```
Sportradar API          Kinexon API
     │                       │
     ▼                       ▼
 assets_raw/            assets_raw/
 (SR events,            (KX positions,
  players, fixtures)     detected shots, sessions)
     │                       │
     └───────────┬───────────┘
                 ▼
          assets_normalized/
          (per-fixture: events, positions,
           detected shots, players, matches)
                 │
                 ▼
          assets_synced/
          (cross-source joins:
           shot_events, players)
                 │
                 ▼
          assets_features/
          (features_xg, features_xs)
                 │
                 ▼
           assets_ml/
           (ml_xg_model, xg_feature_importance)
```

---

## Layer Separation

The codebase is split into two distinct concerns:

| Layer | Location | Responsibility |
|-------|----------|---------------|
| **Business logic** | `src/pipelines/` | Pure Python — DataFrames in, DataFrames out. No Dagster. Testable in isolation. |
| **Orchestration** | `src/hbl_etl_dagster/` | Thin Dagster asset wrappers that call pipeline functions, emit metadata, and persist via IO managers. |

This split means pipeline functions can be called from tests, scripts, or notebooks without any
Dagster dependency. The asset layer only adds context, logging, and IO wiring.

```
src/pipelines/             src/hbl_etl_dagster/
   raw/                       assets_raw/
   normalized/       ←──────  assets_normalized/
   synced/                    assets_synced/
   features/                  assets_features/
   ml/                        assets_ml/
```

---

## Dagster Jobs

There are three jobs, each selecting a different subset of assets:

### `season_raw_refresh_job` (unpartitioned)

Fetches season-level catalog data from both APIs and builds the fixture partition list.
Run once per season or when the fixture list changes.

```
competition_id
season_id
teams_sportradar_raw
fixtures_sportradar_raw      ← populates fixtures_partition_def
teams_kinexon_raw
sessions_kinexon_raw
matches_normalized           ← links Kinexon sessions to SR fixtures
```

### `fixture_raw_backfill_job` (partitioned by `fixture_id`)

Full fixture pipeline from raw ingest through feature generation.
Run once per fixture, or re-run from a specific asset using `scripts/debug/pipeline.py fixture <fixture_id> --from <asset>`.

```
fixture_events_sportradar_raw   players_sportradar_raw
detected_events_kinexon_raw     positions_kinexon_raw
       ↓                                ↓
  match_events_normalized*     match_positions_normalized
  match_detected_shots_normalized
  match_players_normalized
       ↓
   players  (cross-source player ID mapping)
       ↓
   shot_events  (goal ↔ detected-shot sync)
       ↓
   features_xg
```

*`match_events_normalized` fans out into `match_events_normalized_setup` and
`match_events_normalized_goals`, which feed `shot_events`.

### `xg_training_job` (unpartitioned / global)

Reads all `features_xg` rows across all fixtures and trains the XGBoost model.
Must run after `fixture_raw_backfill_job` has populated features for enough fixtures.

```
ml_xg_model          ← reads full features_xg table, trains, persists via file_io_manager
xg_feature_importance
```

---

## Partitioning

Assets are partitioned by `fixture_id` using a `DynamicPartitionsDefinition` named
`"fixture_partitions"`. Partitions are registered by `season_raw_refresh_job` when it
writes `fixtures_sportradar_raw`.

The `DuckDBIOManager` handles partition-scoped reads and writes transparently:
- **Write**: `DELETE WHERE fixture_id = <key>` then `INSERT` the new slice.
- **Read**: `SELECT … WHERE fixture_id = <key>`.

The default partition column is `fixture_id`. Override per-asset with
`metadata={"partition_column": "other_col"}`.

---

## Storage

All DataFrames are stored in a single DuckDB file at `data/hbl_raw.duckdb`.
Each Dagster asset maps to one table (name derived from the asset key).

| IO manager | Key | Use |
|------------|-----|-----|
| `DuckDBIOManager` | `"io_manager"` | All DataFrames (default) |
| `InMemoryIOManager` | `"in_memory_io_manager"` | Non-serializable objects within a run |
| `FilesystemIOManager` | `"file_io_manager"` | Model artifacts on disk |

Model artifacts (sklearn `Pipeline` objects containing the XGBoost classifier) are persisted by
Dagster's `FilesystemIOManager` to the local filesystem, not inside DuckDB.

---

## External Resources

| Resource | Key | Source |
|----------|-----|--------|
| Sportradar REST API | `sportradar_api` | OAuth2, credentials in `.env` |
| Kinexon REST API | `kinexon_api` | API key + session auth, credentials in `.env` |

---

## Cross-Source Synchronization

The hardest part of the pipeline is aligning Sportradar events to Kinexon positional data.
The two sources use different clocks with fixture-level offsets of −35 s to +35 s.

Key sync steps in `src/pipelines/synced/shot_events.py`:

1. **Player ID mapping** (`players.py`) — fuzzy name match constrained to same team.
2. **Pass 1 (drift estimation)** — collect player-matched (SR event, KX shot) pairs within a
   wide window; fit linear clock-drift model via regression.
3. **Pass 2 (final match)** — if drift is active, search in a tight window centered on
   `predicted_kin_ms`; otherwise fall back to the wide window.
4. **Throw refinement** — within the matched scene window, find the last possession-end
   acceleration peak to estimate the actual release timestamp.
5. **Goal-side inference** — assign `goal_position` from goalkeeper positions, per (team, period).

See [shot-detection-analysis.md](shot-detection-analysis.md) for how this pipeline was developed, pitfalls, and fixes.

---

## Asset Checks

Dagster asset checks run alongside materialization to catch schema and integrity issues early:

| Check | What it validates |
|-------|------------------|
| `check_fixtures_have_unique_ids_when_present` | No duplicate fixture rows |
| `check_events_have_unique_event_ids_when_present` | No duplicate event IDs per fixture |
| `check_positions_kinexon_raw_notna` | Position data not empty |
| `check_players_integrity` | Player mapping completeness |
| `check_shot_events_integrity` | No null required fields in shot_events |
| `check_shot_events_sync_result` | Sync quality (fallback rate, time_diff distribution) |

---

## Debug Workflow

For local iteration without the Dagster UI:

```bash
uv run python scripts/debug/pipeline.py fixture <fixture_id>              # full fixture pipeline
uv run python scripts/debug/pipeline.py fixture <fixture_id> --from shot_events  # re-run from asset
uv run python scripts/debug/ml.py evaluate                                 # evaluate trained xG model
```

---

## Related

- [data-model.md](data-model.md) — DuckDB tables, schemas, relationships
- [xg-feature-model-guide.md](xg-feature-model-guide.md) — xG feature engineering, model training, and evaluation protocol
- [dagster-development-guide.md](dagster-development-guide.md) — how to add assets, jobs, IO managers
- [pipeline-development-guide.md](pipeline-development-guide.md) — how to write pipeline functions
