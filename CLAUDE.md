# CLAUDE.md

## Purpose / Project Summary

This is the single authoritative guide for coding agents and contributors working in this repository.
BieLeMetrics builds handball analytics pipelines from Sportradar event data and Kinexon positional
tracking into normalized tables, synced shot/player data, xG features, and ML models.
Primary outputs: expected goal (xG) and expected save (xS) models for professional handball.
Research paper implementation — see README.md.

---

## Stack

- Python `>=3.12`
- Package manager and runner: `uv` (use `uv sync` / `uv run`)
- Orchestration: Dagster
- Storage: DuckDB — primary storage via custom `DuckDBIOManager`; DB file: `data/hbl_raw.duckdb`
- Core libraries: `pandas`, `pyarrow`, `scikit-learn`, `xgboost`, `shapely`
- PyTorch: planned for spatiotemporal xG upgrade (not yet active)

---

## Repository Layout

```
src/
  fetcher_sportradar/       # API fetch helpers for Sportradar
  fetcher_kinexon/          # API fetch helpers for Kinexon
  pipelines/                # Pure Python business logic (no Dagster)
    raw/                    # Raw data parsing
    normalized/             # Normalization transforms
    synced/                 # Cross-source synchronization (e.g. shot_events.py)
    features/               # Feature engineering (calc_xg_features.py, calc_xs_features.py)
    ml/                     # Model train/infer (train_xg.py, infer_xg.py, importance_xg.py)
  hbl_etl_dagster/          # Dagster wrappers only (thin assets, io managers, defs)
    assets_raw/
    assets_normalized/
    assets_synced/
    assets_features/
    assets_ml/
    io_managers.py          # DuckDBIOManager + InMemoryIOManager
    defs.py                 # Definitions, jobs, resources
    resources.py            # API resource configs
apps/
  xg_simulator.py
assets/
  data_samples/             # Sample CSVs for tests/integration checks
test/
  test_pipeline_raw.py
  test_pipeline_synced.py
  integration/
docs/                       # Architecture docs, analysis reports, best practices
scripts/                    # One-off analysis and diagnostic scripts
```

---

## Active Dagster Entry Point

- Active entrypoint: `src/hbl_etl_dagster/defs.py`
- `src/hbl_etl_dagster/defs_debug.py` is only a compatibility re-export if present in older branches.
- Sensors are not part of the active workflow.

### Active asset modules (use these)

- `assets_raw/` — Sportradar + Kinexon raw ingestion
- `assets_normalized/` — normalized match-level tables
- `assets_synced/` — cross-source synced entities
- `assets_features/` — feature engineering
- `assets_ml/` — model training / feature importance / inference

Legacy top-level asset modules (e.g. `assets_sportradar.py`, `assets_sync.py`) were removed.
Treat any references in old branches/commits as historical only.

---

## Architecture Rules

1. **Business logic lives in `src/pipelines/`** — Dagster assets in `hbl_etl_dagster/` are thin wrappers that call pipeline functions.
2. **Dagster assets must not contain business logic** — keep asset files focused on `context`, logging, metadata, and delegation.
3. **Tests target `src/pipelines/`** — not Dagster assets directly.
4. **Partitioning**: Assets are partitioned by `fixture_id` (dynamic partition `fixtures_partition_def`). The `DuckDBIOManager` handles partition-aware reads/writes using a `fixture_id` column by default. Override with `metadata={"partition_column": "other_col"}` on the asset.
5. For new data products, add assets in active `assets_*` subfolders and register them in `src/hbl_etl_dagster/defs.py`.
6. Do not reintroduce dependencies on legacy top-level asset modules.
7. Prefer explicit imports in `defs.py`; avoid wildcard imports.

---

## IO Managers

| Key | Class | Use for |
|-----|-------|---------|
| `"io_manager"` (default) | `DuckDBIOManager` | DataFrames — stores as DuckDB tables partitioned by `fixture_id` |
| `"in_memory_io_manager"` | `InMemoryIOManager` | Non-serializable objects within a single run (e.g. interim models) |
| `"file_io_manager"` | `FilesystemIOManager` | sklearn/XGBoost model artifacts on disk |

- Nested dict/list columns are auto-JSON-serialized on write — avoid relying on this for numeric data.
- Override partition column via `metadata={"partition_column": "other_col"}` on the asset definition.

---

## DuckDB Storage Reference

Primary storage is the DuckDB file `data/hbl_raw.duckdb`.
In practice, each persisted Dagster asset becomes one DuckDB table, with asset-key slashes replaced by underscores.

### How data is organized

- **Season-level tables** are global tables with one row per season entity, fixture, team, or Kinexon session.
- **Fixture-level tables** are partition-aware and usually carry a `fixture_id` column.
- The `DuckDBIOManager` writes fixture partitions with a delete-and-replace strategy for the current `fixture_id`.
- Raw tables preserve more source-specific naming; normalized and synced tables use cleaner analytics-oriented column names.

### Table families currently stored in `hbl_raw.duckdb`

#### 1. Raw ingestion tables

These are closest to the source APIs and retain many original fields.

| Table | What it stores | Representative columns |
|------|-----------------|------------------------|
| `fixtures_sportradar_raw` | Season fixture catalog | `fixture_id`, `seasonId`, `startTimeLocal`, `startTimeUTC`, `round`, `venue`, `competitors`, `attendance` |
| `fixture_events_sportradar_raw` | All Sportradar match events for each fixture | `fixture_id`, `eventId`, `eventTime`, `eventType`, `personId`, `subType`, `clock`, `scores`, `attackType`, `goalKeeperId`, `x`, `y`, `success` |
| `players_sportradar_raw` | Player roster rows from Sportradar | `personId`, `fixture_id`, `nameFullLatin`, `nameGivenLatin`, `nameFamilyLatin`, `dob`, `nationality`, `additionalDetails_height`, `additionalDetails_weight` |
| `teams_sportradar_raw` | Team master data from Sportradar | `entity_id`, `name_full_local`, `name_full_latin`, `code_local`, `code_latin`, `external_id` |
| `teams_kinexon_raw` | Kinexon team/group entities | `id`, `name` |
| `sessions_kinexon_raw` | Kinexon tracking session metadata | `session_id`, `id`, `start_session`, `end_session`, `duration`, `type`, `description`, `team_id`, `facility`, `phases` |
| `detected_events_kinexon_raw` | Kinexon shot detection events | `id`, `fixture_id`, `timestamp_ms`, `game_clock`, `period`, `player_id`, `goalkeeper_id`, `distance`, `speed_ball`, `trajectory`, `shot_position_x`, `shot_position_y`, `success` |
| `positions_kinexon_raw` | Raw Kinexon tracking frames with original column names | `fixture_id`, `session_id`, `ts in ms`, `sensor id`, `mapped id`, `league id`, `group name`, `x in m`, `y in m`, `speed in m/s`, `acceleration in m/s2` |

#### 2. Normalized tables

These tables standardize naming, types, and per-fixture semantics for downstream use.

| Table | What it stores | Representative columns |
|------|-----------------|------------------------|
| `matches_normalized` | Fixture ↔ Kinexon session linkage and match metadata | `season_id`, `fixture_id`, `session_id`, `start_time_local`, `round_number`, `team_name_home`, `team_name_away`, `score_home`, `score_away`, `attendance` |
| `match_events_normalized` | Cleaned Sportradar event stream | `fixture_id`, `event_id`, `event_time`, `event_type`, `person_id`, `sub_type`, `period_id`, `clock`, `attack_type`, `goalkeeper_id`, `x`, `y`, `success`, `empty_net` |
| `match_events_normalized_goals` | Goal-only subset of normalized events | `fixture_id`, `event_id`, `person_id`, `goalkeeper_id`, `attack_type`, `clock`, `x`, `y`, `success` |
| `match_events_normalized_setup` | Setup and technical event subset | `fixture_id`, `event_id`, `event_type`, `person_id`, `sub_type`, `name`, `position` |
| `match_players_normalized` | Cleaned player identities and roster attributes | `fixture_id`, `person_id`, `name_full_latin`, `name_given_latin`, `name_family_latin`, `date_of_birth`, `nationality`, `height`, `weight` |
| `match_detected_shots_normalized` | Cleaned Kinexon shot detections | `fixture_id`, `id`, `timestamp_ms`, `game_clock`, `player_id`, `goalkeeper_id`, `distance`, `speed_ball`, `shot_category`, `trajectory`, `validated` |
| `match_positions_normalized` | Normalized player and ball tracking frames | `fixture_id`, `session_id`, `timestamp_ms`, `sensor_id`, `mapped_id`, `league_id`, `group_name`, `x_m`, `y_m`, `speed_m_s`, `direction`, `acceleration` |

#### 3. Synced cross-source tables

These are the important fusion products joining Sportradar and Kinexon.

| Table | What it stores | Representative columns |
|------|-----------------|------------------------|
| `players` | Per-fixture player identity mapping between Sportradar and Kinexon | `fixture_id`, `entity_id`, `person_id`, `league_id`, `mapped_id`, `session_id`, `team_name`, `team_side`, `position`, `name` |
| `shot_events` | Central joined shot/goal table used for downstream feature creation | `fixture_id`, `event_id`, `person_id`, `person_league_id`, `goalkeeper_person_id`, `goalkeeper_league_id`, `detected_shot_id`, `event_time_ms`, `detected_events_shot_time`, `throw_timestamp_ms`, `time_difference_ms`, `match_method`, `goal_position`, `attack_type`, `sub_type`, `speed_ball` |

#### 4. Feature and model-support tables

These tables feed model training and model interpretation.

| Table | What it stores | Representative columns |
|------|-----------------|------------------------|
| `features_xg` | One row per shot attempt with engineered xG features | `fixture_id`, `event_id`, `shooter_distance_to_goal`, `shooter_distance_to_goalkeeper`, `goalkeeper_distance_to_goal`, `ball_distance_to_goal`, `shot_angle_to_goal`, `closest_defender_distance`, `attack_type`, `sub_type`, `target` |
| `xg_feature_importance` | Aggregate feature importance output | `feature`, `importance_mean`, `importance_std` |

### Key shared identifiers and time columns

These appear across many tables and are the main join keys:

- `fixture_id`: Sportradar fixture UUID; dominant partition key for fixture-scoped tables.
- `event_id`: Sportradar event identifier.
- `person_id`: Sportradar player/person identifier.
- `league_id`: Kinexon-side player identifier used in tracking and shot detections.
- `mapped_id`: Kinexon mapped sensor/player id.
- `session_id`: Kinexon session identifier used to connect tracking with fixtures.
- `timestamp_ms`: Kinexon wall-clock timestamp in milliseconds.
- `event_time_ms`: normalized Sportradar event timestamp in milliseconds.
- `throw_timestamp_ms`: refined ball-release timestamp used by xG feature extraction.

### High-volume tables

- `positions_kinexon_raw` and `match_positions_normalized` are by far the largest tables and hold frame-level tracking data.
- In the current local database snapshot, `match_positions_normalized` contains roughly 175 million rows.
- These tables are expensive to recreate and must be preserved.

### How to inspect the live schema

Use DuckDB directly when you need the exact current schema instead of the stable summary above:

```python
import duckdb

con = duckdb.connect("data/hbl_raw.duckdb", read_only=True)
print(con.execute("SHOW TABLES").fetchall())
print(con.execute("PRAGMA table_info('shot_events')").fetchall())
print(con.execute("SELECT COUNT(*) FROM match_positions_normalized").fetchone())
```

For broader documentation of intended table roles and partitioning, also see `docs/data-model.md`.

---

## Key Dagster Jobs

| Job | Purpose |
|-----|---------|
| `season_raw_refresh_job` | Fetch season-level catalog data (competition, fixtures, teams, sessions) |
| `fixture_raw_backfill_job` | Full fixture pipeline: raw → normalized → synced → features (partitioned by `fixture_id`) |
| `xg_training_job` | Global (unpartitioned) — reads all `features_xg` rows, trains XGBoost xG model |

---

## Known Issues / Tech Debt

- `ml_xs_model` is stubbed out (returns `False`, body commented out).
- JSON-serialized nested columns in DuckDB — plan to avoid for sequence data.

---

## ⚠️ NEVER DROP these DuckDB tables

`positions_kinexon_raw` and `match_positions_normalized` contain raw positional tracking data
fetched from the Kinexon API. Re-fetching takes hours and may not be possible for past seasons.
All other tables are derived and can be safely dropped and recreated.

Safe wipe command (drops everything except position tables):

```python
keep = {'positions_kinexon_raw', 'match_positions_normalized'}
tables = [r[0] for r in con.execute('SHOW TABLES').fetchall()]
for t in [t for t in tables if t not in keep]:
    con.execute(f'DROP TABLE "{t}"')
```

---

## Environment Variables

Configured in `.env` at repo root. Required variables:

```bash
# Kinexon
ENDPOINT_KINEXON_SESSION=""
ENDPOINT_KINEXON_MAIN=""
ENDPOINT_KINEXON_API=""
USERNAME_KINEXON_SESSION=""
USERNAME_KINEXON_MAIN=""
PASSWORD_KINEXON_SESSION=""
PASSWORD_KINEXON_MAIN=""
API_KEY_KINEXON=""

# Sportradar
BASE_URL=""
AUTH_URL=""
CLIENT_ID=""
CLIENT_SECRET=""
CLIENT_ORGANIZATION_ID=""
```

Integration tests require these variables to be set; fixtures auto-skip if they are missing.
Do not hardcode secrets — read them from `.env` and environment only.

---

## Build And Environment Commands

```bash
uv sync --no-dev     # install runtime deps only
uv sync --dev        # install all deps including dev
uv sync              # update lock/resolution
uv build             # optional package build artifact
```

---

## Run Commands

```bash
# Dev server (browser UI)
uv run dagster dev -m hbl_etl_dagster.defs

# Season-level data refresh
uv run dagster job execute -m hbl_etl_dagster.defs -j season_raw_refresh_job

# Fixture pipeline for one partition
uv run dagster job execute -m hbl_etl_dagster.defs -j fixture_raw_backfill_job --partition <fixture_id>
```

### Debug Scripts (preferred for local iteration — no UI required)

The `scripts/debug/` entrypoints run jobs in-process for fast iteration:

```bash
uv run python scripts/debug/pipeline.py season                            # run season_raw_refresh_job
uv run python scripts/debug/pipeline.py fixture                           # random fixture (seed=0)
uv run python scripts/debug/pipeline.py fixture --seed 3                  # different random fixture
uv run python scripts/debug/pipeline.py fixture <fixture_id>              # specific fixture
uv run python scripts/debug/pipeline.py fixture <fixture_id> --from shot_events  # from asset onward
uv run python scripts/debug/ml.py train                                   # run xg_training_job
uv run python scripts/debug/ml.py evaluate                                # evaluate trained xG model
uv run python scripts/debug/ml.py analyze models                          # inspect trained model metadata
```

- Prefer `scripts/debug/pipeline.py` and `scripts/debug/ml.py` over `dagster dev` when iterating locally.
- Use `scripts/debug/pipeline.py fixture <fixture_id> --from <asset_name>` to skip expensive upstream steps and re-run only changed assets.
- Dagster DEBUG logs are suppressed; only INFO and above are shown.

---

## Lint, Format, And Type Commands

```bash
uv run ruff check src/                   # lint source
uv run ruff check src/ test/             # lint source and tests
uv run ruff format src/                  # auto-format source
uv run ruff format src/ test/            # auto-format source and tests
uv run mypy src/                         # type-check source
uv run pre-commit run --all-files        # run pre-commit on all files
```

---

## Test Commands

```bash
uv run pytest test/                                              # run all tests
uv run pytest test/test_pipeline_raw.py test/test_pipeline_synced.py  # unit tests only
uv run pytest test/integration/                                  # integration tests only
uv run pytest test/test_pipeline_raw.py                          # one file
uv run pytest test/test_pipeline_raw.py::test_normalize_season_year   # one test by node id
uv run pytest test/test_pipeline_raw.py -k normalize_season_year      # keyword filter
uv run pytest test/ -x                                           # stop on first failure
uv run pytest test/test_pipeline_synced.py -s                    # show prints/logs
```

---

## Test Data And Test Policy

- Prefer fast tests against pure pipeline functions in `src/pipelines/`.
- Use `assets/data_samples/*.csv` for lightweight data-driven checks.
- Integration tests depend on real API credentials loaded from `.env`.
- Integration fixtures skip automatically if required env vars are missing.
- `test/conftest.py` inserts the project root into `sys.path`; keep imports consistent.
- Avoid coupling new tests to Dagster asset wrappers unless explicitly requested.

---

## Code Style

- Follow existing Python style before introducing new patterns.
- Use `ruff format` output as the formatting source of truth.
- Keep functions focused and composable; prefer small data-transform helpers over monoliths.
- Write docstrings for non-obvious public helpers and pipeline entrypoints.
- Use comments sparingly — only for important context or non-obvious workarounds.
- Prefer ASCII when editing unless the file already uses Unicode meaningfully.

---

## Imports

- Group imports as: standard library, third-party, then local.
- Use explicit imports; do not use wildcard imports.
- Prefer absolute imports rooted at `src.` in repository code.
- In type-heavy modules, `from __future__ import annotations` is common and should be preserved.
- Keep import lists stable and minimal; remove unused imports.

---

## Formatting Conventions

- 4-space indentation.
- One logical step per block.
- Let long calls wrap vertically in the formatter's default style.
- Keep DataFrame pipelines readable; split chained operations when needed.
- Avoid manual alignment or style that fights auto-formatting.

---

## Types

- Add type hints for new or modified function signatures.
- Return `pd.DataFrame` explicitly for DataFrame-producing helpers.
- Use `Optional[...]`, `dict[...]`, `list[...]`, and tuple annotations where they clarify contracts.
- Existing mypy settings are permissive in some areas; still prefer precise typing in changed code.

---

## Naming Conventions

- `snake_case` for variables, functions, and modules.
- `UPPER_SNAKE_CASE` for module-level constants.
- Test names start with `test_` and describe behavior, not implementation details.
- DataFrames use `df_*` naming; keep that convention for consistency.
- Dagster asset names should reflect the dataset they materialize.

---

## Data And Pandas Conventions

- Avoid mutating upstream input DataFrames unless that is the explicit contract.
- Prefer copying before destructive reshaping when inputs may be reused.
- Normalize column names only when needed for downstream consistency.
- Preserve key identifiers: `fixture_id`, `event_id`, `person_id`, and session ids.
- Be careful with nested JSON-like columns; the DuckDB IO manager serializes dict/list columns automatically.
- Keep partition-awareness in mind when reading or writing fixture-scoped tables.

---

## Error Handling And Logging

- Fail fast on invalid required inputs using `ValueError`, `KeyError`, or `RuntimeError`.
- Preserve the current logging style: `logging` module with `logger = logging.getLogger(__name__)`.
- Log useful counts and identifiers for data pipeline steps.
- When catching broad exceptions around API calls, log with `logger.exception(...)` and re-raise.
- Return empty DataFrames intentionally only when that matches existing pipeline semantics.
- Do not silently swallow schema problems.

---

## Dagster-Specific Guidance

- Keep assets focused on context, metadata, and delegating to pipeline functions.
- Register new assets, checks, jobs, or resources in `src/hbl_etl_dagster/defs.py`.
- Use output metadata for row counts, previews, and sanity checks when useful.
- Use the configured IO managers instead of ad hoc persistence.
- Do not add business logic directly into Dagster asset files.

---

## ML And Feature Work

- Active xG feature/model code lives in `src/pipelines/features/` and `src/pipelines/ml/`.
- Prefer fixture-level leakage-safe thinking when changing train/validation logic.
- Keep feature generation deterministic and traceable from input tables.
- Do not couple model code tightly to Dagster asset wrappers.

---

## Planned: Spatiotemporal xG Upgrade

See [.github/prompts/plan-spatiotemporalXgUpgrade.prompt.md](.github/prompts/plan-spatiotemporalXgUpgrade.prompt.md) and [docs/transformers.md](docs/transformers.md) for the full plan. Key decisions:

- **Parallel track**: keep snapshot xG (`features_xg` / `ml_xg_model`) unchanged; add
  `features_xg_sequence` + new PyTorch model alongside.
- New module: `src/pipelines/features/calc_xg_sequence_windows.py` — fixed-length pre-shot
  temporal windows (1.5–2s before release).
- Sequence stored as normalized relational tables in DuckDB (not nested blobs).
- Model training moved to a **global job** (not per-fixture partitioned).
- PyTorch transformer encoder with compact architecture + mandatory ablations.
- Fixture-level train/val split to prevent leakage.

---

## Documentation

Docs live in `docs/`. See the index below:

| File | Content |
|------|---------|
| [docs/architecture.md](docs/architecture.md) | System overview, layer separation, jobs, sync pipeline, asset checks |
| [docs/data-model.md](docs/data-model.md) | DuckDB tables, schemas, key identifiers, partition strategy |
| [docs/xg-feature-model-guide.md](docs/xg-feature-model-guide.md) | xG features, model design, training protocol, and known limitations |
| [docs/transformers.md](docs/transformers.md) | Spatiotemporal transformer xG — plan stub |
| [docs/shot-detection-analysis.md](docs/shot-detection-analysis.md) | Shot-goal sync pipeline review; pitfalls and fixes |
| [docs/pipeline-development-guide.md](docs/pipeline-development-guide.md) | Adding pipeline functions; data conventions |
| [docs/dagster-development-guide.md](docs/dagster-development-guide.md) | Asset patterns; IO manager selection; job structure |
| [docs/testing-guide.md](docs/testing-guide.md) | Test structure; sample data; integration tests |

---

## Practical Agent Workflow

- Read nearby pipeline and test files before changing conventions.
- Prefer the smallest change that fits the existing architecture.
- If you add logic in `src/pipelines/`, update or add focused tests in `test/`.
- If you touch Dagster assets, verify `defs.py` wiring and partition behavior.
- Run targeted tests first, then broader lint/type/test commands as needed.
- For new data products: add pipeline function → add thin asset → register in `defs.py` → add test.
