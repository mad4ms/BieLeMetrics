# BieLeMetrics

BieLeMetrics is a handball analytics pipeline for synchronizing Sportradar event data with Kinexon positional tracking, materializing normalized tables in DuckDB, engineering expected-goal features, and training xG models.

This repository is the codebase behind the paper [Expected Goals Prediction in Professional Handball using Synchronized Event and Positional Data](https://dl.acm.org/doi/10.1145/3606038.3616152).

![Demo GIF](./assets/events/videos/demo.gif)

## What This Repository Does

The pipeline turns raw vendor data into analytics-ready datasets and models:

- fetches season and fixture data from Sportradar and Kinexon
- normalizes raw source payloads into stable match-level tables
- synchronizes players, goals, detected shots, and throw timestamps across sources
- builds xG features from shot context and tracking frames
- trains and evaluates an expected-goal model with fixture-aware validation

Primary outputs live in DuckDB tables such as `matches_normalized`, `players`, `shot_events`, and `features_xg`, plus model artifacts under `data/models/`.

## Stack

- Python `>=3.12`
- package manager / runner: [uv](https://docs.astral.sh/uv/)
- orchestration: Dagster
- storage: DuckDB
- core libraries: pandas, pyarrow, scikit-learn, xgboost, shapely

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/mad4ms/BieLeMetrics.git
cd BieLeMetrics
uv sync --dev
```

### 2. Configure environment variables

Create a `.env` file in the repository root.

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

Integration tests and live ingestion require these credentials. Do not hardcode secrets.

### 3. Verify the environment

```bash
uv run pytest test/test_pipeline_raw.py test/test_pipeline_synced.py
```

## Common Workflows

### Start the Dagster UI

```bash
uv run dagster dev -m hbl_etl_dagster.defs
```

Active Dagster entrypoint:

```bash
src/hbl_etl_dagster/defs.py
```

### Refresh season-level data

This fetches season metadata, fixtures, teams, and Kinexon sessions.

```bash
uv run dagster job execute -m hbl_etl_dagster.defs -j season_raw_refresh_job
```

### Run one fixture pipeline

This executes raw ingestion through normalized tables, sync products, features, and ML-support outputs for a single partition.

```bash
uv run dagster job execute -m hbl_etl_dagster.defs -j fixture_raw_backfill_job --partition <fixture_id>
```

### Use the debug scripts for local iteration

These are usually faster than going through the full Dagster UI.

```bash
uv run python scripts/debug/pipeline.py season
uv run python scripts/debug/pipeline.py fixture
uv run python scripts/debug/pipeline.py fixture <fixture_id>
uv run python scripts/debug/pipeline.py fixture <fixture_id> --from shot_events
uv run python scripts/debug/pipeline.py backfill --skip-season --skip-train --from shot_events --parallel 1
```

### Train and evaluate the xG model

```bash
uv run python scripts/debug/ml.py train
uv run python scripts/debug/ml.py evaluate
uv run python scripts/debug/ml.py analyze models
uv run python scripts/debug/ml.py analyze features
```

## Repository Layout

```text
assets/                  Sample CSVs, demo media, and supporting assets
data/                    DuckDB database and model artifacts
docs/                    Architecture, data model, ML notes, and investigations
notebooks/               Exploratory notebooks and QA views
scripts/                 Debug, diagnostics, and one-off operational scripts
src/
   fetcher_kinexon/       Kinexon API fetch helpers
   fetcher_sportradar/    Sportradar API fetch helpers
   pipelines/             Business logic
      raw/                 Raw ingestion helpers
      normalized/          Source normalization
      synced/              Cross-source synchronization
      features/            Feature engineering
      ml/                  Training, inference, importance analysis
   hbl_etl_dagster/       Thin Dagster asset wrappers, io managers, defs
test/                    Unit and integration tests
```

## Data Model

All materialized pipeline outputs are stored in:

```bash
data/hbl_raw.duckdb
```

Important table families:

- raw ingestion tables: vendor-near payloads and tracking frames
- normalized tables: cleaned match-level entities such as `match_events_normalized` and `match_positions_normalized`
- synced tables: fusion outputs such as `players` and `shot_events`
- feature tables: model inputs such as `features_xg`

See [docs/data-model.md](docs/data-model.md) for the detailed schema reference.

### Critical data safety note

Do not drop these tables unless you explicitly intend to destroy expensive tracking data:

- `positions_kinexon_raw`
- `match_positions_normalized`

They are large and expensive to rebuild.

## Development Commands

### Tests

```bash
uv run pytest test/
uv run pytest test/test_pipeline_raw.py test/test_pipeline_synced.py
uv run pytest test/integration/
```

### Lint and types

```bash
uv run ruff check src/ test/
uv run ruff format src/ test/
uv run mypy src/
uv run pre-commit run --all-files
```

## Project Conventions

- Business logic belongs in `src/pipelines/`
- Dagster asset files should stay thin and delegate into pipeline functions
- Tests should target pipeline logic first, not Dagster wrappers
- Fixture-scoped data is partitioned by `fixture_id`
- Model evaluation should stay fixture-aware to avoid leakage across train and validation

## Documentation

Use the docs folder for deeper reference material:

- [docs/architecture.md](docs/architecture.md) — end-to-end architecture and job layout
- [docs/data-model.md](docs/data-model.md) — tables, identifiers, and partitioning
- [docs/pipeline-development-guide.md](docs/pipeline-development-guide.md) — writing pipeline logic in `src/pipelines/`
- [docs/dagster-development-guide.md](docs/dagster-development-guide.md) — asset wiring, IO managers, and jobs
- [docs/testing-guide.md](docs/testing-guide.md) — unit and integration test conventions
- [docs/xg-feature-model-guide.md](docs/xg-feature-model-guide.md) — xG features, training, evaluation, and limitations
- [docs/shot-detection-analysis.md](docs/shot-detection-analysis.md) — shot sync heuristics and pitfalls
- [docs/transformers.md](docs/transformers.md) — planned spatiotemporal model direction

## Current xG Workflow

The active xG setup is a single global model trained on `features_xg` with fixture-aware validation. If feature generation or upstream shot synchronization changes, rebuild affected fixture partitions and retrain before trusting evaluation metrics.

## Contributing

1. Create a branch from `main`
2. Make the smallest architecture-consistent change that solves the problem
3. Add or update focused tests
4. Run targeted validation before broader checks
5. Open a pull request with a clear problem statement, change summary, and validation notes

## License

MIT. See [LICENSE](LICENSE).
