<!-- Copilot / AI assistant instructions for contributors and agents -->
# BieLeMetrics Dagster ETL — Developer Guide

This document provides detailed instructions for working with the BieLeMetrics codebase. It is designed to help AI agents and developers understand the project structure, data flow, and development workflows.

## 1. Architecture Overview

The project is a Dagster-based ETL pipeline that ingests handball match data from two primary sources: **Sportradar** (play-by-play events, fixtures) and **Kinexon** (positional data, shot_events, sessions).

The codebase enforces a strict separation of concerns:
1.  **Business Logic (`src/events_*`)**: Pure Python functions that handle API interaction, data fetching, and initial parsing. These functions are agnostic of the orchestration layer.
2.  **Orchestration (`src/hbl_etl_dagster`)**: The Dagster layer that defines Assets, Jobs, Sensors, and Resources. It wraps the business logic and manages dependencies, partitioning, and persistence.

**Critical Rule**: Never put API fetching logic directly inside a Dagster asset. Write a function in `src/events_*` and call it from the asset.

---

## 2. Directory Structure & Responsibilities

### `src/events_sportradar/`
Contains logic for interacting with the Sportradar API.
-   `fetch_list_fixtures.py`: Fetches the schedule for a season.
-   `fetch_list_fixture_events.py`: Fetches play-by-play events for a specific match.
-   `fetch_teams.py`: Fetches team metadata.
-   `fetch_competition_id.py` / `fetch_saison_id.py`: Helpers to resolve IDs.

### `src/events_kinexon/`
Contains logic for interacting with the Kinexon API.
-   `fetch_positions_for_fixture.py`: Downloads and parses large positional data files (gzip/zip CSVs).
-   `fetch_events_for_session.py`: Fetches detected events (e.g., shots, passes) from Kinexon's analysis.
-   `fetch_session_id_for_fixtures.py`: Maps Sportradar fixtures to Kinexon session IDs using fuzzy matching on team names.

### `src/hbl_etl_dagster/`
The Dagster application.
-   **Assets (`assets_*.py`)**:
    -   `assets_ids.py`: In-memory assets for `competition_id` and `season_id`.
    -   `assets_sportradar.py`: `teams`, `fixtures` (season-level), and `fixture_events` (partitioned).
    -   `assets_kinexon.py`: `kinexon_positions`, `kinexon_events`.
    -   `assets_sync.py`: `players_merged`, `sportradar_goals_synced`, `sportradar_goals_refined`.
-   **Definitions (`defs.py`)**: Registers all assets, jobs, sensors, and resources.
-   **Sensors (`sensors.py`)**: `fixture_sensor` polls DuckDB for new fixtures and triggers runs for them.
-   **IO Managers (`io_managers.py`)**: `DuckDBIOManager` handles reading/writing DataFrames to `data/hbl.duckdb`. It uses `FileLock` to prevent concurrent write errors.
-   **Resources (`resources.py`)**: Provides `sportradar_api` and `kinexon_api` clients to assets.

---

## 3. Data Flow & Persistence

### The Pipeline
1.  **Season Setup**: `season_refresh_job` runs `teams` and `fixtures` assets.
    -   `fixtures` fetches the schedule and maps Kinexon session IDs.
    -   It updates the `fixture_partitions` dynamic partition definition.
2.  **Fixture Processing**: `fixture_sensor` detects new fixture IDs in the `fixtures` table and launches `fixture_backfill_job` for each new fixture.
3.  **Per-Fixture Assets**:
    -   `fixture_events`: Fetches Sportradar events.
    -   `kinexon_positions`: Fetches Kinexon tracking data (heavy download).
    -   `kinexon_events`: Fetches Kinexon analysis events.
    -   `players_merged`: Syncs player metadata between sources.
    -   `sportradar_goals_synced` / `refined`: Advanced analytics combining both sources.

### DuckDB Persistence
-   **Location**: `data/hbl.duckdb`
-   **Schema**: Table names correspond to asset names (e.g., `fixtures`, `match_events`, `players`).
-   **Incremental Loading**:
    -   `fixture_events` deletes rows for the current `fixture_id` before inserting new ones to ensure idempotency.
    -   `players` table is accumulated globally.
-   **Concurrency**: The `DuckDBIOManager` uses a file lock (`.lock`) to serialize writes. **Do not bypass the IO Manager** for writes if possible.

---

## 4. Development Guidelines

### Adding a New Feature
1.  **Identify the Source**: Is it Sportradar or Kinexon data?
2.  **Implement Logic**: Create a function in `src/events_<source>/` that accepts an API client and returns a DataFrame.
3.  **Create Asset**: Add an `@asset` in `src/hbl_etl_dagster/assets_<source>.py`.
    -   Use `fixtures_partition_def` if it's per-match data.
    -   Add `compute_kind="duckdb"` and `group_name`.
4.  **Register**: Add the asset to `defs.py`.

### Handling Concurrency
-   **Dagster Limits**: The `fixture_sensor` limits concurrent runs via `FIXTURE_RUN_CONCURRENCY_LIMIT` (default: 1) to prevent database locking issues.
-   **No Threads**: Do not use `ThreadPoolExecutor` inside your asset logic. Let Dagster handle parallelism via multiple run workers if needed (though currently throttled).

### Debugging & Testing
-   **Local Run**: Use `dagster dev` to run the UI.
-   **Backfills**: Use the UI or `scripts/backfill_fixture_partitions.py` to re-run specific fixtures.
-   **Logs**: Check Dagster logs for "Error updating fixture partitions" or API failures.

### Common Pitfalls
-   **Missing Session IDs**: If `fixtures` asset fails to map a session ID, downstream Kinexon assets will skip that fixture. Check `fetch_session_id_for_fixtures.py`.
-   **DuckDB Locks**: If you see "IO Error: Cannot open file ...", it means multiple processes are trying to write to DuckDB simultaneously. Ensure the `FileLock` in `io_managers.py` is working and that `fixture_sensor` is throttling runs.

---
**Ignore**: `src/backup/`
**References**: `pyproject.toml`, `data/hbl.duckdb`
