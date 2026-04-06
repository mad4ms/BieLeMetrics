# Data Model

All data is stored in a single DuckDB database at `data/hbl_raw.duckdb`.
Each Dagster asset materializes as one table. Table names match asset names
(slashes in asset keys are replaced with underscores).

---

## Table Overview

### Season-level tables (unpartitioned, no `fixture_id` column)

| Table | Source | Description |
|-------|--------|-------------|
| `competition_id` | Sportradar | Single-row: resolved competition UUID |
| `season_id` | Sportradar | Single-row: resolved season UUID for the target year |
| `teams_sportradar_raw` | Sportradar | All teams in the season |
| `fixtures_sportradar_raw` | Sportradar | All fixtures (matches) with metadata; populates partition list |
| `teams_kinexon_raw` | Kinexon | Kinexon group/team entities |
| `sessions_kinexon_raw` | Kinexon | Kinexon tracking sessions with timestamps and team linkage |
| `matches_normalized` | Both | Season-level fixture ↔ Kinexon session linkage; one row per fixture |

### Fixture-level tables (partitioned by `fixture_id`)

> Partition column: `fixture_id` (unless noted otherwise).

#### Raw ingestion

| Table | Source | Description |
|-------|--------|-------------|
| `fixture_events_sportradar_raw` | Sportradar | All game events (goals, saves, fouls, …) for a fixture |
| `players_sportradar_raw` | Sportradar | Player roster per fixture |
| `detected_events_kinexon_raw` | Kinexon | Shot detection events from Kinexon |
| `positions_kinexon_raw` | Kinexon | **⚠️ NEVER DROP** — raw positional tracking (ball + all players) at ~25 Hz |

#### Normalized (cleaned, typed, renamed)

| Table | Description |
|-------|-------------|
| `match_events_normalized` | All SR events, typed and renamed; includes `event_time_ms` |
| `match_events_normalized_setup` | Subset: setup/technical events (throw-ins, foul calls, …) |
| `match_events_normalized_goals` | Subset: goal events only, with shooter/goalkeeper IDs |
| `match_positions_normalized` | **⚠️ NEVER DROP** — normalized positions with `timestamp_ms`, `x`, `y`, agent roles |
| `match_detected_shots_normalized` | Normalized Kinexon shot detections |
| `match_players_normalized` | Normalized player roster per fixture |

#### Synced (cross-source joined)

| Table | Description |
|-------|-------------|
| `players` | Sportradar ↔ Kinexon player ID mapping per fixture; `person_league_id` ↔ `league_id` |
| `shot_events` | Goals from SR matched to KX detected shots; includes `throw_timestamp_ms`, `goal_position`, `match_method` |

#### Features

| Table | Description |
|-------|-------------|
| `features_xg` | One row per shot attempt; numeric + categorical features + `target` label |

---

## Key Identifiers

| Column | Type | Scope | Description |
|--------|------|-------|-------------|
| `fixture_id` | string UUID | All fixture tables | Sportradar fixture UUID; partition key |
| `event_id` | string | Fixture | Sportradar event UUID within a fixture |
| `detected_shot_id` | string | Fixture | Kinexon shot detection ID |
| `person_id` | string | Fixture | Sportradar person UUID |
| `person_league_id` | string | Fixture | Sportradar player's league-registered ID |
| `league_id` | string | Fixture | Kinexon's internal player identifier |
| `session_id` | string | Season | Kinexon tracking session ID |
| `timestamp_ms` | int64 | Fixture | Kinexon wall-clock timestamp (milliseconds UTC) |
| `event_time_ms` | int64 | Fixture | Sportradar event timestamp (milliseconds UTC) |

---

## Critical Tables — Do Not Drop

```
positions_kinexon_raw       raw Kinexon tracking frames; re-fetching takes hours
match_positions_normalized  derived from above; re-fetching requires raw data
```

All other tables are fully derived from API responses and can be dropped and recreated by
re-running the pipeline. See [AGENTS.md](../AGENTS.md) for the safe wipe command.

---

## Partition Strategy

The IO manager uses a **DELETE + INSERT** strategy per partition:

```sql
DELETE FROM <table> WHERE fixture_id = '<key>';
INSERT INTO <table> BY NAME SELECT * FROM tmp_df;
```

This means re-running a fixture's pipeline safely replaces only that fixture's rows.
The global (non-partitioned) tables use `CREATE OR REPLACE TABLE`.

---

## Schema Evolution

The `DuckDBIOManager` handles forward schema evolution automatically:
- **New columns**: `ALTER TABLE … ADD COLUMN` on first insert that introduces the column.
- **Type widening**: `ALTER TABLE … ALTER COLUMN … TYPE` when the incoming type is wider
  (e.g. `INTEGER` → `DOUBLE`).
- **Column removal**: not handled — removed columns remain in the table as nulls.

---

## Nested Columns

The IO manager auto-JSON-serializes `dict` and `list` columns on write.
Avoid storing nested data in tables intended for numeric ML consumption — use
normalized relational tables instead (one row per atomic observation).

---

## `shot_events` — Central Join Table

`shot_events` is the most important derived table. It links:

- **Sportradar** goal identity (`event_id`, `person_league_id`, `attack_type`, `sub_type`, `success`)
- **Kinexon** shot timing (`detected_shot_id`, `timestamp_ms`, `throw_timestamp_ms`)
- **Sync metadata** (`match_method`, `time_difference_ms`)
- **Enrichment** (`goal_position`, `goalkeeper_league_id`)

The `match_method` column indicates sync quality:
- `"player_time"` — matched on player identity + time (high confidence)
- `"time_priority_override"` — a player-matched shot existed, but a different detected shot was materially closer in time to the predicted Kinexon timestamp
- `"time_only_fallback"` — matched on time alone (lower confidence; monitor this rate)

The `throw_timestamp_ms` column is the refined release timestamp derived from
ball acceleration in the positional data. It is the anchor for xG feature extraction.

---

## `features_xg` — Feature Table

One row per shot attempt. Populated per fixture by `fixture_raw_backfill_job`.
Consumed globally by `xg_training_job`.

### Feature columns

| Column | Type | Description |
|--------|------|-------------|
| `fixture_id` | string | Partition key; used for leakage-safe train/val split |
| `event_id` | string | Links back to `shot_events` |
| `avg_offense_distance_to_goal` | float | Mean distance of attacking players to goal at throw time |
| `avg_defense_distance_to_goal` | float | Mean distance of defending players to goal at throw time |
| `shooter_distance_to_goal` | float | Shooter's distance to goal at throw time |
| `shooter_distance_to_goalkeeper` | float | Distance between shooter and goalkeeper |
| `goalkeeper_distance_to_goal` | float | Goalkeeper's displacement from goal line |
| `ball_distance_to_goal` | float | Ball distance to goal at throw time |
| `ball_distance_to_goalkeeper` | float | Ball distance to goalkeeper |
| `shot_angle_to_goal` | float | Solid angle subtended by goal from shooter position (radians) |
| `ball_angle_to_goal` | float | Angle from ball to goal |
| `angle_ball_to_goalkeeper` | float | Angle between ball–goal vector and ball–GK vector |
| `num_defenders_close` | int | Number of defenders within a cone in front of the shooter |
| `closest_defender_distance` | float | Distance to nearest defender |
| `attack_type` | string | Sportradar attack classification (SET_PLAY, FAST_BREAK, etc.) |
| `sub_type` | string | Sportradar sub-classification |
| `target` | int | Label: 1 = goal scored, 0 = shot saved/missed |

---

## Related

- [architecture.md](architecture.md) — full data flow and job overview
- [ml-model.md](ml-model.md) — how `features_xg` is consumed for training
- [shot-detection-analysis.md](shot-detection-analysis.md) — how `shot_events` is produced
