# Shot Detection Analysis

## Scope

This note reviews the active goal-to-shot sync pipeline in `src/pipelines/synced/shot_events.py` and the player mapping path in `src/pipelines/synced/players.py`.

## Current Pipeline

1. Sportradar goal events are normalized and filtered in `src/pipelines/normalized/match_events.py`.
2. Kinexon detected shots are normalized in `src/pipelines/normalized/match_detected_shots.py`.
3. Players are mapped from Sportradar to Kinexon IDs via fuzzy name matching in `src/pipelines/synced/players.py` — team-constrained since fix.
4. Goals are matched to detected shots in `src/pipelines/synced/shot_events.py:_sync_goals_to_detected_shots`.
5. Throw timestamps are refined from positional data in `src/pipelines/synced/shot_events.py:_refine_throw_times`.
6. Goal side is inferred from goalkeeper positions per period in `src/pipelines/synced/shot_events.py:insert_goal_position`.

## Matching Heuristics

- Goal-to-shot matching uses a fixed asymmetric window of `-30_000 ms` to `+3_000 ms` around the Sportradar goal time.
- Preferred match: same `person_league_id` and closest time, `match_method = "player_time"`.
- Fallback match: closest time only, `match_method = "time_only_fallback"`.
- Matching is one-to-one on both `goal_event_id` and `detected_shot_id` (both deduplicated, smallest time_diff wins).
- Throw-point detection uses possession defined as `dist_pb <= 1.5` and chooses the max `ball_acc` within ±50 ms of the **last** possession-end in the window.

---

## Pitfalls

### 1. The time window is wide enough to hide bad alignment *(monitoring only)*

- A 30-second lookback can attach an earlier possession or even an earlier shot in dense phases.
- This is especially risky when the code falls back to `time_only_fallback`.
- **Mitigation:** `sync_shot_events` logs a warning when any goal's `time_difference_ms` exceeds 5 000 ms.
- No hard limit is enforced; operator review is still required.

### 2. ~~Matching is not one-to-one on detected shots~~ **FIXED**

- Previously, multiple goals could reference the same `detected_shot_id`.
- **Fix:** After sorting by `time_difference_ms`, `_sync_goals_to_detected_shots` now calls
  `drop_duplicates(subset=["detected_shot_id"], keep="first")` before the final left join.
  The goal with the smallest time delta wins the detected shot; other goals fall through as unmatched.

### 3. ~~Player identity mapping is brittle~~ **IMPROVED**

- Fuzzy matching a name against all Kinexon `full_name` values across teams can produce wrong `league_id`.
- **Fix:** `fuzzy_match_player` in `players.py` now first resolves the player's Sportradar `team_name`
  to the closest Kinexon `group_name` (cutoff 0.4), then restricts name matching to that group.
  This prevents cross-team name collisions.

### 4. ~~`frame_tol_ms` is passed but not used~~ **NOT A PITFALL**

- Kinexon's positional tracking system uses a **single shared hardware clock** for all sensors
  (ball + every player). Ball and player samples captured in the same frame share the **exact same**
  `timestamp_ms`.
- The exact-equality join in `_build_player_ball_timeline` is therefore **correct by design**.
  `frame_tol_ms` is retained as a parameter for documentation purposes only.

### 5. ~~Throw detection can confuse passes with shots~~ **IMPROVED**

- Previously, `detect_throw_point` searched all possession-end windows in the ±30 s scene window
  and picked the global max `ball_acc`. An earlier pass with a high acceleration peak could
  overshadow the actual throw.
- **Fix:** The search now uses only the **last** possession-end window (±50 ms around the final
  possession loss before the detected shot). This is the possession end closest to the throw moment
  and avoids earlier pass events. The fallback to `last_possession_fallback_*` is unchanged.
- Without a direction-to-goal filter (removed due to noise), some pass/loose-touch events can still
  be selected in multi-possession windows within a single 30 s scene. This is an accepted limitation.

### 6. ~~Goal-side inference depends on a fragile halftime split~~ **FIXED**

- Previously, half 1 vs half 2 was split using the last first-half goal timestamp. If the defending
  team had **no period-1 goals**, the entire team was skipped and `goal_position` remained `None`
  for all their conceded goals including period 2.
- **Fix:** `insert_goal_position` now iterates over each (team, period) pair independently.
  For each period it queries goalkeeper positions within ±2 minutes of that period's goal events.
  No global split point is required; each period is resolved separately.

### 7. Normalization drops useful signal *(unchanged)*

- `normalize_match_detected_shots(...)` removes ball rows and nulls `validated == 0`.
- That loses the distinction between "explicitly invalid" and "unknown".

---

## Operational Gaps

- The sync pipeline logs coarse counts, but not why individual rows fail refinement.
- The `time_only_fallback` rate is not surfaced as a Dagster asset metadata field (operator must inspect logs).
- `debug_shot.py` provides an interactive per-fixture diagnostic report.

## Highest-Risk Failure Modes

1. Whole fixtures appear "matched" even when the underlying clocks are off by tens of seconds — detectable via the `time_difference_ms > 5 000 ms` warning.
2. Missing or incorrect player mapping drives the system into `time_only_fallback` — mitigated by team-constrained fuzzy matching.
3. Throw timestamps are missing because ball and player timestamps rarely align exactly — **not an issue**; Kinexon uses a shared clock so alignment is guaranteed.
4. Rebounds or clustered shot sequences map to the wrong detected shot — mitigated by one-to-one `detected_shot_id` deduplication.

## Recommended Checks (status)

| Check | Status |
|---|---|
| Track `match_method = "time_only_fallback"` rate | Logged; not surfaced as Dagster metadata |
| Flag large median / p95 `time_difference_ms` | Warning at >5 000 ms |
| Enforce one-to-one `detected_shot_id` matching | **Done** |
| Nearest-within-50ms join for throw refinement | N/A — same clock, exact join is correct |
| Validation report for missing `person_league_id`, `goal_position`, `throw_timestamp_ms` | Available via `debug_shot.py` |

---

## Related

- [timing-async-analysis.md](timing-async-analysis.md) — root cause of Kinexon/Sportradar clock offset
- [shot-sync-drift-findings.md](shot-sync-drift-findings.md) — per-fixture drift investigation and fix proposal
