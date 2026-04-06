# Kinexon vs Sportradar Timing Analysis

## Question

Why do some games appear about 30 seconds asynchronous between Kinexon and Sportradar?

## Short Answer

There is no explicit hard-coded 30-second correction in the active pipeline. The strongest code-level explanation is that the system uses absolute timestamps as the primary truth and then allows a very large `30_000 ms` backward matching window, which can mask or normalize a fixture-level offset instead of detecting it.

## Most Likely Causes

### 1. The sync uses absolute wall-clock timestamps, not match clock reconciliation

- Sportradar goals are matched using `event_time -> event_time_ms` in `src/pipelines/synced/shot_events.py:611`.
- Kinexon detected shots are matched using `timestamp_ms` only in `src/pipelines/synced/shot_events.py:184`.
- Even though Sportradar has `clock` and `period_id`, and Kinexon has `game_clock` and `period`, those fields are not used in the sync logic.
- Relevant code: `src/pipelines/normalized/match_events.py:46`, `src/pipelines/normalized/match_events.py:50`, `src/pipelines/synced/shot_events.py:160`.

Expected symptom:
- A whole fixture can have a near-constant offset while the in-game sequence still looks plausible.

### 2. The `-30s / +3s` shot-match window hides bad alignment

- `tol_before_ms = 30_000` and `tol_after_ms = 3_000` are hard-coded in `src/pipelines/synced/shot_events.py:848`.
- If a fixture is offset by roughly 20-30 seconds, the code can still produce matches instead of surfacing a sync failure.
- Fallback to nearest-by-time makes this worse when player mapping is missing.

Expected symptom:
- Matched rows cluster around large positive `time_difference_ms` values.
- Affected fixtures show many `time_only_fallback` matches.

### 3. Session-to-fixture linking does not strongly validate kickoff time

- `normalize_matches(...)` merges Kinexon sessions to fixtures mainly on team-name pairs, not on nearest start timestamp, in `src/pipelines/normalized/matches.py:239`.
- The alternate Kinexon session helper searches a full-day window and accepts matching sessions without a nearest-start selection step in `src/fetcher_kinexon/fetch_session_id_for_fixtures.py:131`.
- Timing fields that would help validate the session are later dropped in `src/pipelines/normalized/matches.py:95`.

Expected symptom:
- Entire fixtures are uniformly shifted, not just a few isolated goals.
- The linked session is plausible by team names but not by actual kickoff.

### 4. Event publication time may differ from on-court event time

- Sportradar `event_time` is treated as the goal timestamp with no correction layer.
- If `event_time` is a feed/publication timestamp or includes a systematic delay in some matches, the pipeline has no reconciliation step using game clock.

Expected symptom:
- Match clock progression looks right, but absolute event timestamps are consistently late or early.

## Less Likely But Relevant

### 5. Throw refinement uses exact timestamp alignment

- `_build_player_ball_timeline(...)` claims nearest-within-tolerance behavior but effectively requires exact timestamp matches.
- This is more likely to reduce throw detection coverage than to create a clean 30-second fixture offset.
- Relevant code: `src/pipelines/synced/shot_events.py:281`.

### 6. Halftime splitting is heuristic

- Goal direction is split by the timestamp of the last first-half goal, not a true period boundary.
- This can degrade second-half refinement but is unlikely to be the main cause of a fixture-wide 30-second shift.
- Relevant code: `src/pipelines/synced/shot_events.py:786`.

## What To Check First

### Fixture-level diagnostics

- Distribution of `time_difference_ms` per fixture.
- Count and share of `match_method = "time_only_fallback"` per fixture.
- Whether the same `detected_shot_id` is reused within a fixture.

### Clock reconciliation

- Compare Sportradar `period_id + clock` to Kinexon `period + game_clock` for matched events.
- If match clocks align while wall-clock timestamps differ by about 30 seconds, the issue is timestamp semantics, not event ordering.

### Session validation

- For affected fixtures, inspect candidate Kinexon sessions for the same teams on that date.
- Compare session start/end against `start_time_utc` and the earliest/latest position timestamps.

## Likely Root-Cause Ranking

1. No clock-based reconciliation, only absolute timestamp matching.
2. `-30s / +3s` tolerance window masking real offsets.
3. Wrong or weakly validated Kinexon session linkage.
4. Sportradar `event_time` semantics differing from actual on-court timing.

## Practical Interpretation

Right now the pipeline is optimized to recover a match even when timing is messy. That makes it robust for throughput, but it also means a fixture can be materially out of sync without clearly failing. The 30-second pattern is most likely not caused by one isolated bug; it is the interaction of:

- wide fallback tolerances,
- wall-clock-first matching,
- weak session timing validation, and
- lack of clock-based cross-source sanity checks.

---

## 2026-03-21 Update — Detailed Investigation Results

Full findings documented in [shot-sync-drift-findings.md](shot-sync-drift-findings.md).

### Confirmed Root Cause

The **asymmetric Pass 1 window** (`[g_ms − 30 s, g_ms + 3 s]`) is the primary
driver of sync failures. The window was designed for the normal case (KX shot
before SR event), but for several fixtures the Kinexon detected-shot timestamps
arrive **after** the Sportradar event, sometimes by 20–33 s.

### Per-Fixture Classification (28 fixtures)

| Class | Fixtures | Description |
|---|---|---|
| Normal, small offset (≤10 s) | ~18 | KX before SR; Pass 1 works well |
| Normal, large offset (15–33 s) | 3 (`00ec13a6`, `02f4e1a9`, `09e1e335`) | KX well before SR; Pass 1 captures fringe |
| Reversed polarity | 1 (`00ba9627`) | KX ~21 s AFTER SR; Pass 1 captures nothing |
| Mixed / bimodal | 6 | Two offset clusters per fixture (likely halftime session split) |
| No player data | 1 (`0172aa0a`) | league_id mapping broken — independent issue |

### Fix Identified

**Widen Pass 1 window to ±35 s** (symmetric). Pass 2 stays tight.
This recovers 3–6× more player-matched pairs for problem fixtures without
weakening final match quality.

Specifically for fixture `00ba9627`:
- With current window: 11 pairs, residual_std=10 147 ms → drift correction skipped
- With ±35 s window: 72 pairs, dominant cluster at SR−KX≈−21 s → offset correctly estimated → Pass 2 searches at `g_ms + 17 000 ms`

### Secondary Finding

The `time_difference_ms` stored in `shot_events` (and reported by
`analyze_goal_event_timing.py`) is the **residual from the ranking reference**
(`|KX_ts − rank_ref_ms|`), not the raw SR−KX offset. After the Pass 2 ranking
fix (2026-03-21), `rank_ref_ms = predicted_kin_ms` when drift is active, so the
stored value correctly reflects alignment quality — but it is no longer
comparable to `SR − KX` directly. The audit script's CLOCK DRIFT ANALYSIS
section already compensates for this by using `time_difference_ms` from the
pre-synced `shot_events` table (stored before the ranking fix).

---

## Related

- [shot-detection-analysis.md](shot-detection-analysis.md) — sync pipeline review and pitfall catalogue
- [shot-sync-drift-findings.md](shot-sync-drift-findings.md) — per-fixture analysis and proposed symmetric window fix
