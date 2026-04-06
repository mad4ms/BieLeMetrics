# Shot Sync Drift — Investigation Findings

*Investigated: 2026-03-21 | Branch: fix/partition-aware-io-manager*

---

## Summary

The current `_sync_goals_to_detected_shots` logic uses an asymmetric time window
(`tol_before_ms=30 000`, `tol_after_ms=3 000`) for its Pass 1 player-time pair
collection. This design assumes Kinexon always timestamps shots **before**
Sportradar logs the goal. That assumption is wrong for several fixtures, causing
Pass 1 to find too few pairs, making drift correction impossible, and ultimately
producing large numbers of `time_only_fallback` matches or outright unmatched
goals.

---

## How the Sync Works Today

```
Pass 1  player + time  (window: [g_ms − 30s, g_ms + 3s])
        ↓ collect (event_time_ms, SR−KX signed_diff) pairs
        ↓ _estimate_clock_drift  (linear regression)
        ↓ if n_pairs ≥ 5 AND residual_std ≤ 5 000 ms → drift_active = True

Pass 2  full matching
        if drift_active:   window centred on predicted_kin_ms ± tight_tol_ms
        else:              original wide window [g_ms − 30s, g_ms + 3s]
```

The sign convention is `signed_diff = SR_event_time_ms − KX_timestamp_ms`.
A positive offset means Sportradar is later (expected normal case).
A negative offset means Kinexon is later (reversed polarity).

---

## Root Cause: Asymmetric Pass 1 Window

Because `tol_after_ms = 3 000 ms`, Pass 1 only captures player-matched pairs
where the Kinexon shot is at most 3 s **after** the Sportradar event
(`signed_diff ≥ −3 000 ms`).

For fixtures with reversed polarity — where Kinexon timestamps the detected shot
**+20 s or more after** Sportradar — all real player-matched pairs fall outside
the window. Pass 1 collects zero or very few fringe pairs, residual_std is
huge, drift correction is skipped, and Pass 2 uses the same broken window.

---

## Per-Fixture Analysis (28 fixtures, season 2025/26 HBL)

Columns:
- `pairs_curr` — player-matched pairs captured by current `[−30 s, +3 s]` window
- `pairs_wide` — pairs captured by proposed symmetric `[−35 s, +35 s]` window
- `wide_med_s` — median of `signed_diff` with wide window (s); sign: SR−KX
- `wide_std_s` — std of those diffs (s); large = multi-modal / noisy fixture
- `polarity`   — `normal` (KX before SR), `reversed` (KX after SR), `mixed`

```
fixture_id (short)  pairs_curr  pairs_wide  wide_med_s  wide_std_s  polarity
─────────────────────────────────────────────────────────────────────────────
00ba9627               11          72          −21.3 s     12.7 s    reversed
00ec13a6               19          66          +30.8 s     12.7 s    normal*
0172aa0a                0           0             NaN        NaN     no data
02ea68cb               85         101           +3.1 s      7.1 s    normal
02f4e1a9               24          84          +32.7 s     15.9 s    normal*
0375b330               79          92           +3.9 s      8.0 s    normal
04a83daf               67          86           +1.8 s      9.9 s    normal
04f6a2b0               67          90           +6.9 s     10.6 s    normal
05583094               71          90           +5.4 s     10.1 s    normal
05fd729b               87          91           +2.6 s      5.4 s    normal
068c8661               62          83           +3.0 s     12.2 s    normal
0788ecda               80          84           +3.8 s      5.6 s    normal
085c0f47               49          95           −1.9 s     13.4 s    mixed
090fadb2               70          88           +3.7 s     10.5 s    normal
09e1e335               75          87          +16.8 s     10.8 s    normal*
0a94b092               66          79           +4.2 s      9.0 s    normal
0aa76e59               84          88           +3.0 s      4.6 s    normal
0b248d78               73          88           +2.4 s     10.3 s    normal
0c954cad               72          78           +3.8 s      6.1 s    normal
0cd4d4a8               27          78           −5.0 s      6.6 s    mixed
0d1d8d0f               31          89           −3.9 s      7.1 s    mixed
0ec62563               72         105           +7.2 s     14.1 s    normal
0f48d925               46          84           −1.9 s     14.0 s    mixed
0fab2c7c               71          90           +2.7 s     10.1 s    normal
10dfb026               55          90           −1.0 s     12.8 s    mixed
3b60d672               80          85           +3.5 s      5.7 s    normal
540fb714               90          97           +2.6 s      5.4 s    normal
fef92414               12          77           −7.7 s     10.6 s    mixed
─────────────────────────────────────────────────────────────────────────────
```

*`normal*` = normal polarity but large offset (>15 s); window currently reaches
these because the −30 s side is wide enough, but Pass 1 still undershoots the
dominant cluster peak.

### Key observations

1. **The symmetric ±35 s window always captures significantly more pairs.**
   Worst cases: `00ba9627` (11 → 72), `fef92414` (12 → 77), `02f4e1a9` (24 → 84).

2. **`00ba9627` is uniquely reversed** (KX ~21 s after SR). The current window
   sees `signed_diff = SR − KX ≈ −21 s`, which is outside `[−30 s, +3 s]` on the
   positive side — i.e. the *lower* bound `lo = g_ms − 30 000` is fine but the
   *upper* bound `hi = g_ms + 3 000` cuts off the KX shots that arrive at
   `g_ms + 21 000`.

3. **`00ec13a6` and `02f4e1a9`** have a large positive offset (~31 s and ~33 s
   respectively). The current `[−30 s, +3 s]` window captures the fringe of
   the pair distribution rather than the peak, producing noisy pairs and
   high residual_std.

4. **"Mixed" fixtures** (`085c0f47`, `0cd4d4a8`, `0d1d8d0f`, `0f48d925`,
   `10dfb026`, `fef92414`) have bimodal distributions — some goals with KX
   before SR, some with KX after. These likely correspond to two different
   Kinexon sessions (one per half) with different absolute clocks, or a
   session restart at halftime.

5. **`0172aa0a`** has zero player-matched pairs regardless of window size. The
   cause is a broken `league_id` mapping — the Kinexon session linked to this
   fixture likely has no player metadata or uses different player IDs. The window
   fix will not help; this needs session-level investigation.

---

## Detailed Findings: Fixture 00ba9627

*MT Melsungen vs SC DHfK Leipzig, 2025-10-10*

### What the data shows

| Diagnostic | Value |
|---|---|
| Goals (SR) | 87 |
| Detected shots (KX) | 131 |
| Nearest ANY KX shot per goal | median +0.3 s — essentially simultaneous |
| Nearest SAME-PLAYER KX shot | median **+21.7 s** — KX is ~21 s later |
| Player match on nearest ANY shot | 32 / 87 (37%) |
| Player mismatch on nearest ANY shot | 55 / 87 (63%) |

The Kinexon shots that are simultaneous with Sportradar goals are mostly
attributed to the **wrong player**. The shots attributed to the **correct
player** are ~21 s later.

### Per-goal player-matched offset distribution (wide ±35 s window)

```
−34 s │ ·
−30 s │ ····
−26 s │ ····
−24 s │ ···········
−22 s │ ███████████████████████  ← peak (23 pairs)
−20 s │ ·············
−18 s │ ·········
  ... │ scattered positive tail
```

### Why drift correction currently fails

Pass 1 uses `[g_ms − 30 000, g_ms + 3 000]`. The peak of the player-matched
distribution is at `signed_diff ≈ −21 s` (KX is 21 s after SR), which
corresponds to `KX_timestamp = g_ms + 21 000` — outside the `+3 000` upper
bound. Pass 1 captures only 11 fringe pairs from the tail of the distribution.
These 11 pairs span −23 s to +28 s, giving `residual_std = 10 147 ms > 5 000 ms`,
so drift correction is skipped.

With a ±35 s window, Pass 1 would capture 72 pairs. The dominant cluster at
−21 s would produce an estimated `offset ≈ −17 000 ms` (i.e. KX is 17 s after
SR at match midpoint). Pass 2 would then center its search at
`predicted_kin_ms = g_ms + 17 000`, correctly finding the player-attributed shots.

---

## Proposed Fix: Symmetric Pass 1 Window

Change Pass 1 from `[g_ms − 30 s, g_ms + 3 s]` to `[g_ms − 35 s, g_ms + 35 s]`.

Pass 2 stays tight (drift-corrected window `± max(3 × residual_std, 3 s)`).
This is safe because Pass 2 is only widened when drift is active, and a tight
corrected window around the true offset is less likely to produce false positives
than the current fallback wide window.

### Why ±35 s specifically

- The largest observed offset across the dataset is ~33 s (`02f4e1a9`).
- ±35 s captures the dominant cluster for all 28 fixtures.
- The `time_only_fallback` in Pass 2 (no player constraint) is already
  protected by the tight drift-corrected window, so widening Pass 1 does not
  weaken the final matching quality.

### Additional change: `drift_active` threshold

The current `_MAX_RESIDUAL_STD_MS = 5 000` is too strict for fixtures with
genuine multi-modal distributions (wide_std 10–16 s). Consider:
- Keep `_MAX_RESIDUAL_STD_MS = 5 000` as is, but use the wide-window median
  as a **fallback offset** when the linear model fails — i.e. if
  `n_pairs ≥ 10 AND abs(median_signed_diff) > 3 000`, apply a fixed median
  correction even without a linear drift model.

### Caveat: `0172aa0a`

This fixture has zero player-matched pairs regardless of window size. The
cause is a broken `league_id` mapping — the Kinexon session linked to this
fixture likely has no player metadata or uses different player IDs. The window
fix will not help; this needs session-level investigation.

---

## Implementation Location

All changes are localised to `src/pipelines/synced/shot_events.py`:

- `_sync_goals_to_detected_shots`: widen Pass 1 loop bounds
- `_estimate_clock_drift`: optionally add median fallback path
- No changes needed in Dagster assets or IO managers

---

## Related Files

| File | Role |
|---|---|
| `src/pipelines/synced/shot_events.py` | Sync implementation |
| `scripts/analyze_goal_event_timing.py` | Audit script (includes window coverage section) |
| [timing-async-analysis.md](timing-async-analysis.md) | Original async investigation |
| [shot-detection-analysis.md](shot-detection-analysis.md) | Pipeline review and pitfall catalogue |
