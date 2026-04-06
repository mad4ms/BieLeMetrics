"""
Goal-event timing analysis: Sportradar vs Kinexon.

Aggregate mode (default, no fixture_id):
    Reads the pre-synced `shot_events` table and reports timing statistics
    across all fixtures — overall, per fixture, per player, and by match method.

Per-fixture mode (fixture_id supplied):
    Loads raw tables for one fixture, re-runs sync_shot_events in memory,
    and prints a detailed diagnostic report.

Usage:
    uv run python scripts/analyze_goal_event_timing.py
    uv run python scripts/analyze_goal_event_timing.py <fixture_id>
    uv run python scripts/analyze_goal_event_timing.py --seed 3
    uv run python scripts/analyze_goal_event_timing.py --list
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

from src.pipelines.synced.shot_events import (  # noqa: E402
    sync_shot_events,
    _estimate_clock_drift,
    normalize_time,
)

DB_PATH = Path(__file__).parent.parent / "data" / "hbl_raw.duckdb"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _row(label: str, value) -> None:
    print(f"  {label:<40} {value}")


def _pct(num: int, den: int) -> str:
    return f"{100 * num / den:.0f}%" if den else "N/A"


def _stats(series: pd.Series) -> str:
    s = series.dropna().astype(float)
    if s.empty:
        return "N/A"
    return (
        f"min={s.min():.0f}  median={s.median():.0f}  "
        f"p95={s.quantile(0.95):.0f}  max={s.max():.0f}"
    )


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _get_eligible_fixtures(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute("""
        SELECT DISTINCT g.fixture_id
        FROM match_events_normalized_goals g
        JOIN match_detected_shots_normalized s USING (fixture_id)
        JOIN players pl USING (fixture_id)
        WHERE g.event_type = 'goal'
        ORDER BY g.fixture_id
    """).fetchall()
    return [r[0] for r in rows]


def _load_fixture(
    con: duckdb.DuckDBPyConnection, fixture_id: str
) -> dict[str, pd.DataFrame]:
    def q(sql: str, *params: object) -> pd.DataFrame:
        return con.execute(sql, list(params)).df()

    # Load positions only within ±130 s of actual goal events to avoid pulling
    # millions of warm-up/post-match rows that sync_shot_events never needs.
    pos = q(
        """
        SELECT p.*
        FROM match_positions_normalized p
        WHERE p.fixture_id = ?
          AND p.timestamp_ms BETWEEN (
                SELECT epoch_ms(min(event_time::TIMESTAMPTZ)) - 130000
                FROM match_events_normalized_goals
                WHERE fixture_id = ? AND event_type = 'goal'
              ) AND (
                SELECT epoch_ms(max(event_time::TIMESTAMPTZ)) + 130000
                FROM match_events_normalized_goals
                WHERE fixture_id = ? AND event_type = 'goal'
              )
        """,
        fixture_id,
        fixture_id,
        fixture_id,
    )

    return {
        "match": q("SELECT * FROM matches_normalized WHERE fixture_id = ?", fixture_id),
        "goals": q(
            "SELECT * FROM match_events_normalized_goals WHERE fixture_id = ?",
            fixture_id,
        ),
        "shots": q(
            "SELECT * FROM match_detected_shots_normalized WHERE fixture_id = ?",
            fixture_id,
        ),
        "pos": pos,
        "players": q("SELECT * FROM players WHERE fixture_id = ?", fixture_id),
    }


def _query_shot_events(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("""
        SELECT
            fixture_id,
            event_id,
            person_name,
            person_league_id,
            event_time,
            event_time_ms,
            period_id,
            attack_type,
            detected_shot_id,
            detected_events_shot_time,
            time_difference_ms,
            time_diff_event_throw_ms,
            time_diff_detected_shot_throw_ms,
            throw_timestamp_ms,
            distance,
            speed_ball,
            shot_position_x,
            shot_position_y,
            match_method,
            (detected_shot_id IS NOT NULL) AS matched
        FROM shot_events
        ORDER BY fixture_id, event_time_ms
    """).df()


def _query_raw_join(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("""
        WITH goals AS (
            SELECT
                e.fixture_id,
                e.event_id,
                e.event_time,
                epoch_ms(strptime(e.event_time, '%Y-%m-%dT%H:%M:%S%.fZ')) AS event_time_ms,
                e.person_id,
                e.period_id,
                e.attack_type,
                p.league_id AS person_league_id,
                p.name      AS person_name
            FROM match_events_normalized_goals e
            LEFT JOIN players p
                ON e.person_id  = p.person_id
               AND e.fixture_id = p.fixture_id
            WHERE e.event_type = 'goal'
        ),
        shots AS (
            SELECT
                fixture_id,
                league_id,
                timestamp_ms AS kin_timestamp_ms,
                distance,
                speed_ball,
                shot_position_x,
                shot_position_y
            FROM match_detected_shots_normalized
            WHERE event_type = 'detected_shot_handball'
        ),
        candidates AS (
            SELECT
                g.*,
                s.kin_timestamp_ms,
                s.distance,
                s.speed_ball,
                s.shot_position_x,
                s.shot_position_y,
                (g.event_time_ms - s.kin_timestamp_ms)::DOUBLE AS time_difference_ms,
                NULL::DOUBLE AS time_diff_event_throw_ms,
                NULL::DOUBLE AS time_diff_detected_shot_throw_ms,
                NULL::DOUBLE AS throw_timestamp_ms,
                'raw_join'   AS match_method,
                TRUE         AS matched,
                ROW_NUMBER() OVER (
                    PARTITION BY g.event_id
                    ORDER BY ABS(g.event_time_ms - s.kin_timestamp_ms)
                ) AS rn
            FROM goals g
            JOIN shots s
                ON  g.fixture_id       = s.fixture_id
                AND g.person_league_id = s.league_id
                AND ABS(g.event_time_ms - s.kin_timestamp_ms) <= 35000
        )
        SELECT * EXCLUDE (rn)
        FROM candidates
        WHERE rn = 1
        ORDER BY fixture_id, event_time_ms
    """).df()


# ---------------------------------------------------------------------------
# Pass-1 window coverage helpers
# (reproduce SHOT_SYNC_DRIFT_FINDINGS.md analysis)
# ---------------------------------------------------------------------------

_PASS1_CURRENT_BEFORE = 30_000  # ms
_PASS1_CURRENT_AFTER = 3_000  # ms
_PASS1_WIDE = 35_000  # ms  symmetric proposed window


def _collect_pass1_pairs(
    goals: pd.DataFrame,
    shots: pd.DataFrame,
    tol_before_ms: int,
    tol_after_ms: int,
) -> list[int]:
    """Return list of signed_diff values (SR − KX ms) for player-matched pairs."""
    pairs: list[int] = []
    kx = shots.dropna(subset=["timestamp_ms"]).copy()
    kx["timestamp_ms"] = kx["timestamp_ms"].astype(np.int64)
    kx["league_id"] = kx["league_id"].astype("string")
    goals = goals.copy()
    goals["person_league_id"] = goals["person_league_id"].astype("string")

    for _, g in goals.iterrows():
        g_ms = g.get("event_time_ms")
        lid = g.get("person_league_id")
        if pd.isna(g_ms) or pd.isna(lid):
            continue
        lo = int(g_ms) - tol_before_ms
        hi = int(g_ms) + tol_after_ms
        cand = kx[(kx["timestamp_ms"].between(lo, hi)) & (kx["league_id"] == lid)]
        if not cand.empty:
            best_kin = int(
                cand.loc[
                    (cand["timestamp_ms"] - int(g_ms)).abs().idxmin(), "timestamp_ms"
                ]
            )
            pairs.append(int(g_ms) - best_kin)
    return pairs


def _agg_window_coverage(con: duckdb.DuckDBPyConnection) -> None:
    """
    For every eligible fixture, compare Pass-1 pair counts under the current
    [−30 s, +3 s] window versus the proposed ±35 s symmetric window.

    Prints a table flagging fixtures where the current window misses the
    dominant offset cluster (reversed or large-offset polarity).

    All pair-matching is done in a single DuckDB query (range join + window
    function) rather than per-fixture Python loops.
    """
    df = con.execute("""
        WITH goals AS (
            SELECT
                e.fixture_id,
                e.event_id,
                epoch_ms(e.event_time::TIMESTAMPTZ) AS event_time_ms,
                p.league_id AS person_league_id
            FROM match_events_normalized_goals e
            LEFT JOIN players p
                ON  e.person_id  = p.person_id
                AND e.fixture_id = p.fixture_id
            WHERE e.event_type = 'goal'
        ),
        shots AS (
            SELECT fixture_id, league_id, timestamp_ms::BIGINT AS timestamp_ms
            FROM match_detected_shots_normalized
            WHERE event_type = 'detected_shot_handball'
        ),
        wide_candidates AS (
            SELECT
                g.fixture_id,
                g.event_id,
                (g.event_time_ms - s.timestamp_ms) AS signed_diff_ms,
                ROW_NUMBER() OVER (
                    PARTITION BY g.fixture_id, g.event_id
                    ORDER BY ABS(g.event_time_ms - s.timestamp_ms)
                ) AS rn
            FROM goals g
            JOIN shots s
                ON  g.fixture_id       = s.fixture_id
                AND g.person_league_id = s.league_id
                AND s.timestamp_ms BETWEEN g.event_time_ms - 35000 AND g.event_time_ms + 35000
        ),
        wide_best AS (SELECT * FROM wide_candidates WHERE rn = 1),
        curr_candidates AS (
            SELECT
                g.fixture_id,
                g.event_id,
                ROW_NUMBER() OVER (
                    PARTITION BY g.fixture_id, g.event_id
                    ORDER BY ABS(g.event_time_ms - s.timestamp_ms)
                ) AS rn
            FROM goals g
            JOIN shots s
                ON  g.fixture_id       = s.fixture_id
                AND g.person_league_id = s.league_id
                AND s.timestamp_ms BETWEEN g.event_time_ms - 30000 AND g.event_time_ms + 3000
        ),
        curr_best AS (SELECT fixture_id, event_id FROM curr_candidates WHERE rn = 1),
        fixture_stats AS (
            SELECT
                w.fixture_id,
                count(*)                         AS pairs_wide,
                count(c.event_id)                AS pairs_current,
                median(w.signed_diff_ms) / 1000.0 AS wide_median_s,
                stddev(w.signed_diff_ms) / 1000.0 AS wide_std_s
            FROM wide_best w
            LEFT JOIN curr_best c USING (fixture_id, event_id)
            GROUP BY w.fixture_id
        ),
        all_goals AS (
            SELECT fixture_id, count(DISTINCT event_id) AS n_goals
            FROM goals GROUP BY fixture_id
        ),
        all_shots AS (
            SELECT fixture_id, count(*) AS n_shots
            FROM match_detected_shots_normalized GROUP BY fixture_id
        )
        SELECT
            ag.fixture_id,
            ag.n_goals,
            coalesce(aso.n_shots,       0) AS n_shots,
            coalesce(fs.pairs_current,  0) AS pairs_current,
            coalesce(fs.pairs_wide,     0) AS pairs_wide,
            fs.wide_median_s,
            fs.wide_std_s
        FROM all_goals ag
        LEFT JOIN all_shots      aso USING (fixture_id)
        LEFT JOIN fixture_stats  fs  USING (fixture_id)
        ORDER BY ag.fixture_id
    """).df()

    print("=" * 62)
    print("PASS-1 WINDOW COVERAGE ANALYSIS")
    print("  Current window : [−30 s, +3 s]  (asymmetric)")
    print("  Wide window    : [−35 s, +35 s]  (proposed symmetric)")
    print("  signed_diff    = SR_event_time − KX_shot_time  (ms)")
    print("=" * 62)
    hdr = (
        f"  {'fixture_id':<38} {'curr':>5}  {'wide':>5}  "
        f"{'med(s)':>8}  {'std(s)':>6}  {'flag':<10}"
    )
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for _, r in df.sort_values("fixture_id").iterrows():
        curr, wide = int(r["pairs_current"]), int(r["pairs_wide"])
        med = r["wide_median_s"]
        std = r["wide_std_s"]

        # Flag problematic fixtures
        if wide == 0:
            flag = "NO DATA"
        elif wide > 0 and curr == 0:
            flag = "⚠ ALL MISSED"
        elif wide > 0 and curr / wide < 0.4:
            flag = "⚠ UNDERCOUNT"
        elif not np.isnan(med) and abs(med) > 15:
            flag = "⚠ LARGE OFFSET"
        elif not np.isnan(med) and med < -3:
            flag = "⚠ REVERSED"
        else:
            flag = "ok"

        med_str = f"{med:+.1f}" if np.isfinite(med) else "  n/a"
        std_str = f"{std:.1f}" if np.isfinite(std) else "n/a"
        fid_str = str(r["fixture_id"])[:36]
        print(
            f"  {fid_str:<38} {curr:>5}  {wide:>5}  "
            f"{med_str:>8}s  {std_str:>6}s  {flag}"
        )

    n_warn = (df["pairs_wide"] > 0).sum()
    n_ok = df[
        (df["pairs_wide"] > 0)
        & (df["pairs_current"] / df["pairs_wide"].replace(0, np.nan) >= 0.4)
        & (df["wide_median_s"].abs() <= 15)
    ].shape[0]
    print(f"\n  Fixtures with data     : {n_warn}")
    print(f"  Fixtures ok (≥40% coverage, offset ≤15 s): {n_ok}")
    print(f"  Fixtures needing attention: {n_warn - n_ok}")
    print()


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


def run_aggregate(con: duckdb.DuckDBPyConnection) -> None:
    available = {r[0] for r in con.execute("SHOW TABLES").fetchall()}

    if "shot_events" in available:
        df = _query_shot_events(con)
        print(f"Loaded {len(df)} goal events from 'shot_events'.\n")
    else:
        print("'shot_events' not found — falling back to raw join.\n")
        df = _query_raw_join(con)
        print(f"Raw join produced {len(df)} matched goals.\n")

    if df.empty:
        print("No goal events found.")
        return

    _agg_overall(df)
    _agg_by_fixture(df)
    _agg_by_player(df)
    _agg_match_methods(df)
    _agg_clock_drift(df)
    _agg_unmatched(df)
    _agg_window_coverage(con)


def _agg_overall(df: pd.DataFrame) -> None:
    matched = df[df["matched"].fillna(False).astype(bool)]
    td = matched["time_difference_ms"].dropna()

    print("=" * 62)
    print("OVERALL  —  Sportradar event_time  vs  Kinexon detected_shot")
    print("            time_difference_ms = Sportradar − Kinexon  [ms]")
    print("=" * 62)
    print(f"  Total goals              : {len(df)}")
    print(f"  Matched to Kinexon shot  : {len(matched)}")
    print(f"  Unmatched                : {len(df) - len(matched)}")

    if not td.empty:
        print(f"\n  Mean                     : {td.mean():+.0f} ms")
        print(f"  Median                   : {td.median():+.0f} ms")
        print(f"  Std                      : {td.std():.0f} ms")
        print(f"  Min / Max                : {td.min():+.0f} ms  /  {td.max():+.0f} ms")
        print(f"\n  Sportradar LATER  (>0)   : {(td > 0).sum()} goals")
        print(f"  Sportradar EARLIER (<0)  : {(td < 0).sum()} goals")

        buckets = [
            ("< −5 s", td < -5_000),
            ("−5..−2 s", (td >= -5_000) & (td < -2_000)),
            ("−2..0 s", (td >= -2_000) & (td < 0)),
            ("0..+2 s", (td >= 0) & (td < 2_000)),
            ("+2..+5 s", (td >= 2_000) & (td < 5_000)),
            ("> +5 s", td >= 5_000),
        ]
        print("\n  Distribution:")
        for label, mask in buckets:
            n = int(mask.sum())
            print(f"    {label:12s}  {n:3d}  {'#' * n}")

    te = (
        matched["time_diff_event_throw_ms"].dropna()
        if "time_diff_event_throw_ms" in matched.columns
        else pd.Series(dtype=float)
    )
    if not te.empty:
        print("\n  Sportradar event vs refined throw time:")
        print(f"  Mean   : {te.mean():+.0f} ms")
        print(f"  Median : {te.median():+.0f} ms")
        print(f"  Std    : {te.std():.0f} ms")
    print()


def _agg_by_fixture(df: pd.DataFrame) -> None:
    matched = df[df["matched"].fillna(False).astype(bool)]
    if matched.empty:
        return

    summary = (
        matched.groupby("fixture_id")["time_difference_ms"]
        .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
        .reset_index()
        .sort_values("fixture_id")
    )

    print("=" * 62)
    print("BY FIXTURE")
    print("=" * 62)
    hdr = f"  {'fixture_id':<38} {'n':>3}  {'mean':>7}  {'med':>7}  {'std':>6}  {'min':>7}  {'max':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, row in summary.iterrows():
        fid = str(row["fixture_id"])[:36]
        print(
            f"  {fid:<38} {int(row['n']):>3}  {row['mean']:>+7.0f}  "
            f"{row['median']:>+7.0f}  {row['std']:>6.0f}  "
            f"{row['min']:>+7.0f}  {row['max']:>+7.0f}"
        )
    print()


def _agg_by_player(df: pd.DataFrame) -> None:
    matched = df[df["matched"].fillna(False).astype(bool)]
    if matched.empty or "person_name" not in matched.columns:
        return

    summary = (
        matched.groupby(["person_name", "person_league_id"])["time_difference_ms"]
        .agg(n="count", mean="mean", median="median", std="std")
        .reset_index()
        .sort_values("n", ascending=False)
    )

    print("=" * 62)
    print("BY PLAYER  (top 20, sorted by goal count)")
    print("=" * 62)
    print(
        f"  {'player':<25} {'league_id':<16} {'n':>3}  {'mean':>7}  {'med':>7}  {'std':>6}"
    )
    print("  " + "-" * 65)
    for _, row in summary.head(20).iterrows():
        name = str(row.get("person_name") or "")[:24]
        lid = str(row.get("person_league_id") or "")[:15]
        print(
            f"  {name:<25} {lid:<16} {int(row['n']):>3}  "
            f"{row['mean']:>+7.0f}  {row['median']:>+7.0f}  {row['std']:>6.0f}"
        )
    print()


def _agg_match_methods(df: pd.DataFrame) -> None:
    if "match_method" not in df.columns:
        return
    matched = df[df["matched"].fillna(False).astype(bool)]
    if matched.empty:
        return

    print("=" * 62)
    print("MATCH METHOD BREAKDOWN")
    print("=" * 62)
    for method, count in matched["match_method"].value_counts().items():
        td = matched.loc[
            matched["match_method"] == method, "time_difference_ms"
        ].dropna()
        med = f"{td.median():+.0f} ms" if not td.empty else "n/a"
        print(f"  {str(method):<30}  n={count:<4}  median={med}")
    print()


def _agg_clock_drift(df: pd.DataFrame) -> None:
    """
    Per-fixture linear regression of time_difference_ms vs time-into-match.

    Uses only player_time matches (most reliable ground truth). Reports:
      - offset_ms: estimated fixed clock offset (intercept at match midpoint)
      - slope:     drift rate in ms/min (non-zero = clocks running at different speeds)
      - residual_std: spread around the fit (ms)
      - r²:        goodness of fit (1 = perfect linear drift)
    """
    if "match_method" not in df.columns or "event_time_ms" not in df.columns:
        return

    player_matches = df[
        df["matched"].fillna(False).astype(bool)
        & (df["match_method"] == "player_time")
        & df["time_difference_ms"].notna()
        & df["event_time_ms"].notna()
    ].copy()

    if player_matches.empty:
        return

    print("=" * 62)
    print("CLOCK DRIFT ANALYSIS  (player_time matches only)")
    print("  offset = fixed per-fixture clock gap at match midpoint")
    print("  slope  = drift rate in ms/min  (0 = no drift)")
    print("=" * 62)
    hdr = f"  {'fixture_id':<38} {'n':>3}  {'offset':>8}  {'slope':>8}  {'res_std':>7}  {'r²':>5}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    fixture_drifts = []
    for fid, grp in player_matches.groupby("fixture_id"):
        t = grp["event_time_ms"].to_numpy(dtype=float)
        d = grp["time_difference_ms"].to_numpy(dtype=float)
        if len(t) < 2:
            continue

        t_ref = float(np.mean(t))
        t_c = t - t_ref

        if len(t) >= 3:
            slope_raw, offset = np.polyfit(t_c, d, deg=1)
            residuals = d - (offset + slope_raw * t_c)
            res_std = float(np.std(residuals))
            ss_res = float(np.sum(residuals**2))
            ss_tot = float(np.sum((d - np.mean(d)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            slope_per_min = slope_raw * 60_000
        else:
            offset = float(np.median(d))
            slope_per_min = float("nan")
            res_std = float(np.std(d))
            r2 = float("nan")

        fixture_drifts.append((fid, len(grp), offset, slope_per_min, res_std, r2))

        fid_str = str(fid)[:36]
        slope_str = f"{slope_per_min:+.1f}" if np.isfinite(slope_per_min) else "  n/a"
        r2_str = f"{r2:.2f}" if np.isfinite(r2) else "  n/a"
        print(
            f"  {fid_str:<38} {len(grp):>3}  {offset:>+8.0f}  {slope_str:>8}  "
            f"{res_std:>7.0f}  {r2_str:>5}"
        )

    if fixture_drifts:
        offsets = [x[2] for x in fixture_drifts]
        slopes = [x[3] for x in fixture_drifts if np.isfinite(x[3])]
        print(f"\n  offset range : {min(offsets):+.0f} ms  …  {max(offsets):+.0f} ms")
        if slopes:
            print(
                f"  slope range  : {min(slopes):+.1f} ms/min  …  {max(slopes):+.1f} ms/min"
            )
            sig = [s for s in slopes if abs(s) > 500]
            if sig:
                print(
                    f"  Fixtures with |slope| > 500 ms/min: {len(sig)}  "
                    f"(significant drift — consider re-running sync)"
                )
    print()


def _agg_unmatched(df: pd.DataFrame) -> None:
    unmatched = df[~df["matched"].fillna(False).astype(bool)]
    if unmatched.empty:
        return

    cols = [
        c
        for c in ["fixture_id", "event_time", "person_name", "person_league_id"]
        if c in unmatched.columns
    ]
    print("=" * 62)
    print(f"UNMATCHED GOALS  ({len(unmatched)} total)")
    print("=" * 62)
    print(unmatched[cols].to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Per-fixture report
# ---------------------------------------------------------------------------


def run_fixture(fixture_id: str) -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        eligible = _get_eligible_fixtures(con)
        if fixture_id not in eligible:
            print(f"WARNING: {fixture_id} has no matching position+goal+player data.")
            print(f"  Eligible fixtures: {len(eligible)}")
            sys.exit(1)
        data = _load_fixture(con, fixture_id)

    match_row = data["match"].iloc[0] if not data["match"].empty else None

    _header("Fixture")
    _row("ID", fixture_id)
    if match_row is not None:
        _row(
            "Match",
            f"{match_row.get('team_name_home')} vs {match_row.get('team_name_away')}",
        )
        _row("Date (local)", match_row.get("start_time_local"))
    else:
        _row("Match info", "NOT FOUND in matches_normalized")

    _header("Input counts")
    n_goals_raw = len(data["goals"][data["goals"]["event_type"] == "goal"])
    _row("Goal events (Sportradar)", n_goals_raw)
    _row("Detected shots (Kinexon)", len(data["shots"]))
    _row("Position rows", f"{len(data['pos']):,}")
    _row("Players mapped", len(data["players"]))

    # ------------------------------------------------------------------
    # Pre-sync drift analysis: compare Pass-1 window coverage before
    # running the full sync, so problems are visible up front.
    # ------------------------------------------------------------------
    _header("Pass-1 window pre-analysis")
    goals_pre = normalize_time(
        data["goals"][data["goals"]["event_type"] == "goal"].copy()
    )
    goals_pre = goals_pre.merge(
        data["players"][["person_id", "league_id"]].rename(
            columns={"league_id": "person_league_id"}
        ),
        on="person_id",
        how="left",
    )
    shots_pre = data["shots"].copy()
    shots_pre["timestamp_ms"] = pd.to_numeric(
        shots_pre["timestamp_ms"], errors="coerce"
    )

    pairs_curr = _collect_pass1_pairs(
        goals_pre, shots_pre, _PASS1_CURRENT_BEFORE, _PASS1_CURRENT_AFTER
    )
    pairs_wide = _collect_pass1_pairs(goals_pre, shots_pre, _PASS1_WIDE, _PASS1_WIDE)

    _row("player_time pairs (current [−30 s, +3 s])", len(pairs_curr))
    _row("player_time pairs (wide    [−35 s, +35 s])", len(pairs_wide))

    if pairs_wide:
        arr = np.array(pairs_wide, dtype=float)
        _row(
            "wide-window signed_diff median", f"{np.median(arr)/1000:+.1f} s  (SR − KX)"
        )
        _row("wide-window signed_diff std", f"{np.std(arr)/1000:.1f} s")

        # Histogram of wide-window pairs (4 s bins)
        bins = np.arange(
            float(np.floor(arr.min() / 4000) * 4),
            float(np.ceil(arr.max() / 4000) * 4 + 4),
            4,
        )
        counts, edges = np.histogram(arr / 1000, bins=bins)
        peak_bin = edges[int(np.argmax(counts))]
        _row(
            "dominant offset cluster",
            f"{peak_bin:+.0f} s .. {peak_bin+4:+.0f} s  (n={counts.max()})",
        )

        print(f"\n  Signed-diff histogram (4 s bins, {len(pairs_wide)} pairs):")
        max_bar = 40
        for i in range(len(counts)):
            if counts[i] == 0:
                continue
            bar = "█" * int(counts[i] * max_bar / max(counts))
            print(f"    {edges[i]:+6.0f} s  {counts[i]:3d}  {bar}")

        if len(pairs_wide) >= 5:
            ev_ms = np.array(
                [
                    int(g["event_time_ms"])
                    for _, g in goals_pre.iterrows()
                    if not pd.isna(g.get("event_time_ms"))
                    and not pd.isna(g.get("person_league_id"))
                ][: len(pairs_wide)]
            )
            try:
                off, slope, _, res_std = _estimate_clock_drift(
                    ev_ms, np.array(pairs_wide)
                )
                drift_active = res_std <= 5_000
                print("\n  Linear drift estimate:")
                _row("    offset (at match midpoint)", f"{off/1000:+.1f} s")
                _row("    slope", f"{slope*60_000:+.1f} ms/min")
                _row("    residual_std", f"{res_std:.0f} ms")
                _row(
                    "    drift_active (current gate)",
                    "YES" if drift_active else "NO  (res_std > 5 000 ms)",
                )
            except Exception:
                pass
    else:
        _row(
            "  → no player-matched pairs found in either window",
            "⚠ player mapping issue",
        )

    print()
    print("\n  Running sync_shot_events … ", end="", flush=True)
    df = sync_shot_events(
        df_match_normalized=data["match"],
        df_match_events_normalized_goals=data["goals"],
        df_match_detected_shots_normalized=data["shots"],
        df_positions_normalized=data["pos"],
        df_players=data["players"],
    )
    print(f"done ({len(df)} goal rows)")

    if df.empty:
        print("\n  No goal rows produced.")
        return

    n_goals = len(df)

    _header("Shot matching")
    n_matched = df["detected_shot_id"].notna().sum()
    _row(
        "Matched to a detected shot",
        f"{n_matched} / {n_goals}  ({_pct(n_matched, n_goals)})",
    )

    if "match_method" in df.columns:
        for method, cnt in df["match_method"].value_counts().items():
            _row(f"  method={method}", f"{cnt}  ({_pct(cnt, n_matched)})")

    if "time_difference_ms" in df.columns:
        _row("time_difference_ms (matched)", _stats(df["time_difference_ms"]))
        n_large = (df["time_difference_ms"].dropna() > 5_000).sum()
        _row("  > 5 000 ms", f"{n_large}")

    n_dup = df["detected_shot_id"].dropna().duplicated().sum()
    _row("Duplicate detected_shot_id reuse", f"{n_dup}  (should be 0)")

    _header("Player mapping")
    n_league = df["person_league_id"].notna().sum()
    _row(
        "Goals with person_league_id",
        f"{n_league} / {n_goals}  ({_pct(n_league, n_goals)})",
    )
    if "goalkeeper_league_id" in df.columns:
        n_gk = df["goalkeeper_league_id"].notna().sum()
        _row(
            "Goals with goalkeeper_league_id",
            f"{n_gk} / {n_goals}  ({_pct(n_gk, n_goals)})",
        )

    _header("Throw timestamp refinement")
    n_throw = (
        df["throw_timestamp_ms"].notna().sum()
        if "throw_timestamp_ms" in df.columns
        else 0
    )
    _row(
        "Goals with throw_timestamp_ms",
        f"{n_throw} / {n_goals}  ({_pct(n_throw, n_goals)})",
    )

    if "time_diff_event_throw_ms" in df.columns:
        _row("event_time → throw delta (ms)", _stats(df["time_diff_event_throw_ms"]))
    if "time_diff_detected_shot_throw_ms" in df.columns:
        _row(
            "detected_shot → throw delta (ms)",
            _stats(df["time_diff_detected_shot_throw_ms"]),
        )

    if "method" in df.columns:
        for m, cnt in df["method"].value_counts().items():
            _row(f"  throw method={m}", cnt)

    _header("Goal position")
    if "goal_position" in df.columns:
        n_gp = df["goal_position"].notna().sum()
        _row("Goals with goal_position", f"{n_gp} / {n_goals}  ({_pct(n_gp, n_goals)})")
        for pos, cnt in df["goal_position"].value_counts(dropna=False).items():
            label = {0: "left (x=0)", 40: "right (x=40)"}.get(pos, str(pos))
            _row(f"  goal_position={label}", cnt)

    _header("Per-period breakdown")
    if "period_id" in df.columns:
        for period, grp in df.groupby("period_id"):
            n = len(grp)
            n_t = (
                grp["throw_timestamp_ms"].notna().sum()
                if "throw_timestamp_ms" in grp.columns
                else 0
            )
            n_m = grp["detected_shot_id"].notna().sum()
            _row(
                f"Period {int(period)}  (n={n})",
                f"matched={_pct(n_m, n)}  throw={_pct(n_t, n)}",
            )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Goal-event timing analysis: Sportradar vs Kinexon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "fixture_id",
        nargs="?",
        default=None,
        help="Fixture ID for per-fixture diagnostic (omit for aggregate report)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for fixture selection (default: 0)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List eligible fixture IDs and exit"
    )
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    if args.list or args.fixture_id or args.seed != 0:
        with duckdb.connect(str(DB_PATH), read_only=True) as con:
            eligible = _get_eligible_fixtures(con)

        if args.list:
            print(f"Eligible fixtures ({len(eligible)}):")
            for fid in eligible:
                print(f"  {fid}")
            return

        if not eligible:
            print("No eligible fixtures found. Run the pipeline first.")
            sys.exit(1)

        if args.fixture_id:
            run_fixture(args.fixture_id)
        else:
            rng = random.Random(args.seed)
            fixture_id = rng.choice(eligible)
            print(
                f"Selected fixture_id={fixture_id}  (seed={args.seed}, {len(eligible)} eligible)"
            )
            run_fixture(fixture_id)
    else:
        with duckdb.connect(str(DB_PATH), read_only=True) as con:
            run_aggregate(con)


if __name__ == "__main__":
    main()
