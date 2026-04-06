"""
Tests for shot-detection pipeline (src/pipelines/synced/shot_events.py).

Covers the pitfalls documented in SHOT_DETECTION_ANALYSIS.md and verifies that
fixes are in place. Remaining known limitations are documented inline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.pipelines.synced.shot_events import (
    _build_player_ball_timeline,
    _sync_goals_to_detected_shots,
    detect_throw_point,
    insert_goal_position,
    normalize_time,
)

SAMPLES_DIR = Path("assets/data_samples")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_goals(
    n: int = 3, *, base_ms: int = 1_000_000, step_ms: int = 60_000
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [f"goal_{i}" for i in range(n)],
            "event_time_ms": [base_ms + i * step_ms for i in range(n)],
            "person_league_id": [f"player_{i}" for i in range(n)],
        }
    )


def _make_detected(timestamps_ms: list[int], league_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"shot_{i}" for i in range(len(timestamps_ms))],
            "timestamp_ms": timestamps_ms,
            "league_id": league_ids,
        }
    )


def _make_position_rows(
    timestamps_ms: list[int],
    league_id: str,
    xs: list[float],
    ys: list[float],
    *,
    speed: float = 1.0,
    acc: float | None = None,
) -> pd.DataFrame:
    rows = []
    for t, x, y in zip(timestamps_ms, xs, ys):
        row = {
            "timestamp_ms": t,
            "league_id": league_id,
            "x_m": x,
            "y_m": y,
            "speed_m_s": speed,
        }
        if acc is not None:
            row["acceleration"] = acc
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# normalize_time
# ---------------------------------------------------------------------------
class TestNormalizeTime:
    def test_creates_event_time_ms(self):
        df = pd.DataFrame({"event_time": ["2025-01-01T12:00:00+00:00"]})
        out = normalize_time(df)
        assert "event_time_ms" in out.columns
        assert out["event_time_ms"].notna().all()

    def test_parses_offset_aware_iso_string(self):
        df = pd.DataFrame({"event_time": ["2025-08-30 16:02:52.714+02:00"]})
        out = normalize_time(df)
        assert out["event_time_ms"].iloc[0] > 0

    def test_drops_unparseable_rows(self):
        df = pd.DataFrame({"event_time": ["not-a-date", "2025-01-01T12:00:00+00:00"]})
        out = normalize_time(df)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# _sync_goals_to_detected_shots
# ---------------------------------------------------------------------------
class TestSyncGoalsToDetectedShots:
    TOL_BEFORE = 30_000
    TOL_AFTER = 3_000

    def _sync(self, goals, detected, **kw):
        return _sync_goals_to_detected_shots(
            df_goals=goals,
            df_match_detected_shots_normalized=detected,
            tol_before_ms=kw.get("tol_before_ms", self.TOL_BEFORE),
            tol_after_ms=kw.get("tol_after_ms", self.TOL_AFTER),
        )

    # --- row preservation ---
    def test_output_has_same_row_count_as_goals(self):
        goals = _make_goals(5)
        detected = _make_detected([1_000_000 + 30_000 * i for i in range(5)], ["x"] * 5)
        out = self._sync(goals, detected)
        assert len(out) == len(goals)

    def test_empty_detected_returns_all_goals_with_na(self):
        goals = _make_goals(3)
        detected = pd.DataFrame(columns=["id", "timestamp_ms", "league_id"])
        out = self._sync(goals, detected)
        assert len(out) == 3
        assert out["detected_shot_id"].isna().all()

    def test_empty_goals_returns_empty(self):
        goals = pd.DataFrame(columns=["event_id", "event_time_ms", "person_league_id"])
        detected = _make_detected([1_000_000], ["p0"])
        out = self._sync(goals, detected)
        assert out.empty

    # --- uniqueness ---
    def test_no_duplicate_goal_event_id_in_output(self):
        goals = _make_goals(4, base_ms=1_000_000)
        detected = _make_detected([1_000_000], ["player_0"])
        out = self._sync(goals, detected)
        assert not out["goal_event_id"].dropna().duplicated().any()

    # --- match methods ---
    def test_player_time_preferred_when_league_id_matches(self):
        goals = pd.DataFrame(
            {
                "event_id": ["g1"],
                "event_time_ms": [1_000],
                "person_league_id": ["player_A"],
            }
        )
        detected = _make_detected([990, 980], ["player_B", "player_A"])
        out = self._sync(goals, detected)
        assert out["match_method"].iloc[0] == "player_time"
        assert out["detected_shot_id"].iloc[0] == "shot_1"

    def test_time_only_fallback_when_no_player_match(self):
        goals = pd.DataFrame(
            {
                "event_id": ["g1"],
                "event_time_ms": [1_000],
                "person_league_id": ["unknown"],
            }
        )
        detected = _make_detected([990], ["player_B"])
        out = self._sync(goals, detected)
        assert out["match_method"].iloc[0] == "time_only_fallback"

    def test_no_match_when_only_shot_is_outside_window(self):
        goals = pd.DataFrame(
            {"event_id": ["g1"], "event_time_ms": [1_000], "person_league_id": ["p0"]}
        )
        detected = _make_detected([1_000 - 40_000], ["p0"])
        out = self._sync(goals, detected)
        assert out["detected_shot_id"].isna().all()

    # --- Pitfall 2 FIX: one-to-one on detected_shot_id ---
    def test_fix2_detected_shot_id_not_reused_across_goals(self):
        """
        FIXED (Pitfall 2): After sorting by time_diff, detected_shot_id is deduplicated.
        The goal with the smallest time delta wins the shot; the other goal is unmatched.
        """
        goals = pd.DataFrame(
            {
                "event_id": ["g1", "g2"],
                "event_time_ms": [1_000, 1_100],
                "person_league_id": ["p0", "p0"],
            }
        )
        # Single detected shot at t=1000 — within window of both goals
        detected = _make_detected([1_000], ["p0"])
        out = self._sync(goals, detected, tol_before_ms=5_000, tol_after_ms=5_000)
        matched = out.dropna(subset=["detected_shot_id"])
        # Only one goal may claim the shot
        assert len(matched) == 1
        assert matched["detected_shot_id"].nunique() == 1
        # The closer goal (g1, diff=0) wins
        assert matched["event_id"].iloc[0] == "g1"

    # --- Pitfall 1: monitoring via warning (wide window, no hard filter) ---
    def test_pitfall1_large_time_diff_still_matched_but_expected_warning(self):
        """
        Pitfall 1 (monitoring only): A match with time_difference_ms=29_999 is still
        accepted. A warning is logged for values > 5 000 ms but no hard reject occurs.
        """
        goals = pd.DataFrame(
            {"event_id": ["g1"], "event_time_ms": [30_000], "person_league_id": ["p0"]}
        )
        detected = _make_detected([30_000 - 29_999], ["p0"])
        out = self._sync(goals, detected)
        assert out["detected_shot_id"].notna().all()
        assert out["time_difference_ms"].iloc[0] == 29_999

    # --- sample CSV regression ---
    def test_sample_csv_no_duplicate_goal_ids(self):
        goals = pd.read_csv(SAMPLES_DIR / "main.sportradar_goals_synced.csv")
        detected = pd.read_csv(SAMPLES_DIR / "main.kinexon_events.csv")
        goals_input = goals[["event_id", "event_time_ms", "person_league_id"]].copy()
        out = _sync_goals_to_detected_shots(
            df_goals=goals_input,
            df_match_detected_shots_normalized=detected,
            tol_before_ms=30_000,
            tol_after_ms=3_000,
        )
        assert len(out) == len(goals_input)
        assert not out["goal_event_id"].dropna().duplicated().any()

    def test_sample_csv_no_duplicate_detected_shot_ids(self):
        goals = pd.read_csv(SAMPLES_DIR / "main.sportradar_goals_synced.csv")
        detected = pd.read_csv(SAMPLES_DIR / "main.kinexon_events.csv")
        goals_input = goals[["event_id", "event_time_ms", "person_league_id"]].copy()
        out = _sync_goals_to_detected_shots(
            df_goals=goals_input,
            df_match_detected_shots_normalized=detected,
            tol_before_ms=30_000,
            tol_after_ms=3_000,
        )
        matched = out.dropna(subset=["detected_shot_id"])
        assert not matched["detected_shot_id"].duplicated().any()

    def test_sample_csv_match_method_populated(self):
        goals = pd.read_csv(SAMPLES_DIR / "main.sportradar_goals_synced.csv")
        detected = pd.read_csv(SAMPLES_DIR / "main.kinexon_events.csv")
        goals_input = goals[["event_id", "event_time_ms", "person_league_id"]].copy()
        out = _sync_goals_to_detected_shots(
            df_goals=goals_input,
            df_match_detected_shots_normalized=detected,
            tol_before_ms=30_000,
            tol_after_ms=3_000,
        )
        matched = out.dropna(subset=["detected_shot_id"])
        assert set(matched["match_method"].unique()).issubset(
            {"player_time", "time_only_fallback"}
        )


# ---------------------------------------------------------------------------
# detect_throw_point
# ---------------------------------------------------------------------------
def _make_pb_df(
    timestamps_ms: list[int],
    dist_pb: list[float],
    ball_acc: list[float] | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame({"timestamp_ms": timestamps_ms, "dist_pb": dist_pb})
    if ball_acc is not None:
        df["ball_acc"] = ball_acc
    return df


class TestDetectThrowPoint:
    D_POSSESS = 1.5
    X_GOAL = 0

    def _detect(self, df: pd.DataFrame, **kw) -> dict:
        return detect_throw_point(
            df,
            d_possess=kw.get("d_possess", self.D_POSSESS),
            x_pos_goal=kw.get("x_pos_goal", self.X_GOAL),
            min_samples=kw.get("min_samples", 4),
        )

    def test_finds_max_acc_near_last_possession_end(self):
        ms = list(range(0, 500, 50))  # 10 rows, 50ms apart
        dist = [1.0, 1.0, 1.0, 1.0, 1.4, 3.0, 5.0, 6.0, 7.0, 8.0]
        acc = [0.5, 0.5, 0.5, 0.5, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        df = _make_pb_df(ms, dist, acc)
        res = self._detect(df)
        assert res["throw_ts"] is not None
        assert res["method"] == "max_acc_in_possession_windows"
        assert res["throw_idx"] == 4

    def test_no_possession_returns_none_throw(self):
        df = _make_pb_df(list(range(0, 500, 50)), [5.0] * 10)
        res = self._detect(df)
        assert res["throw_ts"] is None

    def test_fallback_to_last_possession_when_no_acc(self):
        dist = [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        df = _make_pb_df(list(range(0, 500, 50)), dist, ball_acc=None)
        res = self._detect(df)
        assert res["throw_ts"] is not None
        assert res["method"] in (
            "last_possession_fallback_no_acc",
            "last_possession_fallback_empty_window",
        )

    def test_respects_min_samples(self):
        df = _make_pb_df([0, 50, 100], [1.0, 1.0, 1.0], [5.0, 5.0, 5.0])
        res = self._detect(df, min_samples=4)
        assert res["throw_ts"] is None

    def test_fix5_last_possession_end_wins_over_earlier_high_acc(self):
        """
        FIXED (Pitfall 5): Only the LAST possession window is searched.
        Even if an earlier burst has higher acceleration (likely a pass),
        the throw point is taken from the last possession end.
        """
        # Two possession bursts:
        #   Burst 1: rows 0-1 at t=0..100ms, acc=10 (earlier, likely a pass)
        #   Burst 2: rows 6-7 at t=600..700ms, acc=3 (later, actual throw)
        ms = list(range(0, 1000, 100))
        dist = [1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 5.0, 5.0]
        acc = [10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0]
        df = _make_pb_df(ms, dist, acc)
        res = self._detect(df)
        # Must pick from the last (burst 2) window, not burst 1
        assert res["throw_idx"] in (
            6,
            7,
        ), "Expected throw at last possession end (burst 2), not earlier high-acc burst"

    def test_result_keys_always_present(self):
        df = _make_pb_df(list(range(0, 500, 50)), [1.0] * 10, [1.0] * 10)
        res = self._detect(df)
        for key in ("n_rows", "last_possession_idx", "throw_idx", "throw_ts", "method"):
            assert key in res


# ---------------------------------------------------------------------------
# _build_player_ball_timeline
# ---------------------------------------------------------------------------
class TestBuildPlayerBallTimeline:
    FRAME_TOL = 50  # ms

    def _scene(self, *dfs) -> pd.DataFrame:
        return pd.concat(list(dfs), ignore_index=True)

    def test_basic_exact_match(self):
        ts = [1000, 2000, 3000]
        ball = _make_position_rows(ts, "ball_EisenBall", [0.0] * 3, [0.0] * 3)
        shooter = _make_position_rows(ts, "shooter_1", [1.0] * 3, [0.0] * 3)
        df_pb = _build_player_ball_timeline(
            self._scene(ball, shooter), "shooter_1", frame_tol_ms=self.FRAME_TOL
        )
        assert len(df_pb) == 3
        assert "dist_pb" in df_pb.columns

    def test_dist_pb_is_euclidean(self):
        ball = _make_position_rows([1000], "ball_x", [3.0], [4.0])
        shooter = _make_position_rows([1000], "shooter_1", [0.0], [0.0])
        df_pb = _build_player_ball_timeline(
            self._scene(ball, shooter), "shooter_1", frame_tol_ms=self.FRAME_TOL
        )
        assert abs(df_pb["dist_pb"].iloc[0] - 5.0) < 1e-6

    def test_same_clock_exact_join_is_correct(self):
        """
        Kinexon uses a single shared hardware clock: ball and player samples for
        the same frame have the exact same timestamp_ms. The exact-equality join
        is correct by design — no nearest-neighbour tolerance is needed.
        """
        ts = [1000, 1050, 1100]  # identical timestamps for both sensors
        ball = _make_position_rows(ts, "ball_x", [0.0] * 3, [0.0] * 3)
        shooter = _make_position_rows(ts, "shooter_1", [1.0] * 3, [0.0] * 3)
        df_pb = _build_player_ball_timeline(
            self._scene(ball, shooter), "shooter_1", frame_tol_ms=self.FRAME_TOL
        )
        # All 3 shared timestamps produce matched rows
        assert len(df_pb) == 3

    def test_ball_identified_by_case_insensitive_ball_in_league_id(self):
        ball = _make_position_rows([1000], "EisenBall3", [0.0], [0.0])
        shooter = _make_position_rows([1000], "player_1", [5.0], [0.0])
        df_pb = _build_player_ball_timeline(
            self._scene(ball, shooter), "player_1", frame_tol_ms=self.FRAME_TOL
        )
        assert len(df_pb) == 1

    def test_returns_empty_when_no_ball_rows(self):
        shooter = _make_position_rows([1000, 2000], "player_1", [1.0, 1.0], [0.0, 0.0])
        df_pb = _build_player_ball_timeline(
            shooter, "player_1", frame_tol_ms=self.FRAME_TOL
        )
        assert df_pb.empty

    def test_returns_empty_when_no_shooter_rows(self):
        ball = _make_position_rows([1000, 2000], "ball_x", [0.0, 0.0], [0.0, 0.0])
        df_pb = _build_player_ball_timeline(
            ball, "missing_player", frame_tol_ms=self.FRAME_TOL
        )
        assert df_pb.empty

    def test_returns_empty_when_scene_empty(self):
        empty = pd.DataFrame(columns=["timestamp_ms", "league_id", "x_m", "y_m"])
        df_pb = _build_player_ball_timeline(empty, "p1", frame_tol_ms=self.FRAME_TOL)
        assert df_pb.empty

    def test_ball_acc_computed_from_speed_when_acceleration_missing(self):
        ts = [0, 100, 200, 300, 400]
        ball = _make_position_rows(ts, "ball_x", [0.0] * 5, [0.0] * 5, speed=5.0)
        ball = ball.drop(columns=["acceleration"], errors="ignore")
        shooter = _make_position_rows(ts, "player_1", [1.0] * 5, [0.0] * 5, speed=1.0)
        df_pb = _build_player_ball_timeline(
            self._scene(ball, shooter), "player_1", frame_tol_ms=self.FRAME_TOL
        )
        assert "ball_acc" in df_pb.columns


# ---------------------------------------------------------------------------
# insert_goal_position
# ---------------------------------------------------------------------------
class TestInsertGoalPosition:
    def _make_goals(self, team, gk_id, period_event_ms):
        return pd.DataFrame(
            {
                "team_name_defense": [team] * len(period_event_ms),
                "goalkeeper_league_id": [gk_id] * len(period_event_ms),
                "period_id": [p for p, _ in period_event_ms],
                "event_time_ms": [t for _, t in period_event_ms],
            }
        )

    def _make_positions(self, timestamps_ms, gk_id, x_vals):
        return pd.DataFrame(
            {
                "timestamp_ms": timestamps_ms,
                "league_id": [gk_id] * len(timestamps_ms),
                "x_m": x_vals,
            }
        )

    # buffer_ms is kept small in these unit tests so that period windows
    # don't overlap. Production default is 120_000 ms (2 minutes).
    BUFFER = 8_000

    def test_period1_left_period2_right(self):
        goals = self._make_goals("Team A", "gk_a", [(1, 10_000), (2, 70_000)])
        pos = self._make_positions(
            [5_000, 6_000, 65_000, 66_000], "gk_a", [2.0, 2.0, 38.0, 38.0]
        )
        out = insert_goal_position(goals, pos, buffer_ms=self.BUFFER)
        assert out.loc[out["period_id"] == 1, "goal_position"].iloc[0] == 0
        assert out.loc[out["period_id"] == 2, "goal_position"].iloc[0] == 40

    def test_period1_right_period2_left(self):
        goals = self._make_goals("Team B", "gk_b", [(1, 10_000), (2, 70_000)])
        pos = self._make_positions(
            [5_000, 6_000, 65_000, 66_000], "gk_b", [38.0, 38.0, 2.0, 2.0]
        )
        out = insert_goal_position(goals, pos, buffer_ms=self.BUFFER)
        assert out.loc[out["period_id"] == 1, "goal_position"].iloc[0] == 40
        assert out.loc[out["period_id"] == 2, "goal_position"].iloc[0] == 0

    def test_fix6_team_with_p2_only_goals_gets_goal_position(self):
        """
        FIXED (Pitfall 6): A defending team that only concedes in period 2 now
        correctly resolves goal_position for period 2. Previously the entire team
        was skipped because df_p1 was empty.
        """
        goals = self._make_goals("Team C", "gk_c", [(2, 70_000)])
        pos = self._make_positions([65_000, 66_000], "gk_c", [38.0, 38.0])
        out = insert_goal_position(goals, pos, buffer_ms=self.BUFFER)
        assert out["goal_position"].iloc[0] == 40

    def test_fix6_periods_resolved_independently(self):
        """
        FIXED (Pitfall 6): Each period is resolved independently without requiring
        the other period's goals to exist. A team with goals in both periods
        gets the correct goal_position for each.
        """
        goals = self._make_goals("Team D", "gk_d", [(1, 10_000), (2, 70_000)])
        pos = self._make_positions(
            [9_000, 10_000, 11_000, 69_000, 70_000, 71_000],
            "gk_d",
            [2.0, 2.0, 2.0, 38.0, 38.0, 38.0],
        )
        out = insert_goal_position(goals, pos, buffer_ms=2_000)
        assert out.loc[out["period_id"] == 1, "goal_position"].iloc[0] == 0
        assert out.loc[out["period_id"] == 2, "goal_position"].iloc[0] == 40

    def test_no_goalkeeper_id_leaves_position_none(self):
        goals = pd.DataFrame(
            {
                "team_name_defense": ["Team E"],
                "goalkeeper_league_id": [np.nan],
                "period_id": [1],
                "event_time_ms": [10_000],
            }
        )
        pos = self._make_positions([5_000], "gk_e", [2.0])
        out = insert_goal_position(goals, pos)
        assert out["goal_position"].isna().all()
