import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from src.hbl_etl_dagster.utils.goal_rendering import (
    OUT_DIR,
    render_goal_with_multifreeze,
)


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------
def _ts_to_ms(series_or_ts):
    s = pd.to_datetime(series_or_ts, utc=True, errors="coerce")
    if isinstance(s, pd.Series):
        # Cast to ms precision before converting to int64.
        # In pandas 3.x, datetime64[us] astype("int64") gives microseconds,
        # so dividing by 10**6 would yield seconds (wrong). Casting to ms first
        # ensures astype("int64") always returns milliseconds since epoch.
        return s.astype("datetime64[ms, UTC]").astype("int64")
    return None if pd.isna(s) else int(s.value // 10**6)


def _fmt_time_ms(ms_val, _pos=None):
    try:
        ts = pd.to_datetime(int(ms_val), unit="ms", utc=True)
        return ts.strftime("%H:%M:%S.%f")[:-3]
    except Exception:
        return ""


def _annot(ax, x, y, text, color, dy: float = 0.0):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(0, 12 + dy),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=1, alpha=0.8),
    )


# -----------------------------------------------------------------------------
# Diagnostics plot around player–ball timeline
# -----------------------------------------------------------------------------
def plot_event_sync(
    df_pb: pd.DataFrame,
    *,
    last_possession_idx: Optional[int],
    event_ms: Optional[int],
    kin_ms: Optional[int],
    throw_ms: Optional[int],
    d_possess: float,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize ball acceleration + player–ball distance with event markers.

    Requires df_pb columns:
      - ts (Timestamp, UTC) OR timestamp_ms
      - ball_acc (float)
      - dist_pb (float)
      - ball_speed (float, optional)
    """
    if df_pb is None or df_pb.empty:
        return

    df_pb = df_pb.copy()
    if "ts" not in df_pb.columns:
        df_pb["ts"] = pd.to_datetime(
            df_pb["timestamp_ms"], unit="ms", utc=True, errors="coerce"
        )

    x_ms = _ts_to_ms(df_pb["ts"])
    acc = df_pb.get("ball_acc", pd.Series([np.nan] * len(df_pb))).to_numpy(dtype=float)
    dist = df_pb.get("dist_pb", pd.Series([np.nan] * len(df_pb))).to_numpy(dtype=float)
    speed = df_pb.get("ball_speed", pd.Series([np.nan] * len(df_pb))).to_numpy(
        dtype=float
    )

    possess = (
        (df_pb["dist_pb"] <= d_possess).to_numpy()
        if "dist_pb" in df_pb.columns
        else np.zeros(len(df_pb), bool)
    )

    fig, axA = plt.subplots(figsize=(11.5, 5.2))
    axA.xaxis.set_major_formatter(FuncFormatter(_fmt_time_ms))

    axA.plot(x_ms, acc, lw=1.6, label="Ball acceleration (m/s²)")
    axA.axhline(0, lw=1)
    axA.set_xlabel("Time (UTC)")
    axA.set_ylabel("Acceleration (m/s²)")

    axD = axA.twinx()
    axD.plot(x_ms, dist, ls="--", lw=1.6, label="Player–ball distance (m)")
    axD.set_ylabel("Distance / speed")

    ymax = np.nanmax(
        [
            np.nanmax(dist) if dist.size else 0.0,
            np.nanmax(speed) if speed.size else 0.0,
        ]
    )
    if np.isfinite(ymax) and ymax > 0:
        poss_mask: List[bool] = [bool(v) for v in possess]
        axD.fill_between(
            x_ms, 0, ymax * 1.1, where=poss_mask, alpha=0.2, label="Possession"
        )

    markers = [
        (event_ms, "SR event", "tab:blue"),
        (kin_ms, "Detected shot", "tab:orange"),
        (throw_ms, "Throw / release", "tab:red"),
    ]
    for ms, lab, col in markers:
        if ms is None:
            continue
        axA.axvline(ms, color=col, ls=":", lw=1.8, label=lab)
        idx = int(np.clip(np.searchsorted(x_ms.to_numpy(), ms), 0, len(acc) - 1))
        y_here = np.nan_to_num(acc[idx], nan=0.0)
        _annot(axA, ms, y_here, lab, col)

    if title:
        axA.set_title(title)

    # Deduplicate legend across twin axes
    lines, labels = axA.get_legend_handles_labels()
    r_lines, r_labels = axD.get_legend_handles_labels()
    seen = set()
    all_lines, all_labels = [], []
    for h, t in list(zip(lines + r_lines, labels + r_labels)):
        if t and t not in seen:
            seen.add(t)
            all_lines.append(h)
            all_labels.append(t)
    axA.legend(all_lines, all_labels, loc="upper right", ncol=2, frameon=False)

    axA.grid(True, alpha=0.25)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show(block=True)


# -----------------------------------------------------------------------------
# Match goal events to detected shot events
# -----------------------------------------------------------------------------
def _sync_goals_to_detected_shots(
    df_goals: pd.DataFrame,
    df_match_detected_shots_normalized: pd.DataFrame,
    *,
    tol_before_ms: int,
    tol_after_ms: int,
) -> pd.DataFrame:
    """
    For each goal, select the closest detected shot within [goal_ms - tol_before, goal_ms + tol_after].

    Strategy:
      1. Try to match by shooter league_id within tolerance.
      2. If no match, fall back to closest shot by time within tolerance (ignoring player).

    Output is row-preserving for df_goals (left join semantics).
    """
    if df_goals.empty or df_match_detected_shots_normalized.empty:
        out = df_goals.copy()
        out["detected_shot_id"] = pd.NA
        out["detected_events_shot_time"] = pd.NaT
        out["time_difference_ms"] = pd.NA
        out["match_method"] = pd.NA
        return out

    kx = df_match_detected_shots_normalized.copy()
    kx["timestamp_ms"] = pd.to_numeric(kx["timestamp_ms"], errors="coerce")
    kx = kx.dropna(subset=["timestamp_ms"]).copy()
    kx["timestamp_ms"] = kx["timestamp_ms"].astype(np.int64)

    # Ensure ids compare reliably
    kx["league_id"] = kx["league_id"].astype("string")

    goals = df_goals.copy()
    goals["person_league_id"] = goals["person_league_id"].astype("string")

    rows: List[Dict[str, Any]] = []

    for _, g in goals.iterrows():
        g_ms = g.get("event_time_ms")
        lid = g.get("person_league_id")
        if pd.isna(g_ms):
            continue

        lo = int(g_ms) - int(tol_before_ms)
        hi = int(g_ms) + int(tol_after_ms)

        # Filter candidates by time window first
        cand_time = kx[kx["timestamp_ms"].between(lo, hi)].copy()

        best = None
        method = None

        if not cand_time.empty:
            # 1. Try strict player match
            if pd.notna(lid):
                cand_player = cand_time[cand_time["league_id"] == lid].copy()
                if not cand_player.empty:
                    cand_player["time_diff"] = (
                        cand_player["timestamp_ms"] - int(g_ms)
                    ).abs()
                    best = cand_player.loc[cand_player["time_diff"].idxmin()]
                    method = "player_time"

            # 2. Fallback to time-only match
            if best is None:
                cand_time["time_diff"] = (cand_time["timestamp_ms"] - int(g_ms)).abs()
                best = cand_time.loc[cand_time["time_diff"].idxmin()]
                method = "time_only_fallback"

        if best is not None:
            rows.append(
                {
                    "goal_event_id": g["event_id"],
                    "detected_shot_id": best.get("id"),
                    "detected_events_shot_time": pd.to_datetime(
                        int(best["timestamp_ms"]), unit="ms", utc=True
                    ),
                    "time_difference_ms": int(best["time_diff"]),
                    "distance": best.get("distance"),
                    "speed_ball": best.get("speed_ball"),
                    "trajectory": best.get("trajectory"),
                    "shot_position_x": best.get("shot_position_x"),
                    "shot_position_y": best.get("shot_position_y"),
                    "hit_position_y": best.get("hit_position_y"),
                    "hit_position_z": best.get("hit_position_z"),
                    "sucess_kinexon": best.get("success"),
                    "shot_category": best.get("shot_category"),
                    "shot_type": best.get("shot_type"),
                    "validated": best.get("validated"),
                    "assisting_mapped_id": best.get("assisting_player_id"),
                    "match_method": method,
                }
            )
    if rows == []:
        df_synced = pd.DataFrame(
            columns=[
                "goal_event_id",
                "detected_shot_id",
                "detected_events_shot_time",
                "time_difference_ms",
                "match_method",
            ]
        )
    else:
        df_synced = (
            pd.DataFrame(rows)
            .sort_values("time_difference_ms")
            .drop_duplicates(subset=["goal_event_id"], keep="first")
        )

    if df_synced["goal_event_id"].duplicated().any():
        raise RuntimeError("Duplicate goal_event_id in detected-shot sync")

    return goals.merge(
        df_synced, left_on="event_id", right_on="goal_event_id", how="left"
    )


# -----------------------------------------------------------------------------
# Build player–ball joined timeline for throw detection
# -----------------------------------------------------------------------------
def _build_player_ball_timeline(
    df_scene: pd.DataFrame,
    shooter_league_id: Any,
    *,
    frame_tol_ms: int,
) -> pd.DataFrame:
    """
    Align ball samples to shooter samples by timestamp (nearest within tolerance).

    Returns df_pb with:
      - timestamp_ms, ts
      - ball_x, ball_y, ball_speed, ball_acc
      - pl_x, pl_y, pl_speed, pl_acc
      - dist_pb
    """
    if df_scene.empty or pd.isna(shooter_league_id):
        return pd.DataFrame()

    shooter_lid = str(shooter_league_id)

    df_scene = df_scene.copy()
    df_scene["timestamp_ms"] = pd.to_numeric(df_scene["timestamp_ms"], errors="coerce")
    df_scene = df_scene.dropna(subset=["timestamp_ms"]).copy()
    df_scene["timestamp_ms"] = df_scene["timestamp_ms"].astype(np.int64)

    if "ts" not in df_scene.columns:
        df_scene["ts"] = pd.to_datetime(
            df_scene["timestamp_ms"], unit="ms", utc=True, errors="coerce"
        )

    df_scene["league_id"] = df_scene["league_id"].astype("string")

    ball_mask = df_scene["league_id"].str.contains("ball", case=False, na=False)
    ball = df_scene[ball_mask].copy()
    shooter = df_scene[df_scene["league_id"] == shooter_lid].copy()
    if ball.empty or shooter.empty:
        return pd.DataFrame()

    keep = ["timestamp_ms", "ts", "x_m", "y_m"]
    if "speed_m_s" in df_scene.columns:
        keep.append("speed_m_s")
    if "acceleration" in df_scene.columns:
        keep.append("acceleration")

    ball = (
        ball[keep]
        .sort_values("timestamp_ms")
        .rename(
            columns={
                "x_m": "ball_x",
                "y_m": "ball_y",
                "speed_m_s": "ball_speed",
                "acceleration": "ball_acc",
            }
        )
    )
    shooter = (
        shooter[keep]
        .sort_values("timestamp_ms")
        .rename(
            columns={
                "x_m": "pl_x",
                "y_m": "pl_y",
                "speed_m_s": "pl_speed",
                "acceleration": "pl_acc",
            }
        )
    )

    df_pb = (
        ball.sort_values(["timestamp_ms"])
        .merge(
            shooter.sort_values(["timestamp_ms"]),
            on=["timestamp_ms", "ts"],  # keep both to guarantee alignment
            how="inner",
            suffixes=("", ""),
        )
        .dropna(subset=["ball_x", "ball_y", "pl_x", "pl_y"])
    )

    if df_pb.empty:
        return df_pb

    df_pb["dist_pb"] = np.hypot(
        df_pb["ball_x"] - df_pb["pl_x"], df_pb["ball_y"] - df_pb["pl_y"]
    )

    # Reconstruct ball_acc from ball_speed if missing
    if "ball_acc" not in df_pb.columns:
        df_pb["ball_acc"] = pd.NA
    if df_pb["ball_acc"].isna().all():
        if "ball_speed" in df_pb.columns:
            v = df_pb["ball_speed"].ffill().fillna(0.0).astype(float)
            dt = df_pb["timestamp_ms"].diff().astype(float) / 1000.0
            dt = dt.replace(0.0, np.nan)
            acc = (v - v.shift(1)) / dt
            df_pb["ball_acc"] = acc.replace([np.inf, -np.inf], np.nan)
        else:
            df_pb["ball_acc"] = pd.NA

    return df_pb.sort_values("timestamp_ms").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Throw point detection
# -----------------------------------------------------------------------------
def detect_throw_point(
    df_pb: pd.DataFrame,
    *,
    d_possess: float,
    x_pos_goal: int,
    min_samples: int = 4,
) -> Dict[str, Optional[object]]:
    """
    Compute throw point based on possession windows and ball acceleration peaks.

    Steps:
      1) Possession is dist_pb <= d_possess
      2) Find ends of all possession windows
      3) Build a ±50ms window around each end timestamp
      4) Choose the row with max ball_acc across all those windows
      5) Fallback to the last possession end row
    """
    out: Dict[str, Optional[object]] = dict(
        n_rows=int(len(df_pb)),
        last_possession_idx=None,
        throw_idx=None,
        throw_ts=None,
        method=None,
    )

    if df_pb.empty or len(df_pb) < min_samples:
        return out

    if "ts" not in df_pb.columns:
        df_pb = df_pb.copy()
        df_pb["ts"] = pd.to_datetime(
            df_pb["timestamp_ms"], unit="ms", utc=True, errors="coerce"
        )

    poss = df_pb["dist_pb"] <= d_possess
    if not poss.any():
        return out

    # Removed strict towards_goal filter as it can be noisy
    # dx = df_pb["ball_x"] - x_pos_goal
    # vx = df_pb["ball_x"].diff() / df_pb["timestamp_ms"].diff()
    # towards_goal = (dx * vx) < 0
    # df_pb = df_pb[towards_goal]

    # End of possession: True followed by False (or end of series)
    is_end = poss & (~poss.shift(-1, fill_value=False))
    end_ts = df_pb.loc[is_end, "ts"]
    if end_ts.empty:
        return out

    # Reference: last possession end row
    last_ts = end_ts.iloc[-1]
    last_row = df_pb[df_pb["ts"] == last_ts].iloc[0]
    out["last_possession_idx"] = int(df_pb.index.get_indexer_for([last_row.name])[0])

    # Search mask: ±50ms around each possession end timestamp
    search_mask = pd.Series(False, index=df_pb.index)
    for t_end in end_ts:
        t0 = t_end - pd.Timedelta(milliseconds=50)
        t1 = t_end + pd.Timedelta(milliseconds=50)
        search_mask |= (df_pb["ts"] >= t0) & (df_pb["ts"] <= t1)

    candidates = df_pb[search_mask]
    if candidates.empty:
        throw_row = last_row
        method = "last_possession_fallback_empty_window"
    else:
        if "ball_acc" not in candidates.columns or candidates["ball_acc"].isna().all():
            throw_row = last_row
            method = "last_possession_fallback_no_acc"
        else:
            best_idx = candidates["ball_acc"].astype(float).idxmax()
            throw_row = candidates.loc[best_idx]
            method = "max_acc_in_possession_windows"

    out["throw_idx"] = int(df_pb.index.get_indexer_for([throw_row.name])[0])
    out["throw_ts"] = throw_row["ts"]
    out["method"] = method
    return out


# -----------------------------------------------------------------------------
# Refine throw time for each goal
# -----------------------------------------------------------------------------
def _refine_throw_times(
    df_goals: pd.DataFrame,
    df_positions: pd.DataFrame,
    *,
    tol_before_ms: int,
    tol_after_ms: int,
    frame_tol_ms: int,
    d_possess: float,
    debug_plot: bool = False,
    debug_plot_max: int = 0,
) -> pd.DataFrame:
    """
    Determine throw timestamp for each goal event.

    Seed time:
      - detected_events_shot_time if present, else event_time_ms
    """
    if df_goals.empty or df_positions.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "throw_timestamp_ms",
                "throw_ts",
                "throw_acceleration",
                "method",
            ]
        )

    pos = df_positions.copy()
    pos["timestamp_ms"] = pd.to_numeric(pos["timestamp_ms"], errors="coerce")
    pos = pos.dropna(subset=["timestamp_ms"]).copy()
    pos["timestamp_ms"] = pos["timestamp_ms"].astype(np.int64)
    pos = pos.sort_values("timestamp_ms").reset_index(drop=True)

    if "ts" not in pos.columns:
        pos["ts"] = pd.to_datetime(
            pos["timestamp_ms"], unit="ms", utc=True, errors="coerce"
        )

    pos["league_id"] = pos["league_id"].astype("string")

    out_rows: List[Dict[str, Any]] = []
    plots_left = int(debug_plot_max)

    for _, goal in df_goals.iterrows():
        det_ts = goal.get("detected_events_shot_time")
        seed_ms = (
            int(det_ts.value // 1_000_000)
            if pd.notna(det_ts)
            else goal.get("event_time_ms")
        )

        shooter_lid = goal.get("person_league_id")
        if pd.isna(seed_ms) or pd.isna(shooter_lid):
            continue

        lo = int(seed_ms) - int(tol_before_ms)
        hi = int(seed_ms) + int(tol_after_ms)

        df_scene = pos[pos["timestamp_ms"].between(lo, hi)].copy()
        if df_scene.empty:
            continue

        df_pb = _build_player_ball_timeline(
            df_scene, shooter_lid, frame_tol_ms=frame_tol_ms
        )
        if df_pb.empty or len(df_pb) < 4:
            continue

        res = detect_throw_point(
            df_pb,
            d_possess=d_possess,
            min_samples=4,
            x_pos_goal=goal.get("goal_position"),
        )
        throw_ts = res.get("throw_ts")
        if throw_ts is None or pd.isna(throw_ts):
            continue

        throw_ms = int(pd.to_datetime(throw_ts, utc=True).value // 10**6)

        # Acceleration at chosen row (if available)
        throw_acc = pd.NA
        try:
            idx_obj = res.get("throw_idx")
            if isinstance(idx_obj, (int, np.integer)) and "ball_acc" in df_pb.columns:
                throw_acc = df_pb.loc[int(idx_obj), "ball_acc"]
        except Exception:
            throw_acc = pd.NA

        if debug_plot and plots_left > 0:
            event_ms = None
            try:
                event_time_ms_val = goal.get("event_time_ms")
                event_ms = (
                    int(event_time_ms_val) if pd.notna(event_time_ms_val) else None
                )
            except Exception:
                event_ms = None

            kin_ms = None
            try:
                kin_ms = int(det_ts.value // 1_000_000) if pd.notna(det_ts) else None
            except Exception:
                kin_ms = None

            last_possession_idx_obj = res.get("last_possession_idx")
            last_possession_idx = (
                int(cast(int, last_possession_idx_obj))
                if isinstance(last_possession_idx_obj, (int, np.integer))
                else None
            )

            plot_event_sync(
                df_pb=df_pb,
                last_possession_idx=last_possession_idx,
                event_ms=event_ms,
                kin_ms=kin_ms,
                throw_ms=throw_ms,
                d_possess=d_possess,
                title=f"event_id={goal.get('event_id')} shooter={goal.get('person_league_id')}",
            )
            plots_left -= 1

        out_rows.append(
            {
                "event_id": goal.get("event_id"),
                "throw_timestamp_ms": throw_ms,
                "throw_ts": pd.to_datetime(throw_ms, unit="ms"),
                "throw_acceleration": throw_acc,
                "method": res.get("method"),
            }
        )

    return pd.DataFrame(out_rows)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def normalize_time(df_goals: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure event_time is datetime and add event_time_ms column.
    """
    out = df_goals.copy()
    out["event_time"] = pd.to_datetime(
        out["event_time"], format="ISO8601", utc=True, errors="coerce"
    )
    out = out.dropna(subset=["event_time"]).copy()
    out["event_time_ms"] = _ts_to_ms(out["event_time"])
    return out


def insert_team_info(
    df_goals: pd.DataFrame,
    df_matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich goals with match team info (home/away entity_id and team_name).
    """
    out = df_goals.copy()

    if df_matches.empty:
        return out

    matches_lkp = (
        df_matches[
            [
                "fixture_id",
                "entity_id_home",
                "entity_id_away",
                "team_name_home",
                "team_name_away",
            ]
        ]
        .sort_values("fixture_id")
        .drop_duplicates(subset=["fixture_id"], keep="first")
    )

    out = out.merge(
        matches_lkp,
        on="fixture_id",
        how="left",
    )

    return out


def insert_player_goalkeeper_info(
    df_goals: pd.DataFrame,
    df_players: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich goals with shooter, goalkeeper, and assisting player info from players DataFrame.
    """
    out = df_goals.copy()

    # Shooter enrichment (person_id -> league_id/mapped_id/team)
    out = out.merge(
        df_players[["person_id", "league_id", "mapped_id", "team_name"]],
        on="person_id",
        how="left",
    ).rename(
        columns={
            "mapped_id": "person_mapped_id",
            "league_id": "person_league_id",
            "team_name": "team_name_offense",
            "name": "person_name",
        }
    )

    # Goalkeeper enrichment (goalkeeper_id -> league_id/mapped_id/name/team)
    # Use distinct names to avoid collisions; do not rename "mapped_id"/"league_id" from the shooter join.
    out = out.merge(
        df_players[["person_id", "league_id", "mapped_id", "name", "team_name"]].rename(
            columns={
                "person_id": "goalkeeper_person_id",
                "league_id": "goalkeeper_league_id",
                "mapped_id": "goalkeeper_mapped_id",
                "name": "goalkeeper_name",
                "team_name": "team_name_defense",
            }
        ),
        left_on="goalkeeper_id",
        right_on="goalkeeper_person_id",
        how="left",
    )

    return out


def insert_assist_info(
    df_goals: pd.DataFrame,
    df_players: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich goals with assisting player info from players DataFrame.
    """
    out = df_goals.copy()

    # Assisting player lookup (assisting_player_id is mapped_id in detected shots)
    players_lkp = (
        df_players[["person_id", "league_id", "mapped_id", "name"]]
        .sort_values("person_id")
        .drop_duplicates(subset=["mapped_id"], keep="first")
    )

    if "assisting_mapped_id" in out.columns:
        out = out.merge(
            players_lkp.rename(
                columns={
                    "mapped_id": "assisting_mapped_id",
                    "league_id": "assisting_league_id",
                    "person_id": "assisting_person_id",
                    "name": "assisting_name",
                }
            ),
            on="assisting_mapped_id",
            how="left",
        )

    return out


def prepare_positions_for_sync(
    df_positions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare positions DataFrame for shot event synchronization.

    Ensures timestamp_ms is numeric and ts column is present.
    """
    out = df_positions.copy()
    out["timestamp_ms"] = pd.to_numeric(out["timestamp_ms"], errors="coerce")
    out = out.dropna(subset=["timestamp_ms"]).copy()
    out["timestamp_ms"] = out["timestamp_ms"].astype(np.int64)

    if "ts" not in out.columns:
        out["ts"] = pd.to_datetime(
            out["timestamp_ms"], unit="ms", utc=True, errors="coerce"
        )

    out = out.sort_values("timestamp_ms").reset_index(drop=True)
    return out


def insert_goal_position(
    df_goals: pd.DataFrame,
    df_positions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Insert goal position based on defending team and period.
    """
    # Compute goal_position per defense team and period (based on goalkeeper median x per half)
    teams = df_goals["team_name_defense"].dropna().unique().tolist()
    goal_pos_map: Dict[str, Dict[int, Optional[int]]] = {
        team: {1: None, 2: None} for team in teams
    }

    for team in teams:
        gk_ids = (
            df_goals.loc[df_goals["team_name_defense"] == team, "goalkeeper_league_id"]
            .dropna()
            .astype("string")
            .unique()
        )
        if len(gk_ids) == 0:
            continue

        df_p1 = df_goals[
            (df_goals["period_id"] == 1) & (df_goals["team_name_defense"] == team)
        ]
        if df_p1.empty:
            continue

        split_ts_ms = int(df_p1.sort_values("event_time_ms").iloc[-1]["event_time_ms"])

        df_pos_p1 = df_positions[df_positions["timestamp_ms"] <= split_ts_ms]
        df_pos_p2 = df_positions[df_positions["timestamp_ms"] > split_ts_ms]

        # Align types for isin
        df_pos_p1_ids = df_pos_p1["league_id"].astype("string")
        df_pos_p2_ids = df_pos_p2["league_id"].astype("string")

        df_gk_p1 = df_pos_p1[df_pos_p1_ids.isin(gk_ids)]
        if not df_gk_p1.empty:
            goal_pos_map[team][1] = 0 if df_gk_p1["x_m"].median() < 20 else 40

        df_gk_p2 = df_pos_p2[df_pos_p2_ids.isin(gk_ids)]
        if not df_gk_p2.empty:
            goal_pos_map[team][2] = 0 if df_gk_p2["x_m"].median() < 20 else 40

    df_goals["goal_position"] = df_goals.apply(
        lambda r: goal_pos_map.get(r["team_name_defense"], {}).get(r["period_id"]),
        axis=1,
    )
    return df_goals


# -----------------------------------------------------------------------------
# sync_shot_events
# -----------------------------------------------------------------------------
def sync_shot_events(
    df_match_normalized: pd.DataFrame,
    df_match_events_normalized_goals: pd.DataFrame,
    df_match_detected_shots_normalized: pd.DataFrame,
    df_positions_normalized: pd.DataFrame,
    df_players: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sync goal events with detected shots and refine throw timestamps from Kinexon positions.

    Produces a goal-centered DataFrame with:
      - goal metadata
      - shooter and goalkeeper mapped ids / league ids
      - matched detected shot metadata
      - inferred throw timestamp + diagnostics deltas
    """
    # Goal rows
    df_goals = df_match_events_normalized_goals[
        df_match_events_normalized_goals["event_type"] == "goal"
    ].copy()

    if df_goals.empty:
        logging.warning("No goal events found.")
        return df_goals
    else:
        logging.info("Processing %d goal events.", len(df_goals))

    # Normalize time
    df_goals = normalize_time(df_goals)
    # Insert team info (entity_id_home/away, team_name_home/away)
    df_goals = insert_team_info(df_goals, df_match_normalized)
    # Insert player, goalkeeper, and assisting player info
    df_goals = insert_player_goalkeeper_info(df_goals, df_players)

    # Match detected shots to goals
    tol_before_ms = 30_000
    tol_after_ms = 3_000
    df_goals = _sync_goals_to_detected_shots(
        df_goals,
        df_match_detected_shots_normalized,
        tol_before_ms=tol_before_ms,
        tol_after_ms=tol_after_ms,
    )

    # insert assisting player info
    df_goals = insert_assist_info(df_goals, df_players)

    logging.info("Matched detected shots for %d goal rows.", len(df_goals))

    # Positions prep
    df_positions = prepare_positions_for_sync(df_positions_normalized)

    # Insert goal position
    df_goals = insert_goal_position(df_goals, df_positions)

    # Refine throw timestamps
    frame_tol_ms = 50
    d_possess = 1.5
    df_throw_events = _refine_throw_times(
        df_goals=df_goals,
        df_positions=df_positions,
        tol_before_ms=tol_before_ms,
        tol_after_ms=tol_after_ms,
        frame_tol_ms=frame_tol_ms,
        d_possess=d_possess,
        debug_plot=False,
        debug_plot_max=15,
    )
    if "event_id" not in df_throw_events.columns:
        logging.warning("No throw timestamps detected.")
        df_goals["throw_timestamp_ms"] = pd.NA
        df_goals["throw_ts"] = pd.NaT
        df_goals["throw_acceleration"] = pd.NA
        df_goals["method"] = pd.NA
        df_goals["detected_events_shot_time"] = pd.NA
        df_goals["time_difference_ms"] = pd.NA
        df_goals["match_method"] = pd.NA
    else:
        logging.info(
            "Refined throw timestamps for %d goal rows.",
            len(df_throw_events),
        )
        df_goals = df_goals.merge(df_throw_events, on="event_id", how="left")

    # Deltas
    if (
        "detected_events_shot_time" in df_goals.columns
        and "throw_timestamp_ms" in df_goals.columns
        and not df_goals["detected_events_shot_time"].isna().all()
    ):
        df_goals["time_diff_detected_shot_throw_ms"] = (
            df_goals["detected_events_shot_time"]
            .astype("datetime64[ms, UTC]")
            .astype("int64")
        ) - df_goals["throw_timestamp_ms"]

    if "event_time_ms" in df_goals.columns and "throw_timestamp_ms" in df_goals.columns:
        df_goals["time_diff_event_throw_ms"] = (
            df_goals["event_time_ms"] - df_goals["throw_timestamp_ms"]
        )

    return df_goals


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------
def render_shot_event(
    df_goals: pd.DataFrame,
    df_positions: pd.DataFrame,
    *,
    max_events: int = 5,
    require_throw_ts: bool = True,
    event_time_col: str = "event_time",
    throw_ts_col: str = "throw_ts",
    detected_shot_ts_col: str = "detected_events_shot_time",
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    if df_goals.empty:
        logging.warning("render_shot_event: df_goals is empty.")
        return df_goals.head(0)

    if df_positions.empty:
        logging.warning("render_shot_event: df_positions is empty.")
        return df_goals.head(0)

    df = df_goals.copy()

    if require_throw_ts and throw_ts_col in df.columns:
        df = df[df[throw_ts_col].notna()]

    if df.empty:
        logging.warning("No events with usable throw timestamps.")
        return df.head(0)

    sort_cols = [c for c in [event_time_col, throw_ts_col] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    df = df.head(max_events)

    rendered_records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        event_id = row.get("event_id")
        logging.info("Rendering shot event %s", event_id)

        freeze_markers: List[Dict[str, Any]] = []

        if event_time_col in row and pd.notna(row[event_time_col]):
            freeze_markers.append(
                {
                    "name": "SR event_time",
                    "ts": row[event_time_col],
                    "color": (0, 0, 255),
                    "seconds": 0.5,
                }
            )

        if detected_shot_ts_col in row and pd.notna(row[detected_shot_ts_col]):
            freeze_markers.append(
                {
                    "name": "Detected shot",
                    "ts": row[detected_shot_ts_col],
                    "color": (0, 255, 255),
                    "seconds": 0.5,
                }
            )

        if throw_ts_col in row and pd.notna(row[throw_ts_col]):
            freeze_markers.append(
                {
                    "name": "Throw / release",
                    "ts": row[throw_ts_col],
                    "color": (0, 255, 0),
                    "seconds": 1.0,
                }
            )

        if not freeze_markers:
            logging.warning("Event %s: no freeze markers, skipping.", event_id)
            continue

        out_path = render_goal_with_multifreeze(
            df_positions=df_positions,
            row_goal=row,
            freeze_markers=freeze_markers,
            out_dir=output_dir,
            show=True,
        )

        if out_path is None:
            logging.warning("Rendering failed for event %s", event_id)
            continue

        rendered_records.append(
            {
                "event_id": event_id,
                "video_path": str(out_path),
                "preview_image_path": str(out_path.with_suffix(".png")),
            }
        )

    if not rendered_records:
        logging.warning("No renders produced.")
        return df.head(0)

    return pd.DataFrame(rendered_records)


if __name__ == "__main__":
    import duckdb

    logging.basicConfig(level=logging.INFO)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fixture_id = "02ea68cb-5ca6-11f0-a329-51a43a77a48d"
    con_duckdb = "./data/hbl_raw.duckdb"
    # Tables to load:
    # matches_normalized
    # match_events_normalized_goals
    # match_detected_shots_normalized
    # positions_normalized
    # players
    db = duckdb.connect(con_duckdb)
    df_matches = db.execute(
        """
        SELECT *
        FROM matches_normalized
        """
    ).df()

    list_fixture_ids = (
        df_matches.sort_values(by="start_time_local", ascending=True)
        .loc[:, "fixture_id"]
        .unique()
        .tolist()
    )

    for fixture_id in list_fixture_ids:
        logging.info("Processing fixture_id=%s", fixture_id)

        df_match = df_matches[df_matches["fixture_id"] == fixture_id].copy()

        # log info about the match
        if df_match.empty:
            logging.warning("No match data for fixture_id=%s, skipping.", fixture_id)
            continue
        else:
            logging.info(
                "\n\n\nMatch info: %s vs %s on %s",
                df_match.iloc[0]["team_name_home"],
                df_match.iloc[0]["team_name_away"],
                df_match.iloc[0]["start_time_local"],
            )

        df_goals = db.execute(
            f"""
            SELECT *
            FROM match_events_normalized_goals
            WHERE fixture_id = '{fixture_id}'
            AND event_type = 'goal'
            """
        ).df()

        df_detected_shots = db.execute(
            f"""
            SELECT *
            FROM match_detected_shots_normalized
            WHERE fixture_id = '{fixture_id}'
            """
        ).df()
        df_positions = db.execute(
            f"""
            SELECT *
            FROM match_positions_normalized
            WHERE fixture_id = '{fixture_id}'
            """
        ).df()
        df_players = db.execute(
            f"""
            SELECT *
            FROM players
            WHERE fixture_id = '{fixture_id}'
            """
        ).df()

        # check if positions data is available
        if df_positions.empty:
            logging.warning(
                "No positions data for fixture_id=%s, skipping.", fixture_id
            )
            continue

        df_synced = sync_shot_events(
            df_match_normalized=df_matches,
            df_match_events_normalized_goals=df_goals,
            df_match_detected_shots_normalized=df_detected_shots,
            df_positions_normalized=df_positions,
            df_players=df_players,
        )

        # df_renders = render_shot_event(
        #     df_goals=df_synced,
        #     df_positions=df_positions,
        #     max_events=5,
        #     require_throw_ts=False,
        #     output_dir=OUT_DIR,
        # )

        # log coverage of synced events
        n_synced = df_synced["throw_timestamp_ms"].notna().sum()
        logging.info(
            "Number of synced events with throw timestamp: %d of total %d",
            n_synced,
            len(df_synced),
        )
        logging.info(
            "Coverage: %.2f%%",
            (n_synced / len(df_synced)) * 100.0 if len(df_synced) > 0 else 0.0,
        )
        # coverage of goalkeeper_league_id
        n_gk_league_id = df_synced["goalkeeper_league_id"].notna().sum()
        logging.info(
            "Number of events with goalkeeper_league_id: %d of total %d",
            n_gk_league_id,
            len(df_synced),
        )
        # time difference of event_time_ms - throw_timestamp_ms
        if "time_diff_event_throw_ms" in df_synced.columns:
            diffs = df_synced["time_diff_event_throw_ms"].dropna().astype(float)
            if not diffs.empty:
                logging.info(
                    "Time difference event_time_ms - throw_timestamp_ms: mean=%.2f ms, median=%.2f ms, std=%.2f ms",
                    diffs.mean(),
                    diffs.median(),
                    diffs.std(),
                )
