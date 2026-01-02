# time_sync.py
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def _merge_closest_time(
    goals: pd.DataFrame,
    kinexon: pd.DataFrame,
    tolerance_ms: int,
    *,
    goal_time_col: str = "event_time_ms",
    kin_time_col: str = "timestamp_ms",
    kin_prefix: str = "kin_",
) -> Optional[pd.DataFrame]:
    """
    Row-preserving 1D time matching (nearest neighbor in time).

    - Returns the same number of rows as `goals`.
    - Adds:
        - kinexon_match_index (int, -1 if unmatched)
        - time_diff_ms (float, NaN if unmatched)  [signed: kin - goal]
        - matched (bool)
        - all kinexon columns prefixed with `kin_prefix` (NaN if unmatched)

    If `goals` is empty, returns an empty DataFrame.
    If `kinexon` is empty, returns `goals` with match metadata (no kin columns).
    """
    if goals is None or goals.empty:
        return pd.DataFrame()

    out = goals.copy()

    # Ensure goal times numeric
    out[goal_time_col] = pd.to_numeric(out[goal_time_col], errors="coerce")

    if kinexon is None or kinexon.empty:
        out["kinexon_match_index"] = -1
        out["time_diff_ms"] = np.nan
        out["matched"] = False
        return out

    kin = kinexon.copy()
    kin[kin_time_col] = pd.to_numeric(kin[kin_time_col], errors="coerce")
    kin = kin.dropna(subset=[kin_time_col]).copy()

    if kin.empty:
        out["kinexon_match_index"] = -1
        out["time_diff_ms"] = np.nan
        out["matched"] = False
        return out

    # Sort Kinexon by time for deterministic behavior
    kin = kin.sort_values(kin_time_col, kind="mergesort").reset_index(drop=True)
    kin_times = kin[kin_time_col].to_numpy(dtype="int64")

    goal_times = out[goal_time_col].to_numpy()
    valid_goal_mask = ~np.isnan(goal_times)

    # Default outputs: unmatched
    match_idx = np.full(len(out), -1, dtype="int64")
    time_diff = np.full(len(out), np.nan, dtype="float64")

    if valid_goal_mask.any():
        gt = goal_times[valid_goal_mask].astype("int64")

        idx_right = np.searchsorted(kin_times, gt, side="left")
        idx_left = idx_right - 1

        inf = np.iinfo(np.int64).max
        delta_right = np.full(len(gt), inf, dtype="int64")
        delta_left = np.full(len(gt), inf, dtype="int64")

        r_ok = idx_right < len(kin_times)
        l_ok = idx_left >= 0

        delta_right[r_ok] = kin_times[idx_right[r_ok]] - gt[r_ok]
        delta_left[l_ok] = kin_times[idx_left[l_ok]] - gt[l_ok]

        # Choose nearer; tie-break to LEFT (earlier kin timestamp) for determinism
        choose_right = np.abs(delta_right) < np.abs(delta_left)
        chosen = np.where(choose_right, idx_right, idx_left)
        chosen_delta = np.where(choose_right, delta_right, delta_left)

        within_tol = np.abs(chosen_delta) <= tolerance_ms
        chosen = np.where(within_tol, chosen, -1)
        chosen_delta = np.where(within_tol, chosen_delta, np.nan)

        match_idx[valid_goal_mask] = chosen.astype("int64")
        time_diff[valid_goal_mask] = chosen_delta.astype("float64")

    out["kinexon_match_index"] = match_idx
    out["time_diff_ms"] = time_diff
    out["matched"] = match_idx != -1

    # Attach kinexon columns (NA for unmatched) with prefix
    kin_pref = kin.add_prefix(kin_prefix)

    kin_pref = kin.add_prefix(kin_prefix).convert_dtypes()

    # Schema-first: correct dtypes from the start
    kin_attached = pd.DataFrame(
        index=out.index,
        columns=kin_pref.columns,
    ).astype(kin_pref.dtypes.to_dict())

    matched_mask = match_idx != -1
    if matched_mask.any():
        kin_attached.loc[matched_mask, :] = kin_pref.iloc[
            match_idx[matched_mask]
        ].to_numpy()

    return pd.concat(
        [out.reset_index(drop=True), kin_attached.reset_index(drop=True)],
        axis=1,
    )


def sync_goals_with_kinexon(
    df_events: pd.DataFrame,
    df_kinexon: pd.DataFrame,
    tolerance_ms: int = 30_000,
    *,
    goal_time_col: str = "event_time_ms",
    kin_time_col: str = "timestamp_ms",
    goal_player_col: str = "person_league_id",
    kin_player_col: str = "league_id",
) -> pd.DataFrame:
    """
    Row-preserving goal -> Kinexon sync for already fixture-filtered inputs.

    Strategy:
    1) If player columns exist: try player-restricted time match first.
    2) For rows still unmatched: fall back to time-only match against all Kinexon.
    3) Always returns len(output) == len(df_events) (unless df_events is empty).

    Adds a `match_mode` column:
      - "player+time"
      - "time_only"
      - "time_only_fallback"
      - "no_kinexon"
    """
    if df_events is None or df_events.empty:
        return pd.DataFrame()

    if df_kinexon is None or df_kinexon.empty:
        out = df_events.copy()
        out["kinexon_match_index"] = -1
        out["time_diff_ms"] = np.nan
        out["matched"] = False
        out["match_mode"] = "no_kinexon"
        return out

    ev = df_events.copy().reset_index(drop=False).rename(columns={"index": "__row"})
    kin = df_kinexon.copy()

    has_player_matching = (goal_player_col in ev.columns) and (
        kin_player_col in kin.columns
    )

    # Fast path: no player matching available -> time-only for all rows
    if not has_player_matching:
        out = _merge_closest_time(
            ev.drop(columns=["__row"]),
            kin,
            tolerance_ms,
            goal_time_col=goal_time_col,
            kin_time_col=kin_time_col,
        )
        out.insert(0, "__row", ev["__row"].to_numpy())
        out["match_mode"] = np.where(out["matched"], "time_only", "time_only")
        return (
            out.sort_values("__row", kind="mergesort")
            .drop(columns=["__row"])
            .reset_index(drop=True)
        )

    parts: List[pd.DataFrame] = []

    # groupby(dropna=False) to avoid silently dropping NaN player ids
    for pid, goals_grp in ev.groupby(goal_player_col, dropna=False, sort=False):
        goals_base = goals_grp.drop(columns=["__row"]).reset_index(drop=True)

        if pd.isna(pid):
            # No shooter id -> time-only
            merged = _merge_closest_time(
                goals_base,
                kin,
                tolerance_ms,
                goal_time_col=goal_time_col,
                kin_time_col=kin_time_col,
            )
            merged["match_mode"] = np.where(merged["matched"], "time_only", "time_only")
        else:
            kin_sub = kin[kin[kin_player_col] == pid]
            if kin_sub.empty:
                # No kinexon rows for this player -> time-only
                merged = _merge_closest_time(
                    goals_base,
                    kin,
                    tolerance_ms,
                    goal_time_col=goal_time_col,
                    kin_time_col=kin_time_col,
                )
                merged["match_mode"] = np.where(
                    merged["matched"], "time_only", "time_only"
                )
            else:
                # First pass: strict player+time
                merged_player = _merge_closest_time(
                    goals_base,
                    kin_sub,
                    tolerance_ms,
                    goal_time_col=goal_time_col,
                    kin_time_col=kin_time_col,
                )
                merged_player["match_mode"] = np.where(
                    merged_player["matched"],
                    "player+time",
                    "time_only_fallback",
                )

                # Second pass: for unmatched rows, fall back to time-only on full kinexon
                unmatched_mask = ~merged_player["matched"]
                if unmatched_mask.any():
                    fallback = _merge_closest_time(
                        goals_base.loc[unmatched_mask].copy(),
                        kin,
                        tolerance_ms,
                        goal_time_col=goal_time_col,
                        kin_time_col=kin_time_col,
                    )
                    fallback["match_mode"] = np.where(
                        fallback["matched"],
                        "time_only_fallback",
                        "time_only_fallback",
                    )

                    # Replace match-related columns for unmatched rows
                    replace_cols = [
                        "kinexon_match_index",
                        "time_diff_ms",
                        "matched",
                    ] + [c for c in fallback.columns if c.startswith("kin_")]
                    idx = merged_player.index[unmatched_mask]
                    rhs = fallback[replace_cols].copy()
                    rhs.index = idx
                    merged_player.loc[idx, replace_cols] = rhs
                    merged_player.loc[idx, "match_mode"] = fallback[
                        "match_mode"
                    ].to_numpy()

                merged = merged_player

        merged.insert(0, "__row", goals_grp["__row"].to_numpy())
        parts.append(merged)

    out = pd.concat(parts, ignore_index=True)
    out = (
        out.sort_values("__row", kind="mergesort")
        .drop(columns=["__row"])
        .reset_index(drop=True)
    )

    # Hard invariant: row-preserving
    if len(out) != len(df_events):
        raise RuntimeError(f"Row-count changed: in={len(df_events)} out={len(out)}")

    return out
