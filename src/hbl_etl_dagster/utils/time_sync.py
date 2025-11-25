import pandas as pd
import numpy as np
from typing import Optional


def _merge_closest_time(
    goals: pd.DataFrame,
    kinexon: pd.DataFrame,
    tolerance_ms: int,
) -> Optional[pd.DataFrame]:
    """
    Helper to perform 1D time matching between two dataframes.
    """
    if kinexon.empty or goals.empty:
        return None

    # Sort Kinexon by time
    kinexon = kinexon.sort_values("timestamp_ms").reset_index(drop=True)
    kinexon_times = kinexon["timestamp_ms"].to_numpy(dtype="int64")
    goal_times = goals["event_time_ms"].to_numpy(dtype="int64")

    # Find insertion points
    idx_right = np.searchsorted(kinexon_times, goal_times, side="left")
    idx_left = (idx_right - 1).clip(min=0)

    best_indices = []
    best_deltas = []

    len_kin = len(kinexon_times)

    for i, goal_time in enumerate(goal_times):
        best_idx, best_delta = -1, float("inf")

        r = idx_right[i]
        l = idx_left[i]

        # Check right
        if r < len_kin:
            delta = kinexon_times[r] - goal_time
            if abs(delta) < abs(best_delta):
                best_delta = delta
                best_idx = r

        # Check left
        if l < len_kin:
            delta = kinexon_times[l] - goal_time
            if abs(delta) < abs(best_delta):
                best_delta = delta
                best_idx = l

        if abs(best_delta) <= tolerance_ms:
            best_indices.append(best_idx)
            best_deltas.append(int(best_delta))
        else:
            best_indices.append(None)
            best_deltas.append(None)

    matched = goals.copy()
    matched["kinexon_match_index"] = best_indices
    matched["time_diff_ms"] = best_deltas

    matched = matched.dropna(subset=["kinexon_match_index"])
    if matched.empty:
        return None

    matched["kinexon_match_index"] = matched["kinexon_match_index"].astype(int)

    # Fetch kinexon rows
    kinexon_subset = kinexon.iloc[matched["kinexon_match_index"]].add_prefix(
        "kin_"
    )

    return pd.concat(
        [
            matched.reset_index(drop=True),
            kinexon_subset.reset_index(drop=True),
        ],
        axis=1,
    )


def sync_goals_with_kinexon(
    df_events: pd.DataFrame,
    df_kinexon: pd.DataFrame,
    tolerance_ms: int = 30_000,
) -> pd.DataFrame:
    """
    Match Sportradar goal events to nearest Kinexon events by fixture and timestamp.
    Enforces strict player ID matching if 'person_league_id' and 'league_id' are present.

    Args:
        df_events: Sportradar events DataFrame (must contain 'eventTime_ms', 'fixtureId').
        df_kinexon: Kinexon events DataFrame (must contain 'timestamp_ms', 'fixture_id').
        tolerance_ms: Maximum allowed time difference in milliseconds.

    Returns:
        DataFrame containing goals merged with the nearest Kinexon event data.
    """
    if df_events.empty or df_kinexon.empty:
        return pd.DataFrame()

    all_synced_fixtures = []

    # Ensure types for matching
    df_events = df_events.copy()
    df_kinexon = df_kinexon.copy()

    # Ensure fixture IDs are strings for consistent grouping
    df_events["fixture_id"] = df_events["fixture_id"].astype(str)
    df_kinexon["fixture_id"] = df_kinexon["fixture_id"].astype(str)

    for fixture_id, goals_grp in df_events.groupby("fixture_id"):
        kinexon_grp = df_kinexon[df_kinexon["fixture_id"] == fixture_id]
        if kinexon_grp.empty:
            continue

        # Strict matching by player if columns exist
        if (
            "person_league_id" in goals_grp.columns
            and "league_id" in kinexon_grp.columns
        ):
            # Iterate over each player who scored
            for league_id, player_goals in goals_grp.groupby(
                "person_league_id"
            ):
                player_kinexon = kinexon_grp[
                    kinexon_grp["league_id"] == league_id
                ]
                merged = _merge_closest_time(
                    player_goals, player_kinexon, tolerance_ms
                )
                if merged is not None:
                    all_synced_fixtures.append(merged)
        else:
            # Fallback: match purely by time within fixture (legacy behavior)
            merged = _merge_closest_time(goals_grp, kinexon_grp, tolerance_ms)
            if merged is not None:
                all_synced_fixtures.append(merged)

    if not all_synced_fixtures:
        return pd.DataFrame()

    return pd.concat(all_synced_fixtures, ignore_index=True)
