# features_xs.py
import difflib
import logging
from typing import List

import numpy as np
import pandas as pd


def _positions_for_timestamp(
    positions_by_timestamp: pd.DataFrame, timestamp_ms: object
) -> pd.DataFrame:
    try:
        df_positions_shot = positions_by_timestamp.loc[timestamp_ms]
    except KeyError:
        return positions_by_timestamp.iloc[0:0].copy()

    if isinstance(df_positions_shot, pd.Series):
        return df_positions_shot.to_frame().T

    return df_positions_shot


def calculate_xs_features(
    df_match_normalized: pd.DataFrame,
    df_shot_events: pd.DataFrame,
    df_positions_normalized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate goalkeeper-centric features for xS (expected save).

    Assumptions:
    - Only ON-TARGET shots are considered
    - target = 1 means SAVE, 0 means GOAL
    - Features are goalkeeper- and ball-centric (no shooter leakage)
    """

    features: List[dict] = []

    dropped_no_ball = 0
    dropped_no_gk = 0

    position_cols = ["timestamp_ms", "group_name", "league_id", "x_m", "y_m"]
    available_position_cols = [
        col for col in position_cols if col in df_positions_normalized.columns
    ]
    df_positions_view = df_positions_normalized[available_position_cols].copy()

    # Remap Kinexon group_name → Sportradar team name (same fix as calc_xg_features).
    if "group_name" in df_positions_view.columns:
        sportradar_teams = set()
        for col in ("team_name_offense", "team_name_defense", "team_name_home"):
            if col in df_shot_events.columns:
                sportradar_teams.update(df_shot_events[col].dropna().unique())
        sportradar_teams.discard(None)
        kinexon_groups = [
            g
            for g in df_positions_view["group_name"].dropna().unique()
            if "ball" not in str(g).lower()
        ]
        group_name_map: dict[str, str] = {}
        for kg in kinexon_groups:
            matches = difflib.get_close_matches(
                kg, list(sportradar_teams), n=1, cutoff=0.5
            )
            group_name_map[kg] = matches[0] if matches else kg
        if group_name_map:
            df_positions_view["group_name"] = df_positions_view["group_name"].map(
                lambda g: group_name_map.get(g, g)
            )
            for kg, sr in group_name_map.items():
                if kg != sr:
                    logging.getLogger(__name__).info(
                        "group_name remapped: %r → %r", kg, sr
                    )

    if "timestamp_ms" in df_positions_view.columns:
        df_positions_view["timestamp_ms"] = pd.to_numeric(
            df_positions_view["timestamp_ms"], errors="coerce"
        )
        positions_by_timestamp = df_positions_view.set_index("timestamp_ms", drop=False)
    else:
        positions_by_timestamp = df_positions_view

    for _, shot in df_shot_events.iterrows():
        # --- xS only defined for on-target shots ---
        if shot.get("on_target") is False:
            continue

        timestamp_ms = shot["throw_timestamp_ms"]
        fixture_id = shot["fixture_id"]
        event_id = shot["event_id"]

        # --- snapshot positions at shot time ---
        df_pos = _positions_for_timestamp(positions_by_timestamp, timestamp_ms)

        if df_pos.empty:
            dropped_no_ball += 1
            dropped_no_gk += 1
            continue

        # --- defense team positions ---
        df_defense = df_pos[df_pos["group_name"] == shot["team_name_defense"]]

        # --- ball position ---
        df_ball = df_pos[df_pos["league_id"].str.contains("ball", case=False, na=False)]

        # --- goalkeeper position ---
        df_gk = df_defense[df_defense["league_id"] == shot["goalkeeper_league_id"]]

        if df_ball.empty:
            dropped_no_ball += 1
            continue

        if df_gk.empty:
            dropped_no_gk += 1
            continue

        # --- extract coordinates ---
        ball_x, ball_y = df_ball[["x_m", "y_m"]].iloc[0]
        gk_x, gk_y = df_gk[["x_m", "y_m"]].iloc[0]

        goal_x = shot["goal_position"]
        goal_y = 10.0  # handball goal center

        # --- geometric features ---
        goalkeeper_distance_to_goal = np.hypot(gk_x - goal_x, gk_y - goal_y)

        ball_distance_to_goalkeeper = np.hypot(ball_x - gk_x, ball_y - gk_y)

        ball_distance_to_goal = np.hypot(ball_x - goal_x, ball_y - goal_y)

        # --- target: 1 = save, 0 = goal ---
        target = int(shot["success"] == 0)

        features.append(
            {
                "fixture_id": fixture_id,
                "event_id": event_id,
                "goalkeeper_distance_to_goal": goalkeeper_distance_to_goal,
                "ball_distance_to_goalkeeper": ball_distance_to_goalkeeper,
                "ball_distance_to_goal": ball_distance_to_goal,
                "target": target,
            }
        )

    df_features = pd.DataFrame(features)

    # --- enforce numeric hygiene ---
    numeric_cols = [
        "goalkeeper_distance_to_goal",
        "ball_distance_to_goalkeeper",
        "ball_distance_to_goal",
    ]
    if not df_features.empty:
        df_features[numeric_cols] = df_features[numeric_cols].astype("float64")

    logging.info(
        "xS feature extraction complete: %d rows | dropped_no_ball=%d | dropped_no_gk=%d",
        len(df_features),
        dropped_no_ball,
        dropped_no_gk,
    )

    return df_features
