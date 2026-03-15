import difflib
import logging

import numpy as np
import pandas as pd

GOAL_WIDTH = 3.0
HALF_GOAL = GOAL_WIDTH / 2.0


def shot_angle_to_goal(
    shooter_x: float, shooter_y: float, goal_x: float, goal_y: float = 10.0
) -> float:
    left_post = (goal_x, goal_y - HALF_GOAL)
    right_post = (goal_x, goal_y + HALF_GOAL)

    angle_left = np.arctan2(left_post[1] - shooter_y, left_post[0] - shooter_x)
    angle_right = np.arctan2(right_post[1] - shooter_y, right_post[0] - shooter_x)
    return float(abs(angle_right - angle_left))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0:
        return np.nan
    c = np.clip(float(np.dot(v1, v2) / n), -1.0, 1.0)
    return float(np.arccos(c))


def point_to_segment_distance(px, py, ax, ay, bx, by) -> float:
    """Distance from P to segment AB."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom == 0:
        return float(np.hypot(apx, apy))
    t = (apx * abx + apy * aby) / denom
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    return float(np.hypot(px - cx, py - cy))


def projection_along_ray(px, py, ax, ay, bx, by) -> float:
    """
    Scalar projection of AP onto AB (ray direction). Positive means "in front" of A towards B.
    Not clipped to segment length.
    """
    ab = np.array([bx - ax, by - ay], dtype=float)
    ap = np.array([px - ax, py - ay], dtype=float)
    denom = float(np.linalg.norm(ab))
    if denom == 0:
        return np.nan
    return float(np.dot(ap, ab) / denom)


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


def calculate_xg_features(
    df_match_normalized: pd.DataFrame,
    df_shot_events: pd.DataFrame,
    df_positions_normalized: pd.DataFrame,
    goal_y: float = 10.0,
    cone_half_angle_deg: float = 20.0,
) -> pd.DataFrame:
    """
    Adds obstruction + GK coverage + defensive compactness features.
    Assumes df_positions_normalized already corresponds to the fixture passed in.
    """
    attendance = df_match_normalized.get("attendance", pd.Series([np.nan])).iloc[0]
    attendance = int(attendance) if pd.notna(attendance) else None

    cone_half_angle = np.deg2rad(cone_half_angle_deg)

    position_cols = ["timestamp_ms", "group_name", "league_id", "x_m", "y_m"]
    available_position_cols = [
        col for col in position_cols if col in df_positions_normalized.columns
    ]
    df_positions_view = df_positions_normalized[available_position_cols].copy()

    # Remap Kinexon group_name → Sportradar team name.
    # The two sources use different casing/punctuation (e.g. "Frisch Auf! Göppingen" vs
    # "FRISCH AUF Göppingen"), so an exact match in the per-shot loop always fails for
    # mismatched teams. Build a one-time fuzzy mapping using the team names that actually
    # appear in shot_events.
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
            matches = difflib.get_close_matches(kg, sportradar_teams, n=1, cutoff=0.5)
            if matches:
                group_name_map[kg] = matches[0]
            else:
                group_name_map[kg] = kg
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

    features_list = []
    for _, shot in df_shot_events.iterrows():
        event_id = shot["event_id"]
        timestamp_ms = shot["throw_timestamp_ms"]
        team_name_offense = shot["team_name_offense"]
        team_name_defense = shot["team_name_defense"]

        fixture_id = shot["fixture_id"]
        is_home_team = team_name_offense == shot.get("team_name_home")

        # positions at timestamp
        df_positions_shot = _positions_for_timestamp(
            positions_by_timestamp, timestamp_ms
        )
        df_offense = df_positions_shot[
            df_positions_shot["group_name"] == team_name_offense
        ]
        df_defense = df_positions_shot[
            df_positions_shot["group_name"] == team_name_defense
        ]

        goal_x = shot["goal_position"]

        # offense/defense avg distances
        offense_distances = np.hypot(
            df_offense["x_m"] - goal_x, df_offense["y_m"] - goal_y
        )
        defense_distances = np.hypot(
            df_defense["x_m"] - goal_x, df_defense["y_m"] - goal_y
        )
        avg_offense_distance = (
            float(offense_distances.mean()) if len(offense_distances) else np.nan
        )
        avg_defense_distance = (
            float(defense_distances.mean()) if len(defense_distances) else np.nan
        )

        # shooter position
        shooter_id = shot["person_league_id"]
        df_shooter = df_offense[df_offense["league_id"] == shooter_id]
        if not df_shooter.empty:
            shooter_x = float(df_shooter["x_m"].iloc[0])
            shooter_y = float(df_shooter["y_m"].iloc[0])
        else:
            shooter_x = np.nan
            shooter_y = np.nan

        # goalkeeper position
        df_goalkeeper = df_defense[
            df_defense["league_id"] == shot.get("goalkeeper_league_id")
        ]
        if not df_goalkeeper.empty:
            goalkeeper_x = float(df_goalkeeper["x_m"].iloc[0])
            goalkeeper_y = float(df_goalkeeper["y_m"].iloc[0])
        else:
            goalkeeper_x = np.nan
            goalkeeper_y = np.nan

        # ball position (optional)
        df_ball = df_positions_shot[
            df_positions_shot["league_id"]
            .astype(str)
            .str.contains("ball", case=False, na=False)
        ]
        if not df_ball.empty:
            ball_x = float(df_ball["x_m"].iloc[0])
            ball_y = float(df_ball["y_m"].iloc[0])
        else:
            ball_x = np.nan
            ball_y = np.nan

        # geometry core
        shooter_distance_to_goal = (
            float(np.hypot(shooter_x - goal_x, shooter_y - goal_y))
            if np.isfinite(shooter_x)
            else np.nan
        )
        shot_angle = (
            shot_angle_to_goal(shooter_x, shooter_y, goal_x, goal_y)
            if np.isfinite(shooter_x)
            else np.nan
        )
        shooter_lateral_offset = (
            float(abs(shooter_y - goal_y)) if np.isfinite(shooter_y) else np.nan
        )

        # components (helpful for wings/side)
        shooter_to_goal_dx = (
            float(goal_x - shooter_x) if np.isfinite(shooter_x) else np.nan
        )
        shooter_to_goal_dy = (
            float(goal_y - shooter_y) if np.isfinite(shooter_y) else np.nan
        )
        shooter_y_signed = (
            float(shooter_y - goal_y) if np.isfinite(shooter_y) else np.nan
        )
        shooter_distance_to_goal_sq = (
            float(shooter_distance_to_goal**2)
            if np.isfinite(shooter_distance_to_goal)
            else np.nan
        )

        # GK features
        shooter_distance_to_goalkeeper = (
            float(np.hypot(shooter_x - goalkeeper_x, shooter_y - goalkeeper_y))
            if np.isfinite(shooter_x) and np.isfinite(goalkeeper_x)
            else np.nan
        )
        goalkeeper_distance_to_goal = (
            float(np.hypot(goalkeeper_x - goal_x, goalkeeper_y - goal_y))
            if np.isfinite(goalkeeper_x)
            else np.nan
        )
        gk_lateral_offset = (
            float(abs(goalkeeper_y - goal_y)) if np.isfinite(goalkeeper_y) else np.nan
        )

        # angle between shooter->goal and shooter->GK
        if np.isfinite(shooter_x) and np.isfinite(goalkeeper_x):
            v_goal = np.array([goal_x - shooter_x, goal_y - shooter_y], dtype=float)
            v_gk = np.array(
                [goalkeeper_x - shooter_x, goalkeeper_y - shooter_y],
                dtype=float,
            )
            gk_angle_from_shooter = angle_between(v_goal, v_gk)
            gk_along_shotline_dist = projection_along_ray(
                goalkeeper_x,
                goalkeeper_y,
                shooter_x,
                shooter_y,
                goal_x,
                goal_y,
            )
        else:
            gk_angle_from_shooter = np.nan
            gk_along_shotline_dist = np.nan

        # Ball features (keep your originals but robust)
        if np.isfinite(ball_x):
            ball_distance_to_goal = float(np.hypot(ball_x - goal_x, ball_y - goal_y))
            ball_angle = shot_angle_to_goal(ball_x, ball_y, goal_x, goal_y)
            if np.isfinite(goalkeeper_x):
                ball_distance_to_goalkeeper = float(
                    np.hypot(ball_x - goalkeeper_x, ball_y - goalkeeper_y)
                )
                angle_ball_gk = angle_between(
                    np.array([goal_x - ball_x, goal_y - ball_y], dtype=float),
                    np.array(
                        [goalkeeper_x - ball_x, goalkeeper_y - ball_y],
                        dtype=float,
                    ),
                )
            else:
                ball_distance_to_goalkeeper = np.nan
                angle_ball_gk = np.nan
        else:
            ball_distance_to_goal = np.nan
            ball_angle = np.nan
            ball_distance_to_goalkeeper = np.nan
            angle_ball_gk = np.nan

        # Defender proximity (keep yours)
        if np.isfinite(shooter_x) and not df_defense.empty:
            d_shooter = np.hypot(
                df_defense["x_m"] - shooter_x, df_defense["y_m"] - shooter_y
            )
            num_defenders_close = int((d_shooter <= 2.0).sum())
            closest_defender_distance = float(d_shooter.min())
        else:
            num_defenders_close = np.nan
            closest_defender_distance = np.nan

        # NEW: shot-line obstruction features
        if np.isfinite(shooter_x) and not df_defense.empty:
            # distance of each defender to the shooter->goal segment
            dist_to_line = []
            along_line = []
            in_cone = 0

            v_goal = np.array([goal_x - shooter_x, goal_y - shooter_y], dtype=float)
            for _, d in df_defense.iterrows():
                dx = float(d["x_m"])
                dy = float(d["y_m"])

                # distance to segment shooter->goal
                dist = point_to_segment_distance(
                    dx, dy, shooter_x, shooter_y, goal_x, goal_y
                )
                dist_to_line.append(dist)

                # projection distance along ray shooter->goal
                along = projection_along_ray(
                    dx, dy, shooter_x, shooter_y, goal_x, goal_y
                )
                along_line.append(along)

                # cone membership (directional obstruction)
                v_def = np.array([dx - shooter_x, dy - shooter_y], dtype=float)
                ang = angle_between(v_goal, v_def)
                if np.isfinite(ang) and ang <= cone_half_angle and along > 0:
                    in_cone += 1

            min_defender_dist_to_shotline = (
                float(np.min(dist_to_line)) if dist_to_line else np.nan
            )
            # only defenders in front of shooter
            along_pos = [a for a in along_line if np.isfinite(a) and a > 0]
            min_defender_along_shotline = (
                float(np.min(along_pos)) if along_pos else np.nan
            )
            n_defenders_in_shot_cone = int(in_cone)
        else:
            min_defender_dist_to_shotline = np.nan
            min_defender_along_shotline = np.nan
            n_defenders_in_shot_cone = np.nan

        # NEW: defensive compactness
        if not df_defense.empty:
            cx = float(df_defense["x_m"].mean())
            cy = float(df_defense["y_m"].mean())
            defense_centroid_dist_to_goal = float(np.hypot(cx - goal_x, cy - goal_y))
            defense_spread = float(
                np.hypot(df_defense["x_m"] - cx, df_defense["y_m"] - cy).mean()
            )
        else:
            defense_centroid_dist_to_goal = np.nan
            defense_spread = np.nan

        # categoricals
        attack_type = shot.get("attack_type")
        sub_type = shot.get("sub_type")
        # keep unknowns as "OTHER" instead of None (information-preserving)
        allowed = {"PIVOT", "BREAK_THROUGH", "FAST_BREAK"}
        attack_type = attack_type if attack_type in allowed else "OTHER"
        sub_type = sub_type if pd.notna(sub_type) else "OTHER"

        target = int(shot["success"])

        features_list.append(
            {
                "fixture_id": fixture_id,
                "event_id": event_id,
                # existing
                "avg_offense_distance_to_goal": avg_offense_distance,
                "avg_defense_distance_to_goal": avg_defense_distance,
                "shooter_distance_to_goal": shooter_distance_to_goal,
                "shooter_distance_to_goalkeeper": shooter_distance_to_goalkeeper,
                "goalkeeper_distance_to_goal": goalkeeper_distance_to_goal,
                "ball_distance_to_goal": ball_distance_to_goal,
                "ball_distance_to_goalkeeper": ball_distance_to_goalkeeper,
                "shot_angle_to_goal": shot_angle,
                "ball_angle_to_goal": ball_angle,
                "angle_ball_to_goalkeeper": angle_ball_gk,
                "num_defenders_close": num_defenders_close,
                "closest_defender_distance": closest_defender_distance,
                "shooter_lateral_offset": shooter_lateral_offset,
                "attack_type": attack_type,
                "sub_type": sub_type,
                # NEW: context
                "is_home_team": int(bool(is_home_team)),
                "attendance": attendance if attendance is not None else np.nan,
                # NEW: richer geometry
                "shooter_to_goal_dx": shooter_to_goal_dx,
                "shooter_to_goal_dy": shooter_to_goal_dy,
                "shooter_y_signed": shooter_y_signed,
                "shooter_distance_to_goal_sq": shooter_distance_to_goal_sq,
                # NEW: GK coverage
                "gk_lateral_offset": gk_lateral_offset,
                "gk_angle_from_shooter": gk_angle_from_shooter,
                "gk_along_shotline_dist": gk_along_shotline_dist,
                # NEW: obstruction
                "n_defenders_in_shot_cone": n_defenders_in_shot_cone,
                "min_defender_dist_to_shotline": min_defender_dist_to_shotline,
                "min_defender_along_shotline": min_defender_along_shotline,
                # NEW: defense shape
                "defense_centroid_dist_to_goal": defense_centroid_dist_to_goal,
                "defense_spread": defense_spread,
                # label
                "target": target,
            }
        )

    df_features = pd.DataFrame(features_list)

    # Ensure numeric dtype for all numeric cols (safe cast)
    for c in df_features.columns:
        if c in {"fixture_id", "event_id", "attack_type", "sub_type"}:
            continue
        df_features[c] = pd.to_numeric(df_features[c], errors="coerce")

    return df_features


if __name__ == "__main__":
    import duckdb

    logging.basicConfig(level=logging.INFO)
    fixture_id = "00ba9627-5ca6-11f0-ac5e-5389986df98b"
    con_duckdb = "./data/hbl_raw.duckdb"

    with duckdb.connect(con_duckdb) as db:
        df_match_normalized = db.execute(
            f"SELECT * FROM matches_normalized WHERE fixture_id = '{fixture_id}'"
        ).df()
        df_shot_events = db.execute(
            f"SELECT * FROM shot_events WHERE fixture_id = '{fixture_id}'"
        ).df()
        df_positions_normalized = db.execute(
            f"SELECT * FROM match_positions_normalized WHERE fixture_id = '{fixture_id}'"
        ).df()

    df_xg_features = calculate_xg_features(
        df_match_normalized=df_match_normalized,
        df_shot_events=df_shot_events,
        df_positions_normalized=df_positions_normalized,
    )
    print(df_xg_features.head())
    print(f"Calculated xG features for {len(df_xg_features)} shot events.")
