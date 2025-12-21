import logging
from typing import Dict, Optional, Tuple, List

import pandas as pd
import numpy as np


GOAL_WIDTH = 3.0
HALF_GOAL = GOAL_WIDTH / 2.0


def shot_angle_to_goal(
    shooter_x: float,
    shooter_y: float,
    goal_x: float,
    goal_y: float = 10.0,
) -> float:
    """
    Compute opening angle (radians) between the two goal posts.
    """

    left_post = (goal_x, goal_y - HALF_GOAL)
    right_post = (goal_x, goal_y + HALF_GOAL)

    angle_left = np.arctan2(
        left_post[1] - shooter_y,
        left_post[0] - shooter_x,
    )
    angle_right = np.arctan2(
        right_post[1] - shooter_y,
        right_post[0] - shooter_x,
    )

    return abs(angle_right - angle_left)


def angle_ball_goalkeeper(
    ball_x,
    ball_y,
    goalkeeper_x,
    goalkeeper_y,
    goal_x,
    goal_y=10.0,
):
    v_goal = np.array([goal_x - ball_x, goal_y - ball_y])
    v_gk = np.array([goalkeeper_x - ball_x, goalkeeper_y - ball_y])

    dot = np.dot(v_goal, v_gk)
    norm = np.linalg.norm(v_goal) * np.linalg.norm(v_gk)

    if norm == 0:
        return np.nan

    cos_theta = np.clip(dot / norm, -1.0, 1.0)
    return np.arccos(cos_theta)  # radians


def calculate_xg_features(
    df_match_normalized: pd.DataFrame,
    df_shot_events: pd.DataFrame,
    df_positions_normalized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate base features from position data for xG Calculation.

    :param df_match_normalized: Normalized match data.
    :param shot_events: DataFrame containing shot event data.
    :param df_positions_normalized: Normalized player position data.
    :return: DataFrame with calculated xG features.
    """

    # global features
    # attenance
    attendance = df_match_normalized["attendance"].iloc[0]
    # to int
    attendance = int(attendance) if pd.notna(attendance) else None

    # iterate over each shot event and calculate features
    features_list = []
    for _, shot in df_shot_events.iterrows():
        event_id = shot["event_id"]
        timestamp_ms = shot["throw_timestamp_ms"]
        team_name_offense = shot["team_name_offense"]
        team_name_defense = shot["team_name_defense"]

        shot["team_name"] = team_name_offense

        is_home_team = shot["team_name"] == shot["team_name_home"]
        fixture_id = shot["fixture_id"]

        # Filter positions for the specific fixture, period, and timestamp
        df_positions_shot = df_positions_normalized[
            df_positions_normalized["timestamp_ms"] == timestamp_ms
        ]

        # Further filter for offense and defense teams
        df_offense = df_positions_shot[
            df_positions_shot["group_name"] == team_name_offense
        ]
        df_defense = df_positions_shot[
            df_positions_shot["group_name"] == team_name_defense
        ]

        # Calculate features (example: average distance to goal for offense players)
        goal_x, goal_y = (
            shot["goal_position"],
            10,  # handball goal center at y=10
        )
        offense_distances = (
            (df_offense["x_m"] - goal_x) ** 2
            + (df_offense["y_m"] - goal_y) ** 2
        ) ** 0.5
        avg_offense_distance = (
            offense_distances.mean() if not offense_distances.empty else None
        )
        # defense distance
        defense_distances = (
            (df_defense["x_m"] - goal_x) ** 2
            + (df_defense["y_m"] - goal_y) ** 2
        ) ** 0.5
        avg_defense_distance = (
            defense_distances.mean() if not defense_distances.empty else None
        )

        # calc distance of shot["person_league_id"] to goal
        shooter_id = shot["person_league_id"]
        df_shooter = df_offense[df_offense["league_id"] == shooter_id]
        if not df_shooter.empty:
            shooter_x = df_shooter["x_m"].iloc[0]
            shooter_y = df_shooter["y_m"].iloc[0]
            shooter_distance_to_goal = (
                (shooter_x - goal_x) ** 2 + (shooter_y - goal_y) ** 2
            ) ** 0.5

            shot_angle = shot_angle_to_goal(
                shooter_x=shooter_x,
                shooter_y=shooter_y,
                goal_x=goal_x,
                goal_y=10.0,
            )
        else:
            shooter_distance_to_goal = None
            shot_angle = None
            shooter_x = None
            shooter_y = None

        # calc distance of shot["person_league_id"] to goalkeeper
        df_goalkeeper = df_defense[
            df_defense["league_id"] == shot["goalkeeper_league_id"]
        ]
        if not df_goalkeeper.empty and not df_shooter.empty:
            goalkeeper_x = df_goalkeeper["x_m"].iloc[0]
            goalkeeper_y = df_goalkeeper["y_m"].iloc[0]
            shooter_distance_to_goalkeeper = (
                (shooter_x - goalkeeper_x) ** 2
                + (shooter_y - goalkeeper_y) ** 2
            ) ** 0.5
            # goalkeeper distance to goal
            goalkeeper_distance_to_goal = (
                (goalkeeper_x - goal_x) ** 2 + (goalkeeper_y - goal_y) ** 2
            ) ** 0.5

            shooter_lateral_offset = abs(shooter_y - goal_y)

        else:
            shooter_distance_to_goalkeeper = None
            goalkeeper_distance_to_goal = None
            goalkeeper_x = None
            goalkeeper_y = None
            shooter_lateral_offset = None

        # league_id_ball is where league_id contains ball or Ball
        df_ball = df_positions_shot[
            df_positions_shot["league_id"].str.contains(
                "ball", case=False, na=False
            )
        ]

        # Ballkeeper angle to goal
        if not df_ball.empty and not df_goalkeeper.empty:
            ball_x = df_ball["x_m"].iloc[0]
            ball_y = df_ball["y_m"].iloc[0]
        else:
            ball_x = None
            ball_y = None

        # ball
        if not df_ball.empty:

            ball_x = df_ball["x_m"].iloc[0]
            ball_y = df_ball["y_m"].iloc[0]

            # distance ball to goal
            ball_distance_to_goal = (
                (ball_x - goal_x) ** 2 + (ball_y - 10) ** 2
            ) ** 0.5

            ball_angle = shot_angle_to_goal(
                shooter_x=ball_x,
                shooter_y=ball_y,
                goal_x=goal_x,
                goal_y=10.0,
            )
            if not df_goalkeeper.empty:
                goalkeeper_x = df_goalkeeper["x_m"].iloc[0]
                goalkeeper_y = df_goalkeeper["y_m"].iloc[0]
                ball_distance_to_goalkeeper = (
                    (ball_x - goalkeeper_x) ** 2 + (ball_y - goalkeeper_y) ** 2
                ) ** 0.5
                angle_ball_gk = angle_ball_goalkeeper(
                    ball_x=ball_x,
                    ball_y=ball_y,
                    goalkeeper_x=goalkeeper_x,
                    goalkeeper_y=goalkeeper_y,
                    goal_x=goal_x,
                    goal_y=10.0,
                )
            else:
                ball_distance_to_goalkeeper = None
                angle_ball_gk = None
        else:
            ball_angle = None
            ball_distance_to_goal = None
            ball_distance_to_goalkeeper = None
            angle_ball_gk = None

        if shooter_x is not None and shooter_y is not None:
            # number of defenders within 2 meters of shooter
            num_defenders_close = len(
                df_defense[
                    (
                        (df_defense["x_m"] - shooter_x) ** 2
                        + (df_defense["y_m"] - shooter_y) ** 2
                    )
                    ** 0.5
                    <= 2.0
                ]
            )
        else:
            num_defenders_close = None

        # distance closest defender to shooter
        if (
            not df_defense.empty
            and shooter_x is not None
            and shooter_y is not None
        ):
            defender_distances = (
                (df_defense["x_m"] - shooter_x) ** 2
                + (df_defense["y_m"] - shooter_y) ** 2
            ) ** 0.5
            closest_defender_distance = defender_distances.min()
        else:
            closest_defender_distance = None

        # attack type
        attack_type = shot["attack_type"]
        # sub_type
        sub_type = shot["sub_type"]
        # set attack_type to None if not in expected values
        if attack_type not in ["PIVOT", "BREAK_THROUGH", "FAST_BREAK"]:
            attack_type = None

        # insert target
        target = shot["success"]

        features_list.append(
            {
                "fixture_id": fixture_id,
                "event_id": event_id,
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
                "target": target,
                # Add more features as needed
            }
        )
    df_features = pd.DataFrame(features_list)
    return df_features


if __name__ == "__main__":
    import duckdb

    logging.basicConfig(level=logging.INFO)
    fixture_id = "00ba9627-5ca6-11f0-ac5e-5389986df98b"
    con_duckdb = "./data/hbl_raw.duckdb"
    # Tables to load:
    # match_normalized: pd.DataFrame,
    # shot_events: pd.DataFrame,
    # positions_normalized: pd.DataFrame,
    db = duckdb.connect(con_duckdb)
    df_match_normalized = db.execute(
        f"""
        SELECT *
        FROM matches_normalized
        WHERE fixture_id = '{fixture_id}'
        """
    ).df()
    df_shot_events = db.execute(
        f"""
        SELECT *
        FROM shot_events
        WHERE fixture_id = '{fixture_id}'
        """
    ).df()
    df_positions_normalized = db.execute(
        f"""
        SELECT *
        FROM match_positions_normalized
        WHERE fixture_id = '{fixture_id}'
        """
    ).df()
    df_xg_features = calculate_xg_features(
        df_match_normalized=df_match_normalized,
        df_shot_events=df_shot_events,
        df_positions_normalized=df_positions_normalized,
    )
    print(df_xg_features.head())
    print(f"Calculated xG features for {len(df_xg_features)} shot events.")
