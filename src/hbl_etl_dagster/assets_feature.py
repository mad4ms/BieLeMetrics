from dagster import (
    asset,
    AssetExecutionContext,
    MetadataValue,
)
import pandas as pd
import numpy as np
from .assets_sportradar_slow import fixtures_partition_def
from .utils.metadata import preview_metadata
from typing import Dict, Any

GOAL_Y: float = 10.0
GOAL_WIDTH: float = 3.0  # meters between posts


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x1 - x2
    dy = y1 - y2
    return float(np.hypot(dx, dy))


def angle_from_to(
    x_from: float, y_from: float, x_to: float, y_to: float
) -> float:
    """Angle (in rad) of vector from (x_from, y_from) to (x_to, y_to)."""
    return float(np.arctan2(y_to - y_from, x_to - x_from))


def handball_shot_angle_deg(
    x_from: float,
    y_from: float,
    goal_x: float,
    goal_y: float,
) -> float:
    """
    Handball shot angle in degrees relative to the goal center.

    Convention:
    - 90°: perfectly in front of goal (on the center line y = goal_y)
    - 0°: shot from the side / almost parallel to the goal line

    Works for both goals at (0, 10) and (40, 10) on a 40x20 court.
    """
    dx = abs(goal_x - x_from)
    dy = abs(goal_y - y_from)

    if dx == 0.0 and dy == 0.0:
        # Player at goal center: define as 90°
        return 90.0

    # angle between shot vector and horizontal (0° center, 90° corner)
    theta_deg = float(np.degrees(np.arctan2(dy, dx)))  # in [0, 90]

    # invert to match handball convention: 90° = center, 0° = corner
    return 90.0 - theta_deg


def point_in_triangle(
    px: float,
    py: float,
    tri: np.ndarray,
    include_boundary: bool = True,
) -> bool:
    """
    Barycentric point-in-triangle test.

    tri: array of shape (3, 2) with vertices [v0, v1, v2].
    """
    (x1, y1), (x2, y2), (x3, y3) = tri

    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if denom == 0:
        # Degenerate triangle
        return False

    a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
    b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
    c = 1.0 - a - b

    if include_boundary:
        return (a >= 0.0) and (b >= 0.0) and (c >= 0.0)
    else:
        return (a > 0.0) and (b > 0.0) and (c > 0.0)


@asset(
    partitions_def=fixtures_partition_def,
    required_resource_keys={"io_manager"},
    group_name="feature_data",
    compute_kind="duckdb",
    description="Feature calculation.",
    metadata={"partition_column": "fixture_id"},
)
def features_at_throw_time(
    context: AssetExecutionContext,
    sportradar_goals_refined: pd.DataFrame,
    positions_for_throw_time: pd.DataFrame,
    fixtures_sportradar: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate features at throw time for a single fixture partition."""

    duckdb_io_manager = context.resources.io_manager
    fixture_id = str(context.partition_key)

    # --- Load positions for this fixture from DuckDB ---------------------------------
    try:
        with duckdb_io_manager._conn() as con:
            df_positions = con.execute(
                "SELECT * FROM kinexon_positions WHERE fixtureId = ?",
                [fixture_id],
            ).df()
    except Exception as e:  # noqa: BLE001
        context.log.error(
            f"Error fetching positions for fixture {fixture_id}: {e}"
        )
        return pd.DataFrame()

    if df_positions.empty:
        context.log.warning(
            f"No positions found for fixture {fixture_id}. Skipping refinement."
        )
        return pd.DataFrame()

    # --- Filter inputs to the current fixture ----------------------------------------
    sportradar_goals_refined = sportradar_goals_refined.copy()
    sportradar_goals_refined["fixture_id"] = sportradar_goals_refined[
        "fixture_id"
    ].astype(str)
    df_events = sportradar_goals_refined[
        sportradar_goals_refined["fixture_id"] == fixture_id
    ]

    if df_events.empty:
        context.log.warning(f"No match events found for fixture {fixture_id}.")
        return pd.DataFrame()

    positions_for_throw_time = positions_for_throw_time.copy()
    positions_for_throw_time["fixture_id"] = positions_for_throw_time[
        "fixture_id"
    ].astype(str)
    positions_for_throw_time = positions_for_throw_time[
        positions_for_throw_time["fixture_id"] == fixture_id
    ]

    fixtures_sportradar = fixtures_sportradar.copy()
    fixtures_sportradar["fixture_id"] = fixtures_sportradar[
        "fixture_id"
    ].astype(str)
    fixture_info = fixtures_sportradar[
        fixtures_sportradar["fixture_id"] == fixture_id
    ]

    context.log.info(
        "Calculating features for fixture %s with %d events and %d positions.",
        fixture_id,
        len(df_events),
        len(df_positions),
    )
    context.log.info(
        "Length of positions_for_throw_time: %d", len(positions_for_throw_time)
    )

    # --- Restrict to throw (goal) events and merge positional snapshot info ----------
    throw_events = df_events[df_events["event_type"] == "goal"].copy()

    throw_events = throw_events.merge(
        positions_for_throw_time,
        how="left",
        left_on=["event_id", "person_league_id"],
        right_on=["event_id", "league id"],
        suffixes=("", "_pos"),
    )

    context.log.info("Length after merge: %d", len(throw_events))

    # --- Feature calculation ---------------------------------------------------------
    total_events = len(throw_events)
    processed_events = 0
    skipped_events = 0
    features_list: list[Dict[str, Any]] = []

    # Optionally pre-group positions by timestamp to avoid repeated filtering
    positions_by_ts = dict(tuple(df_positions.groupby("ts in ms")))

    for _, event in throw_events.iterrows():
        event_time = event.get("refined_throw_ts_ms")

        positions_at_event = positions_by_ts.get(event_time, pd.DataFrame())
        if positions_at_event.empty:
            context.log.debug(
                "No positions found at throw time %s for fixture %s.",
                event_time,
                fixture_id,
            )
            skipped_events += 1
            continue

        kin_distance = event.get("kin_distance")
        id_thrower = event["person_league_id"]
        id_goalkeeper = event["goalkeeper_league_id"]

        position_thrower = positions_at_event[
            positions_at_event["league id"] == id_thrower
        ]
        position_goalkeeper = positions_at_event[
            positions_at_event["league id"] == id_goalkeeper
        ]

        if position_thrower.empty or position_goalkeeper.empty:
            skipped_events += 1
            continue

        # ball id is where group name == "Ball"
        position_ball = positions_at_event[
            positions_at_event["group name"] == "Ball"
        ]
        if position_ball.empty:
            skipped_events += 1
            continue

        thrower_x = float(position_thrower.iloc[0]["x in m"])
        thrower_y = float(position_thrower.iloc[0]["y in m"])
        gk_x = float(position_goalkeeper.iloc[0]["x in m"])
        gk_y = float(position_goalkeeper.iloc[0]["y in m"])
        ball_x = float(position_ball.iloc[0]["x in m"])
        ball_y = float(position_ball.iloc[0]["y in m"])

        goal_x = float(event["goal_position"])
        goal_y = GOAL_Y

        # Distances
        dist_tg = euclidean_distance(thrower_x, thrower_y, gk_x, gk_y)
        dist_th = euclidean_distance(thrower_x, thrower_y, goal_x, goal_y)
        dist_gk = euclidean_distance(gk_x, gk_y, goal_x, goal_y)

        # Angles: from player/ball TO goal
        angle_player_goal = angle_from_to(thrower_x, thrower_y, goal_x, goal_y)
        angle_ball_goal = angle_from_to(ball_x, ball_y, goal_x, goal_y)

        angle_player_straight = handball_shot_angle_deg(
            thrower_x, thrower_y, goal_x, goal_y
        )
        angle_ball_straight = handball_shot_angle_deg(
            ball_x, ball_y, goal_x, goal_y
        )

        # Triangle: ball + two posts
        half_width = GOAL_WIDTH / 2.0
        goal_left = (goal_x, goal_y - half_width)
        goal_right = (goal_x, goal_y + half_width)

        triangle = np.array(
            [
                [ball_x, ball_y],
                [goal_left[0], goal_left[1]],
                [goal_right[0], goal_right[1]],
            ],
            dtype=float,
        )

        # Count other players (exclude thrower, goalkeeper, and ball)
        other_positions = positions_at_event[
            (positions_at_event["group name"] != "Ball")
            & ~positions_at_event["league id"].isin(
                [id_thrower, id_goalkeeper]
            )
        ]

        name_team_thrower = position_thrower["group name"].iloc[0]

        # Defense players only (excluding goalkeeper)
        other_positions_defense = other_positions[
            (other_positions["group name"] != name_team_thrower)
            & (other_positions["league id"] != id_goalkeeper)
        ]

        num_players_in_triangle = 0
        for _, pos in other_positions_defense.iterrows():
            px = float(pos["x in m"])
            py = float(pos["y in m"])
            if point_in_triangle(px, py, triangle, include_boundary=True):
                num_players_in_triangle += 1

        # number of players close to the thrower (within 1.5 meters)
        num_players_near_thrower = 0

        for _, pos in other_positions_defense.iterrows():
            px = float(pos["x in m"])
            py = float(pos["y in m"])
            dist_to_thrower = euclidean_distance(px, py, thrower_x, thrower_y)
            if dist_to_thrower <= 1.5:
                num_players_near_thrower += 1

        # player speed
        speed_thrower = float(position_thrower.iloc[0]["speed in m/s"])
        # total distance of player already covered until throw time
        distance_covered_thrower = float(
            position_thrower.iloc[0]["total distance in m"]
        )

        features_list.append(
            {
                "fixture_id": fixture_id,
                "event_id": event["event_id"],
                "distance_thrower_goalkeeper": dist_tg,
                "distance_thrower_goal": dist_th,
                "distance_goalkeeper_goal": dist_gk,
                "angle_player_goal": angle_player_goal,
                "angle_ball_goal": angle_ball_goal,
                "angle_player_straight": np.degrees(angle_player_straight),
                "angle_ball_straight": np.degrees(angle_ball_straight),
                "num_players_in_triangle": num_players_in_triangle,
                "num_players_near_thrower": num_players_near_thrower,
                "speed_thrower": speed_thrower,
                "distance_covered_thrower": distance_covered_thrower,
                "kinexon_distance": kin_distance,
                "attack_type": event.get("attack_type", "unknown"),
                "success": event.get("success", False),
            }
        )

        processed_events += 1

    features_df = pd.DataFrame(features_list)

    # --- Logging & metadata ----------------------------------------------------------
    if total_events > 0:
        success_rate = processed_events / total_events
        success_rate_str = f"{success_rate:.1%}"
    else:
        success_rate = 0.0
        success_rate_str = "N/A"

    context.log.info(
        "Processed %d/%d throw events (%.1f%% success rate).",
        processed_events,
        total_events,
        success_rate * 100.0 if total_events > 0 else 0.0,
    )

    metadata: Dict[str, Any] = {
        "fixture_id": fixture_id,
        "total_events": total_events,
        "processed_events": processed_events,
        "skipped_events": skipped_events,
        "success_rate": success_rate_str,
        "num_features": len(features_df),
        **preview_metadata(features_df),
    }

    context.add_output_metadata(metadata)
    context.log.info("Calculated features for fixture %s.", fixture_id)

    return features_df
