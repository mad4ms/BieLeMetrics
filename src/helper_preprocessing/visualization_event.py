from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from scipy.spatial import voronoi_plot_2d
from scipy.spatial import Voronoi

# import seaborn as sns
from typing import List, Tuple, Dict
import cv2
import importlib

from shapely.geometry import Polygon, Point
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
import cv2
import os

try:
    from .game_event import GameEvent
except ImportError:
    try:
        from game_event import GameEvent
    except ImportError:
        from helper_preprocessing.game_event import GameEvent


def plot_event_syncing(event_game: GameEvent):
    # Plot the distance between the player and the ball
    #  over time with matplotlib, but with original data
    df_player_ball = event_game.data_kinexon_event_player_ball
    ## shift acceleration by 3 capture the throw
    # df_player_ball["acceleration"] = df_player_ball["acceleration"].shift(
    #     -3
    # )

    if event_game.event_time_throw is None:
        return

    plt.figure()
    plt.plot(df_player_ball["time"], df_player_ball["acceleration"], color="b")
    # plot distance on other y axis
    plt.twinx()
    plt.plot(df_player_ball["time"], df_player_ball["distance"], color="r")
    # Gray area for distance < 2.5 m
    plt.fill_between(
        df_player_ball["time"],
        0,
        25,
        where=df_player_ball["distance"] < 2.5,
        color="gray",
        alpha=0.2,
    )
    time_last_in_df = df_player_ball["time"].iloc[-1]

    # Light read area for 1.5 seconds before time last in df
    plt.fill_between(
        df_player_ball["time"],
        0,
        25,
        where=(
            df_player_ball["time"]
            > time_last_in_df - pd.Timedelta(1.5, unit="s")
        ),
        color="red",
        alpha=0.2,
    )

    # Gray area for direction of ball
    # plt.fill_between(
    #     df_player_ball["ts in ms"],
    #     0,
    #     25,
    #     where=df_player_ball["direction"] == 1,
    #     color="green",
    #     alpha=0.2,
    # )

    if event_game.event_time_throw is not None:
        # Add a vertical line for the throw
        plt.axvline(x=event_game.event_time_throw, color="r", linestyle="--")

    # Plot peaks
    for peak in event_game.peaks:
        plt.axvline(
            x=df_player_ball["time"].iloc[peak], color="g", linestyle="--"
        )
    if len(event_game.peaks) > 0:
        # Highlight the last peak
        plt.axvline(
            x=df_player_ball["time"].iloc[event_game.peaks[-1]],
            color="r",
            linestyle="--",
            label="Last peak",
        )

    plt.xlabel("Time in ms")
    plt.ylabel("Distance in m")
    plt.title(f"Distance between player and ball for {event_game.name_player}")
    plt.show(block=True)


def render_event(event_game: GameEvent):

    # check if type is in
    types_to_render = [
        "score_change",
        "shot_saved",
        "shot_off_target",
        "shot_blocked",
        "seven_m_missed",
    ]
    if event_game.event_type not in types_to_render:
        return

    img_name = "./assets/handballfeld.png"
    # Create empty image
    img = cv2.imread(img_name)
    # check if img is None
    if img is None:
        img = cv2.imread("../" + img_name)
    plt.imshow(img, extent=[0, 40, 0, 20])

    # get image width and height
    height, width, _ = img.shape

    print(f"Image width: {width}, height: {height}.")
    # if the full width results in 40 m, get the scale factor
    scale = width / 40

    name_event = f"event_{int(event_game.event_id)}"

    # name_event = f"Test"

    # fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    name_file = os.path.join("assets", "events", "videos", f"{name_event}.mp4")

    # create a writer with img.shape
    writer = cv2.VideoWriter(
        name_file,
        fourcc,
        30,
        (img.shape[1], img.shape[0]),
    )

    first_event = False

    # group by time in ms
    df_kinexon_grouped = event_game.df_kinexon_event.groupby("time")

    for ts, group in df_kinexon_grouped:

        img_draw = img.copy()
        # empty image with same shape as img
        img_overlay = np.zeros_like(img_draw)

        # Draw attack direction
        if event_game.attack_direction == "left":
            cv2.arrowedLine(
                img_overlay,
                (int(5 * scale), int(19 * scale)),
                (int(35 * scale), int(19 * scale)),
                (0, 0, 255),
                3,
            )
        else:
            cv2.arrowedLine(
                img_overlay,
                (int(35 * scale), int(19 * scale)),
                (int(5 * scale), int(19 * scale)),
                (0, 0, 255),
                3,
            )

        # Check if row_sportradar["id_player"] is nan
        if pd.isna(event_game.id_player):
            continue

        # Iterate over the grouped dataframe
        for index, row in group.iterrows():
            if row["group_id"] != 3:  # ball
                if int(event_game.id_player.split(":")[-1]) == int(
                    row["league_id"]
                ):
                    cv2.circle(
                        img_draw,
                        (
                            int(row["pos_x"] * scale),
                            int(row["pos_y"] * scale),
                        ),
                        17,
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        img_draw,
                        "PL",
                        (
                            int(row["pos_x"] * scale) - 10,
                            int(row["pos_y"] * scale) - 10,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )
                if event_game.id_goalkeeper is not None and int(
                    event_game.id_goalkeeper.split(":")[-1]
                ) == int(row["league_id"]):
                    cv2.circle(
                        img_draw,
                        (
                            int(row["pos_x"] * scale),
                            int(row["pos_y"] * scale),
                        ),
                        17,
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        img_draw,
                        "GK",
                        (
                            int(row["pos_x"] * scale) - 10,
                            int(row["pos_y"] * scale) - 10,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )
            if row["group_id"] == 3:
                cv2.circle(
                    img_draw,
                    (int(row["pos_x"] * scale), int(row["pos_y"] * scale)),
                    15,
                    (0, 0, 255),
                    -1,
                )
            elif row["group_id"] == 2:
                cv2.circle(
                    img_draw,
                    (int(row["pos_x"] * scale), int(row["pos_y"] * scale)),
                    10,
                    (0, 255, 0),
                    -1,
                )
            elif row["group_id"] == 1:
                cv2.circle(
                    img_draw,
                    (int(row["pos_x"] * scale), int(row["pos_y"] * scale)),
                    10,
                    (255, 0, 0),
                    -1,
                )

        # Pause 3 seconds at the time of the throw
        if (
            event_game.event_time_throw is not None
            and ts == event_game.event_time_throw
        ):
            # get x,y pos of ball
            ball_x = int(
                group[group["group_id"] == 3]["pos_x"].values[0] * scale
            )
            ball_y = int(
                group[group["group_id"] == 3]["pos_y"].values[0] * scale
            )

            # goal post left, because handball goal is 3m wide
            goal_post_left_upper = (0, 11.5 * scale)
            goal_post_left_lower = (0, 8.5 * scale)
            # goal post right
            goal_post_right_upper = (40 * scale, 11.5 * scale)
            goal_post_right_lower = (40 * scale, 8.5 * scale)
            # Draw triangle between ball and goal posts and fill it
            if event_game.attack_direction == "right":
                points = np.array(
                    [
                        [ball_x, ball_y],
                        [goal_post_left_upper[0], goal_post_left_upper[1]],
                        [goal_post_left_lower[0], goal_post_left_lower[1]],
                    ]
                )
            else:
                points = np.array(
                    [
                        [ball_x, ball_y],
                        [goal_post_right_upper[0], goal_post_right_upper[1]],
                        [goal_post_right_lower[0], goal_post_right_lower[1]],
                    ]
                )

            points = points.astype(int)
            cv2.fillPoly(img_overlay, [points], (0, 255, 0))

            img_draw = plot_text(img_draw, event_game)

            # # calc voronoi
            # vor, player_positions_home, player_positions_away, bb = (
            #     calculate_voronoi(row_sportradar)
            # )
            # if vor is not None:

            #     img_overlay = plot_voronoi_on_image(
            #         img_draw,
            #         vor,
            #         player_positions_home,
            #         player_positions_away,
            #         row_sportradar,
            #         bb,
            #     )

            # Draw angle from ball to goal center
            if event_game.angle_ball_goal is not None:

                angle = event_game.angle_ball_goal
                if event_game.attack_direction == "right":
                    point_goal_center = (40 * scale, 10 * scale)
                else:
                    point_goal_center = (0, 10 * scale)

                point_ball = (ball_x, ball_y)

                # # Draw line from ball to goal center
                # cv2.line(
                #     img_overlay,
                #     point_ball,
                #     point_goal_center,
                #     (255, 255, 255),
                #     2,
                # )

                # # Draw angle
                # cv2.putText(
                #     img_overlay,
                #     f"{angle:.2f} degrees",
                #     (int(ball_x + 10), int(ball_y + 10)),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.6,
                #     (255, 255, 255),
                #     1,
                # )

            # add weight to overlay
            img_draw = cv2.addWeighted(img_draw, 1, img_overlay, 0.5, 0)

            # write image 20 times
            for i in range(20):
                writer.write(img_draw)

            cv2.imshow("image", img_draw)
            cv2.waitKey(1500)

        # img_draw = plot_text(img_draw, row_sportradar)

        # show
        img_draw = plot_text(img_draw, event_game)

        img_draw = cv2.addWeighted(img_draw, 1, img_overlay, 0.5, 0)
        cv2.imshow("image", img_draw)
        cv2.waitKey(1)
        # Resize the image
        img_draw = cv2.resize(img_draw, (width, height))
        writer.write(img_draw)

    writer.release()
    cv2.destroyAllWindows()


def plot_voronoi_on_image(
    img: np.ndarray,
    vor: Voronoi,
    player_positions_home: list,
    player_positions_away: list,
    df_row_with_positions: pd.Series,
    bounding_box: Polygon,
) -> np.ndarray:
    """
    Plot the Voronoi diagram with team-specific color coding, clipped to the handball field dimensions,
    and additional game details on an image.

    Args:
    img (np.ndarray): Image to plot the Voronoi diagram on.
    vor (Voronoi): The Voronoi object computed from player positions.
    player_positions_home (list): List of home team player positions.
    player_positions_away (list): List of away team player positions.
    df_row_with_positions (pd.Series): A series containing additional game details like ball position.
    bounding_box (Polygon): Bounding box representing the field limits.

    Returns:
    np.ndarray: Image with the Voronoi diagram and additional game details.
    """
    # img_draw is empty image with same shape as img
    img_draw = np.zeros_like(img)

    scale = img.shape[1] / 40  # assuming field length is 40 meters

    # Total list of player positions combining both teams
    all_player_positions = player_positions_home + player_positions_away

    # scale bounding box like so:     scaled_points = [(x * scale, y * scale) for x, y in polygon.exterior.coords]
    bounding_box = Polygon(
        [(x * scale, y * scale) for x, y in bounding_box.exterior.coords]
    )

    # Color the Voronoi regions clipped to the bounding box
    for point_index, region_index in enumerate(vor.point_region):
        # Get the region of the current point
        region = vor.regions[region_index]
        # Check if the region is valid and not infinite
        if not -1 in region:
            # Get the polygon points and clip it to the bounding box
            polygon_points = [vor.vertices[i] for i in region]
            scaled_polygon_points = (np.array(polygon_points) * scale).astype(
                int
            )

            # multiply polygon points with scale and convert to integers
            # polygon_points = (np.array(polygon_points) * scale).astype(int)
            polygon = Polygon(scaled_polygon_points)
            clipped_polygon = polygon.intersection(bounding_box)
            if not clipped_polygon.is_empty:
                color = (
                    (0, 0, 255)
                    if point_index < len(player_positions_home)
                    else (255, 0, 0)
                )
                cv2.fillPoly(
                    img_draw,
                    [np.array(clipped_polygon.exterior.coords, np.int32)],
                    color,
                )

    return img_draw


def calculate_voronoi(
    df_row_with_positions: pd.Series,
) -> (Voronoi, list, list, Polygon):
    """
    Calculate the Voronoi partition based on player positions and clip it to a bounding box.

    Args:
    df_row_with_positions (pd.Series): A series containing player positions.

    Returns:
    Tuple[Voronoi, list, list, Polygon]: Voronoi object, lists of player positions for home and away teams, and bounding box.
    """
    player_positions_home = []
    player_positions_away = []
    for i in range(1, 10):
        if (
            f"x_1_{i}" in df_row_with_positions
            and f"y_1_{i}" in df_row_with_positions
        ):
            player_positions_home.append(
                [
                    df_row_with_positions[f"x_1_{i}"],
                    df_row_with_positions[f"y_1_{i}"],
                ]
            )
        if (
            f"x_2_{i}" in df_row_with_positions
            and f"y_2_{i}" in df_row_with_positions
        ):
            player_positions_away.append(
                [
                    df_row_with_positions[f"x_2_{i}"],
                    df_row_with_positions[f"y_2_{i}"],
                ]
            )

    if not player_positions_home or not player_positions_away:
        return None, [], [], None

    # Combine home and away positions for the Voronoi calculation
    all_positions = player_positions_home + player_positions_away

    # Define the bounding box (field boundaries)
    margin = 1  # margin added to ensure all points are within the bounding box
    min_x, max_x = 0 - margin, 40 + margin
    min_y, max_y = 0 - margin, 20 + margin
    bounding_box = Polygon(
        [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
    )

    # Extend points for bounding box
    points_extended = all_positions + [
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
        [max_x, min_y],
    ]
    vor = Voronoi(points_extended)

    return vor, player_positions_home, player_positions_away, bounding_box


def plot_text(img_draw, event_game: GameEvent):
    # Event type
    cv2.putText(
        img_draw,
        f"Event: {event_game.event_type} at {event_game.event_time_tagged} with ({int(event_game.home_score)}:{int(event_game.away_score)})",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    # Player and GK name
    cv2.putText(
        img_draw,
        f"Player: {event_game.name_player} and GK: {event_game.name_goalkeeper}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    # Distance to goal
    if event_game.distance_player_goal is not None:
        cv2.putText(
            img_draw,
            f"Distance Player <-> Goal: {event_game.distance_player_goal:.2f}m",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    # Distance to goalkeeper
    if event_game.distance_goalkeeper_goal is not None:
        cv2.putText(
            img_draw,
            f"Distance GK <-> Goal: {event_game.distance_goalkeeper_goal:.2f}m",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    # Angle to goal
    if event_game.angle_ball_goal is not None:
        # Angle to goal
        cv2.putText(
            img_draw,
            f"Angle Ball <-> Goal: {event_game.angle_ball_goal:.2f} degrees",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
    # Speed of ball
    if event_game.speed_ball_at_throw is not None:
        cv2.putText(
            img_draw,
            f"Speed of ball: {event_game.speed_ball_at_throw * 3.6:.2f} km/h or {event_game.speed_ball_at_throw:.2f} m/s",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    # Distance to goal
    # if (
    #     "distance_player_to_goal" in row_sportradar
    #     and row_sportradar["distance_player_to_goal"] is not None
    # ):
    #     cv2.putText(
    #         img_draw,
    #         f"Distance Player <-> Goal: {row_sportradar['distance_player_to_goal']:.2f}m",
    #         (10, 60),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.6,
    #         (255, 255, 255),
    #         1,
    #     )

    # # Distance to goalkeeper
    # if (
    #     "distance_goalkeeper_to_goal" in row_sportradar
    #     and row_sportradar["distance_goalkeeper_to_goal"] is not None
    # ):
    #     cv2.putText(
    #         img_draw,
    #         f"Distance GK <-> Goal: {row_sportradar['distance_goalkeeper_to_goal']:.2f}m",
    #         (10, 80),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.6,
    #         (255, 255, 255),
    #         1,
    #     )

    # # Angle to goal
    # if (
    #     "angle_ball_goal" in row_sportradar
    #     and row_sportradar["angle_ball_goal"] is not None
    # ):
    #     # Angle to goal
    #     cv2.putText(
    #         img_draw,
    #         f"Angle Ball <-> Goal: {row_sportradar['angle_ball_goal']:.2f} degrees",
    #         (10, 100),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.6,
    #         (255, 255, 255),
    #         1,
    #     )

    #     #  with force {int(row_sportradar['force_of_player'])} N and kinetic energy {int(row_sportradar['kinetic_energy_of_player'])} J
    # if (
    #     "force_of_player" in row_sportradar
    #     and row_sportradar["force_of_player"] is not None
    # ):
    #     cv2.putText(
    #         img_draw,
    #         f"Force: {int(row_sportradar['force_of_player'])} N, Energy: {int(row_sportradar['kinetic_energy_of_player'])} J",
    #         (10, 120),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.6,
    #         (255, 255, 255),
    #         1,
    #     )

    return img_draw
