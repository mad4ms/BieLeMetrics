import pandas as pd
from scipy.signal import find_peaks
from typing import Optional, Tuple, List


def find_timestamp_of_throw(
    event_type: str, data_kinexon_event: pd.DataFrame, id_player: str
) -> Optional[Tuple[pd.Timestamp, pd.DataFrame, List]]:
    """
    Finds the timestamp of the throw event in the kinexon data.

    Args:
        event_type (str): The type of the event.
        data_kinexon_event (pd.DataFrame): The kinexon event data.
        id_player (str): The player's id who threw the ball.

    Returns:
        pd.Timestamp: The timestamp of the throw event, if found. Otherwise, None.
    """
    # Determine if the event type is a relevant throw event
    relevant_event_types = [
        "score_change",
        "shot_off_target",
        "shot_blocked",
        "shot_saved",
        "seven_m_missed",
    ]

    if event_type not in relevant_event_types:
        return None, None, None

    data_kinexon_event["time"] = pd.to_datetime(data_kinexon_event["time"])

    # Extract most used ball id
    df_value_count_ball_ids = data_kinexon_event["league_id"].value_counts()

    # Set current ball ID to the one with the most occurrences that have "ball" or "Ball" in the name
    id_ball = df_value_count_ball_ids[
        df_value_count_ball_ids.index.astype(str).str.contains("ball|Ball")
    ]
    if not id_ball.empty and id_player is not None:
        id_ball = id_ball.idxmax()
    else:
        # No throw can be found without a ball
        return None, None, None

    data_kinexon_event_player = data_kinexon_event[
        data_kinexon_event["league_id"] == id_player.split(":")[-1]
    ]
    data_kinexon_event_player = data_kinexon_event_player[
        ["time", "pos_x", "pos_y"]
    ]

    data_kinexon_event_ball = data_kinexon_event[
        data_kinexon_event["league_id"] == id_ball
    ]
    data_kinexon_event_ball = data_kinexon_event_ball[
        ["time", "pos_x", "pos_y"]
    ]

    # Merge the data of the thrower and the ball, based on the timestamp
    data_kinexon_event_player_ball = pd.merge_asof(
        data_kinexon_event_player,
        data_kinexon_event_ball,
        on="time",
        suffixes=("", "_ball"),
    )

    # Add distance to the ball
    data_kinexon_event_player_ball["distance"] = (
        (
            data_kinexon_event_player_ball["pos_x"]
            - data_kinexon_event_player_ball["pos_x_ball"]
        )
        ** 2
        + (
            data_kinexon_event_player_ball["pos_y"]
            - data_kinexon_event_player_ball["pos_y_ball"]
        )
        ** 2
    ) ** 0.5

    # Add speed to the ball
    data_kinexon_event_player_ball["time_diff"] = (
        data_kinexon_event_player_ball["time"]
        - data_kinexon_event_player_ball["time"].shift()
    )

    # Convert time difference to seconds
    data_kinexon_event_player_ball["time_diff_seconds"] = (
        data_kinexon_event_player_ball["time_diff"].dt.total_seconds()
    )

    # Calculate speed
    data_kinexon_event_player_ball["speed"] = (
        data_kinexon_event_player_ball["distance"]
        / data_kinexon_event_player_ball["time"].diff().dt.total_seconds()
    )

    # Add acceleration to the ball
    data_kinexon_event_player_ball["acceleration"] = (
        data_kinexon_event_player_ball["distance"].diff()
        / data_kinexon_event_player_ball["time"].diff().dt.total_seconds()
    )

    # Shift acceleration by 3 to capture the throw
    data_kinexon_event_player_ball["acceleration"] = (
        data_kinexon_event_player_ball["acceleration"].shift(-3)
    )

    # Get peaks in the acceleration
    peaks, _ = find_peaks(
        data_kinexon_event_player_ball["acceleration"], distance=10
    )

    # Make empty list for peaks if None
    if peaks is None:
        peaks = []

        # Clean peaks
    peaks = clean_peaks(peaks, data_kinexon_event_player_ball)

    if peaks is None or len(peaks) == 0:
        print(f"! No peaks found for event {event_type}: ❌")
        return None, None, None

    # Return the timestamp of the throw
    event_time_throw = data_kinexon_event_player_ball["time"].iloc[peaks[-1]]

    print(f"\t> Throw time for event {event_type}: ✅")
    return event_time_throw, data_kinexon_event_player_ball, peaks


def clean_peaks(
    peaks: list, data_kinexon_event_player_ball: pd.DataFrame
) -> list:
    """
    Removes peaks with distance > 2.5 m and earlier than 1.5 seconds before the event.

    Args:
    peaks (list): The indices of the peaks in the distance.
    df_ball (pd.DataFrame): The ball data frame.

    Returns:
    list: The cleaned indices of the peaks in the distance.
    """

    # Remove peaks with distance > 2.5 m
    peaks = [
        peak
        for peak in peaks
        if data_kinexon_event_player_ball["distance"].iloc[peak] < 2.5
    ]
    # Remove peaks with direction not towards the goal
    peaks = [
        peak
        for peak in peaks
        # if self.data_kinexon_event_player_ball["direction"].iloc[peak]
    ]

    # check if there are any peaks
    if len(peaks) == 0:
        print("\tNo peaks found.")
        return []

    # event_time_tagged is last event time in data_kinexon_event_player_ball
    event_time_tagged = data_kinexon_event_player_ball["time"].iloc[-1]
    # find indices of peaks in df_ball that are
    # 3 seconds before row["time"]
    idx_event_threshold = data_kinexon_event_player_ball[
        data_kinexon_event_player_ball["time"]
        < event_time_tagged - pd.Timedelta(seconds=1.5)
    ].index[-1]

    # remove peaks if located in df_ball before idx_event_threshold
    peaks = [
        peak
        for peak in peaks
        if data_kinexon_event_player_ball.index[peak] < idx_event_threshold
    ]

    # # Check if row["type"] is score_changed or shot_missed
    # if row["type"] in ["score_change", "shot_off_target"]:
    #     # Check if at any point in df_ball, the ball was behind the goal line
    #     if df_ball["x in m"].min() < 0:
    #         # Get indices of peaks where the ball was behind the goal line
    #         idx_ball_behind_goal = df_ball[
    #             df_ball["x in m"] < 0
    #         ].index.tolist()

    #         peaks = [
    #             peak
    #             for peak in peaks
    #             if df_ball.index[peak] < idx_ball_behind_goal
    #         ]

    return peaks
