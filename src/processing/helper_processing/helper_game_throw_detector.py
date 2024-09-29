import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import math


class ThrowEventDetector:
    def __init__(
        self,
        event_type: str,
        data_kinexon_event: pd.DataFrame,
        id_player: str,
        attack_direction: str,
    ):
        """
        Initialize the ThrowEventDetector object.

        Args:
            event_type (str): The type of the event.
            data_kinexon_event (pd.DataFrame): The Kinexon event data.
            id_player (str): The player's id who threw the ball.
        """
        self.event_type = event_type
        self.attack_direction = attack_direction
        self.data_kinexon_event = data_kinexon_event.copy()
        self.id_player = id_player
        self.event_time_throw = None
        self.data_kinexon_event_player_ball = None
        self.peaks = []

        if attack_direction not in ["left", "right"]:
            print("Attack direction must be either 'left' or 'right'.")

        self.goal_position = (
            (40, 10) if attack_direction == "left" else (0, 10)
        )

        # Process the event data to find the throw
        self._prepare_data()

    def _prepare_data(self):
        """Prepare the data and check if event type is relevant for throw detection."""
        relevant_event_types = [
            "score_change",
            "shot_off_target",
            "shot_blocked",
            "shot_saved",
            "seven_m_missed",
        ]

        if self.event_type not in relevant_event_types:
            return None

        self.data_kinexon_event["time"] = pd.to_datetime(
            self.data_kinexon_event["time"], dayfirst=True
        )

        # Extract ball id
        df_value_count_ball_ids = self.data_kinexon_event[
            "league_id"
        ].value_counts()
        id_ball = df_value_count_ball_ids[
            df_value_count_ball_ids.index.astype(str).str.contains("ball|Ball")
        ]
        if not id_ball.empty and self.id_player is not None:
            self.id_ball = id_ball.idxmax()
        else:
            print(
                f"! No ball (league_id does not contain 'Ball' or 'ball') found for event {self.event_type}: ❌"
            )
            # Check if field "number" contains 99 as it is the default value for the ball
            if self.data_kinexon_event["number"].eq(99).any():
                self.id_ball = self.data_kinexon_event[
                    self.data_kinexon_event["number"] == 99
                ]["league_id"].values[0]
            else:
                print(
                    f"! No ball (number is not 99) found for event {self.event_type}: ❌"
                )
                self.id_ball = None
            return None

    def find_throw_timestamp(
        self,
    ) -> Optional[Tuple[pd.Timestamp, pd.DataFrame, List]]:
        """
        Finds the timestamp of the throw event in the kinexon data.
        """
        if self.id_player is None or self.id_ball is None:
            return None, None, None

        # Merge player and ball data based on time
        self.data_kinexon_event_player_ball = (
            self._merge_player_and_ball_data()
        )

        # Compute distances, speeds, and acceleration
        self._calculate_motion_features()

        # Find acceleration peaks (representing the throw)
        self.peaks = self._find_peaks_acceleration()

        # Clean the peaks
        self.peaks = self._clean_peaks(self.peaks)

        if len(self.peaks) == 0:
            print(f"! No peaks found for event {self.event_type}: ❌")
            return None, None, None

        self.event_time_throw = self.data_kinexon_event_player_ball[
            "time"
        ].iloc[self.peaks[-1]]
        print(f"\t> Throw time: {self.event_time_throw}: ✅")
        return (
            self.event_time_throw,
            self.data_kinexon_event_player_ball,
            self.peaks,
        )

    def _merge_player_and_ball_data(self) -> pd.DataFrame:
        """Merge player and ball data based on time."""
        data_kinexon_event_player = self.data_kinexon_event[
            self.data_kinexon_event["league_id"]
            == self.id_player.split(":")[-1]
        ][["time", "pos_x", "pos_y"]]

        data_kinexon_event_ball = self.data_kinexon_event[
            self.data_kinexon_event["league_id"] == self.id_ball
        ][["time", "pos_x", "pos_y"]]

        # Merge the player and ball data
        return pd.merge_asof(
            data_kinexon_event_player,
            data_kinexon_event_ball,
            on="time",
            suffixes=("", "_ball"),
        )

    def _calculate_distance_player_to_ball(self):
        """Calculate the distance between the player and the ball."""
        self.data_kinexon_event_player_ball["distance"] = (
            (
                self.data_kinexon_event_player_ball["pos_x"]
                - self.data_kinexon_event_player_ball["pos_x_ball"]
            )
            ** 2
            + (
                self.data_kinexon_event_player_ball["pos_y"]
                - self.data_kinexon_event_player_ball["pos_y_ball"]
            )
            ** 2
        ) ** 0.5

        return self.data_kinexon_event_player_ball["distance"]

    def _calculate_speed_player_to_ball(self):
        """Calculate the speed between the player and the ball."""
        self.data_kinexon_event_player_ball["speed"] = (
            self.data_kinexon_event_player_ball["distance"]
            / self.data_kinexon_event_player_ball["time_diff_seconds"]
        )
        return self.data_kinexon_event_player_ball["speed"]

    def _calculate_motion_features(self):
        """Calculate distance, speed, and acceleration between player and ball."""

        # Calculate the distance between the player and the ball
        self.data_kinexon_event_player_ball["distance"] = (
            self._calculate_distance_player_to_ball()
        )

        # Calculate the time difference between each row
        self.data_kinexon_event_player_ball["time_diff_seconds"] = (
            self.data_kinexon_event_player_ball["time"]
            .diff()
            .dt.total_seconds()
        )

        # Calculate the speed between the player and the ball
        self.data_kinexon_event_player_ball["speed"] = (
            self._calculate_speed_player_to_ball()
        )
        # Smooth the speed data
        self.data_kinexon_event_player_ball["speed_smoothed"] = (
            self.data_kinexon_event_player_ball["speed"]
            .rolling(window=5, min_periods=1)
            .mean()
        )
        # Calculate the acceleration between the player and the ball
        self.data_kinexon_event_player_ball["acceleration"] = (
            self.data_kinexon_event_player_ball["distance"].diff()
            / self.data_kinexon_event_player_ball["time_diff_seconds"]
            # ).shift(-3)
        )
        # smooth the acceleration data
        self.data_kinexon_event_player_ball["acceleration_smoothed"] = (
            self.data_kinexon_event_player_ball["acceleration"]
            .rolling(window=2, min_periods=1)
            .mean()
        )

        # Apply rolling smoothing to ball's x and y coordinates
        self.data_kinexon_event_player_ball["smoothed_pos_x_ball"] = (
            self.data_kinexon_event_player_ball["pos_x_ball"]
            .rolling(window=5)
            .mean()
        )
        self.data_kinexon_event_player_ball["smoothed_pos_y_ball"] = (
            self.data_kinexon_event_player_ball["pos_y_ball"]
            .rolling(window=5)
            .mean()
        )

        # Calculate smoothed distance to the goal
        self.data_kinexon_event_player_ball["smoothed_distance_ball_goal"] = (
            (
                self.data_kinexon_event_player_ball["smoothed_pos_x_ball"]
                - self.goal_position[0]
            )
            ** 2
            + (
                self.data_kinexon_event_player_ball["smoothed_pos_y_ball"]
                - self.goal_position[1]
            )
            ** 2
        ) ** 0.5

        # Check if the ball is moving towards the goal by checking the smoothed distance difference
        self.data_kinexon_event_player_ball["ball_moving_towards_goal"] = (
            self.data_kinexon_event_player_ball[
                "smoothed_distance_ball_goal"
            ].diff()
            < 0
        )

        # Determine if the ball is moving towards the goal based on the X-coordinate
        goal_x = self.goal_position[0]

        # Check direction of ball's x-movement relative to the goal (using smoothed X-coordinates)
        if (
            goal_x
            < self.data_kinexon_event_player_ball["smoothed_pos_x_ball"].mean()
        ):
            # If the goal is to the left of the ball, check if the ball is moving left (x decreasing)
            self.data_kinexon_event_player_ball[
                "ball_moving_towards_goal_x"
            ] = (
                self.data_kinexon_event_player_ball[
                    "smoothed_pos_x_ball"
                ].diff()
                < 0
            ).astype(
                int
            )
        else:
            # If the goal is to the right of the ball, check if the ball is moving right (x increasing)
            self.data_kinexon_event_player_ball[
                "ball_moving_towards_goal_x"
            ] = (
                self.data_kinexon_event_player_ball[
                    "smoothed_pos_x_ball"
                ].diff()
                > 0
            ).astype(
                int
            )

        # Combine direction check with smoothed distance change to refine the condition
        self.data_kinexon_event_player_ball["ball_moving_towards_goal"] = (
            (
                self.data_kinexon_event_player_ball[
                    "smoothed_distance_ball_goal"
                ].diff()
                < 0
            )
            & (
                self.data_kinexon_event_player_ball[
                    "ball_moving_towards_goal_x"
                ]
                == 1
            )
        ).astype(int)

    def _find_peaks_speed(self) -> List[int]:
        """Find peaks in the speed data."""
        peaks, _ = find_peaks(
            self.data_kinexon_event_player_ball["speed"], distance=10
        )
        return peaks if peaks is not None else []

    def _find_peaks_acceleration(self) -> List[int]:
        """Find peaks in the acceleration data."""
        peaks, _ = find_peaks(
            self.data_kinexon_event_player_ball["acceleration"],
            distance=5,
        )
        return peaks if peaks is not None else []

    def _remove_peaks_not_moving_towards_goal(
        self, peaks: List[int]
    ) -> List[int]:
        """Remove peaks where the ball is not moving towards the goal."""
        cleaned_peaks = []

        for peak in peaks:
            # Check if the ball is moving towards the goal at the peak and within the +-5 index range
            if (
                self.data_kinexon_event_player_ball["ball_moving_towards_goal"]
                .iloc[
                    # peak
                    max(peak - 2, 0) : min(
                        peak + 2, len(self.data_kinexon_event_player_ball)
                    )
                ]
                .any()  # Check if any value in the range is 1 (i.e., moving towards the goal)
            ):
                cleaned_peaks.append(peak)
            #     print(
            #         f'Found potential peak at index {peak} at time {self.data_kinexon_event_player_ball["time"].iloc[peak]}'
            #     )
            # else:
            # cleaned_peaks.append(peak)
            # print(
            #     f'Removed potential peak at index {peak} at time {self.data_kinexon_event_player_ball["time"].iloc[peak]}'
            # )

        return cleaned_peaks

    def _remove_peaks_above_player_to_ball_distance_threshold(
        self, peaks: List[int], threshold_dist: float = 2.5
    ) -> List[int]:
        """Remove peaks where the distance between player and ball is below a threshold."""
        cleaned_peaks = []

        for peak in peaks:
            if (
                self.data_kinexon_event_player_ball["distance"].iloc[peak]
                < threshold_dist
            ):
                cleaned_peaks.append(peak)

        return cleaned_peaks

    def _remove_peaks_too_close_to_tagging_time(
        self, peaks: List[int], threshold_time: float = 1.5
    ) -> List[int]:
        """Remove peaks that are too close to the event tagging time."""
        # Find the index of the event time tagged
        event_time_tagged = self.data_kinexon_event_player_ball["time"]
        if event_time_tagged.empty:
            return []
        else:
            event_time_tagged = event_time_tagged.iloc[-1]

        # Find the index of the event time tagged - threshold_time
        idx_event_threshold = self.data_kinexon_event_player_ball[
            self.data_kinexon_event_player_ball["time"]
            < event_time_tagged - pd.Timedelta(seconds=threshold_time)
        ]
        if idx_event_threshold.empty:
            return []
        else:
            idx_event_threshold = idx_event_threshold.index[-1]

        # Remove peaks that are too close to the tagging time
        cleaned_peaks = []
        for peak in peaks:
            if (
                self.data_kinexon_event_player_ball.index[peak]
                < idx_event_threshold
            ):
                cleaned_peaks.append(peak)

        return cleaned_peaks

    def _clean_peaks(self, peaks: List[int]) -> List[int]:
        """Clean the peaks by removing irrelevant ones."""

        # Remove peaks where the ball is not moving towards the goal
        # Remove peaks where the distance to the goal is too large
        peaks = self._remove_peaks_above_player_to_ball_distance_threshold(
            peaks, threshold_dist=1.2
        )
        # Remove peaks that are too close to the tagging time
        peaks = self._remove_peaks_too_close_to_tagging_time(peaks)

        # peaks = self._remove_peaks_not_moving_towards_goal(peaks)

        return peaks


# Example usage:
# throw_detector = ThrowEventDetector(event_type, data_kinexon_event, id_player)
# throw_time, merged_data, peaks = throw_detector.find_throw_timestamp()
# throw_detector.plot_event_syncing()
