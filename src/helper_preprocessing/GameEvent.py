import os
import json
import pandas as pd
from typing import Optional, Dict
import pickle
from glob import glob
from scipy.signal import find_peaks
import math
import numpy as np

from matplotlib import pyplot as plt


class GameEvent:
    def __init__(
        self,
        dict_event: Dict,
        df_kinexon: pd.DataFrame,
        plot_sync: bool = False,
        render_video_event: bool = False,
        path_result_csv: str = None,
    ):
        self.dict_event = dict_event
        # self.df_kinexon = df_kinexon
        self.event_id = dict_event.get("id")
        self.event_type = dict_event.get("type")

        self.path_result_csv = path_result_csv
        # add hack to convert time to datetime
        # and add 2 hours to the time because of the time zone
        self.event_time_tagged = pd.to_datetime(
            dict_event.get("time")
        ).replace(tzinfo=None) + pd.to_timedelta(2, unit="h")
        # subtract 15 seconds from the tagged time to get the start time of the event
        self.event_time_start = self.event_time_tagged - pd.Timedelta(
            seconds=15
        )
        self.match_time = dict_event.get("match_time")
        self.match_clock = dict_event.get("match_clock")
        self.match_clock_in_s = dict_event.get("match_clock_in_s", None)
        self.competitor = dict_event.get("competitor")
        self.home_score = dict_event.get("home_score")
        self.away_score = dict_event.get("away_score")
        self.scorer = dict_event.get("scorer")
        self.assists = dict_event.get("assists", [])
        self.zone = dict_event.get("zone")
        if self.event_type == "score_change":
            self.player = dict_event.get("scorer")
        else:
            self.player = dict_event.get("player")

        self.players = dict_event.get("players", [])
        self.shot_type = dict_event.get("shot_type")
        self.method = dict_event.get("method")
        self.outcome = dict_event.get("outcome")
        self.suspension_minutes = dict_event.get("suspension_minutes")
        self.attack_direction = dict_event.get("attack_direction")

        self.id_goalkeeper = dict_event.get("id_goalkeeper")

        self.event_time_throw = None

        # Define needed variables
        self.name_player = None
        self.id_player = None
        self.pos_x_player = None
        self.pos_y_player = None
        # Ball
        self.id_ball = None
        self.pos_x_ball = None
        self.pos_y_ball = None
        # Blocker
        self.name_blocker = None
        self.id_blocker = None
        self.pos_x_blocker = None
        self.pos_y_blocker = None
        # Goalkeeper
        self.name_goalkeeper = None
        # self.id_goalkeeper = None
        self.pos_x_goalkeeper = None
        self.pos_y_goalkeeper = None
        # Assists
        self.name_assist = None
        self.id_assist = None
        self.pos_x_assist = None
        self.pos_y_assist = None

        self.peaks = None

        self.data_kinexon_event_player_ball = None

        print(
            f"Event type: {self.event_type} - Score: {self.home_score}-{self.away_score} at time {self.match_clock}"
        )

        # Extract kinexon data for the event
        self.data_kinexon_event = self._get_data_kinexon(df_kinexon)

        # Determine attack direction

    def process_event(self) -> None:
        self._extract_nested_player()
        self._extract_nested_players()
        self._extract_nested_assist()

        self.find_timestamp_of_throw()
        # Now set the information of the player, blocker, goalkeeper and assist
        self.set_throw_information()
        self._calculate_speed_acceleration_direction()
        self.init_features()
        self.set_features()

        self.write_features()

        print(
            f"\tDetails: Scorer: {self.name_player} - Assist: {self.name_assist} - Player: {self.name_player} - Blocker: {self.name_blocker} - Goalkeeper: {self.name_goalkeeper}"
        )

    def write_features(self) -> None:
        """Writes or appends the features of the GameEvent to a CSV file."""
        # Define the path to the CSV file
        path_features = self.path_result_csv

        # Check if the file already exists
        if os.path.exists(path_features):
            # Load the existing CSV file
            df_features = pd.read_csv(path_features)
        else:
            # Create a new DataFrame if the file does not exist
            df_features = pd.DataFrame()

        # Create a new row with the features of the GameEvent
        row = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_time_tagged": self.event_time_tagged,
            "event_time_start": self.event_time_start,
            "event_time_throw": self.event_time_throw,
            "id_player": self.id_player,
            "id_goalkeeper": self.id_goalkeeper,
            "id_blocker": self.id_blocker,
            "id_assist": self.id_assist,
            "match_time": self.match_time,
            "match_clock": self.match_clock,
            "competitor": self.competitor,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "scorer": self.name_player,
            "assists": self.name_assist,
            "zone": self.zone,
            # "players": self.players,
            "shot_type": self.shot_type,
            "method": self.method,
            "outcome": self.outcome,
            "suspension_minutes": self.suspension_minutes,
            "attack_direction": self.attack_direction,
            "distance_player_goal": self.distance_player_goal,
            "distance_ball_goal": self.distance_ball_goal,
            "distance_goalkeeper_goal": self.distance_goalkeeper_goal,
            "distance_player_goalkeeper": self.distance_player_goalkeeper,
            "distance_nearest_defender": self.distance_nearest_defender,
            "num_defenders_close": self.num_defenders_close,
            "angle_ball_goal": self.angle_ball_goal,
            # "speed_throw": self.speed_throw,
            "score_difference": self.score_difference,
            "match_clock_normalized": self.match_clock_normalized,
            "was_goal": 1 if self.event_type == "score_change" else 0,
        }

        # in row, replance None with empty string
        for key, value in row.items():
            if value is None or pd.isnull(value):
                row[key] = ""

        # Append the row to the DataFrame
        # Fill NA values in 'row' with a default value (e.g., 0 or empty string)
        # row_filled = pd.DataFrame(row).fillna("")

        # Concatenate the filled DataFrame
        df_features = pd.concat(
            [df_features, pd.DataFrame([row])], ignore_index=True
        )

        # Save the DataFrame to the CSV file
        df_features.to_csv(path_features, index=False)

    def init_features(self) -> None:
        """Initializes the features of the GameEvent."""
        self.distance_player_goal = None
        self.distance_ball_goal = None
        self.distance_goalkeeper_goal = None
        self.distance_player_goalkeeper = None
        self.distance_nearest_defender = None
        self.num_defenders_close = None
        self.angle_ball_goal = None

        # score difference, but difference between home and away score
        if self.competitor == "home":
            self.score_difference = self.home_score - self.away_score
        else:
            self.score_difference = self.away_score - self.home_score

        # Normalized match clock is time in seconds divided by 3600
        if self.match_clock_in_s is not None:  # it is currently in seconds
            self.match_clock_normalized = self.match_clock_in_s / 3600
        else:
            self.match_clock_normalized = None

    def set_features(self) -> None:
        """Sets the features of the GameEvent."""

        if self.event_type not in [
            "score_change",
            "shot_off_target",
            "shot_blocked",
            "shot_saved",
            "seven_m_missed",
        ]:
            return

        if self.event_type == "shot_blocked":
            print("Here.")
            pass
        # get throw time
        time_throw = self.event_time_throw
        # get player id
        id_player = self.id_player.split(":")[-1]
        # get ball id
        id_ball = self.id_ball.split(":")[-1]
        # get goalkeeper id
        if self.id_goalkeeper:
            id_goalkeeper = self.id_goalkeeper.split(":")[-1]

        # attack direction
        attack_direction = self.attack_direction
        # Get the goal position
        if attack_direction == "right":
            goal_position_x = 0
            goal_position_y = 10
        else:
            goal_position_x = 40
            goal_position_y = 10

        # get moment of the throw
        df_moment_throw = self.data_kinexon_event[
            self.data_kinexon_event["time"] == time_throw
        ].copy()

        # Now around this throw, we can smooth speed and acceleration of player and ball
        df_moment_throw.loc[:, "speed_player"] = (
            df_moment_throw["speed"]
            .rolling(window=5, min_periods=1, center=True)
            .mean()
        )

        # Get boolean index of player
        idx_player_mask = df_moment_throw["league_id"] == id_player

        # Ensure there are matching rows before proceeding
        if idx_player_mask.any():
            # Find the index of the player
            idx_player = idx_player_mask.idxmax()

            # find the row in df_moment_throw where the player is located
            row = df_moment_throw.loc[idx_player]

            # Safely compute distance
            self.distance_player_goal = (
                (
                    df_moment_throw["pos_x"].loc[int(idx_player)]
                    - goal_position_x
                )
                ** 2
                + (df_moment_throw["pos_y"].loc[idx_player] - goal_position_y)
                ** 2
            ) ** 0.5
        else:
            self.distance_player_goal = None

        # calculate distance of ball to goal
        idx_ball = df_moment_throw["league_id"] == id_ball

        if idx_ball.any():
            idx_ball = idx_ball.idxmax()

            self.distance_ball_goal = (
                (df_moment_throw["pos_x"].loc[idx_ball] - goal_position_x) ** 2
                + (df_moment_throw["pos_y"].loc[idx_ball] - goal_position_y)
                ** 2
            ) ** 0.5

        # calculate distance of goalkeeper to goal
        if self.id_goalkeeper:
            idx_goalkeeper = df_moment_throw["league_id"] == id_goalkeeper

            if idx_goalkeeper.any():
                idx_goalkeeper = idx_goalkeeper.idxmax()
                self.distance_goalkeeper_goal = (
                    (
                        df_moment_throw["pos_x"].loc[idx_goalkeeper]
                        - goal_position_x
                    )
                    ** 2
                    + (
                        df_moment_throw["pos_y"].loc[idx_goalkeeper]
                        - goal_position_y
                    )
                    ** 2
                ) ** 0.5

                if idx_player_mask.any():
                    self.distance_player_goalkeeper = (
                        (
                            df_moment_throw["pos_x"].loc[idx_player]
                            - df_moment_throw["pos_x"].loc[idx_goalkeeper]
                        )
                        ** 2
                        + (
                            df_moment_throw["pos_y"].loc[idx_player]
                            - df_moment_throw["pos_y"].loc[idx_goalkeeper]
                        )
                        ** 2
                    ) ** 0.5

            # calculate distance of nearest defender to player
            self.distance_nearest_defender = 1000
            # number of defenders close to the player (< 1.5m)
            self.num_defenders_close = 0
            for idx, player in df_moment_throw.iterrows():
                # insert "home" or "away" to the player id based on the group_id where 1 is home and 2 is away
                player["competitor"] = (
                    "home" if player["group_id"] == 1 else "away"
                )
                if (
                    player["league_id"] == id_player
                    or player["league_id"] == id_goalkeeper
                    or player["league_id"] == id_ball
                    or player["competitor"] != self.competitor
                ):
                    continue
                distance = (
                    (
                        player["pos_x"]
                        - df_moment_throw["pos_x"].loc[idx_player]
                    )
                    ** 2
                    + (
                        player["pos_y"]
                        - df_moment_throw["pos_y"].loc[idx_player]
                    )
                    ** 2
                ) ** 0.5
                if distance < self.distance_nearest_defender:
                    self.distance_nearest_defender = distance
                if distance < 1.5:
                    self.num_defenders_close += 1

        # angle of the throw, use atan2 to get the angle in radians (of the ball and the goal)
        if idx_ball.any():
            angle_throw = (
                math.atan2(
                    goal_position_y - df_moment_throw["pos_y"].loc[idx_ball],
                    goal_position_x - df_moment_throw["pos_x"].loc[idx_ball],
                )
                * 180
                / math.pi
            )

            # speed of throw
            self.speed_throw = df_moment_throw["speed"].loc[idx_ball]

            if angle_throw < 0:
                angle_throw += 360
            if angle_throw > 180:
                angle_throw = 360 - angle_throw

            self.angle_ball_goal = abs(90 - angle_throw)
        else:
            self.angle_ball_goal = None

    def set_throw_information(self) -> None:
        """Sets the timestamp of the throw event."""
        if not self.event_time_throw:
            return

        # print event type and throw time
        # print(
        #     f"GameEvent: Event type: {self.event_type} - Throw time: {self.event_time_throw}"
        # )

        # Create dataframe for the moment of the throw
        df_moment_throw = self.data_kinexon_event[
            self.data_kinexon_event["time"] == self.event_time_throw
        ]

        # print(df_moment_throw)

        id_player = self.id_player.split(":")[-1]

        # Set the position of the player based on the id of the player and the time of the throw event
        self.pos_x_player = df_moment_throw[
            df_moment_throw["league_id"] == id_player
        ]["pos_x"].iloc[0]

        self.pos_y_player = df_moment_throw[
            df_moment_throw["league_id"] == id_player
        ]["pos_y"].iloc[0]

        # Set the position of the ball based on the id of the ball and the time of the throw event
        self.pos_x_ball = df_moment_throw[
            df_moment_throw["league_id"] == self.id_ball
        ]["pos_x"].iloc[0]

        self.pos_y_ball = df_moment_throw[
            df_moment_throw["league_id"] == self.id_ball
        ]["pos_y"].iloc[0]

        # Set the position of the blocker based on the id of the blocker and the time of the throw event
        if self.id_blocker:
            id_blocker = self.id_blocker.split(":")[-1]
            if id_blocker not in df_moment_throw["league_id"].values:
                pass
            else:
                self.pos_x_blocker = df_moment_throw[
                    df_moment_throw["league_id"] == id_blocker
                ]["pos_x"].iloc[0]

                self.pos_y_blocker = df_moment_throw[
                    df_moment_throw["league_id"] == id_blocker
                ]["pos_y"].iloc[0]

        # Set the position of the goalkeeper based on the id of the goalkeeper and the time of the throw event
        if self.id_goalkeeper:
            id_goalkeeper = self.id_goalkeeper.split(":")[-1]
            # Check if id goalkeeper is in the data
            if id_goalkeeper not in df_moment_throw["league_id"].values:
                pass
            else:
                self.pos_x_goalkeeper = df_moment_throw[
                    df_moment_throw["league_id"] == id_goalkeeper
                ]["pos_x"].iloc[0]

                self.pos_y_goalkeeper = df_moment_throw[
                    df_moment_throw["league_id"] == id_goalkeeper
                ]["pos_y"].iloc[0]

    def _extract_nested_assist(self) -> None:
        """Extracts the nested assist from the event data."""
        if not self.assists:
            return
        self.id_assist = self.assists[0].get("id")
        self.name_assist = self.assists[0].get("name")

    def _extract_nested_player(self) -> None:
        """Extracts the nested player from the event data."""
        if self.player is None:
            return
        self.id_player = self.player.get("id")
        self.name_player = self.player.get("name")

    def _extract_nested_players(self) -> None:
        """Extracts the nested players from the event data."""

        # Example: [{'id': 'sr:player:124865', 'name': 'Quenstedt, Dario', 'type': 'goalkeeper'}]
        for player in self.players:
            # check if type is in player
            if "type" not in player:
                continue

            if player["type"] == "shot":
                self.name_player = player["name"]
                self.id_player = player["id"]
            elif player["type"] == "blocker":
                self.name_blocker = player["name"]
                self.id_blocker = player["id"]
            elif player["type"] == "saved":
                self.name_goalkeeper = player["name"]
                self.id_goalkeeper = player["id"]
            elif player["type"] == "goalkeeper":
                self.name_goalkeeper = player["name"]
                self.id_goalkeeper = player["id"]

    def _clean_peaks(self, peaks: list) -> list:
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
            if self.data_kinexon_event_player_ball["distance"].iloc[peak] < 2.5
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
        # find indices of peaks in df_ball that are
        # 3 seconds before row["time"]
        idx_event_threshold = self.data_kinexon_event_player_ball[
            self.data_kinexon_event_player_ball["time"]
            < self.event_time_tagged - pd.Timedelta(seconds=1.5)
        ].index[-1]

        # remove peaks if located in df_ball before idx_event_threshold
        peaks = [
            peak
            for peak in peaks
            if self.data_kinexon_event_player_ball.index[peak]
            < idx_event_threshold
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

    def _calculate_speed_acceleration_direction(
        self, smoothing_window: int = 5
    ) -> None:
        """
        Calculate the speed, acceleration, direction in the Kinexon data with smoothing.

        Args:
        df_kinexon (pd.DataFrame): The Kinexon data.
        smoothing_window (int): The window size for smoothing. Default is 5.

        Returns:
        None
        """

        # Sort the dataframe to ensure correct diff calculations
        df_kinexon = self.data_kinexon_event.sort_values(
            by=["league_id", "time"]
        )

        # Apply rolling mean to smooth the x and y positions
        df_kinexon["x_smoothed"] = df_kinexon.groupby("league_id")[
            "pos_x"
        ].transform(
            lambda x: x.rolling(window=smoothing_window, min_periods=1).mean()
        )
        df_kinexon["y_smoothed"] = df_kinexon.groupby("league_id")[
            "pos_y"
        ].transform(
            lambda y: y.rolling(window=smoothing_window, min_periods=1).mean()
        )

        # Calculate the differences in smoothed position and time
        df_kinexon["x_diff"] = df_kinexon.groupby("league_id")[
            "x_smoothed"
        ].diff()
        df_kinexon["y_diff"] = df_kinexon.groupby("league_id")[
            "y_smoothed"
        ].diff()
        df_kinexon["time_diff"] = (
            df_kinexon.groupby("league_id")["time"].diff().dt.total_seconds()
        )

        # Calculate speed
        df_kinexon["speed"] = (
            np.sqrt(df_kinexon["x_diff"] ** 2 + df_kinexon["y_diff"] ** 2)
            / df_kinexon["time_diff"]
        )

        # Calculate acceleration
        df_kinexon["acceleration"] = (
            df_kinexon.groupby("league_id")["speed"].diff()
            / df_kinexon["time_diff"]
        )

        # Calculate direction in degrees
        df_kinexon["direction"] = np.degrees(
            np.arctan2(df_kinexon["y_diff"], df_kinexon["x_diff"])
        )
        # Normalize direction to make it between 0 and 360 degrees
        df_kinexon["direction"] = (df_kinexon["direction"] + 360) % 360

        # Insert the calculated speed, acceleration, and direction back into the original dataframe
        self.data_kinexon_event = df_kinexon

    def find_timestamp_of_throw(self) -> Optional[pd.Timestamp]:
        """Finds the timestamp of the throw event in the kinexon data."""
        # determine the player who threw the ball

        # check if type is one of the following
        if self.event_type in [
            "score_change",
            "shot_off_target",
            "shot_blocked",
            "shot_saved",
            "seven_m_missed",
        ]:
            pass
        else:
            return None

        # Extract most used ball id
        df_value_count_ball_ids = self.data_kinexon_event[
            "league_id"
        ].value_counts()

        # Set current ball ID to the one with the most occurrences that have "ball" or "Ball" in the name
        self.id_ball = df_value_count_ball_ids[
            df_value_count_ball_ids.index.str.contains("ball|Ball")
        ].idxmax()

        self.data_kinexon_event_player = self.data_kinexon_event[
            self.data_kinexon_event["league_id"]
            == self.id_player.split(":")[-1]
        ]

        self.data_kinexon_event_player = self.data_kinexon_event_player[
            ["time", "pos_x", "pos_y"]
        ]

        self.data_kinexon_event_ball = self.data_kinexon_event[
            self.data_kinexon_event["league_id"] == self.id_ball
        ]

        # only columns of the ball data that are needed
        self.data_kinexon_event_ball = self.data_kinexon_event_ball[
            ["time", "pos_x", "pos_y"]
        ]

        # merge the data of the thrower and the ball, based on the timestamp, rename the columns of the ball data with suffix "_ball"
        self.data_kinexon_event_player_ball = pd.merge_asof(
            self.data_kinexon_event_player,
            self.data_kinexon_event_ball,
            on="time",
            suffixes=("", "_ball"),
        )

        # add distance to the ball
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

        # Add speed to the ball
        self.data_kinexon_event_player_ball["time_diff"] = (
            self.data_kinexon_event_player_ball["time"]
            - self.data_kinexon_event_player_ball["time"].shift()
        )

        # Convert time difference to seconds
        self.data_kinexon_event_player_ball["time_diff_seconds"] = (
            self.data_kinexon_event_player_ball["time_diff"].dt.total_seconds()
        )

        # Calculate speed
        self.data_kinexon_event_player_ball["speed"] = (
            self.data_kinexon_event_player_ball["distance"]
            / self.data_kinexon_event_player_ball["time"]
            .diff()
            .dt.total_seconds()
        )
        # add acceleration to the ball
        self.data_kinexon_event_player_ball["acceleration"] = (
            self.data_kinexon_event_player_ball["distance"].diff()
            / self.data_kinexon_event_player_ball["time"]
            .diff()
            .dt.total_seconds()
        )

        # for some reason we need to shift acceleration by 3 to capture the throw
        self.data_kinexon_event_player_ball["acceleration"] = (
            self.data_kinexon_event_player_ball["acceleration"].shift(-3)
        )

        # get peaks
        peaks, _ = find_peaks(
            self.data_kinexon_event_player_ball["acceleration"], distance=10
        )
        # Make empty list for peaks if None
        if peaks is None:
            peaks = []

        # Clean peaks
        self.peaks = self._clean_peaks(peaks)

        # Check if there are any peaks
        if self.peaks is None or len(self.peaks) == 0:
            print("\tNo peaks found.")
            return None

        self.event_time_throw = self.data_kinexon_event_player_ball[
            "time"
        ].iloc[self.peaks[-1]]

        # print time
        print(
            f"Found throw time for event {self.event_type}: Throw time: {self.event_time_throw}"
        )

    def _get_data_kinexon(
        self, df_kinexon: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Retrieves kinexon data for the event's start and tagged time."""
        if df_kinexon is not None:
            # print(f"event_time_start: {self.event_time_start}")
            # print(f"event_time_tagged: {self.event_time_tagged}")
            # print(df_kinexon["time"].head())

            diff = df_kinexon["time"] - self.event_time_start
            idx_start = diff.abs().idxmin()

            diff = df_kinexon["time"] - self.event_time_tagged
            idx_tagged = diff.abs().idxmin()

            data_kinexon = df_kinexon.loc[idx_start:idx_tagged]

            if not data_kinexon.empty:
                return data_kinexon

        return None

    def save_event(self, directory: str):
        """Saves the GameEvent and its kinexon data to files."""
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save kinexon data to a pickle file
        kinexon_file_path = os.path.join(
            directory, f"event_{self.event_id}_positions.csv"
        )
        if self.data_kinexon_event is not None:
            self.data_kinexon_event.to_csv(kinexon_file_path, index=False)

        # Save event data to a JSON file
        event_file_path = os.path.join(
            directory, f"event_{self.event_id}_sportradar.json"
        )
        with open(event_file_path, "w") as f:
            json.dump(self.to_dict(kinexon_file_path), f, indent=4)

        print(f"Event {self.event_id} saved to {directory}.")

    def to_dict(self, kinexon_file_path: str) -> Dict:
        """Converts the GameEvent instance to a dictionary for saving."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_time_tagged": self.event_time_tagged.isoformat(),
            "event_time_start": self.event_time_start.isoformat(),
            "match_time": self.match_time,
            "match_clock": self.match_clock,
            "competitor": self.competitor,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "scorer": self.scorer,
            "assists": self.assists,
            "zone": self.zone,
            "players": self.players,
            "shot_type": self.shot_type,
            "method": self.method,
            "outcome": self.outcome,
            "suspension_minutes": self.suspension_minutes,
            "kinexon_file_path": kinexon_file_path,  # Path to the kinexon data file
        }

    @staticmethod
    def load_event(directory: str, event_id: int) -> "GameEvent":
        """Loads a GameEvent and its kinexon data from files."""
        # Load event data from JSON file
        event_file_path = os.path.join(
            directory, f"event_{event_id}_sportradar.json"
        )
        with open(event_file_path, "r") as f:
            event_data = json.load(f)

        # Load kinexon data from pickle file
        kinexon_file_path = event_data.get("kinexon_file_path")
        if kinexon_file_path:
            with open(os.path.join(directory, kinexon_file_path), "rb") as f:
                df_kinexon = pickle.load(f)
        else:
            df_kinexon = pd.DataFrame()

        # Create a GameEvent instance from the loaded data
        return GameEvent(event_data, df_kinexon)


# Example Usage
if __name__ == "__main__":
    # Assuming you have event_data and df_kinexon available
    event_data = {
        "id": 1525671897,
        "type": "score_change",
        "time": "2023-08-24T17:08:56+00:00",
        "match_time": 6,
        "match_clock": "5:24",
        "competitor": "away",
        "home_score": 2,
        "away_score": 4,
        "scorer": {"id": "sr:player:1262014", "name": "Buchner, Vincent"},
        "assists": [
            {
                "id": "sr:player:925808",
                "name": "Michalczik, Marian",
                "type": "primary",
            }
        ],
        "zone": "wing_left",
        "players": [
            {
                "id": "sr:player:1291472",
                "name": "Ferlin, Klemen",
                "type": "goalkeeper",
            }
        ],
    }

    df_kinexon = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2023-08-24T17:08:41", "2023-08-24T17:08:56"]
            ),
            "some_metric": [1.5, 2.0],
            "another_metric": [3.4, 3.9],
        }
    )

    # Create a GameEvent
    event = GameEvent(event_data, df_kinexon)

    # Save the event and kinexon data
    event.save_event(directory="events")

    # Load the event back
    loaded_event = GameEvent.load_event(
        directory="events", event_id=1525671897
    )
    print(loaded_event.to_dict("events"))
