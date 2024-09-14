import pandas as pd
import os
import math

try:
    from helper_find_throw_point import find_timestamp_of_throw
except ImportError:
    from helper_preprocessing.helper_find_throw_point import (
        find_timestamp_of_throw,
    )


class GameEvent:
    """Class representing a game event."""

    def __init__(self, event: dict, path_to_kinexon_scene: str) -> None:
        """
        Initialize the GameEvent object with the event data.

        Args:
            event (dict): Dictionary containing event data.
        """
        self.event = event

        if event["type"] == "shot_blocked":
            pass

        self.path_to_kinexon_scene = path_to_kinexon_scene
        # Initialize event metadata
        self._init_event_metadata()

        # Initialize kinexon metadata
        self._init_kinexon_metadata()

        # Find the throw point
        (
            self.event_time_throw,
            self.data_kinexon_event_player_ball,
            self.peaks,
        ) = find_timestamp_of_throw(
            self.event_type, self.df_kinexon_event, self.id_player
        )
        # Extract throw moment
        self.df_moment_throw = self.df_kinexon_event[
            self.df_kinexon_event["time"] == self.event_time_throw
        ]

        # Initialize metadata of the throw event
        self._init_metadata_throw()

        self._calc_features_throw()

    def _calc_features_throw(self) -> None:
        """Calculate features of the throw event."""

        self.distance_player_goal = None
        self.distance_player_goalkeeper = None
        self.distance_player_blocker = None
        self.distance_goalkeeper_goal = None
        self.angle_throw = None
        self.number_defenders_close = None
        self.distance_nearest_defender = None
        self.angle_ball_goal = None
        self.speed_ball_at_throw = None
        self.id_nearest_defender = None
        self.name_nearest_defender = None

        if not self.id_player:
            return

        # Get the goal position
        if self.attack_direction == "right":
            goal_position_x = 0
            goal_position_y = 10
        else:
            goal_position_x = 40
            goal_position_y = 10

        if self.throw_player_pos_x is not None:
            self.distance_player_goal = (
                (self.throw_player_pos_x - goal_position_x) ** 2
                + (self.throw_player_pos_y - goal_position_y) ** 2
            ) ** 0.5
            print(f"\t> Distance player goal: {self.distance_player_goal} ✅")

        if (
            self.throw_player_pos_x is not None
            and self.pos_x_goalkeeper is not None
        ):
            self.distance_player_goalkeeper = (
                (self.throw_player_pos_x - self.pos_x_goalkeeper) ** 2
                + (self.throw_player_pos_y - self.pos_y_goalkeeper) ** 2
            ) ** 0.5
            print(
                f"\t> Distance player goalkeeper: {self.distance_player_goalkeeper} ✅"
            )

        if (
            self.throw_player_pos_x is not None
            and self.pos_x_blocker is not None
        ):
            self.distance_player_blocker = (
                (self.throw_player_pos_x - self.pos_x_blocker) ** 2
                + (self.throw_player_pos_y - self.pos_y_blocker) ** 2
            ) ** 0.5
            print(
                f"\t> Distance player blocker: {self.distance_player_blocker} ✅"
            )

        if self.pos_x_goalkeeper is not None:
            self.distance_goalkeeper_goal = (
                (self.pos_x_goalkeeper - goal_position_x) ** 2
                + (self.pos_y_goalkeeper - goal_position_y) ** 2
            ) ** 0.5
            print(
                f"\t> Distance goalkeeper goal: {self.distance_goalkeeper_goal} ✅"
            )

        # Get the angle of the throw using atan2
        if (
            self.throw_player_pos_x is not None
            and self.throw_player_pos_y is not None
            and self.throw_ball_pos_x is not None
            and self.throw_ball_pos_y is not None
        ):
            angle_throw = (
                math.atan2(
                    self.throw_ball_pos_y - self.throw_player_pos_y,
                    self.throw_ball_pos_x - self.throw_player_pos_x,
                )
                * 180
                / math.pi
            )
            if angle_throw < 0:
                angle_throw += 360
            if angle_throw > 180:
                angle_throw = 360 - angle_throw

            self.angle_ball_goal = abs(90 - angle_throw)
            print(f"\t> Angle Ball <-> Goal: {self.angle_ball_goal} ✅")

        # get speed of the throw up too 200 ms after the throw
        # Get speed of the throw up to 200 ms after the throw for league_id == 3 (ball)
        if not self.event_time_throw is None:
            df_speed_after_throw = self.df_kinexon_event[
                (self.df_kinexon_event["time"] >= self.event_time_throw)
                & (
                    self.df_kinexon_event["time"]
                    <= self.event_time_throw + pd.Timedelta(200, unit="ms")
                )
                & (
                    self.df_kinexon_event["group_id"] == 3
                )  # Filter for league_id == 3 (ball)
            ]

            # Calculate average speed
            if not df_speed_after_throw.empty:
                avg_speed = df_speed_after_throw["speed"].max()
                self.speed_ball_at_throw = avg_speed
                print(
                    f"\t> Average speed of ball after throw: {self.speed_ball_at_throw} m/s ✅"
                )

        if not self.df_moment_throw.empty:
            # Calculate distance to nearest defender
            self.distance_nearest_defender = 1000
            self.number_defenders_close = 0

            for index, row in self.df_moment_throw.iterrows():
                row["competitor"] = "home" if row["group_id"] == 1 else "away"

                if (
                    self.id_goalkeeper is None
                    or self.id_player is None
                    or row["league_id"] == self.id_player.split(":")[-1]
                    or row["league_id"] == self.id_goalkeeper.split(":")[-1]
                    or row["league_id"] == self.id_ball
                    or row["competitor"] != self.competitor
                ):
                    continue

                distance = (
                    (self.throw_player_pos_x - row["pos_x"]) ** 2
                    + (self.throw_player_pos_y - row["pos_y"]) ** 2
                ) ** 0.5

                if distance < self.distance_nearest_defender:
                    self.distance_nearest_defender = distance
                    self.id_nearest_defender = row["league_id"]
                    self.name_nearest_defender = row["full_name"]

                if distance < 1.5:
                    self.number_defenders_close += 1

            print(
                f"\t> Distance nearest defender with name {self.name_nearest_defender}: {self.distance_nearest_defender} ✅"
            )
        pass

    def _init_metadata_throw(self) -> None:
        """Initialize the metadata of the throw event."""

        self.throw_player_pos_x = None
        self.throw_player_pos_y = None
        self.throw_ball_pos_x = None
        self.throw_ball_pos_y = None
        self.pos_x_blocker = None
        self.pos_y_blocker = None
        self.pos_x_goalkeeper = None
        self.pos_y_goalkeeper = None

        # Check if the player id exists
        if not self.id_player:
            return

        # Extract the player id without the prefix
        id_player = self.id_player.split(":")[-1]

        # Use the following to extract ball id
        # Extract most used ball id from the kinexon data
        df_value_count_ball_ids = self.df_kinexon_event[
            "league_id"
        ].value_counts()
        try:
            # Set current ball ID to the one with the most occurrences that have "ball" or "Ball" in the name
            self.id_ball = df_value_count_ball_ids[
                df_value_count_ball_ids.index.str.contains("ball|Ball")
            ].idxmax()
        except AttributeError:
            self.id_ball = None

        if self.df_moment_throw.empty or self.id_ball is None:
            return

        # Extract the player and ball data
        self.throw_player_pos_x = self.df_moment_throw[
            self.df_moment_throw["league_id"] == id_player
        ]["pos_x"].values[0]

        self.throw_player_pos_y = self.df_moment_throw[
            self.df_moment_throw["league_id"] == id_player
        ]["pos_y"].values[0]

        self.throw_ball_pos_x = self.df_moment_throw[
            self.df_moment_throw["league_id"] == self.id_ball
        ]["pos_x"].values[0]

        self.throw_ball_pos_y = self.df_moment_throw[
            self.df_moment_throw["league_id"] == self.id_ball
        ]["pos_y"].values[0]

        # Check if the blocker id exists
        if self.id_blocker:
            id_blocker = self.id_blocker.split(":")[-1]
            # Check if the blocker id is in the data
            if id_blocker not in self.df_moment_throw["league_id"].values:
                pass
            else:
                self.pos_x_blocker = self.df_moment_throw[
                    self.df_moment_throw["league_id"] == id_blocker
                ]["pos_x"].iloc[0]

                self.pos_y_blocker = self.df_moment_throw[
                    self.df_moment_throw["league_id"] == id_blocker
                ]["pos_y"].iloc[0]
        else:
            self.pos_x_blocker = None
            self.pos_y_blocker = None

        # Set the position of the goalkeeper based on the id of the goalkeeper and the time of the throw event
        if self.id_goalkeeper:
            id_goalkeeper = self.id_goalkeeper.split(":")[-1]
            # Check if the goalkeeper id is in the data
            if id_goalkeeper not in self.df_moment_throw["league_id"].values:
                pass
            else:
                self.pos_x_goalkeeper = self.df_moment_throw[
                    self.df_moment_throw["league_id"] == id_goalkeeper
                ]["pos_x"].iloc[0]

                self.pos_y_goalkeeper = self.df_moment_throw[
                    self.df_moment_throw["league_id"] == id_goalkeeper
                ]["pos_y"].iloc[0]

        else:
            self.pos_x_goalkeeper = None
            self.pos_y_goalkeeper = None

    def _init_event_metadata(self) -> None:
        """Initialize event metadata from the event dictionary."""
        # Sportradar event metadata
        self.event_id = self.event.get("id")
        self.event_type = self.event.get("type")

        # Convert 'time' to a pandas Timestamp
        self.event_time_tagged = pd.to_datetime(
            self.event.get("time")
        ).replace(tzinfo=None) + pd.Timedelta(hours=2)

        # Subtracting 15 seconds from event_time_tagged to get the event start time
        self.event_time_start = self.event_time_tagged - pd.Timedelta(
            seconds=15
        )

        self.match_time = self.event.get("match_time")
        self.match_clock = self.event.get("match_clock")

        self.attack_direction = self.event.get("attack_direction")

        # If match_clock exists, convert it to seconds
        if self.match_clock:
            minutes, seconds = map(int, self.match_clock.split(":"))
            self.match_clock_in_s = minutes * 60 + seconds
        else:
            self.match_clock_in_s = None

        self.competitor = self.event.get("competitor")
        self.home_score = self.event.get("home_score")
        self.away_score = self.event.get("away_score")

        # Optional fields (may be None in some events)
        self.player = self.event.get("player")
        if self.player is not None:
            self.id_player = self.player.get("id")
            self.name_player = self.player.get("name")
        else:
            self.id_player = None
            self.name_player = None

        self.shot_type = self.event.get("shot_type")
        self.players = self.event.get(
            "players", []
        )  # Defaults to an empty list if not present

        self.name_blocker = None
        self.id_blocker = None
        self.name_goalkeeper = self.event.get("name_goalkeeper")
        self.id_goalkeeper = self.event.get("id_goalkeeper")
        self.id_assist = None
        self.name_assist = None

        if self.players is not None:
            for player in self.players:
                # check if type is in player
                if "type" not in player:
                    continue

                if player["type"] == "shot":
                    self.name_player = player["name"]
                    self.id_player = player["id"]
                elif player["type"] == "blocked":
                    self.name_blocker = player["name"]
                    self.id_blocker = player["id"]
                elif player["type"] == "saved":
                    self.name_goalkeeper = player["name"]
                    self.id_goalkeeper = player["id"]
                elif player["type"] == "goalkeeper":
                    self.name_goalkeeper = player["name"]
                    self.id_goalkeeper = player["id"]

        self.suspension_minutes = self.event.get("suspension_minutes")

        # Scoring-related fields
        self.scorer = self.event.get("scorer")
        if self.scorer is not None:
            self.player = self.scorer
            self.id_player = self.player.get("id")
            self.name_player = self.player.get("name")
        else:
            self.scorer = None

        self.method = self.event.get("method")
        self.zone = self.event.get("zone")
        self.assists = self.event.get(
            "assists", []
        )  # Defaults to an empty list if not present
        if self.assists:
            self.id_assist = self.assists[0].get("id")
            self.name_assist = self.assists[0].get("name")
        else:
            self.id_assist = None
            self.name_assist = None

        self.outcome = self.event.get("outcome")

        # Period and team info
        self.period = self.event.get("period")
        self.break_name = self.event.get("break_name")
        self.team_home = self.event.get("team_home")
        self.team_away = self.event.get("team_away")

        # Additional metadata
        self.gameday = self.event.get("gameday")
        self.competition_name = self.event.get("competition_name")

    def _init_kinexon_metadata(self) -> None:
        """Initialize Kinexon metadata from the event dictionary."""
        # Cut the kinexon data to the event time
        # Check if the file exists
        if os.path.exists(self.path_to_kinexon_scene):
            df_kinexon_event = pd.read_csv(self.path_to_kinexon_scene)

            self.df_kinexon_event = df_kinexon_event
        else:
            self.df_kinexon_event = None

    def _check_meta_data(self) -> None:
        """Check if the metadata is correct."""
        # Check if the event time is within the kinexon scene
        # if self.kinexon_scene is not None:
        #     kinexon_scene_time = self.kinexon_scene["time"].values
        #     assert (
        #         self.event_time_start in kinexon_scene_time
        #     ), f"Event time not in kinexon scene: {self.event_time_start}"
        pass

    def to_dict(self) -> dict:
        """Return a dictionary representation of the GameEvent."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_time_tagged": self.event_time_tagged,
            "event_time_start": self.event_time_start,
            "match_time": self.match_time,
            "match_clock": self.match_clock,
            "match_clock_in_s": self.match_clock_in_s,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "competitor": self.competitor,
            "player_id": self.id_player,
            "player_name": self.name_player,
            "blocker_id": self.id_blocker,
            "blocker_name": self.name_blocker,
            "goalkeeper_id": self.id_goalkeeper,
            "goalkeeper_name": self.name_goalkeeper,
            "assist_id": self.id_assist,
            "assist_name": self.name_assist,
            "attack_direction": self.attack_direction,
            "scorer": self.scorer,
            "zone": self.zone,
            "period": self.period,
            "outcome": self.outcome,
            "gameday": self.gameday,
            "competition_name": self.competition_name,
            "distance_player_goal": self.distance_player_goal,
            "distance_player_goalkeeper": self.distance_player_goalkeeper,
            "distance_player_blocker": self.distance_player_blocker,
            "distance_goalkeeper_goal": self.distance_goalkeeper_goal,
            "angle_ball_goal": self.angle_ball_goal,
            "speed_ball_at_throw": self.speed_ball_at_throw,
            "distance_nearest_defender": self.distance_nearest_defender,
            "number_defenders_close": self.number_defenders_close,
            # "num_defenders_in_goal_angle": self.num_defenders_in_goal_angle,  # Assuming this is calculated as discussed before
            "throw_player_pos_x": self.throw_player_pos_x,
            "throw_player_pos_y": self.throw_player_pos_y,
            "throw_ball_pos_x": self.throw_ball_pos_x,
            "throw_ball_pos_y": self.throw_ball_pos_y,
            "goalkeeper_pos_x": self.pos_x_goalkeeper,
            "goalkeeper_pos_y": self.pos_y_goalkeeper,
            "blocker_pos_x": self.pos_x_blocker,
            "blocker_pos_y": self.pos_y_blocker,
        }

    def __str__(self) -> str:
        """String representation of the GameEvent object."""
        return f"GameEvent({self.event_id}, {self.event_type}, {self.event_time_tagged}, {self.home_score}-{self.away_score})"
