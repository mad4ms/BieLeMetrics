import pandas as pd
from typing import Dict, Any
import math


class GameEventFeatures:
    def __init__(self, event_dict: Dict[str, Any], data_kinexon: pd.DataFrame):
        self.event_dict = event_dict
        self.data_kinexon_throw = data_kinexon

        ### Initialize features with None that can later be used to train an xG model
        ### Mind there this class is meant for the exact moment of the throw of the ball
        # Get important event information
        self.throw_player_pos_x = event_dict.get("pos_x_player")
        self.throw_player_pos_y = event_dict.get("pos_y_player")
        self.throw_ball_pos_x = event_dict.get("pos_x_ball")
        self.throw_ball_pos_y = event_dict.get("pos_y_ball")
        self.pos_x_goalkeeper = event_dict.get("pos_x_goalkeeper")
        self.pos_y_goalkeeper = event_dict.get("pos_y_goalkeeper")
        self.pos_x_blocker = event_dict.get("pos_x_blocker")
        self.pos_y_blocker = event_dict.get("pos_y_blocker")
        # and the IDs
        self.id_throw_player = (
            event_dict.get("id_player").split(":")[-1]
            if event_dict.get("id_player")
            else None
        )
        self.id_goalkeeper = (
            event_dict.get("id_goalkeeper").split(":")[-1]
            if event_dict.get("id_goalkeeper")
            else None
        )
        self.id_blocker = (
            event_dict.get("id_blocker").split(":")[-1]
            if event_dict.get("id_blocker")
            else None
        )
        self.id_ball = event_dict.get("id_ball")
        self.competitor = event_dict.get("competitor")

        # some other metadata
        self.home_score = event_dict.get("home_score")
        self.away_score = event_dict.get("away_score")

        # Define goal postion based on attacking direction
        self.goal_position = (
            (0, 10) if event_dict["attack_direction"] == "right" else (40, 10)
        )
        self.goal_position_x = self.goal_position[0]
        self.goal_position_y = self.goal_position[1]

        # Distances player_to_
        self.distance_player_to_goal = None
        self.distance_player_to_goalkeeper = None
        self.distance_player_to_blocker = None
        self.distance_player_to_nearst_opponent = None
        self.id_nearst_opponent = None
        self.distance_player_to_nearest_teammate = None
        self.id_nearest_teammate = None

        # Distances goalkeeper_to_
        self.distance_goalkeeper_to_goal = None

        # Angles
        self.angle_player_to_goal = None
        self.angle_ball_to_goal = None

        # Speeds
        self.speed_player = None
        self.speed_ball = None

        # Number of opponents between player and goal
        self.num_opponents_between_player_and_goal = None
        self.num_opponents_close_to_player = None  # idk, like 1,5m? #TODO

        # home advantage
        self.home_advantage = None  # Can be 1 or 0

    def calculate_features(self):

        self._calculate_distance_player_to_goal()
        self._calculate_distance_player_to_goalkeeper()
        self._calculate_distance_player_to_blocker()
        self._calculate_distance_player_to_nearst_opponent()
        self._calculate_distance_player_to_nearest_teammate()
        self._calculate_distance_goalkeeper_to_goal()
        self._calculate_angle_player_to_goal()
        self._calculate_angle_ball_to_goal()
        self._calc_speeds()

        pass

    def _calculate_distance_player_to_goal(self):
        if self.throw_player_pos_x is not None:
            self.distance_player_to_goal = (
                (self.throw_player_pos_x - self.goal_position_x) ** 2
                + (self.throw_player_pos_y - self.goal_position_y) ** 2
            ) ** 0.5
            print(
                f"\t> Distance player <--> goal: {self.distance_player_to_goal:.2f} ✅"
            )
        else:
            print("\t> Distance player <--> goal: None ❌")

    def _calculate_distance_player_to_goalkeeper(self):
        if (
            self.throw_player_pos_x is not None
            and self.pos_x_goalkeeper is not None
        ):
            self.distance_player_to_goalkeeper = (
                (self.throw_player_pos_x - self.pos_x_goalkeeper) ** 2
                + (self.throw_player_pos_y - self.pos_y_goalkeeper) ** 2
            ) ** 0.5
            print(
                f"\t> Distance player <--> goalkeeper: {self.distance_player_to_goalkeeper:.2f} ✅"
            )
        else:
            print("\t> Distance player <--> goalkeeper: None ❌")

    def _calculate_distance_player_to_blocker(self):
        if (
            self.throw_player_pos_x is not None
            and self.pos_x_blocker is not None
        ):
            self.distance_player_to_blocker = (
                (self.throw_player_pos_x - self.pos_x_blocker) ** 2
                + (self.throw_player_pos_y - self.pos_y_blocker) ** 2
            ) ** 0.5
            print(
                f"\t> Distance player <--> blocker: {self.distance_player_to_blocker:.2f} ✅"
            )
        else:
            print("\t> Distance player <--> blocker: None ❌")

    def _calculate_distance_player_to_nearst_opponent(self):
        self.distance_player_to_nearst_opponent = 50
        self.num_opponents_close_to_player = 0
        self.id_nearst_opponent = None

        if not self.data_kinexon_throw.empty:
            for index, row in self.data_kinexon_throw.iterrows():
                row["competitor"] = "home" if row["group_id"] == 1 else "away"

                if (
                    self.id_goalkeeper is None
                    or self.id_throw_player is None
                    or row["league_id"] == self.id_throw_player.split(":")[-1]
                    or row["league_id"] == self.id_goalkeeper.split(":")[-1]
                    or row["league_id"] == self.id_ball
                    or row["competitor"] == self.competitor
                ):
                    continue
                distance = (
                    (self.throw_player_pos_x - row["pos_x"]) ** 2
                    + (self.throw_player_pos_y - row["pos_y"]) ** 2
                ) ** 0.5
                if distance < self.distance_player_to_nearst_opponent:
                    self.distance_player_to_nearst_opponent = distance
                    self.id_nearst_opponent = row["league_id"]

                if distance < 1.5:
                    self.num_opponents_close_to_player += 1

            print(
                f"\t> Distance player <--> nearest opponent: {self.distance_player_to_nearst_opponent:.2f} ✅"
            )
        else:
            print("\t> Distance player <--> nearest opponent: None ❌")

    def _calculate_distance_player_to_nearest_teammate(self):
        self.distance_player_to_nearest_teammate = 50
        self.id_nearest_teammate = None

        if not self.data_kinexon_throw.empty:
            for index, row in self.data_kinexon_throw.iterrows():
                row["competitor"] = "home" if row["group_id"] == 1 else "away"

                if (
                    self.id_goalkeeper is None
                    or self.id_throw_player is None
                    or row["league_id"] == self.id_throw_player.split(":")[-1]
                    or row["league_id"] == self.id_goalkeeper.split(":")[-1]
                    or row["league_id"] == self.id_ball
                    or row["competitor"] != self.competitor
                ):
                    continue
                distance = (
                    (self.throw_player_pos_x - row["pos_x"]) ** 2
                    + (self.throw_player_pos_y - row["pos_y"]) ** 2
                ) ** 0.5
                if distance < self.distance_player_to_nearest_teammate:
                    self.distance_player_to_nearest_teammate = distance
                    self.id_nearest_teammate = row["league_id"]

            print(
                f"\t> Distance player <--> nearest teammate: {self.distance_player_to_nearest_teammate:.2f} ✅"
            )
        else:
            print("\t> Distance player <--> nearest teammate: None ❌")

    def _calculate_distance_goalkeeper_to_goal(self):
        if self.pos_x_goalkeeper is not None:
            self.distance_goalkeeper_to_goal = (
                (self.pos_x_goalkeeper - self.goal_position_x) ** 2
                + (self.pos_y_goalkeeper - self.goal_position_y) ** 2
            ) ** 0.5
            print(
                f"\t> Distance goalkeeper <--> goal: {self.distance_goalkeeper_to_goal:.2f} ✅"
            )
        else:
            print("\t> Distance goalkeeper <--> goal: None ❌")

    def _calculate_angle_ball_to_goal(self):
        #  Get the angle of the throw using atan2
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

            self.angle_ball_to_goal = abs(90 - angle_throw)

            print(
                f"\t> Angle ball <--> goal: {self.angle_ball_to_goal:.2f} ✅"
            )
        else:
            print("\t> Angle ball <--> goal: None ❌")

    def _calculate_angle_player_to_goal(self):
        if (
            self.throw_player_pos_x is not None
            and self.throw_player_pos_y is not None
        ):
            angle_player = (
                math.atan2(
                    self.goal_position_y - self.throw_player_pos_y,
                    self.goal_position_x - self.throw_player_pos_x,
                )
                * 180
                / math.pi
            )
            if angle_player < 0:
                angle_player += 360
            if angle_player > 180:
                angle_player = 360 - angle_player

            self.angle_player_to_goal = abs(90 - angle_player)

            print(
                f"\t> Angle player <--> goal: {self.angle_player_to_goal:.2f} ✅"
            )
        else:
            print("\t> Angle player <--> goal: None ❌")

    def _calc_speeds(self):
        # Find id_player in data_kinexon_throw
        row_player = self.data_kinexon_throw[
            self.data_kinexon_throw["league_id"] == self.id_throw_player
        ]
        # Check if the player is in the data and access the "speed" column to get the speed
        if not row_player.empty:
            self.speed_player = row_player["speed"].values[0]
            print(f"\t> Speed player: {self.speed_player:.2f} ✅")
        else:
            print("\t> Speed player: None ❌")

        # Find id_ball in data_kinexon_throw
        row_ball = self.data_kinexon_throw[
            self.data_kinexon_throw["league_id"] == self.id_ball
        ]
        # Check if the ball is in the data and access the "speed" column to get the speed
        if not row_ball.empty:
            self.speed_ball = row_ball["speed"].values[0]
            if not pd.isna(self.speed_ball):
                print(f"\t> Speed ball: {self.speed_ball:.2f} ✅")
            else:
                print("\t> Speed ball: NaN ❌")
        else:
            print("\t> Speed ball: None ❌")

    def get_features(self):
        return {
            "distance_player_to_goal": self.distance_player_to_goal,
            "distance_player_to_goalkeeper": self.distance_player_to_goalkeeper,
            "distance_player_to_blocker": self.distance_player_to_blocker,
            "distance_player_to_nearst_opponent": self.distance_player_to_nearst_opponent,
            "id_nearst_opponent": self.id_nearst_opponent,
            "distance_player_to_nearest_teammate": self.distance_player_to_nearest_teammate,
            "id_nearest_teammate": self.id_nearest_teammate,
            "distance_goalkeeper_to_goal": self.distance_goalkeeper_to_goal,
            "angle_player_to_goal": self.angle_player_to_goal,
            "angle_ball_to_goal": self.angle_ball_to_goal,
            "num_opponents_between_player_and_goal": self.num_opponents_between_player_and_goal,
            "num_opponents_close_to_player": self.num_opponents_close_to_player,
            "home_advantage": self.home_advantage,
            "speed_player": self.speed_player,
            "speed_ball": self.speed_ball,
        }
