import pandas as pd
import os
import math
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.core.game_event_features import GameEventFeatures
from src.processing.helper_processing.helper_game_throw_detector import (
    ThrowEventDetector,
)

import time


class GameEvent:
    """Class representing a game event."""

    def __init__(self, event: dict, path_to_kinexon_scene: str) -> None:
        """
        Initialize the GameEvent object with the event data.

        Args:
            event (dict): Dictionary containing event data.
        """
        self.event = event
        self.path_to_kinexon_scene = path_to_kinexon_scene

        self.throw_player_pos_x = None
        self.throw_player_pos_y = None
        self.throw_ball_pos_x = None
        self.throw_ball_pos_y = None
        self.pos_x_blocker = None
        self.pos_y_blocker = None
        self.pos_x_goalkeeper = None
        self.pos_y_goalkeeper = None

        self.event_time_throw = None

        # IDs
        self.id_player = None
        self.id_blocker = None
        self.id_goalkeeper = None
        self.id_assist = None
        self.id_ball = None

        # Metadata
        self.event_id = None
        self.event_type = None
        self.event_time_tagged = None
        self.event_time_start = None
        self.match_time = None
        self.match_clock = None
        self.match_clock_in_s = None
        self.name_team_attack = None
        self.name_team_defense = None

        self.dict_features = {}

        # Initialize event metadata
        self._init_metadata_event_sportradar()

        # Initialize kinexon metadata
        self._init_metadata_event_kinexon()

        # print known metadata
        print(
            f"> New event {self.event_id} type: {self.event_type} - Score: {self.home_score}-{self.away_score} - Event time: {self.event_time_tagged} - Player: {self.name_player} (Team: {self.name_team_attack}) - Goalkeeper: {self.name_goalkeeper} (Team: {self.name_team_defense})"
        )
        print(
            f"\t> Length positional data: {len(self.df_kinexon_event)} in file: {self.path_to_kinexon_scene}"
        )

        # Check metadata
        self._check_meta_data()

        # process
        self._process_event()

        # self._calc_features_throw()

    def _process_event(self) -> None:
        """Process the event data."""
        relevant_event_types = [
            "score_change",
            "shot_off_target",
            "shot_blocked",
            "shot_saved",
            "seven_m_missed",
        ]

        if self.event_type not in relevant_event_types:
            return
        # Initialize the throw detector
        throw_detector = ThrowEventDetector(
            self.event_type,
            self.df_kinexon_event,
            self.id_player,
            self.attack_direction,
        )
        # Find the throw timestamp
        (
            self.event_time_throw,
            self.data_kinexon_event_player_ball,
            self.peaks,
        ) = throw_detector.find_throw_timestamp()

        # check if throw point is found and in the kinexon evnet data
        if self.event_time_throw is not None:
            diff_throw = self.event_time_throw - self.event_time_start
            if (
                diff_throw.total_seconds() > 15
                or diff_throw.total_seconds() < -15
            ):
                print(
                    f"\t> Throw time check: {self.event_time_throw} - Event time: {self.event_time_start} - Difference: {diff_throw.total_seconds()} s ❌"
                )
            else:
                print(
                    f"\t> Throw time check: {self.event_time_throw} - Event time: {self.event_time_start} - Difference: {diff_throw.total_seconds()} s ✅"
                )

        self.df_kinexon_event["time"] = pd.to_datetime(
            self.df_kinexon_event["time"]
        )

        # Extract throw moment
        self.df_moment_throw = self.df_kinexon_event[
            self.df_kinexon_event["time"] == self.event_time_throw
        ]

        # Initialize metadata of the throw event
        self._init_metadata_throw()

        # Delegate feature calculation to GameEventFeatures
        feature_calculator = GameEventFeatures(
            self.to_dict(), self.df_moment_throw
        )
        feature_calculator.calculate_features()

        self.dict_features = feature_calculator.get_features()

    def _init_metadata_throw(self) -> None:
        """Initialize the metadata of the throw event."""

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

    def _init_metadata_event_sportradar(self) -> None:
        """Initialize event metadata from the event dictionary."""
        # Sportradar event metadata
        self.event_id = self.event.get("id")
        self.event_type = self.event.get("type")

        # Convert 'time' to a pandas Timestamp and add 2 hours to convert to CET
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
        # If scorer is not None, set player to scorer as the scorer is always the throwing player
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
        self.team_home = self.event.get("name_team_home")
        self.team_away = self.event.get("name_team_away")

        self.name_team_attack = (
            self.team_home if self.competitor == "home" else self.team_away
        )
        self.name_team_defense = (
            self.team_home if self.competitor == "away" else self.team_away
        )

        # Additional metadata
        self.gameday = self.event.get("gameday")
        self.competition_name = self.event.get("competition_name")

    def _init_metadata_event_kinexon(self) -> None:
        """Initialize Kinexon metadata from the event dictionary."""
        if os.path.exists(self.path_to_kinexon_scene):
            self.df_kinexon_event = pd.read_csv(self.path_to_kinexon_scene)
        else:
            self.df_kinexon_event = None

    def _check_meta_data(self) -> None:
        """Check if the metadata is correct."""
        # Check the time difference between the event and the kinexon data
        if self.df_kinexon_event is not None:
            diff = (
                pd.to_datetime(self.df_kinexon_event["time"].iloc[0])
                - self.event_time_start
            )
            if diff.total_seconds() > 15 or diff.total_seconds() < -15:
                print(
                    f"\t !!! Event ID: {self.event_id}: {self.event_type} - Event time: {self.event_time_start} - Kinexon time: {self.df_kinexon_event['time'].iloc[0]} - Difference: {diff.total_seconds()} s"
                )
                # raise ValueError(
                #     "Time difference between event and kinexon data is too large."
                # )

    def to_dict(self) -> dict:
        """Return a dictionary representation of the GameEvent."""
        dict_res = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_time_tagged": self.event_time_tagged,
            "event_time_start": self.event_time_start,
            "event_time_throw": self.event_time_throw,
            "match_time": self.match_time,
            "match_clock": self.match_clock,
            "match_clock_in_s": self.match_clock_in_s,
            "name_team_home": self.team_home,
            "name_team_away": self.team_away,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "competitor": self.competitor,
            "id_ball": self.id_ball,
            "id_player": self.id_player,
            "name_player": self.name_player,
            "id_blocker": self.id_blocker,
            "name_blocker": self.name_blocker,
            "id_goalkeeper": self.id_goalkeeper,
            "name_goalkeeper": self.name_goalkeeper,
            "id_assist": self.id_assist,
            "name_assist": self.name_assist,
            "attack_direction": self.attack_direction,
            "gameday": self.gameday,
            "competition_name": self.competition_name,
            "pos_x_player": self.throw_player_pos_x,
            "pos_y_player": self.throw_player_pos_y,
            "pos_x_ball": self.throw_ball_pos_x,
            "pos_y_ball": self.throw_ball_pos_y,
            "pos_x_goalkeeper": self.pos_x_goalkeeper,
            "pos_y_goalkeeper": self.pos_y_goalkeeper,
            "pos_x_blocker": self.pos_x_blocker,
            "pos_y_blocker": self.pos_y_blocker,
        }

        if self.dict_features:
            dict_res.update(self.dict_features)

        return dict_res

    def __str__(self) -> str:
        """String representation of the GameEvent object."""
        return f"GameEvent({self.event_id}, {self.event_type}, {self.event_time_tagged}, {self.home_score}-{self.away_score})"


if __name__ == "__main__":
    # Test the GameEvent class
    path_to_kinexon_scene = (
        "./data/events/match_42307421/event_1525662873_positions.csv"
    )
    event = {
        "id": "1234",
        "type": "throw",
        "time": "2022-01-01 12:00:00",
        "match_time": "12:34",
        "match_clock": "12:34",
        "attack_direction": "right",
        "competitor": "home",
        "home_score": 0,
        "away_score": 0,
        "player": {"id": "player1", "name": "Player 1"},
        "shot_type": "jump shot",
        "players": [
            {"type": "shot", "id": "player1", "name": "Player 1"},
            {"type": "blocked", "id": "player2", "name": "Player 2"},
            {"type": "saved", "id": "player3", "name": "Player 3"},
            {"type": "goalkeeper", "id": "player4", "name": "Player 4"},
        ],
        "suspension_minutes": 2,
        "scorer": {"id": "player1", "name": "Player 1"},
        "method": "jump shot",
        "zone": "left wing",
        "assists": [{"id": "player2", "name": "Player 2"}],
        "outcome": "goal",
        "period": 1,
        "break_name": "timeout",
        "team_home": "Team A",
        "team_away": "Team B",
        "gameday": "2022-01-01",
        "competition_name": "Test Competition",
    }

    game_event = GameEvent(event, path_to_kinexon_scene)
    print(game_event.to_dict())
    print(game_event)
    print("GameEvent class test completed.")
