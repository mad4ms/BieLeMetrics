"""
A class for projecting SportRadar JSON data as a Python class.
"""

import datetime
from typing import List, Optional
import shutil

import sys
import os
import json
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

from src.models.helper_sportradar.Competitors import (
    CompetitorStats,
)
from src.models.helper_sportradar.EventContext import (
    SportEventContext,
)
from src.models.helper_sportradar.EventConditions import (
    SportEventConditions,
)
from src.models.helper_sportradar.EventStatus import EventStatus

from src.models.helper_sportradar.Venue import Venue

from src.models.helper_sportradar.EventTimeline import (
    EventTimeline,
)


class DataClassGameEvents:
    """
    Contains all game information from SportRadar JSON data.
    """

    def __init__(self, path_to_data: str) -> None:
        self.path_to_data = path_to_data
        self.id = None
        self.start_time = None
        self.sport_event_context = None
        self.sport_event_conditions = None
        self.sport_event_status = None
        self.competitors = None
        self.venue = None
        self.timeline = None
        self.competitor_home = None
        self.competitor_away = None
        self.date_event = None
        self.id_match = None
        self.id_match_str = None
        self.gameday = None
        self.gameday_str = None
        self.path_to_silver_location = None
        self.name_file = None
        self.timestamp_in_pos_data = None

        # Load and set data
        if path_to_data is not None and os.path.exists(path_to_data):
            with open(path_to_data, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.set_data(data)
        else:
            raise FileNotFoundError(f"File not found: {path_to_data}")

    def set_data(self, data: dict) -> None:
        """Sets data attributes based on the provided data dictionary."""
        ### TOP LEVEL 0: Sport Event
        data_sport_event = data.get("sport_event", {})
        self.id = data_sport_event.get("id", "")
        self.start_time = self._parse_time(data_sport_event.get("start_time"))

        # Context
        data_sport_event_context = data_sport_event.get(
            "sport_event_context", {}
        )
        self.sport_event_context = SportEventContext(data_sport_event_context)

        # Conditions
        data_sport_event_conditions = data_sport_event.get(
            "sport_event_conditions", {}
        )
        self.sport_event_conditions = SportEventConditions(
            data_sport_event_conditions
        )

        ### TOP LEVEL 1: Sport Event Status
        data_sport_event_status = data.get("sport_event_status", {})
        self.sport_event_status = EventStatus(data_sport_event_status)

        ### TOP LEVEL 2: Statistics
        data_statistics = data.get("statistics", {})
        data_statistics_total = data_statistics.get("totals", {})
        data_competitors = data_statistics_total.get("competitors", [])
        self.competitors: List[CompetitorStats] = [
            CompetitorStats(competitor) for competitor in data_competitors
        ]

        # Venue
        data_venue = data_sport_event.get("venue", {})
        self.venue = Venue(data_venue)

        ### TOP LEVEL 3: Timeline
        data_timeline = data.get("timeline", [])
        self.timeline = EventTimeline(data_timeline)

        # Derived attributes
        self.competitor_home = self.get_team_home()
        self.competitor_away = self.get_team_away()
        self.date_event = self.get_datetime()
        self.id_match = self.id
        self.id_match_str = str(self.id_match).split(":")[-1]
        self.gameday = self.get_gameday()
        self.gameday_str = f"{self.gameday:02d}"

    @staticmethod
    def _parse_time(time_str: Optional[str]) -> Optional[datetime.datetime]:
        if not time_str:
            return None
        try:
            return datetime.datetime.fromisoformat(
                time_str.replace("Z", "+00:00")
            )
        except ValueError:
            return None

    def get_competitor_by_id(
        self, competitor_id: str
    ) -> Optional[CompetitorStats]:
        """
        Retrieve a competitor's statistics by competitor ID.
        """
        for competitor in self.competitors:
            if competitor.id == competitor_id:
                return competitor
        return None

    def get_gameday(self) -> str:
        """
        Retrieve the gameday from the path.
        """
        return self.sport_event_context.round_number

    def get_gameday_str(self) -> str:
        """
        Retrieve the gameday from the path.
        """
        gameday = self.sport_event_context.round_number
        return f"{gameday:02d}"

    def get_datetime(self) -> datetime.datetime:
        """
        Retrieve the datetime of the game.
        """
        return self.start_time

    def get_team_home(self) -> str:
        """
        Retrieve the home team.
        """
        name_team_home = [
            comp.name for comp in self.competitors if comp.qualifier == "home"
        ][0]
        return name_team_home

    def get_team_away(self) -> str:
        """
        Retrieve the away team.
        """
        name_team_away = [
            comp.name for comp in self.competitors if comp.qualifier == "away"
        ][0]
        return name_team_away


if __name__ == "__main__":

    path_test_data = "./data/raw/gameday_01/sportradar/2023-08-24_gd_01_id_42307421_teams_HCErlangen_vs_TSVHannover-Burgdorf_sportradar.json"
    # load json data

    # initialize GameEventData object
    game_data = DataClassGameEvents(path_test_data)

    # print game data
    print(game_data.id)
    print(game_data.start_time)
    print(game_data.sport_event_context)
    print(game_data.sport_event_conditions)
    print(game_data.sport_event_status)
    print(game_data.competitors)
    print(game_data.venue)
    for event in game_data.timeline.events:
        print(event)
