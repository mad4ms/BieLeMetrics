"""Class to extract data from SportRadar API."""

import os
import yaml
import json
import logging
from typing import Optional, Any, Dict, List

from sportradar import Handball

logging.basicConfig(level=logging.INFO)


class DownloaderDataEvents:
    """
    Extracts data from SportRadar API.
    Necessary API keys:
    - SportRadar API key: "API_KEY_SPORTRADAR"
    """

    def __init__(self, api_key: str = os.getenv("API_KEY_SPORTRADAR")) -> None:
        self.api_key = api_key
        self.client_sportradar = Handball.Handball(api_key)
        # load config/config.yaml
        self.config = None
        self.load_config()

    def download_game_data(self, game_id: str) -> Dict[str, Any]:
        """
        Download game data from Sportradar API.
        """
        data_sport_event = self.client_sportradar.get_sport_event_timeline(
            game_id
        ).json()

        # remove field "generated_at" from data if it exists
        if "generated_at" in data_sport_event:
            del data_sport_event["generated_at"]

        return data_sport_event

    def load_config(self):
        """
        Load config from config/config.yaml.
        """
        with open("config/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

    def get_competition_id(self) -> str:
        """
        Get tournament ID from SportRadar API.
        """
        # Fetch all tournaments from API
        competitions_sportradar = (
            self.client_sportradar.get_competitions().json()
        )["competitions"]

        id_competition = None

        # Check if config/config.yaml is loaded
        assert self.config["sportradar"]["competition"] is not None
        assert self.config["sportradar"]["gender"] is not None

        for tournament in competitions_sportradar:
            # Skip if name not in config or gender not in config
            if (
                tournament["name"] in self.config["sportradar"]["competition"]
                and tournament["gender"] in self.config["sportradar"]["gender"]
            ):
                id_competition = tournament["id"]
                logging.info(
                    "> Found tournament: %s with ID: %s",
                    tournament["name"],
                    id_competition,
                )
                break
            else:
                logging.info(
                    "> Skipping tournament: %s with ID: %s",
                    tournament["name"],
                    tournament["id"],
                )

        return id_competition

    def get_game_ids_of_season(self) -> List[str]:
        """
        Get game IDs from SportRadar API.
        """

        id_competition = self.get_competition_id()

        assert id_competition is not None

        # Fetch all seasons for the tournament
        seasons_for_competition = (
            self.client_sportradar.get_seasons_for_competition(
                id_competition
            ).json()
        )["seasons"]

        id_season = None

        for season in seasons_for_competition:
            # Fetch id of the season
            if season["year"] == self.config["sportradar"]["season"]:
                id_season = season["id"]
                break

        assert id_season is not None

        # Fetch season summaries (divided into multiple requests if needed)
        season_summaries = []
        offset = 0
        limit = 100

        while True:
            response = self.client_sportradar.get_season_summaries(
                id_season, offset, limit
            ).json()
            season_summaries.extend(response["summaries"])
            offset += limit

            if len(response["summaries"]) < limit:
                break

        # List of game IDs
        game_ids = [
            summary["sport_event"]["id"] for summary in season_summaries
        ]

        for summary in season_summaries:
            logging.info(
                "> Found game ID : %s with data: %s",
                summary["sport_event"]["id"],
                summary["sport_event"],
            )

        return game_ids

    def save_game_data(self, game_id_str: str, data: Dict[str, Any]) -> None:
        """
        Save game data to a JSON file.
        """
        if ":" in game_id_str:
            game_id_str = game_id_str.split(":")[-1]
        # Path to data directory
        path_to_data = self.config["etl"]["path_raw"] + "data_events/"
        # Create dir if it does not exist
        os.makedirs(path_to_data, exist_ok=True)
        # Save data to JSON file
        with open(f"{path_to_data}{game_id_str}.json", "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Initialize LoaderDataEvents
    loader = DownloaderDataEvents()

    # Get game IDs
    # game_ids = loader.get_game_ids_of_season()
    # print(game_ids)
    # get random id
    # game_id_str = game_ids[0]
    game_id = "sr:sport_event:42307421"
    game_id_str = game_id.split(":")[-1]
    # Get game events
    game_data = loader.download_game_data(game_id_str)

    # Save game data
    loader.save_game_data(game_id_str, game_data)
