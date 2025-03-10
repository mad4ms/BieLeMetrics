"""Class to extract data from SportRadar API."""

import os
import yaml
import json
import logging
from typing import Optional, Any, Dict, List

from sportradar import Handball
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)


class DownloaderDataEvents:
    """
    Extracts data from SportRadar API.
    Necessary API keys:
    - SportRadar API key: "API_KEY_SPORTRADAR"
    """

    def __init__(self, api_key: str = os.getenv("API_KEY_SPORTRADAR")) -> None:
        """
        Initialize the DownloaderDataEvents class.

        :param api_key: API key for SportRadar API
        """
        self.api_key = api_key
        self.client_sportradar = Handball.Handball(api_key)
        self.config = None
        self.load_config()

    def load_config(self) -> None:
        """
        Load configuration from config/config.yaml.
        """
        config_path = "config/config.yaml"
        if not os.path.exists(config_path):
            logging.error("Configuration file not found: %s", config_path)
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        logging.info("Configuration loaded successfully.")

    def download_game_data(self, game_id: str) -> Dict[str, Any]:
        """
        Download game data from SportRadar API.

        :param game_id: ID of the game to download data for
        :return: Dictionary containing game data
        """
        data_sport_event = self.client_sportradar.get_sport_event_timeline(
            game_id
        ).json()

        # Remove field "generated_at" from data if it exists
        data_sport_event.pop("generated_at", None)

        return data_sport_event

    def get_competition_id(self) -> str:
        """
        Get competition ID from SportRadar API based on configuration.

        :return: Competition ID
        """
        competitions = (
            self.client_sportradar.get_competitions()
            .json()
            .get("competitions", [])
        )
        competition_name = self.config["sportradar"]["competition"]
        competition_gender = self.config["sportradar"]["gender"]

        for competition in competitions:
            if (
                competition["name"] in competition_name
                and competition["gender"] in competition_gender
            ):
                logging.info(
                    "Found competition: %s with ID: %s",
                    competition["name"],
                    competition["id"],
                )
                return competition["id"]
            else:
                logging.info(
                    "Skipping competition: %s with ID: %s",
                    competition["name"],
                    competition["id"],
                )

        logging.error("No matching competition found.")
        raise ValueError("No matching competition found.")

    def get_game_ids_of_season(self) -> List[str]:
        """
        Get game IDs for the specified season from SportRadar API.

        :return: List of game IDs
        """
        competition_id = self.get_competition_id()
        seasons = (
            self.client_sportradar.get_seasons_for_competition(competition_id)
            .json()
            .get("seasons", [])
        )
        season_year = self.config["sportradar"]["season"]

        season_id = next(
            (
                season["id"]
                for season in seasons
                if season["year"] == season_year
            ),
            None,
        )
        if not season_id:
            logging.error("No matching season found.")
            raise ValueError("No matching season found.")

        game_ids = []
        offset = 0
        limit = 100

        while True:
            response = self.client_sportradar.get_season_summaries(
                season_id, offset, limit
            ).json()
            summaries = response.get("summaries", [])
            game_ids.extend(
                summary["sport_event"]["id"] for summary in summaries
            )
            offset += limit

            if len(summaries) < limit:
                break

        logging.info(
            "Found %d game IDs for season %s.", len(game_ids), season_year
        )
        return game_ids

    def construct_file_name_and_path(
        self, game_id: str, metadata: Dict[str, Any]
    ) -> str:
        """
        Construct a file name based on the game ID and metadata.

        :param game_id: ID of the game
        :param metadata: Metadata of the game
        :return: File name
        """
        path_to_data = os.path.join(
            self.config["etl"]["path_raw"],
            f"gameday_{metadata["gameday"]}",
            "data_events",
        )
        os.makedirs(path_to_data, exist_ok=True)

        game_id_str = game_id.split(":")[-1]
        team_home = metadata["team_home"]
        team_away = metadata["team_away"]
        time_start = metadata["time_start"]  # '2023-08-24T17:00:00+00:00'
        date_start = time_start.split("T")[0]  # '2023-08-24'
        return os.path.join(
            path_to_data,
            f"{date_start}_id_{game_id_str}_{team_home}_{team_away}.json",
        )

    def save_game_data(self, data: Dict[str, Any], file_path: str) -> None:
        """
        Save game data to a JSON file.

        :param game_id: ID of the game
        :param data: Data to save
        """

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logging.info("Game data saved to %s", file_path)

    def extract_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from the game data.

        :param data: Game data
        :return: Metadata
        """
        metadata = {
            "team_home": data["statistics"]["totals"]["competitors"][0][
                "abbreviation"
            ],
            "team_away": data["statistics"]["totals"]["competitors"][1][
                "abbreviation"
            ],
            "time_start": data["sport_event"]["start_time"],
            "gameday": str(
                data["sport_event"]["sport_event_context"]["round"]["number"]
            ).zfill(2),
        }
        return metadata

    def download_and_save_event_data(self, game_id: str) -> None:
        """
        Download and save game data for a given game ID.

        :param game_id: ID of the game
        """
        game_data = self.download_game_data(game_id)
        metadata = self.extract_metadata(game_data)
        file_path = self.construct_file_name_and_path(game_id, metadata)
        self.save_game_data(game_data, file_path)


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Initialize DownloaderDataEvents
    downloader = DownloaderDataEvents()

    # Example usage
    list_games = downloader.get_game_ids_of_season()

    logging.info("Game IDs: %s", list_games)

    downloader.download_and_save_event_data(list_games[0])
