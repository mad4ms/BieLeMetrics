import logging
import os
import sys
import yaml
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src2.download.data_positions.DownloaderDataPositions import (
    DownloaderDataPositions,
)
from src2.download.data_events.DownloaderDataEvents import DownloaderDataEvents

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        logging.error("Configuration file not found: %s", config_path)
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logging.info("Configuration loaded successfully.")
    return config


def download_event_data(game_id: str, downloader: DownloaderDataEvents):
    """
    Download and save data for a single game.

    :param game_id: The ID of the game.
    :param downloader: Instance of DownloaderDataEvents.
    """
    try:
        logging.info("Downloading game ID: %s", game_id)
        downloader.download_and_save_event_data(game_id)
    except Exception as e:
        logging.error("Error downloading game ID %s: %s", game_id, e)


def download_position_data(event: dict, downloader: DownloaderDataPositions):
    """
    Download and save position data for a given event.

    :param event: Dictionary containing event details.
    :param downloader: DownloaderDataPositions instance.
    """
    try:
        session_id = event.get("session_id", "Unknown")
        description = event.get("description", "No description")
        logging.info(
            "Downloading event ID: %s, Description: %s",
            session_id,
            description,
        )

        downloader.download_and_save_position_data(event)
    except Exception as e:
        logging.error("Error downloading event ID %s: %s", session_id, e)


def download_season_data():
    """Download all game data and position data for the season."""
    load_dotenv()  # Load environment variables

    # Initialize downloaders
    downloader_positions = DownloaderDataPositions()
    downloader_events = DownloaderDataEvents()

    # Retrieve all game IDs in the season
    # game_ids = downloader_events.get_game_ids_of_season()
    # if not game_ids:
    #     logging.warning("No game IDs found for the season.")
    # else:
    #     for game_id in game_ids:
    #         download_event_data(game_id, downloader_events)

    # logging.info("All event data downloads completed.")

    # Retrieve team IDs from configuration
    team_ids = downloader_positions.get_team_ids_from_config()
    if not team_ids:
        logging.warning("No team IDs found in configuration.")
        return

    for team_id in team_ids:
        logging.info("Fetching event IDs for Team ID: %s", team_id)
        events_kinexon = downloader_positions.get_kinexon_events_for_team_id(
            team_id
        )

        if not events_kinexon:
            logging.warning("No events found for Team ID: %s", team_id)
            continue

        for event in events_kinexon:
            download_position_data(event, downloader_positions)

    logging.info("All position data downloads completed.")


if __name__ == "__main__":
    download_season_data()
