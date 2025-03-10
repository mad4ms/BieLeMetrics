import os
import sys
import logging


# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

from src.etl_pipeline.models.DownloaderDataPositions import (
    DownloaderDataPositions,
)
from src.etl_pipeline.models.DownloaderDataEvents import DownloaderDataEvents

logging.basicConfig(level=logging.INFO)


def load_data_positions() -> None:
    """
    Load data from Kinexon API and save it to CSV files.
    """
    loader = DownloaderDataPositions()
    team_data = loader.get_team_ids_from_config()
    for team in team_data:
        event_ids = loader.get_kinexon_event_ids_for_team_id(team)
        for id_event in event_ids:
            data = loader.download_game_data(id_event)

            loader.save_game_data(id_event, data)

        logging.info("Team IDs: %s", team_data)
        logging.info("Event IDs: %s", event_ids)


def load_data_events() -> None:
    """
    Load data from Sportradar API and save it to JSON files.
    """
    loader = DownloaderDataEvents()
    game_ids = loader.get_game_ids_of_season()  # season is set in config
    for game_id in game_ids:
        # Download from SportRadarAPI
        data = loader.download_game_data(game_id)
        # Save game to bronze layer (raw data)
        loader.save_game_data(game_id, data)

    logging.info("Game IDs: %s", game_ids)


def extract_data() -> None:
    """
    Extract data from APIs.
    """
    load_data_positions()
    load_data_events()


def main() -> None:
    """
    Main function. Parameters specified in config.yaml.
    """
    extract_data()


if __name__ == "__main__":
    main()
