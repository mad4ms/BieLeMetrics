""" 
This script downloads the data from the source and saves it in the data/raw folder. 
Parameters are specified in the config.yaml file. """

import logging

from src.step_0_download.DownloaderDataEvents import DownloaderDataEvents
from src.step_0_download.DownloaderDataPositions import DownloaderDataPositions


def download_data_positions() -> None:
    """
    Load data from Kinexon API and save it to CSV files.
    """
    loader = DownloaderDataPositions()
    # Specify all teams or a specific team in config.yaml
    team_data = loader.get_team_ids_from_config()
    for team in team_data:
        # Get event IDs for a team (weird API structure, but okay)
        event_ids = loader.get_kinexon_event_ids_for_team_id(team)
        # Download data for each event for a team
        for id_event in event_ids:
            data = loader.download_game_data(id_event)
            # Save data to raw layer
            loader.save_game_data(id_event, data)

        logging.info("Team IDs: %s", team_data)
        logging.info("Event IDs: %s", event_ids)


def download_data_events() -> None:
    """
    Load data from Sportradar API and save it to JSON files.
    """
    loader = DownloaderDataEvents()
    game_ids = loader.get_game_ids_of_season()  # season is set in config
    for game_id in game_ids:
        # Download from SportRadarAPI
        data = loader.download_game_data(game_id)
        # Save game to raw layer (raw data)
        loader.save_game_data(game_id, data)

    logging.info("Game IDs: %s", game_ids)


def download_data() -> None:
    """
    Download data from APIs.
    """
    download_data_positions()
    download_data_events()


def main() -> None:
    """
    Main function. Parameters specified in config.yaml.
    """
    download_data()


if __name__ == "__main__":
    main()
