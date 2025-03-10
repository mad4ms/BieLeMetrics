import os
import io
import json
import logging
import yaml
from difflib import SequenceMatcher
from typing import Optional, Tuple, List, Dict
import pandas as pd
from dotenv import load_dotenv

import bielemetrics_kinexon_api_wrapper as kinexon_api


class DownloaderDataPositions:
    """
    Extracts data from Kinexon API.
    Necessary API keys:
    - USERNAME_KINEXON_SESSION
    - PASSWORD_KINEXON_SESSION
    - ENDPOINT_KINEXON_SESSION
    - USERNAME_KINEXON_MAIN
    - PASSWORD_KINEXON_MAIN
    - ENDPOINT_KINEXON_MAIN
    """

    def __init__(self) -> None:
        load_dotenv()
        self.credentials = kinexon_api.load_credentials()
        self.session = kinexon_api.login(self.credentials)

    def get_team_ids_from_config(self) -> List[Dict[str, str]]:
        """
        Get team IDs from config/config.yaml.
        """
        with open("config/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        return [team["id"] for team in self.config["kinexon"]["team_data"]]

    def get_kinexon_event_ids_for_team_id(self, id_team: str) -> List[str]:
        """
        Get event IDs from Kinexon API for a given team.
        """
        event_ids = kinexon_api.fetch_event_ids(
            self.session,
            self.credentials["ENDPOINT_KINEXON_API"],
            id_team,
            self.config["kinexon"]["time_min"],
            self.config["kinexon"]["time_max"],
        )
        list_ids = [event["session_id"] for event in event_ids]

        for event in event_ids:
            logging.info(
                "Event ID: %s and name %s",
                event["session_id"],
                event["description"],
            )

        return list_ids

    def download_game_data(self, id_event: str) -> pd.DataFrame:
        """
        Download Kinexon data from Kinexon API.
        """
        data = kinexon_api.fetch_game_csv_data(
            self.session,
            self.credentials["ENDPOINT_KINEXON_API"],
            id_event,
        )

        data = pd.read_csv(io.BytesIO(data), delimiter=";")

        return data

    def save_game_data(self, id_event: str, data: pd.DataFrame) -> None:
        """
        Save Kinexon data to CSV file.
        """
        path_to_data_bronze = (
            self.config["etl"]["path_raw"] + "data_positions/"
        )
        os.makedirs(path_to_data_bronze, exist_ok=True)

        data.to_csv(f"{path_to_data_bronze}{id_event}.csv", index=False)


def main() -> None:
    """
    Main function.
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


if __name__ == "__main__":
    # Example usage
    main()
