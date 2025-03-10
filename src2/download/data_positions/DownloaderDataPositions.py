import os
import io
import json
import logging
import yaml
import glob
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

    def get_kinexon_events_for_team_id(self, id_team: str) -> List[str]:
        """
        Get event IDs from Kinexon API for a given team.
        """
        events_kinexon = kinexon_api.fetch_event_ids(
            self.session,
            self.credentials["ENDPOINT_KINEXON_API"],
            id_team,
            self.config["kinexon"]["time_min"],
            self.config["kinexon"]["time_max"],
        )
        # list_ids = [event["session_id"] for event in event_ids]

        # for event in events_kinexon:
        #     logging.info(
        #         "Event ID: %s and name %s",
        #         event["session_id"],
        #         event["description"],
        #     )

        return events_kinexon

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

    def find_filename_path_of_event_data(self, event: dict) -> Optional[str]:
        """
        Find the filename of the event data.
        """
        # example path of event data: data\raw\gameday_01\data_events\2023-08-24_id_42307421_HCE_HAN.json

        # use glob to list all files in the directory data/raw
        list_of_files = glob.glob(
            "data/raw/**/data_events/*.json", recursive=True
        )
        # date of the current position event
        date_event = event["start_session"].split(" ")[0]
        logging.info("Date of event: %s", date_event)
        # only keep the files that contain the date of the current event
        list_of_files = [file for file in list_of_files if date_event in file]
        logging.warning("Files with date: %s", list_of_files)

        # find abbreviation of home team
        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()

        team_home = None
        try:
            team_home = event["description"].split(" vs")[0]
            logging.info("Team home: %s", team_home)
            # find in config.yaml the abbreviation of the home team
            team_data = self.config["kinexon"]["team_data"]
            team_abb = None
            for team in team_data:
                # if team["name"] == team_home:
                similarity = similar(team["name"], team_home)
                if similarity > 0.7:
                    team_abb = team["abbreviation"]
                    break
            if team_abb is None:
                raise ValueError(
                    f"Home team {team_home} abbreviation not found in config.yaml"
                )
        except IndexError as e:
            logging.error(
                "Team not found in config.yaml. Error: %s.",
                e,
            )
            return None
        except AttributeError as e:
            logging.error(
                "Error in finding team abbreviation or problem with team name %s. Error: %s.",
                team_home,
                e,
            )
            return None

        logging.info("Team home abbreviation: %s", team_abb)
        # only keep the files that contain the abbreviation of the home team
        list_of_files = [file for file in list_of_files if team_abb in file]
        # usually, there should be only one file left
        if len(list_of_files) == 1:
            return list_of_files[0]
        elif len(list_of_files) == 0:
            logging.error("No file found for event wit abb %s", team_abb)
            return None
        else:
            # show the files that are left
            logging.error(
                "Error! Multiple files found for event: %s", list_of_files
            )
            return None

    def construct_filepath_and_name(self, event: dict) -> str:
        """
        Construct file name for Kinexon data.
        """
        path_file_event_data = self.find_filename_path_of_event_data(event)

        if path_file_event_data is not None:
            logging.info(
                "Found corresponding path to event file: %s",
                path_file_event_data,
            )
            # replace data_events with data_positions
            path_file_event_data = path_file_event_data.replace(
                "data_events", "data_positions"
            )

            # make sure folder exists
            os.makedirs(os.path.dirname(path_file_event_data), exist_ok=True)
            # replace .json with .csv
            path_file_event_data = path_file_event_data.replace(
                ".json", f"_sid_{event["session_id"]}.csv"
            )
            return path_file_event_data
        else:
            logging.error("Error in constructing file path.")
            return ""

    def save_game_data(self, data: pd.DataFrame, path_file: str) -> None:
        """
        Save Kinexon data to CSV file.
        """

        data.to_csv(path_file, index=False)

    def download_and_save_position_data(self, event: dict) -> None:
        """
        Download and save Kinexon data to CSV file.
        """
        path_file = self.construct_filepath_and_name(event)
        if os.path.exists(path_file):
            logging.info("Data already downloaded. Skipping download.")
        elif path_file == "":
            logging.error("Error in constructing file path.")

        else:
            if True:
                data = self.download_game_data(event["session_id"])
                self.save_game_data(data, path_file)
            else:
                logging.info("Skipping download of data. Path: %s", path_file)


def main() -> None:
    """
    Main function.
    """
    loader = DownloaderDataPositions()
    team_data = loader.get_team_ids_from_config()
    for team in team_data:
        event_ids = loader.get_kinexon_events_for_team_id(team)
        for id_event in event_ids:
            # data = loader.download_game_data(id_event["session_id"])
            loader.download_and_save_position_data(id_event)

        logging.info("Team IDs: %s", team_data)
        logging.info("Event IDs: %s", event_ids)


if __name__ == "__main__":
    # Example usage
    main()
