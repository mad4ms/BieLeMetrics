"""
A class for projecting SportRadar JSON data as a Python class.
"""

import datetime
from typing import List, Dict

import os
import pandas as pd
import yaml
import logging

from src.models.DataClassGameEvents import DataClassGameEvents

logging.basicConfig(level=logging.INFO)


class DataClassGamePositions:
    """
    Contains all game information from SportRadar JSON data.
    """

    def __init__(self, path_to_kinexon_file: str) -> None:

        # Check if the file exists
        if path_to_kinexon_file is None or not os.path.exists(
            path_to_kinexon_file
        ):
            raise FileNotFoundError(f"File not found: {path_to_kinexon_file}")
        self.path_to_kinexon_file = path_to_kinexon_file

        self.data_positions = pd.read_csv(path_to_kinexon_file)

        # Check if the data is empty
        if self.data_positions.empty:
            raise ValueError("The data is empty.")
        else:
            logging.info("Data %s loaded successfully.", path_to_kinexon_file)

        # Read the mapping from the config file
        mapping = self.read_mapping_from_config()
        # Replace the column names
        self.replace_or_drop_column_names(mapping)

        # format format="%d.%m.%Y %H:%M:%S.%f" for the column "time"
        self.data_positions["time"] = pd.to_datetime(
            self.data_positions["time"], format="%d.%m.%Y %H:%M:%S.%f"
        )

        # Set additional attributes derived from the sport event data
        self.competitor_home = None
        self.competitor_away = None
        self.id_match = None
        self.gameday = None
        self.date_event = None
        self.path_name_to_silver_location = None
        self.name_file = None

    def replace_or_drop_column_names(
        self, mapping: List[Dict[str, str]]
    ) -> None:
        """
        Replace or drop column names in the data based on the provided mapping.
        """
        # Flatten the mapping list into a single dictionary
        flattened_mapping = {}
        for map_entry in mapping:
            flattened_mapping.update(map_entry)

        # Before renaming, check if all columns in the mapping are present in the data
        missing_columns = [
            col
            for col in flattened_mapping.keys()
            if col not in self.data_positions.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The following columns are missing in the data: {missing_columns}"
            )

        # Iterate over the flattened mapping and process replacements or drops
        for key, value in flattened_mapping.items():
            if value is None:
                # Drop the column if the value is None
                self.data_positions.drop(columns=key, inplace=True)
            else:
                # Rename the column if a valid new name is provided
                self.data_positions.rename(columns={key: value}, inplace=True)

    def read_mapping_from_config(self) -> Dict[str, str]:
        """
        Read the column mapping from the config file.
        """
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        return config["kinexon"]["mapping_column"]

    def add_event_info(self, event_info: DataClassGameEvents) -> None:
        """
        Add event information to the class.
        """
        self.competitor_home = event_info.competitor_home
        self.competitor_away = event_info.competitor_away
        self.id_match = event_info.id_match.split(":")[-1]
        self.gameday = event_info.gameday
        self.date_event = event_info.date_event
        # set rest as well
        self.event_info = event_info

    def calc_attack_direction_for_halftime(self) -> None:
        """
        Add the attack direction for each half of the game.
        """
        # We are gonna calc this by clustering the goalkeeper position
        # We assume that the goalkeeper is always in the half court of the team
        # that he is defending.
        list_goalkeeper_ids_home = []
        for event in self.event_info.timeline:
            # Check for home team
            if event["team"] == self.competitor_away and event["match_clock"] == "00:00":
                list_goalkeeper_ids_home.append(event["player_id"])

            pass


    def add_additional_info(
        self,
        competitor_home: str,
        competitor_away: str,
        id_match,
        gameday_str,
        date_event,
    ) -> None:
        """
        Add additional information to the class.
        """
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.competitor_home = competitor_home
        self.competitor_away = competitor_away
        self.id_match = id_match
        self.id_match_str = str(id_match).split(":")[-1]
        self.gameday = gameday_str
        self.date_event = date_event

    def save_to_silver(self) -> None:
        """
        Save the data to the silver location.
        """
        self.create_path_name_to_silver_location()
        self.save_data()

    def create_path_name_to_silver_location(self) -> None:
        """
        Create the path to the silver location.
        """
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.path_to_silver_location = (
            config["etl"]["path_silver"]
            + "data_positions/"
            + f"season_{config['sportradar']['season'].replace('/','-')}/"
            + f"gameday_{self.gameday}/"
        )
        self.name_file = f"{self.date_event.date()}_id_{self.id_match_str}_{self.competitor_home}_vs_{self.competitor_away}.csv"

        # make sure the path exists
        os.makedirs(self.path_to_silver_location, exist_ok=True)
        # create the new path
        self.path_name_to_silver_location = (
            self.path_to_silver_location + self.name_file
        )

    def save_data(self) -> None:
        """
        Save the data to a CSV file.
        """
        df_to_save = self.data_positions.copy()
        # add the additional information
        df_to_save["competitor_home"] = self.competitor_home
        df_to_save["competitor_away"] = self.competitor_away
        df_to_save["id_match"] = self.id_match
        df_to_save["gameday"] = self.gameday
        df_to_save["date_event"] = self.date_event

        df_to_save.to_csv(self.path_name_to_silver_location, index=False)
