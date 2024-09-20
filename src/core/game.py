import os
import json
from typing import Optional, Tuple, List, Dict
from configparser import ConfigParser
import pandas as pd
import warnings

from src.processing.helper_processing.helper_game_processing import (
    add_missing_goalkeeper_to_events,
    add_scores_to_events,
    calc_attack_direction,
    insert_names_competitors,
)


class Game:
    def __init__(self, path_file_sportradar: str, path_file_kinexon: str):
        """
        Initialize the Game object with paths to Sportradar and Kinexon data.
        """
        self.path_file_sportradar = path_file_sportradar
        self.path_file_kinexon = path_file_kinexon

        # Extract match ID from Kinexon file path
        self.match_id = self._extract_match_id()

        # Generate event path and ensure directory exists
        self.path_events = self._generate_event_path()
        os.makedirs(self.path_events, exist_ok=True)

        # Load configurations and Sportradar data
        self.kinexon_config = self._load_config("config_fields_kinexon.cfg")
        self.dict_sportradar = self._load_sportradar_json()

        # Load Kinexon data, prefer Parquet for performance
        self.df_kinexon = self._load_kinexon_data()

        # Ensure data integrity
        assert not self.df_kinexon.empty, "Kinexon dataframe is empty"
        assert self.dict_sportradar, "Sportradar JSON is empty"

        # Clean the data if necessary
        self.df_kinexon, self.dict_sportradar = self.clean_game_data()

        # Create event-specific Kinexon snippets
        self.dict_kinexon_path_by_event_id = {}
        self.create_kinexon_snippets_from_events()
        # with kinexon snippets, add additional event data
        self.process_additional_event_data()
        # and save updated Sportradar events
        self.create_sportradar_event_from_events()

    def _extract_match_id(self) -> str:
        """
        Extract the match ID from the Kinexon file path.
        """
        return self.path_file_kinexon.split("_id_")[1].split("_")[0]

    def _generate_event_path(self) -> str:
        """
        Generate the event directory path from Kinexon file path.
        """
        base_path = self.path_file_kinexon.split("raw")[0]
        return os.path.join(base_path, "events", f"match_{self.match_id}")

    def _load_config(self, config_path: str) -> Dict[str, str]:
        """
        Load and parse configuration file for column mappings.
        """
        config = ConfigParser()
        config.read(config_path)
        return {k: v for k, v in config.items("fields")}

    def _load_sportradar_json(self) -> Dict:
        """
        Load the Sportradar JSON data from the provided file path.
        """
        with open(self.path_file_sportradar, "r", encoding="utf-8") as file:
            return json.load(file)

    def _load_kinexon_data(self) -> pd.DataFrame:
        """
        Load Kinexon data, using Parquet if available, else fallback to CSV.
        """
        parquet_path = self.path_file_kinexon.replace(".csv", ".parquet")
        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path, engine="pyarrow")

        df_kinexon = pd.read_csv(self.path_file_kinexon)

        # Attempt to save as Parquet for future use
        try:
            df_kinexon.to_parquet(parquet_path)
        except Exception:
            df_kinexon["league id"] = df_kinexon["league id"].astype(str)
            df_kinexon.to_parquet(parquet_path)

        return df_kinexon

    def clean_game_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean the Kinexon data based on the loaded configuration.
        """
        # Retain only columns present in the configuration and rename
        df_kinexon_cleaned = self.df_kinexon[
            [
                col
                for col in self.df_kinexon.columns
                if col in self.kinexon_config
            ]
        ].rename(columns=self.kinexon_config)

        # Sportradar data does not require cleaning at this point
        return df_kinexon_cleaned, self.dict_sportradar

    def create_kinexon_snippets_from_events(self) -> None:
        """
        Create event-specific Kinexon snippets and save them to CSV files.
        """
        for dict_event in self.dict_sportradar["timeline"]:
            event_id = dict_event["id"]
            path_event = os.path.join(
                self.path_events, f"event_{event_id}_positions.csv"
            )

            # Skip if the event file already exists
            if os.path.exists(path_event):
                dict_event["path_kinexon"] = path_event
                self.dict_kinexon_path_by_event_id[event_id] = (
                    dict_event  # Update the dictionary
                )
                continue
            # Extract the event time and create a 15-second window
            event_time_tagged = pd.to_datetime(dict_event.get("time")).replace(
                tzinfo=None
            ) + pd.to_timedelta(2, unit="h")
            # Start 15 seconds before the event
            event_time_start = event_time_tagged - pd.Timedelta(seconds=15)
            # Convert Kinexon timestamps to datetime objects
            self.df_kinexon["time"] = pd.to_datetime(self.df_kinexon["time"])

            # Find the closest timestamps in the Kinexon data
            idx_start = (
                (self.df_kinexon["time"] - event_time_start).abs().idxmin()
            )
            idx_tagged = (
                (self.df_kinexon["time"] - event_time_tagged).abs().idxmin()
            )

            # Extract the Kinexon data for the event and save to CSV
            data_kinexon = self.df_kinexon.loc[idx_start:idx_tagged]
            if not data_kinexon.empty:
                data_kinexon.to_csv(path_event, index=False)
                dict_event["path_kinexon"] = path_event

            # Update the event-to-Kinexon path dictionary
            self.dict_kinexon_path_by_event_id[event_id] = dict_event

        print(f">>> Saved Kinexon data for match {self.match_id}.")

    def create_sportradar_event_from_events(self) -> None:
        """
        Saves the enriched Sportradar event to a JSON file.
        """
        for dict_event in self.dict_sportradar["timeline"]:
            event_id = dict_event["id"]
            path_event = os.path.join(
                self.path_events, f"event_{event_id}_sportradar.json"
            )
            # Skip if the event file already exists
            if os.path.exists(path_event):
                continue

            # Save the Sportradar event to a JSON file
            with open(path_event, "w", encoding="utf-8") as file:
                json.dump(dict_event, file, ensure_ascii=False, indent=4)

        print(f">>> Saved Sportradar data for match {self.match_id}.")

    def process_additional_event_data(self):
        """
        Process and add additional event data like goalkeepers, scores, and attack directions.
        """
        add_missing_goalkeeper_to_events(self.dict_kinexon_path_by_event_id)
        add_scores_to_events(self.dict_kinexon_path_by_event_id)
        calc_attack_direction(self.dict_kinexon_path_by_event_id)
        insert_names_competitors(
            self.dict_kinexon_path_by_event_id, self.dict_sportradar
        )
