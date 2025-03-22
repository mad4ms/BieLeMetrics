"""Synchronize event and position data based on timestamps."""

import os
import sys
import json
import logging
import yaml
import glob
from typing import List, Tuple
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src2.sync.DataClassGameEvents import DataClassGameEvents
from src2.sync.DataClassGamePositions import DataClassGamePositions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataCutter:
    """
    Synchronizes event and position data.
    """

    CONFIG_PATH = "config/config.yaml"
    TIME_DIFFERENCE_THRESHOLD = 15  # seconds
    CUT_TIME_BEFORE_EVENT = 15  # seconds

    def __init__(self) -> None:
        load_dotenv()
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """
        Load configuration from the YAML configuration file.
        """
        if not os.path.exists(self.CONFIG_PATH):
            logging.error("Configuration file not found: %s", self.CONFIG_PATH)
            raise FileNotFoundError(
                f"Configuration file not found: {self.CONFIG_PATH}"
            )

        with open(self.CONFIG_PATH, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        logging.info("Configuration loaded successfully.")
        return config

    def find_matching_files(self, path_data: str) -> List[Tuple[str, str]]:
        """
        Identify matching JSON and CSV file pairs based on unique identifiers.
        """
        file_list = glob.glob(
            os.path.join(path_data, "**", "*.json"), recursive=True
        )
        file_list += glob.glob(
            os.path.join(path_data, "**", "*.csv"), recursive=True
        )

        file_mapping = {}
        for file in file_list:
            filename = os.path.basename(file)
            if "id_" in filename:
                try:
                    file_id = filename.split("id_")[1].split("_")[0]
                    extension = os.path.splitext(file)[1]

                    if file_id not in file_mapping:
                        file_mapping[file_id] = {"json": None, "csv": None}

                    if extension == ".json":
                        file_mapping[file_id]["json"] = file
                    elif extension == ".csv":
                        file_mapping[file_id]["csv"] = file
                except IndexError:
                    continue

        return [
            (entry["json"], entry["csv"])
            for entry in file_mapping.values()
            if entry["json"] and entry["csv"]
        ]

    @staticmethod
    def find_closest_timestamp(
        event_time: pd.Timestamp, data_positions: pd.DataFrame
    ) -> pd.Timestamp:
        """
        Locate the closest timestamp in position data to the event timestamp.
        """
        timestamps = pd.to_datetime(
            data_positions["time"], format="%d.%m.%Y %H:%M:%S.%f"
        )
        time_diff = (timestamps - event_time.replace(tzinfo=None)).abs()
        return timestamps[time_diff.idxmin()]

    def calc_attack_direction(
        self,
        data_events: DataClassGameEvents,
        data_positions: DataClassGamePositions,
    ) -> pd.DataFrame:
        """
        Calculate the attack direction based on the player positions.
        """
        # get list of goalkeepers
        list_goalkeepers_home_ht1 = []
        for event in data_events.timeline.events:
            if (
                (event.type == "shot_saved" and event.competitor == "away")
                or (
                    event.type == "shot_off_target"
                    and event.competitor == "away"
                )
                or (
                    event.type == "score_change" and event.competitor == "away"
                )
            ):
                match_clock_minutes = int(event.match_clock.split(":")[0])
                if match_clock_minutes < 30:
                    logging.info(
                        "%s: Found goalkeeper: %s, %s at time %s",
                        event.type,
                        event.name_goalkeeper,
                        event.competitor,
                        event.match_clock,
                    )
                    list_goalkeepers_home_ht1.append(str(event.id_goalkeeper))
                    time_last_event = event.time


        # Collect goalkeeper positions and cluster them
        df_goalkeepers = data_positions.data_positions[
            data_positions.data_positions["id_league"].isin(
                list_goalkeepers_home_ht1
            )
        ]
        df_goalkeepers = df_goalkeepers[["id_league", "x", "y", "time"]]
        # print first and last entry of time of df_goalkeepers
        logging.info("First entry of goalkeeper data: %s", df_goalkeepers.head(1))
        logging.info("Last entry of goalkeeper data: %s", df_goalkeepers.tail(1))
        # remove rows after time of last event
        df_goalkeepers = df_goalkeepers[
            df_goalkeepers["time"] <= time_last_event
        ]

        # print(df_goalkeepers)
        # Determine by the average x-coordinate if the goalkeeper is left or right
        mean_x = df_goalkeepers["x"].mean()
        logging.info("Mean x-coordinate of goalkeepers: %.2f", mean_x)
        attack_direction = (
            "left" if df_goalkeepers["x"].mean() < 20 else "right"
        )
        # add attack direction to events
        for event in data_events.timeline.events:
            if not event.match_clock:
                continue
            match_clock_minutes = int(event.match_clock.split(":")[0])
            if match_clock_minutes < 30:
                event.attack_direction = attack_direction
            else:
                event.attack_direction = "right" if attack_direction == "left" else "left"

            # show
            logging.info(
                "Event %s: %s, %s at time %s, attack direction: %s",
                event.type,
                event.name_goalkeeper,
                event.competitor,
                event.match_clock,
                event.attack_direction,
            )



    def sync_data(self) -> None:
        """
        Synchronize event and position data based on timestamps.
        """
        matching_files = self.find_matching_files(
            self.config["etl"]["path_raw"]
        )
        matching_files = [
            pair
            for pair in matching_files
            if "vs" not in pair[0] and "vs" not in pair[1]
        ]

        first_event_processed = False

        for json_file, csv_file in matching_files:
            logging.info(
                "Processing file pair:\nJSON: %s,\nCSV: %s", json_file, csv_file
            )

            events = DataClassGameEvents(json_file)
            positions = DataClassGamePositions(csv_file)
            positions.add_event_info(events)
            self.calc_attack_direction(events, positions)

            for event in events.timeline.events:
                event_time = event.time
                
                closest_time = self.find_closest_timestamp(
                    event_time, positions.data_positions
                )
                time_diff = abs((closest_time - event_time).total_seconds())

                logging.info(
                    "[Event %d/%d] Type: %s | Event Time: %s | Closest Position Time: %s | Difference: %.2f sec",
                    events.timeline.events.index(event) + 1,
                    len(events.timeline.events),
                    event.type,
                    event_time,
                    closest_time,
                    time_diff,
                )

                if time_diff > self.TIME_DIFFERENCE_THRESHOLD:
                    logging.warning(
                        "Time difference exceeds threshold: %.2f sec",
                        time_diff,
                    )
                    continue

                cut_start_time = (
                    closest_time
                    - pd.Timedelta(seconds=self.CUT_TIME_BEFORE_EVENT)
                ).replace(tzinfo=None)
                cut_positions = positions.data_positions[
                    (positions.data_positions["time"] >= cut_start_time)
                    & (positions.data_positions["time"] <= closest_time)
                ].copy()
                # insert event id
                cut_positions.loc[:, "event_id"] = event.id

                path_file = os.path.dirname(csv_file)
                # Unify folder style
                path_file = path_file.replace("\\", "/")

                output_dir = csv_file.replace(".csv", "").replace("\\", "/").replace(
                    self.config["etl"]["path_raw"],
                    self.config["etl"]["path_cut"],
                )
                os.makedirs(output_dir, exist_ok=True)
                output_file = (
                    f"{output_dir}/{os.path.basename(csv_file)}_{event.id}.csv"
                )
                cut_positions.to_csv(output_file, index=False)

                logging.info("Saved cut data positions: %s", output_file)

                if not first_event_processed:
                    with open(json_file, "r", encoding="utf-8") as f:
                        json_data = json.load(f)

                    json_output_file = os.path.join(
                        output_dir, os.path.basename(json_file)
                    )
                    with open(json_output_file, "w", encoding="utf-8") as f:
                        json.dump(json_data, f)

                    first_event_processed = True
                    logging.info(
                        "Saved first event JSON file: %s", json_output_file
                    )


if __name__ == "__main__":
    DataCutter().sync_data()
