""" ETL Pipeline - Transform Data Module """

import os
import sys
import pandas as pd
import yaml
import logging
import glob
import shutil

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

from src.etl_pipeline.models.DataClassGameEvents import DataClassGameEvents
from src.etl_pipeline.models.DataClassGamePositions import (
    DataClassGamePositions,
)

logging.basicConfig(level=logging.INFO)


def find_data_position_file(
    path_to_data_positions: str, data_game_events: DataClassGameEvents
) -> str:
    """
    Find the data position file.
    """
    files_position = glob.glob(
        path_to_data_positions + "*.csv",
        recursive=True,
    )

    # Get date of the game
    date_event_events = data_game_events.get_datetime()
    # Get team home
    team_home = data_game_events.get_team_home()
    # assert that team_home is not []
    assert team_home, "Team home is empty"

    # Get team away
    team_away = data_game_events.get_team_away()
    # assert that team_away is not []
    assert team_away, "Team away is empty"

    for file in files_position:
        # read the first 1000 lines
        data = pd.read_csv(file, nrows=1000)

        # get uniques of group name
        group_name = data["group name"].unique()
        # field is formatted local time: 24.08.2023 19:00:52.800
        date_event_positions = pd.to_datetime(
            data["formatted local time"], format="%d.%m.%Y %H:%M:%S.%f"
        )
        # get the first date
        date_event_positions = date_event_positions.iloc[0]

        # check if the date of the game is the same as the date of the positions
        if date_event_events.date() == date_event_positions.date():
            # Check if the team names are in the group name
            if team_home in group_name and team_away in group_name:
                print("Found the correct file, renaming and copying ...")
                return file


def find_timestamp_in_data_positions(
    time_event: pd.Timestamp, data_positions: pd.DataFrame
) -> pd.Timestamp:
    """
    Find the timestamp in the data positions.
    """
    # Get the timestamp of the data positions
    timestamp_positions = pd.to_datetime(
        data_positions["time"], format="%d.%m.%Y %H:%M:%S.%f"
    )
    # Ensure time_event is tz-naive
    time_event_naive = time_event.replace(tzinfo=None)

    # Get the time difference
    time_diff = timestamp_positions - time_event_naive
    # Get the index of the minimum time difference
    idx = time_diff.abs().idxmin()
    # Get the timestamp
    timestamp = timestamp_positions[idx]

    return timestamp


def transform_data(path_to_data_game_events: str) -> None:
    """
    Transform data from CSV files.
    """

    # Open config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Get the path to the data positions
    data_game_events = DataClassGameEvents(path_to_data_game_events)
    # Find the data position file
    path_to_data_game_positions = find_data_position_file(
        config["etl"]["path_raw"] + "data_positions/",
        data_game_events,
    )
    data_game_positions = DataClassGamePositions(path_to_data_game_positions)

    data_game_positions.add_additional_info(
        data_game_events.get_team_home(),
        data_game_events.get_team_away(),
        data_game_events.id,
        data_game_events.get_gameday_str(),
        data_game_events.get_datetime(),
    )

    timestamp_in_pos_data = find_timestamp_in_data_positions(
        data_game_events.get_datetime(), data_game_positions.data_positions
    )
    # set to data_game_events
    data_game_events.timestamp_in_pos_data = timestamp_in_pos_data

    # save the data to the silver location
    data_game_positions.save_to_silver()

    # Save the data to the silver location
    data_game_events.save_to_silver()


def generate_folder_structure_silver() -> None:
    """
    Generate folder structure for the silver layer.
    """
    # Load config/config.yaml
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create folder structure
    os.makedirs(config["etl"]["path_silver"], exist_ok=True)
    # Now follow the structure season -> gameday -> game
    os.makedirs(
        config["etl"]["path_silver"]
        + f"season_{config["sportradar"]["season"].replace("/","-")}",
        exist_ok=True,
    )


def main() -> None:
    """
    Main function.
    """
    # Load config/config.yaml
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Generate folder structure for the silver layer
    generate_folder_structure_silver()

    # list of all game events
    list_game_events = []

    # Get all JSON files in the directory
    files = glob.glob(
        config["etl"]["path_raw"] + "data_events/*.json",
        recursive=True,
    )

    for file in files:
        # Transform data
        transform_data(file)


if __name__ == "__main__":
    main()
