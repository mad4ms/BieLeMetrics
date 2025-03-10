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


def generate_folder_structure_gold() -> None:
    """
    Generate folder structure for the gold layer.
    """
    # Load config/config.yaml
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create folder structure
    os.makedirs(config["etl"]["path_gold"], exist_ok=True)
    # Now follow the structure season -> gameday -> game
    os.makedirs(
        config["etl"]["path_gold"]
        + f"season_{config["sportradar"]["season"].replace("/","-")}",
        exist_ok=True,
    )


def load_data(
    path_to_data_game_events: str, path_to_data_game_positions
) -> None:
    """
    Load data from the silver layer.
    """
    # Load config/config.yaml
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Get the path to the data events
    data_game_events = DataClassGameEvents(path_to_data_game_events)
    data_game_positions = DataClassGamePositions(path_to_data_game_positions)

    data_game_positions.add_additional_info(
        data_game_events.get_team_home(),
        data_game_events.get_team_away(),
        data_game_events.id,
        data_game_events.get_gameday_str(),
        data_game_events.get_datetime(),
    )
