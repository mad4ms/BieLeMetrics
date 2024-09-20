import os
import io
import json
import sys
from typing import Optional, Tuple, List, Dict
from configparser import ConfigParser
import glob
import pandas as pd
import argparse
import warnings

# append the path of the src directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.core.game import Game
from src.core.game_event import GameEvent
from src.core.game_event_features import GameEventFeatures

from src.processing.helper_processing.visualization_event import render_event

warnings.simplefilter(action="ignore", category=FutureWarning)


class GameProcessing:
    """
    GameProcessing class to process game data.
    """

    def __init__(
        self,
        path_file_sportradar: str,
        path_file_kinexon: str,
        render_events: bool = False,
    ):
        """
        Initialize the GameProcessing object with paths to Sportradar and Kinexon data.
        """
        self.path_file_sportradar = path_file_sportradar
        self.path_file_kinexon = path_file_kinexon
        self.game = Game(path_file_sportradar, path_file_kinexon)

        # TODO proper implementation of result path
        self.path_file_result = (
            path_file_kinexon.replace("_kinexon.csv", "_features.csv")
            .replace("raw/", "processed/")
            .replace("raw\\", "processed/")
        )

        self.render_events = render_events
        # initialize list of game events
        self.list_game_events = []
        # process game data
        self.process_game()
        # Save the game events
        self.save_game_events()

    def process_game(self) -> None:
        """
        Process the game data.
        """
        print(f"Processing game data for match ID: {self.game.match_id}")

        # process game events
        for (
            event_id,
            dict_event,
        ) in self.game.dict_kinexon_path_by_event_id.items():
            game_event = GameEvent(
                dict_event,
                self.game.dict_kinexon_path_by_event_id[event_id][
                    "path_kinexon"
                ],
            )
            # game_event.process_event()
            self.list_game_events.append(game_event)

            if self.render_events:
                render_event(game_event)

    def save_game_events(self) -> None:
        """
        Save the game events.
        """
        df_result = pd.DataFrame()

        print("Saving game events...")
        for game_event in self.list_game_events:
            df_game_event = pd.DataFrame([game_event.to_dict()])

            df_result = pd.concat(
                [df_result, df_game_event], ignore_index=True
            )

        # makedir of basepath
        os.makedirs(os.path.dirname(self.path_file_result), exist_ok=True)

        # Save the game events to CSV
        df_result.to_csv(self.path_file_result, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path_file_sportradar", help="path to sportradar file")
    parser.add_argument("path_file_kinexon", help="path to kinexon file")

    # args = parser.parse_args()
    # path_file_kinexon = args.path_file_kinexon
    # path_file_sportradar = args.path_file_sportradar

    path_file_sportradar = "./data/raw/gameday_01/sportradar/2023-08-24_gd_01_id_42307421_teams_HCErlangen_vs_TSVHannover-Burgdorf_sportradar.json"
    path_file_kinexon = "./data/raw/gameday_01/kinexon/2023-08-24_gd_01_id_42307421_teams_HCErlangen_vs_TSVHannover-Burgdorf_kinexon.csv"

    game_processing = GameProcessing(path_file_sportradar, path_file_kinexon)
