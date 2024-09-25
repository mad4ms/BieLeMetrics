import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.processing.game_processor import GameProcessor
import argparse

# example is:
# path_file_sportradar = "2023-08-24_gd_01_id_42307421_HCErlangen_vs_TSVHannover-Burgdorf_sportradar.json"
# path_file_kinexon = "2023-08-24_gd_01_id_42307421_HCErlangen_vs_TSVHannover-Burgdorf_kinexon.csv"

# game_data_preprocessing = GameDataPreprocessing(
#     path_file_sportradar, path_file_kinexon
# )
# game_data_preprocessing.clean_game_data()
# game_data_preprocessing.extract_game_data_to_events()


def process_game(path_file_sportradar: str, path_file_kinexon: str) -> None:
    """
    Process a game by cleaning and extracting data.
    """
    game_data_preprocessing = GameProcessor(
        path_file_sportradar, path_file_kinexon
    )


def main():
    # get gameid from args
    parser = argparse.ArgumentParser()
    parser.add_argument("path_file_sportradar", help="path to sportradar file")
    parser.add_argument("path_file_kinexon", help="path to kinexon file")
    args = parser.parse_args()

    path_file_kinexon = args.path_file_kinexon
    path_file_sportradar = args.path_file_sportradar

    # path_file_sportradar = "data/raw/gameday_01/sportradar/2023-08-28_gd_01_id_42307435_teams_SCDHfKLeipzig_vs_FuchseBerlin_sportradar.json"
    # path_file_kinexon = "data/raw/gameday_01/kinexon/2023-08-28_gd_01_id_42307435_teams_SCDHfKLeipzig_vs_FuchseBerlin_kinexon.csv"

    process_game(
        path_file_sportradar,
        path_file_kinexon,
    )


if __name__ == "__main__":
    main()
