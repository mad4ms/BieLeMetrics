import subprocess
import os
from glob import glob
import re
from collections import defaultdict
import platform


import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

# Import the process_game function
from src.process_game import process_game

DEBUG_MODE = False
id_game = ""

START_AFTER_MODE = False
START_AFTER_ID = ""


# Function to process a single game sequentially
def process_game_sequential(path_file_sportradar: str, path_file_kinexon: str):
    """Process a single game using the process_game script."""

    global START_AFTER_MODE  # Declare that you are using the global variable

    if DEBUG_MODE:
        if not id_game in path_file_sportradar:
            print(
                f"Skip game with Sportradar: {path_file_sportradar} and Kinexon: {path_file_kinexon}"
            )
            return
        print(
            f"Processing game with Sportradar: {path_file_sportradar} and Kinexon: {path_file_kinexon}"
        )

    if START_AFTER_MODE:
        if not START_AFTER_ID in path_file_sportradar:
            return
        else:
            START_AFTER_MODE = False

    process_game(path_file_sportradar, path_file_kinexon)
    print(
        f"Processed game with Sportradar: {path_file_sportradar} and Kinexon: {path_file_kinexon}"
    )
    # except Exception as e:
    #     print(f"Failed to process game: {e}")


# Function to process all games for a specific game day sequentially
def process_games_for_gameday(game_data):
    """Process all games for a specific game day sequentially."""
    for game in game_data:
        process_game_sequential(game["sportradar"], game["kinexon"])


# Function to extract gameday from filename
def extract_gameday(file_name: str) -> int:
    """Extract the gameday number from the file name using regex."""
    match = re.search(r"_gd_(\d+)_", file_name)
    if match:
        return int(match.group(1))
    return None


# Function to process games by game day sequentially
def run_sequential_processing_by_gameday(path_to_files):
    """Finds files, groups them by gameday, and processes them sequentially."""
    list_files_sportradar = glob(
        f"{path_to_files}/**/*sportradar.json", recursive=True
    )
    list_files_kinexon = glob(
        f"{path_to_files}/**/*kinexon.csv", recursive=True
    )

    print(f"Found {len(list_files_sportradar)} Sportradar files")
    print(f"Found {len(list_files_kinexon)} Kinexon files")

    # Create a mapping from gameday to list of game data
    games_by_gameday = defaultdict(list)

    # Pair the Sportradar and Kinexon files together and group them by gameday
    for sportradar_file, kinexon_file in zip(
        list_files_sportradar, list_files_kinexon
    ):
        gameday = extract_gameday(sportradar_file)
        if gameday is not None:
            games_by_gameday[gameday].append(
                {"sportradar": sportradar_file, "kinexon": kinexon_file}
            )

    # Sort the game days and process each gameday sequentially
    for gameday in sorted(games_by_gameday.keys()):
        game_data = games_by_gameday[gameday]
        print(f"Starting processing for Game Day {gameday}")
        process_games_for_gameday(game_data)
        print(f"Completed processing for Game Day {gameday}\n")


if __name__ == "__main__":
    # Path to the raw files directory
    path_to_files = "data/raw"
    run_sequential_processing_by_gameday(path_to_files)
