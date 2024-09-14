import subprocess
import os
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from collections import defaultdict
import platform


# Function to process a single game by calling the process_game.py script
from process_game import process_game


def process_game_sequential(
    game_id, path_file_sportradar: str, path_file_kinexon: str
):
    process_game(path_file_sportradar, path_file_kinexon)


# Function to process a single game by calling the process_game.py script
def process_game_threaded(
    game_id, path_file_sportradar: str, path_file_kinexon: str
):
    try:
        system = platform.system()

        # Windows-specific PowerShell script
        if system == "Windows":
            # Create a unique .ps1 file for each game_id
            ps1_filename = f"process_{game_id}.ps1".replace(":", "_")

            # PowerShell script content: activate conda, run Python script for processing
            ps1_content = f"""
            conda activate env_github_bielemetrics
            python src/process_game.py {path_file_sportradar} {path_file_kinexon}
            exit
            """

            print(
                f"Processing game {game_id} with Sportradar file {path_file_sportradar}"
            )

            # Write the PowerShell commands to the .ps1 file
            with open(ps1_filename, "w") as ps1_file:
                ps1_file.write(ps1_content)

            # Use subprocess to open a new PowerShell window and wait for it to finish
            ls_output = subprocess.Popen(
                [
                    "powershell",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    f"Start-Process powershell -ArgumentList '-ExecutionPolicy Bypass -File {ps1_filename}' -Wait -WindowStyle Normal",
                ]
            )

            # Wait for the PowerShell process to complete
            ls_output.wait()

            print(f"Game {game_id} processing completed.")

            # Optionally clean up the .ps1 file after execution
            os.remove(ps1_filename)

        # Linux and macOS-specific shell script
        else:
            # Create a unique .sh file for each game_id
            sh_filename = f"process_{game_id}.sh"

            # Shell script content: activate conda, run Python script for processing
            sh_content = f"""
            source activate env_github_bielemetrics
            python src/process_game.py {path_file_sportradar} {path_file_kinexon}
            """

            print(
                f"Processing game {game_id} with Sportradar file {path_file_sportradar}"
            )

            # Write the shell commands to the .sh file
            with open(sh_filename, "w") as sh_file:
                sh_file.write(sh_content)

            # Make the script executable
            os.chmod(sh_filename, 0o755)

            # Use subprocess to open a new shell and wait for it to finish
            ls_output = subprocess.Popen(
                ["/bin/bash", sh_filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for the shell process to complete
            stdout, stderr = ls_output.communicate()

            if ls_output.returncode != 0:
                print(f"Error processing game {game_id}: {stderr.decode()}")
            else:
                print(f"Game {game_id} processing completed.")

            # Optionally clean up the .sh file after execution
            os.remove(sh_filename)

    except subprocess.CalledProcessError as e:
        print(f"Failed to process game {game_id}: {e}")


# Function to process all games for a specific game day in parallel
def process_games_for_gameday(game_data):
    # Iterate over all game data (a list of dictionaries) and process each in parallel
    with ThreadPoolExecutor(max_workers=len(game_data)) as executor:
        futures = {
            executor.submit(
                process_game_threaded, idx, game["sportradar"], game["kinexon"]
            ): idx
            for idx, game in enumerate(game_data)
        }

        for future in as_completed(futures):
            game_id = futures[future]
            try:
                future.result()  # Raises any exceptions caught in the thread
            except Exception as e:
                print(f"Error processing game {game_id}: {e}")


# Function to extract gameday from filename
def extract_gameday(file_name: str) -> int:
    # Use regex to extract the gameday from the filename
    match = re.search(r"_gd_(\d+)_", file_name)
    if match:
        return int(match.group(1))
    return None


# Function to manage parallel processing for each game day
def run_parallel_processing_by_gameday(path_to_files):
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

    # Sort the game days and process each gameday in batches
    for gameday in sorted(games_by_gameday.keys()):
        game_data = games_by_gameday[gameday]
        print(game_data)
        print(f"Starting processing for Game Day {gameday}")
        process_games_for_gameday(
            game_data
        )  # Process the games for this game day
        print(f"Completed processing for Game Day {gameday}\n")


if __name__ == "__main__":
    # Path to the raw files directory
    path_to_files = "data/raw"
    run_parallel_processing_by_gameday(path_to_files)
