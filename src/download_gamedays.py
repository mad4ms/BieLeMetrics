import pandas as pd
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from download_game_by_id import download_game
from dotenv import load_dotenv


# Function to load the CSV data
def load_game_data(file_path):
    df = pd.read_csv(file_path)
    return df[["ID", "Game Day"]]


# Function to download a single game by calling the download_game_by_id.py script
import subprocess
import os


def download_game_sequential(game_id):
    # call python script to download game
    download_game(game_id)


def download_game_threaded(game_id):
    try:
        # Create a unique .ps1 file for each game_id
        ps1_filename = f"download_{game_id}.ps1".replace(":", "_")

        # PowerShell script content: activate conda, run Python script
        ps1_content = f"""
        conda activate env_github_bielemetrics
        python src/download_game_by_id.py {game_id}
        Start-Sleep -Seconds 5
        exit
        """

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

        print(f"Game {game_id} download completed.")

        # Optionally clean up the .ps1 file after execution (can be done after all executions or on program exit)
        os.remove(ps1_filename)

    except subprocess.CalledProcessError as e:
        print(f"Failed to download game {game_id}: {e}")


def download_games_for_gameday_sequential(game_ids):
    for game_id in game_ids:
        download_game_sequential(game_id)


# Function to download all games for a specific game day in parallel
def download_games_for_gameday(game_ids):
    # debug
    # download_game(game_ids[0])
    # return

    with ThreadPoolExecutor(max_workers=len(game_ids)) as executor:
        futures = {
            executor.submit(download_game_threaded, game_id): game_id
            for game_id in game_ids
        }

        for future in as_completed(futures):
            game_id = futures[future]
            try:
                future.result()  # Raises any exceptions caught in the thread
            except Exception as e:
                print(f"Error downloading game {game_id}: {e}")


# Main function to manage parallel downloads for each game day
def run_parallel_downloads_by_gameday(file_path):
    df = load_game_data(file_path)
    grouped = df.groupby("Game Day")["ID"].apply(list).to_dict()

    for game_day, game_ids in grouped.items():
        if game_day < 10:
            continue

        print(f"Starting download for Game Day {game_day}")
        download_games_for_gameday(game_ids)
        print(f"Completed downloads for Game Day {game_day}\n")
        # break


if __name__ == "__main__":
    load_dotenv()
    file_path = "./games_23-24.csv"  # Path to your CSV file
    run_parallel_downloads_by_gameday(file_path)
