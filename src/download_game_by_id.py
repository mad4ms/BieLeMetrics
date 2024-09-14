from helper_download.GameDataManager import GameDataManager
from bielemetrics_kinexon_api_wrapper import (
    login,
    fetch_team_ids,
    fetch_game_csv_data,
    fetch_event_ids,
    load_credentials,
)
import argparse
from dotenv import load_dotenv


def download_game(game_id):

    load_dotenv()
    credentials = load_credentials()
    game_manager = GameDataManager(credentials)

    # only downloads and stores, to not further process the data
    _, _ = game_manager.download_game_data(
        game_id, True, path_local="./data/raw"
    )


def main():
    # get gameid from args
    parser = argparse.ArgumentParser()
    parser.add_argument("game_id", help="game id to download data for")
    args = parser.parse_args()
    game_id = args.game_id
    # game_id = "sr:sport_event:42307453"

    download_game(game_id)


if __name__ == "__main__":
    main()
