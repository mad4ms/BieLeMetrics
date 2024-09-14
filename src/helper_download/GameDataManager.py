import os
import io
import json
from difflib import SequenceMatcher
from typing import Optional, Tuple, List, Dict
import pandas as pd
from dotenv import load_dotenv
import certifi
import urllib3
from glob import glob
from bielemetrics_kinexon_api_wrapper import (
    login,
    fetch_team_ids,
    fetch_game_csv_data,
    fetch_event_ids,
    load_credentials,
)
from sportradar import Handball

try:
    from storage_connectors.NextCloudConnector import (
        NextcloudConnector,
    )  # noqa
except ImportError:
    from .storage_connectors.NextCloudConnector import (
        NextcloudConnector,
    )

# argsparse
import argparse


class GameDataManager:
    def __init__(self, credentials: dict):
        self.credentials = credentials
        self.http = urllib3.PoolManager(
            cert_reqs="CERT_REQUIRED", ca_certs=certifi.where()
        )
        self.nc_connector = NextcloudConnector(credentials)

    @staticmethod
    def calculate_similarity(a: str, b: str) -> float:
        """
        Calculate the similarity between two strings.
        """
        return SequenceMatcher(None, a, b).ratio()

    def get_available_team_ids(self) -> List[str]:
        """
        Fetch available team IDs from Kinexon API.
        """
        login_session = login(self.credentials)
        team_data = fetch_team_ids(login_session)
        return [team["id"] for team in team_data]

    def download_sportradar_game_data(self, game_id: str) -> dict:
        """
        Download game data from Sportradar API.
        """
        client_sportradar = Handball.Handball(
            self.credentials["API_KEY_SPORTRADAR"]
        )
        timeline_sport_event = client_sportradar.get_sport_event_timeline(
            game_id
        ).json()

        sport_event = timeline_sport_event["sport_event"]
        date_sport_event = sport_event["start_time"].split("T")[0]
        team_home = sport_event["competitors"][0]["name"].replace(" ", "")
        team_away = sport_event["competitors"][1]["name"].replace(" ", "")

        print(
            f"Downloading data for game {game_id} between {team_home} and {team_away} on {date_sport_event}"
        )

        return timeline_sport_event

    def download_stored_game_data(
        self, game_id: int, path_local: str = "."
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Download game data from both Sportradar and Kinexon APIs.
        """
        json_sportradar = self.nc_connector.download_file_by_game_id(
            game_id, extension=".json", path_local=path_local
        )
        df_kinexon = self.nc_connector.download_file_by_game_id(
            game_id, extension=".csv", path_local=path_local
        )
        return df_kinexon, json_sportradar

    def store_and_upload_game_data(
        self,
        df_kinexon: pd.DataFrame,
        json_sportradar: dict,
        path_local: str,
    ) -> None:
        """
        Upload game data to Nextcloud.
        """
        game_info = self._extract_game_info(json_sportradar)
        filename_base = self._create_filename(game_info)

        gameday_path = (
            f"{self.nc_connector.path_base}/gameday_{game_info['gameday']}"
        )
        self.nc_connector.create_folders(gameday_path)

        path_local_sportradar = (
            f"{path_local}/gameday_{game_info['gameday']}/sportradar/"
        )

        os.makedirs(path_local_sportradar, exist_ok=True)

        try:
            self.nc_connector.upload_file(
                f"{filename_base}_sportradar.json",
                json_sportradar,
                f"{gameday_path}/data_sportradar_json",
                path_local=path_local_sportradar,
            )
        except Exception as e:
            print(f"Failed to upload Sportradar data: {e}")
        # self.nc_connector.upload_file(
        #     f"{filename_base}_sportradar.csv",
        #     df_sportradar,
        #     f"{gameday_path}/data_sportradar_csv",
        # )
        path_local_kinexon = (
            f"{path_local}/gameday_{game_info['gameday']}/kinexon/"
        )

        os.makedirs(path_local_kinexon, exist_ok=True)

        try:
            self.nc_connector.upload_file(
                f"{filename_base}_kinexon.csv",
                df_kinexon,
                f"{gameday_path}/data_kinexon_csv",
                path_local=path_local_kinexon,
            )
        except Exception as e:
            print(f"Failed to upload Kinexon data: {e}")

    def _extract_game_info(self, dict_sportradar: Dict) -> Dict[str, str]:
        """
        Extract game information from DataFrame.
        """
        return {
            "id_game": dict_sportradar["sport_event"]["id"],
            "datetime_game": dict_sportradar["sport_event"][
                "start_time"
            ].split("T")[0],
            "team_home": dict_sportradar["sport_event"]["competitors"][0][
                "name"
            ].replace(" ", ""),
            "team_away": dict_sportradar["sport_event"]["competitors"][1][
                "name"
            ].replace(" ", ""),
            "gameday": str(
                dict_sportradar["sport_event"]["sport_event_context"]["round"][
                    "number"
                ]
            ).zfill(2),
            "season": dict_sportradar["sport_event"]["sport_event_context"][
                "season"
            ]["name"]
            .replace("Bundesliga ", "")
            .replace("/", "-"),
        }

    def _create_filename(self, game_info: dict) -> str:
        """
        Create a generic filename based on game information.
        """
        return (
            f"{game_info['datetime_game']}"
            + f"_gd_{game_info['gameday']}"
            + f"_id_{game_info['id_game'].split(':')[-1]}"
            + f"_teams_{game_info['team_home']}"
            + f"_vs_{game_info['team_away']}"
        )

    def check_data_exists(self, game_id: int) -> bool:
        """
        Check if data for the game already exists in Nextcloud.
        """
        return self.nc_connector.check_file_exists(game_id)

    def download_kinexon_game_data(self, game_id: int) -> pd.DataFrame:
        """
        Download game data from Kinexon API.
        """

        session = login(self.credentials)
        csv_data = fetch_game_csv_data(
            session, self.credentials["ENDPOINT_KINEXON_API"], game_id
        )
        session.close()

        # Convert the CSV data to a DataFrame
        df_kinexon = pd.read_csv(io.BytesIO(csv_data), delimiter=";")
        # df_kinexon.to_csv(f"{game_id}_kinexon_debug_save.csv", index=False)

        if isinstance(csv_data, bytes):
            return pd.read_csv(io.BytesIO(csv_data), delimiter=";")
        return pd.DataFrame()

    def find_kinexon_game_id(self, dict_sportradar: dict) -> Optional[int]:
        """
        Find the corresponding Kinexon game ID by comparing data with Sportradar.
        """
        date = dict_sportradar["sport_event"]["start_time"].split("T")[0]
        team_home = dict_sportradar["sport_event"]["competitors"][0][
            "name"
        ].replace(" ", "")
        min_time, max_time = f"{date} 00:00:00", f"{date} 23:59:59"
        team_ids = self.get_available_team_ids()
        session = login(self.credentials)

        for team_id in team_ids:
            event_ids = fetch_event_ids(
                session,
                self.credentials["ENDPOINT_KINEXON_API"],
                team_id,
                min_time,
                max_time,
            )
            for game in event_ids:
                game_team_home = (
                    game["description"]
                    .split("vs.")[0]
                    .strip()
                    .replace(" ", "")
                )
                similar = self.calculate_similarity(
                    team_home.lower(), game_team_home.lower()
                )
                if similar > 0.6:
                    print(
                        f"[Kinexon] FOUND Kinexon Match: {game['description']}"
                    )
                    return game["session_id"]

        print(f"We have a problem: {team_home} - {date}")
        return None

    def download_game_data(
        self,
        game_id_str: str,
        save_to_storage: bool = True,
        path_local: str = ".",
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Download game data from both Sportradar and Kinexon APIs.
        """
        # find all files recursively in the local storage that end with the "_sportradar.json"
        list_files_local_sportradar = glob(
            f"{path_local}/**/*_sportradar.json", recursive=True
        )
        list_files_local_kinexon = glob(
            f"{path_local}/**/*_kinexon.csv", recursive=True
        )

        # make sure to convert the game_id to an integer
        game_id = int(game_id_str.split(":")[-1])

        # check if the game data is already stored locally
        file_exists_locally = False
        for file in list_files_local_sportradar:
            if str(game_id) in file:
                with open(file, "r", encoding="utf-8") as f:
                    json_sportradar = json.load(f)
                file_exists_locally = True
                print(f"Found local Sportradar data for game {game_id}")

        if not file_exists_locally:
            json_sportradar = self.download_sportradar_game_data(game_id_str)

        kinexon_game_id = self.find_kinexon_game_id(json_sportradar)
        file_exists_on_remote_storage = self.check_data_exists(game_id)
        file_exists_locally = False
        for file in list_files_local_kinexon:
            if str(game_id) in file:
                df_kinexon = pd.read_csv(file)
                file_exists_locally = True

        if (
            kinexon_game_id
            and not file_exists_on_remote_storage
            and not file_exists_locally
        ):
            df_kinexon = self.download_kinexon_game_data(kinexon_game_id)

            if save_to_storage:
                self.store_and_upload_game_data(
                    df_kinexon, json_sportradar, path_local=path_local
                )
        elif file_exists_on_remote_storage and not file_exists_locally:
            print(
                f"Data for game {game_id} exists. Downloading for local usage ..."
            )
            df_kinexon, json_sportradar = self.download_stored_game_data(
                game_id, path_local=path_local
            )

        return df_kinexon, json_sportradar

    def list_downloadable_games_of_season(
        self, wanted_season: str
    ) -> pd.DataFrame:
        """
        List all games of a season that can be downloaded and return as a DataFrame.
        """
        client_sportradar = Handball.Handball(
            self.credentials["API_KEY_SPORTRADAR"]
        )

        # Fetch all tournaments from API
        tournaments = client_sportradar.get_competitions().json()

        # Filter for the desired season
        name_competition = "Bundesliga"
        id_competition = None
        for tournament in tournaments["competitions"]:
            if (
                name_competition in tournament["name"]
                and tournament["gender"] == "men"
            ):
                print(f'Found competition: {tournament["name"]}')
                id_competition = tournament["id"]
                # Fetch all seasons for current competition
                seasons = client_sportradar.get_seasons_for_competition(
                    id_competition
                ).json()
                for season in seasons["seasons"]:
                    if wanted_season in season["name"]:
                        print(f'Found season: {season["name"]}')
                        id_season = season["id"]
                        break

        # Fetch season summaries (divided into multiple requests if needed)
        season_summaries0 = client_sportradar.get_season_summaries(
            id_season, 0, 100
        ).json()
        season_summaries1 = client_sportradar.get_season_summaries(
            id_season, 100, 100
        ).json()
        season_summaries2 = client_sportradar.get_season_summaries(
            id_season, 200, 100
        ).json()
        season_summaries3 = client_sportradar.get_season_summaries(
            id_season, 300, 100
        ).json()

        # Concatenate the season summaries
        season_summaries = season_summaries0
        season_summaries["summaries"].extend(season_summaries1["summaries"])
        season_summaries["summaries"].extend(season_summaries2["summaries"])
        season_summaries["summaries"].extend(season_summaries3["summaries"])

        # Initialize a list to store game data
        game_data = []

        # Extract relevant data from season summaries and prepare for DataFrame
        for summary in season_summaries["summaries"]:
            sport_event = summary["sport_event"]
            id_sport_event = sport_event["id"]
            date_sport_event = sport_event["start_time"]
            comps_sport_event = sport_event["competitors"]
            team_1 = comps_sport_event[0]["name"]
            team_2 = comps_sport_event[1]["name"]
            current_game_day = sport_event["sport_event_context"]["round"][
                "number"
            ]

            # Append extracted data to the game_data list
            game_data.append(
                {
                    "ID": id_sport_event,
                    "Date": date_sport_event,
                    "Team 1": team_1,
                    "Team 2": team_2,
                    "Game Day": current_game_day,
                }
            )

        # Create a DataFrame from the game_data list
        df_games = pd.DataFrame(game_data)

        return df_games


if __name__ == "__main__":
    load_dotenv()
    credentials = load_credentials()
    game_manager = GameDataManager(credentials)

    game_id = "sr:sport_event:42307423"
    df_kinexon, json_sportradar = game_manager.download_game_data(
        game_id, True, path_local="./data/raw"
    )

    # # get gameid from args
    # parser = argparse.ArgumentParser()
    # parser.add_argument("game_id", help="game id to download data for")
    # args = parser.parse_args()

    # game_id = args.game_id

    # df_games = game_manager.list_downloadable_games_of_season("23/24")
    # df_games.to_csv("games_23-24.csv", index=False)
    # exit()

    # now use subprocesses to download the data for each game

    # game_id = "sr:sport_event:42307421"
