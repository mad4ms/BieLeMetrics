import os
import io
import json
from typing import Optional, Tuple, List, Dict
from configparser import ConfigParser
import glob
import pandas as pd

try:
    from .GameEvent import GameEvent
    from .visualization_event import plot_event_syncing
    from .visualization_event import render_event
except ImportError:
    from GameEvent import GameEvent
    from visualization_event import plot_event_syncing
    from visualization_event import render_event




class GameDataPreprocessing:
    def __init__(self, path_file_sportradar: str, path_file_kinexon: str):
        self.path_file_sportradar = path_file_sportradar
        self.path_file_kinexon = path_file_kinexon

        # Construct path_file_result for csv output of features
        self.path_file_result = path_file_kinexon.replace(
            "_kinexon.csv", "_features.csv"
        )

        # Load the config files
        self.kinexon_config = self._load_config("config_fields_kinexon.cfg")
        # self.sportradar_config = self._load_config(
        #     "config_fields_sportradar.cfg"
        # )

        # load sportradar json
        with open(path_file_sportradar, "r", encoding="utf-8") as file:
            self.dict_sportradar = json.load(file)

        # assert that the sportradar json is not empty
        assert self.dict_sportradar, "Sportradar JSON is empty"

        # load kinexon csv
        self.df_kinexon = pd.read_csv(path_file_kinexon)
        # save as parquet
        self.df_kinexon.to_parquet(
            path_file_kinexon.replace(".csv", ".parquet")
        )

        # load parquet
        self.df_kinexon = pd.read_parquet(
            path_file_kinexon.replace(".csv", ".parquet"), engine="pyarrow"
        )

        # assert that the kinexon dataframe is not empty
        assert not self.df_kinexon.empty, "Kinexon dataframe is empty"

    def _load_config(self, config_path: str) -> Dict[str, str]:
        """
        Load configuration file that maps original column names to desired column names.
        """
        config = ConfigParser()
        config.read(config_path)
        return {k: v for k, v in config.items("fields")}

    def _add_scores_to_events(
        self, sportradar_events_timeline: List[Dict]
    ) -> List[Dict]:
        """
        Add the scores to the events in the timeline.
        """
        home_score = 0
        away_score = 0

        for event in sportradar_events_timeline:
            # Update the score if the event changes it
            if "home_score" in event:
                home_score = event.get("home_score")
                away_score = event.get("away_score")

            # Forward fill the score to the current event
            event["home_score"] = home_score
            event["away_score"] = away_score

        return sportradar_events_timeline

    def _add_goalkeeper_for_events(
        self, sportradar_events_timeline: List[Dict], path_files_kinexon: str
    ) -> List[Dict]:
        """
        Add the goalkeeper to the events in the timeline.
        """
        df_sportradar = pd.DataFrame(sportradar_events_timeline)

        list_goalkeeper_home_ids = []
        list_goalkeeper_away_ids = []

        list_events_without_goalkeeper = []

        for event in sportradar_events_timeline:
            if not "players" in event:
                event["id_goalkeeper"] = None
                list_events_without_goalkeeper.append(event)
                continue

            # Check if the event has a goalkeeper
            for player in event["players"]:
                if player["type"] == "goalkeeper" or player["type"] == "saved":
                    event["id_goalkeeper"] = player["id"].split(":")[-1]
                    # Check if goalkeeper is from home or away team
                    if event["competitor"] == "home":
                        # append to away team
                        if player["id"] not in list_goalkeeper_away_ids:
                            list_goalkeeper_away_ids.append(player["id"])
                    else:
                        # append to home team
                        if player["id"] not in list_goalkeeper_home_ids:
                            list_goalkeeper_home_ids.append(player["id"])

                else:
                    list_events_without_goalkeeper.append(event)

        # Clear duplicate events from list_events_without_goalkeeper
        list_events_without_goalkeeper = list(
            {v["id"]: v for v in list_events_without_goalkeeper}.values()
        )

        # Now iterate over the events that don't have a goalkeeper and insert the goalkeeper
        # based on the list of goalkeepers
        # Iterate over the event ids and read the kinexon data
        for event in list_events_without_goalkeeper:
            event_id = event["id"]
            # Find file
            kin_event_for_sportradar_event = glob.glob(
                os.path.join(path_files_kinexon, f"*{event_id}*.csv")
            )

            # event = sportradar_events_timeline[df_sportradar["id"] == event_id]

            # Check if the file is found
            if len(kin_event_for_sportradar_event) == 0:
                print(f"! File for event {event_id} not found.")
                continue
            # Read the file
            df_kinexon_event = pd.read_csv(kin_event_for_sportradar_event[0])
            # if the competitor in the event is home, it has the group_id of 1 in the df_kinexon_event
            # if the competitor in the event is away, it has the group_id of 2 in the df_kinexon_event

            if not "competitor" in event:
                print(
                    f"! Competitor for {event_id} not found. Type was {event["type"]}"
                )
                continue

            if event["competitor"] == "home":
                # check for the goalkeeper in the away team
                for player_id in list_goalkeeper_away_ids:
                    df_goalkeeper = df_kinexon_event[
                        df_kinexon_event["league_id"] == player_id.split(":")[-1]
                    ]
                    if len(df_goalkeeper) > 0:
                        # print(f">> Away Goalkeeper {player_id} data for {event_id} found. Type was {event["type"]}")
                        break
                # Check if the goalkeeper data is available
                if len(df_goalkeeper) == 0:
                    print(f"! Goalkeeper data for {event_id} not found. Type was {event["type"]}")
                    continue
            else:
                # check for the goalkeeper in the home team
                for player_id in list_goalkeeper_home_ids:
                    df_goalkeeper = df_kinexon_event[
                        df_kinexon_event["league_id"] == player_id.split(":")[-1]
                    ]
                    if len(df_goalkeeper) > 0:
                        # print(f">> Home Goalkeeper {player_id}  data for {event_id} found. Type was {event["type"]}")
                        break
                # Check if the goalkeeper data is available
                if len(df_goalkeeper) == 0:
                    print(f"! Goalkeeper data for {event_id} not found. Type was {event["type"]}")
                    continue

            # insert the goalkeeper data into the sportradar event
            event["id_goalkeeper"] = df_goalkeeper["league_id"].values[0]

        return sportradar_events_timeline

    def _add_attack_direction_to_events(
        self, sportradar_events_timeline: List[Dict], path_files_kinexon: str
    ) -> List[Dict]:
        """
        Add the attack direction to the events in the timeline.
        """
        df_sportradar = pd.DataFrame(sportradar_events_timeline)

        df_sportradar["match_clock"] = pd.to_numeric(
            df_sportradar["match_clock"].str.replace(":", "")
        )

        # Filter for the first half and the away team
        # because the goalkeeper is from the home team in this row then
        df_sportradar_ht1 = df_sportradar[
            (df_sportradar["match_clock"] <= 3000)
            & (df_sportradar["competitor"] == "home")
        ].copy()

        list_event_ids = df_sportradar_ht1["id"].tolist()

        pos_goalkeeper_home_x = []

        # Iterate over the event ids and read the kinexon data
        for event_id in list_event_ids:
            # Find file
            kin_event_for_sportradar_event = glob.glob(
                os.path.join(path_files_kinexon, f"*{event_id}*.csv")
            )
            # Check if the file is found
            if len(kin_event_for_sportradar_event) == 0:
                print(f"File for event {event_id} not found.")
                continue
            # Read the file
            df_kinexon_event = pd.read_csv(kin_event_for_sportradar_event[0])
            # Get the league id for the goalkeeper from sportradar
            field_event_players = df_sportradar_ht1[
                df_sportradar_ht1["id"] == event_id
            ]["players"].values[0]

            # check if field_event_players is a list, else continue
            if not isinstance(field_event_players, list):
                continue

            for player in field_event_players:
                # check if type is goalkeeper or saved
                if player["type"] == "goalkeeper" or player["type"] == "saved":
                    league_id_goalkeeper = player["id"].split(":")[-1]
                    break

            # league_id_goalkeeper = df_sportradar_ht1[
            #     df_sportradar_ht1["id"] == event_id
            # ]["id_goalkeeper"].values[0]

            # Check if league_id_goalkeeper is not null
            if pd.isnull(league_id_goalkeeper):
                # print(f"Goalkeeper id for {event_id} not found.")
                continue

            # Get the goalkeeper data from the kinexon data
            df_goalkeeper = df_kinexon_event[
                df_kinexon_event["league_id"] == str(int(league_id_goalkeeper))
            ].copy()
            # Check if the goalkeeper data is available
            if len(df_goalkeeper) == 0:
                # print(f"Goalkeeper data for {event_id} not found.")
                continue

            # get the last position of the goalkeeper
            last_goalkeeper_position = df_goalkeeper.iloc[-1]

            # Append the goalkeeper position to the list
            pos_goalkeeper_home_x.append(last_goalkeeper_position["pos_x"])

        # Determine the attack direction based on the
        # goalkeeper positions in the first half
        if len(pos_goalkeeper_home_x) == 0:
            print("No goalkeeper data found.")
            return df_sportradar
        # Determine the attack direction based on the goalkeeper position
        if sum(pos_goalkeeper_home_x) / len(pos_goalkeeper_home_x) < 20:
            attack_direction = "right"
        else:
            attack_direction = "left"

        # Add the attack direction to the events
        # For the first half, the home team attack direction is the same as the goalkeeper
        for event in sportradar_events_timeline:
            if not "match_clock" in event or not "competitor" in event:
                event["attack_direction"] = None
                event["match_clock_in_s"] = None
                event["competitor"] = None
                continue

            # match_clock is in the format m:ss or mm:ss, so we need to split and convert
            minutes, seconds = map(int, event["match_clock"].split(":"))
            # convert match_clock to total seconds
            event["match_clock_in_s"] = minutes * 60 + seconds

            if (
                event["match_clock_in_s"] <= 1800
                and event["competitor"] == "home"
            ):
                event["attack_direction"] = attack_direction
            elif (
                event["match_clock_in_s"] <= 1800
                and event["competitor"] == "away"
            ):
                event["attack_direction"] = (
                    "right" if attack_direction == "left" else "left"
                )
            elif (
                event["match_clock_in_s"] > 1800
                and event["competitor"] == "home"
            ):
                event["attack_direction"] = (
                    "right" if attack_direction == "left" else "left"
                )
            elif (
                event["match_clock_in_s"] > 1800
                and event["competitor"] == "away"
            ):
                event["attack_direction"] = attack_direction

        return sportradar_events_timeline

    # def _calculate_attack_direction(self, df_goalkeeper: pd.DataFrame) -> str:
    #     """
    #     Calculate the attack direction based on the goalkeeper's position.
    #     """
    #     # Your code to calculate the attack direction goes here

    #     return attack_direction

    def extract_game_data_to_events(self) -> List[Dict]:
        """
        Process game data from Sportradar and Kinexon.
        """
        # Extract single events from sportradar data
        sportradar_events_timeline = self.dict_sportradar["timeline"]
        # and convert ts in ms to datetime in kinexon data, which is in ms since epoch
        self.df_kinexon["time"] = pd.to_datetime(self.df_kinexon["time"])
        self.df_kinexon["league_id"] = self.df_kinexon["league_id"].astype(str)

        id_match = self.dict_sportradar["sport_event"]["id"]
        id_match_int = id_match.split(":")[-1]

        list_game_events = []

        # Create and save the sportradar events, but do not process them yet
        # as we need to add additional information first
        for event in sportradar_events_timeline:
            game_event = GameEvent(
                event,
                self.df_kinexon,
                plot_sync=False,
                render_video_event=False,
                path_result_csv=self.path_file_result,
            )

            # Save the event and kinexon data
            game_event.save_event(
                directory=f"./data/events/match_{id_match.split(':')[-1]}"
            )

        # Add additional information to the events
        sportradar_events_timeline = self._add_scores_to_events(
            sportradar_events_timeline
        )
        # Add goalkeeper to the events
        sportradar_events_timeline = self._add_goalkeeper_for_events(
            sportradar_events_timeline, f"./data/events/match_{id_match_int}/"
        )
        # Add attack direction to the events
        sportradar_events_timeline = self._add_attack_direction_to_events(
            sportradar_events_timeline, f"./data/events/match_{id_match_int}/"
        )

        # and again after adding information, we process the events
        for event in sportradar_events_timeline:
            game_event = GameEvent(
                event,
                self.df_kinexon,
                plot_sync=False,
                render_video_event=False,
                path_result_csv=self.path_file_result,
            )
            # HERE we process the event
            game_event.process_event()

            # Overwrite and Save the event and kinexon data
            game_event.save_event(
                directory=f"./data/events/match_{id_match.split(':')[-1]}"
            )

            # plot sync
            # plot_event_syncing(game_event)
            if True:
                render_event(game_event)

            # Save the event and kinexon data
            # game_event.save_event(
            #     directory=f"./data/events/match_{id_match.split(':')[-1]}"
            # )

    def clean_kinexon_data(self, df_kinexon: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and rename the Kinexon data columns based on the config file.
        """
        # remove columns that are not in the config file
        df_kinexon = df_kinexon[
            [col for col in df_kinexon.columns if col in self.kinexon_config]
        ]
        # and rename the columns based on the config file
        df_kinexon = df_kinexon.rename(columns=self.kinexon_config)

        return df_kinexon

    def clean_game_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean both Kinexon and Sportradar data and return the cleaned versions.
        """
        self.df_kinexon = self.clean_kinexon_data(self.df_kinexon)

        # Dict sportradar does not need to be cleaned

        return self.df_kinexon, self.dict_sportradar


if __name__ == "__main__":
    path_file_sportradar = "./data/raw/2023-08-24_gd_01_id_42307421_HCErlangen_vs_TSVHannover-Burgdorf_sportradar.json"
    path_file_kinexon = "./data/raw/2023-08-24_gd_01_id_42307421_HCErlangen_vs_TSVHannover-Burgdorf_kinexon.csv"

    game_data_preprocessing = GameDataPreprocessing(
        path_file_sportradar, path_file_kinexon
    )
    game_data_preprocessing.clean_game_data()
    game_data_preprocessing.extract_game_data_to_events()
