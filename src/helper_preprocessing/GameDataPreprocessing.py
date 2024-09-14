import os
import io
import json
from typing import Optional, Tuple, List, Dict
from configparser import ConfigParser
import glob
import pandas as pd
import argparse
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

try:
    from .game_event import GameEvent
    from .visualization_event import plot_event_syncing
    from .visualization_event import render_event
except ImportError:
    try:
        from game_event import GameEvent
        from visualization_event import plot_event_syncing
        from visualization_event import render_event
    except ImportError:
        from helper_preprocessing.game_event import GameEvent
        from helper_preprocessing.visualization_event import plot_event_syncing
        from helper_preprocessing.visualization_event import render_event




class GameDataPreprocessing:
    def __init__(self, path_file_sportradar: str, path_file_kinexon: str):
        self.path_file_sportradar = path_file_sportradar
        self.path_file_kinexon = path_file_kinexon

        # Construct path_file_result for csv output of features
        self.path_file_result = path_file_kinexon.replace(
            "_kinexon.csv", "_features.csv"
        ).replace("raw/", "processed/").replace("raw\\", "processed/")

        # extract ID from path_file_kinexon by splitting and selecting "_id_MYID"
        self.match_id = path_file_kinexon.split("_id_")[1].split("_")[0]

        self.path_files_events = None
        # split after "raw" to get the path to the events and relpace raw with events
        self.path_events = path_file_kinexon.split("raw")[0] + "events/"
        # now add the match id to the path
        self.path_events = os.path.join(self.path_events, f"match_{self.match_id}")

        # Create the directory if it does not exist
        if not os.path.exists(self.path_events):
            os.makedirs(self.path_events, exist_ok=True)


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

        # Check if parquet file exists
        if os.path.exists(path_file_kinexon.replace(".csv", ".parquet")):
            # load parquet
            self.df_kinexon = pd.read_parquet(
                path_file_kinexon.replace(".csv", ".parquet"), engine="pyarrow"
            )
        else:
            # load kinexon csv
            self.df_kinexon = pd.read_csv(path_file_kinexon)

            # print uniques of league id
            print(f"Unique league ids: {self.df_kinexon['league id'].astype(str).unique()}")

            # Convert column league id to string
            # self.df_kinexon["league id"] = self.df_kinexon["league id"].astype(str)
            # save as parquet
            try:
                self.df_kinexon.to_parquet(
                    path_file_kinexon.replace(".csv", ".parquet")
                )
            except Exception as e:
                # try to convert league id to string
                self.df_kinexon["league id"] = self.df_kinexon["league id"].astype(str)
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
    
    def create_kinexon_snippets_from_events(self) -> None:
        """
        Create singular event data from the Sportradar and  Kinexon data.
        """


        # create dict with event id as key and the path to the kinexon data as value
        self.dict_event_id_kinexon_path = {}
        # Get event ID from iterating over sportradar events
        for dict_event in self.dict_sportradar["timeline"]:
            event_id = dict_event["id"]
            # break

            if self.df_kinexon is not None:
                event_time_tagged = pd.to_datetime(dict_event.get("time")).replace(
                    tzinfo=None
                ) + pd.to_timedelta(2, unit="h")
                event_time_start = event_time_tagged - pd.Timedelta(seconds=15)
                self.df_kinexon["time"] = pd.to_datetime(self.df_kinexon["time"])

                diff =self. df_kinexon["time"] - event_time_start
                # print(f'Difference of {event_id} is {diff}')
                idx_start = diff.abs().idxmin()

                diff = self.df_kinexon["time"] - event_time_tagged
                idx_tagged = diff.abs().idxmin()

                data_kinexon = self.df_kinexon.loc[idx_start:idx_tagged]

                if not data_kinexon.empty:
                    # return data_kinexon
                    pass

                path_event = f"{self.path_events}/event_{event_id}_positions.csv"

                # add to dict_event
                dict_event["path_kinexon"] = path_event

                self.dict_event_id_kinexon_path[event_id] = dict_event

                # check if the file already exists
                if os.path.exists(path_event):
                    # print(f"File for event {event_id} already exists.")
                    continue
                # Save Kinexon data for the event
                data_kinexon.to_csv(path_event, index=False)

        print(f'>>> Saved Kinexon data for match {self.match_id}.')

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
    
    def calc_attack_direction(self) -> None:
        """
        Calculate the attack direction based on the goalkeeper's position.
        """
        # IDs of goalkeepers in the first half for team home and away
        list_goalkeeper_home_ids = []
        list_goalkeeper_positions_home = []

        for event_id, dict_event in self.dict_event_id_kinexon_path.items():
            if not "id_goalkeeper" in dict_event or not "competitor" in dict_event:
                continue

            if dict_event["competitor"] == "home" and dict_event["match_time"] <= 30:
                list_goalkeeper_home_ids.append(dict_event["id_goalkeeper"])

        # Iterate over the event ids and read the kinexon data
        for event_id, dict_event in self.dict_event_id_kinexon_path.items():
            if not "id_goalkeeper" in dict_event:
                continue

            # Find file
            kin_event_for_sportradar_event = glob.glob(
                os.path.join(dict_event["path_kinexon"])
            )

            # Check if the file is found
            if len(kin_event_for_sportradar_event) == 0:
                print(f"! File for event {event_id} not found.")
                continue
            # Read the file
            df_kinexon_event = pd.read_csv(kin_event_for_sportradar_event[0])

            # Get the goalkeeper data from the kinexon data
            df_goalkeeper = df_kinexon_event[
                df_kinexon_event["league_id"] == dict_event["id_goalkeeper"]
            ].copy()
            # Check if the goalkeeper data is available
            if len(df_goalkeeper) == 0:
                # print(f"! Goalkeeper data for {event_id} not found.")
                continue

            # get the last position of the goalkeeper
            last_goalkeeper_position = df_goalkeeper.iloc[-1]

            # Append the goalkeeper position to the list
            list_goalkeeper_positions_home.append(last_goalkeeper_position["pos_x"])

        # Determine the attack direction based on the
        # goalkeeper positions in the first half
        if len(list_goalkeeper_positions_home) == 0:
            # print("No goalkeeper data found.")
            return
        
        # print the average position of the goalkeeper

        print(f"Average position of the goalkeeper: {sum(list_goalkeeper_positions_home) / len(list_goalkeeper_positions_home)}")
        
        # Determine the attack direction based on the goalkeeper position
        if sum(list_goalkeeper_positions_home) / len(list_goalkeeper_positions_home) < 20:
            # Then the goalkeeper is positioned on the left side, therefore the attack direction is right
            attack_direction = "right"
        else:
            # Then the goalkeeper is positioned on the right side, therefore the attack direction is left
           attack_direction = "left"

        # Add the attack direction to the events
        # For the first half, the home team attack direction is the same as the goalkeeper
        for event_id, dict_event in self.dict_event_id_kinexon_path.items():
            if not "match_time" in dict_event or not "competitor" in dict_event:
                dict_event["attack_direction"] = None
                dict_event["match_time"] = None
                dict_event["competitor"] = None
                continue

            if (
                dict_event["match_time"] <= 30
                and dict_event["competitor"] == "home"
            ):
                dict_event["attack_direction"] = attack_direction
            elif (
                dict_event["match_time"] <= 30
                and dict_event["competitor"] == "away"
            ):
                dict_event["attack_direction"] = (
                    "right" if attack_direction == "left" else "left"
                )
            elif (
                dict_event["match_time"] > 30
                and dict_event["competitor"] == "home"
            ):
                dict_event["attack_direction"] = (
                    "right" if attack_direction == "left" else "left"
                )
            elif (
                dict_event["match_time"] > 30
                and dict_event["competitor"] == "away"
            ):
                dict_event["attack_direction"] = attack_direction

            # for sanity check, print the match_time, competitor and attack_direction
            # print(f"Match time: {dict_event['match_time']}, Competitor: {dict_event['competitor']}, Attack direction: {dict_event['attack_direction']}")
        # print unique attack directions of team home for the first half
        attack_dir_home_first_half = list(set([
            dict_event["attack_direction"]
            for event_id, dict_event in self.dict_event_id_kinexon_path.items()
            if dict_event["competitor"] == "home" and dict_event["match_time"] <= 30
        ]))
        print(f'Unique attack directions of team home for the first half: {attack_dir_home_first_half}')

        # print unique attack directions of team away for the first half
        attack_dir_away_first_half = list(set([
            dict_event["attack_direction"]
            for event_id, dict_event in self.dict_event_id_kinexon_path.items()
            if dict_event["competitor"] == "away" and dict_event["match_time"] <= 30
        ]))
        print(f'Unique attack directions of team away for the first half: {attack_dir_away_first_half}')

        # print unique attack directions of team home for the second half
        attack_dir_home_second_half = list(set([
            dict_event["attack_direction"]
            for event_id, dict_event in self.dict_event_id_kinexon_path.items()
            if dict_event["competitor"] == "home" and dict_event["match_time"] > 30
        ]))
        print(f'Unique attack directions of team home for the second half: {attack_dir_home_second_half}')

        # print unique attack directions of team away for the second half
        attack_dir_away_second_half = list(set([
            dict_event["attack_direction"]
            for event_id, dict_event in self.dict_event_id_kinexon_path.items()
            if dict_event["competitor"] == "away" and dict_event["match_time"] > 30
        ]))
        print(f'Unique attack directions of team away for the second half: {attack_dir_away_second_half}')
    
    def add_missing_goalkeeper_to_events(self):
        """
        Add missing goalkeeper to the events.
        """
        list_goalkeeper_home_ids = []
        list_goalkeeper_away_ids = []

        list_events_without_goalkeeper = []

        relevant_event_types = [
            "score_change",
            "shot_off_target",
            "shot_blocked",
            "shot_saved",
            "seven_m_missed",
        ]



        for event_id, dict_event in self.dict_event_id_kinexon_path.items():
            if dict_event["type"] not in relevant_event_types:
                # print(f'! Event {event_id} is not a relevant event type {dict_event["type"]}.')
                dict_event["id_goalkeeper"] = None
                continue
            if not "players" in dict_event:
                dict_event["id_goalkeeper"] = None
                list_events_without_goalkeeper.append(dict_event)
                continue

            # Check if the event has a goalkeeper
            for player in dict_event["players"]:
                if player["type"] == "goalkeeper" or player["type"] == "saved":
                    dict_event["id_goalkeeper"] = player["id"].split(":")[-1]
                    # Check if goalkeeper is from home or away team
                    if dict_event["competitor"] == "home":
                        # append to away team
                        if player["id"] not in list_goalkeeper_away_ids:
                            list_goalkeeper_away_ids.append(player["id"])
                    else:
                        # append to home team
                        if player["id"] not in list_goalkeeper_home_ids:
                            list_goalkeeper_home_ids.append(player["id"])

                else:
                    list_events_without_goalkeeper.append(dict_event)

        print(f"Found following gk for team home: {list_goalkeeper_home_ids}")
        print(f"Found following gk for team away: {list_goalkeeper_away_ids}")

        # Clear duplicate events from list_events_without_goalkeeper
        list_events_without_goalkeeper = list(
            {v["id"]: v for v in list_events_without_goalkeeper}.values()
        )

        # Now iterate over the events that don't have a goalkeeper and insert the goalkeeper
        # based on the list of goalkeepers
        # Iterate over the event ids and read the kinexon data
        for dict_event in list_events_without_goalkeeper:
            event_id = dict_event["id"]
            # Find file
            kin_event_for_sportradar_event = glob.glob(
                os.path.join(dict_event["path_kinexon"])
            )

            # check if "shot_blocked" is in the event type
            if dict_event["type"] == "shot_blocked":
                # print(f"Event {event_id} is a shot blocked event.")
                pass

            # event = sportradar_events_timeline[df_sportradar["id"] == event_id]

            # Check if the file is found
            if len(kin_event_for_sportradar_event) == 0:
                print(f"! File for event {event_id} not found.")
                continue
            # Read the file
            df_kinexon_event = pd.read_csv(kin_event_for_sportradar_event[0])
            # if the competitor in the event is home, it has the group_id of 1 in the df_kinexon_event
            # if the competitor in the event is away, it has the group_id of 2 in the df_kinexon_event

            if not "competitor" in dict_event:
                print(
                    f"! Competitor for {event_id} not found. Type was {dict_event["type"]}"
                )
                continue

            if dict_event["competitor"] == "home":
                # check for the goalkeeper in the away team
                for player_id in list_goalkeeper_away_ids:
                    df_goalkeeper = df_kinexon_event[
                        df_kinexon_event["league_id"] == player_id.split(":")[-1]
                    ]
                    if len(df_goalkeeper) > 0:
                        # print(f">> Away Goalkeeper {player_id} data for {event_id} found. Type was {dict_event["type"]}")
                        break
                # Check if the goalkeeper data is available
                if len(df_goalkeeper) == 0:
                    print(f"! Goalkeeper data for {event_id} not found. Type was {dict_event["type"]}")
                    continue
            else:
                # check for the goalkeeper in the home team
                for player_id in list_goalkeeper_home_ids:
                    df_goalkeeper = df_kinexon_event[
                        df_kinexon_event["league_id"] == player_id.split(":")[-1]
                    ]
                    if len(df_goalkeeper) > 0:
                        # print(f">> Home Goalkeeper {player_id}  data for {event_id} found. Type was {dict
                        break
                # Check if the goalkeeper data is available
                if len(df_goalkeeper) == 0:
                    print(f"! Goalkeeper data for {event_id} not found. Type was {dict_event["type"]}")
                    continue

            # insert the goalkeeper data into the sportradar event
            dict_event["id_goalkeeper"] = df_goalkeeper["league_id"].values[0]
            dict_event["name_goalkeeper"] = df_goalkeeper["full_name"].values[0]

        # return list_events_without_goalkeeper
    
    def add_scores_to_events(self) -> None:
        """
        Add the scores to the events in the timeline.
        """
        home_score = 0
        away_score = 0

        for event_id, dict_event in self.dict_event_id_kinexon_path.items():
            # Update the score if the event changes it
            if "home_score" in dict_event:
                home_score = dict_event.get("home_score")
                away_score = dict_event.get("away_score")

            # Forward fill the score to the current event
            dict_event["home_score"] = home_score
            dict_event["away_score"] = away_score

    def process_game_events(self) -> List[Dict]:
        """
        Process the game events.
        """
        self.add_missing_goalkeeper_to_events()

        self.add_scores_to_events()

        self.calc_attack_direction()

        df_result = pd.DataFrame()
        # Your code to process the game events goes here
        # iterate over dict_event_id_kinexon_path
        for index, (event_id, dict_event) in enumerate(self.dict_event_id_kinexon_path.items()):
            # print(f"Processing event number {index + 1} of {len(self.dict_event_id_kinexon_path)}")
            print(f"> ({index + 1} / {len(self.dict_event_id_kinexon_path)})\tType:\t{dict_event['type']}\tID:\t{event_id}")     

            
            # create the game event
            game_event = GameEvent(
            dict_event,
            path_to_kinexon_scene=dict_event["path_kinexon"],
            )

            df_game_event = pd.DataFrame([game_event.to_dict()])

            df_result = pd.concat([df_result, df_game_event], ignore_index=True)


            if game_event.event_time_throw:
                pass
                # print(f"Throw point found for event {event_id}.")
                # plot_event_syncing(game_event)
            render_event(game_event)

            # # process the game event
            # game_event.process_event()
            # # save the game event
            # game_event.save_event(
            #     directory=f"./data/events/match_{self.match_id}"


        # create dir of path_file_result
        os.makedirs(os.path.dirname(self.path_file_result), exist_ok=True)
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

    game_data_preprocessing = GameDataPreprocessing(
        path_file_sportradar, path_file_kinexon
    )
    game_data_preprocessing.clean_game_data()
    game_data_preprocessing.create_kinexon_snippets_from_events()
    game_data_preprocessing.process_game_events()
    # game_data_preprocessing.extract_game_data_to_events()
