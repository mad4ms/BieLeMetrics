import os
import pandas as pd
import glob
import json
import numpy as np

def add_missing_goalkeeper_to_events(dict_event_id_kinexon_path: dict) -> None:
    """
    Add missing goalkeeper to the events in the dictionary.
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

    for event_id, dict_event in dict_event_id_kinexon_path.items():
        if dict_event["type"] not in relevant_event_types:
            dict_event["id_goalkeeper"] = None
            continue
        if "players" not in dict_event:
            dict_event["id_goalkeeper"] = None
            list_events_without_goalkeeper.append(dict_event)
            continue

        for player in dict_event["players"]:
            if player["type"] in ["goalkeeper", "saved"]:
                dict_event["id_goalkeeper"] = player["id"].split(":")[-1]
                if dict_event["competitor"] == "home":
                    if player["id"] not in list_goalkeeper_away_ids:
                        list_goalkeeper_away_ids.append(player["id"])
                else:
                    if player["id"] not in list_goalkeeper_home_ids:
                        list_goalkeeper_home_ids.append(player["id"])
            else:
                list_events_without_goalkeeper.append(dict_event)

    # Now, try to fill in missing goalkeepers
    for dict_event in list_events_without_goalkeeper:
        event_id = dict_event["id"]
        kin_event_for_sportradar_event = glob.glob(
            os.path.join(dict_event["path_kinexon"])
        )

        if len(kin_event_for_sportradar_event) == 0:
            print(f"! File for event {event_id} not found.")
            continue

        df_kinexon_event = pd.read_csv(kin_event_for_sportradar_event[0])

        if dict_event["competitor"] == "home":
            # Check goalkeeper in away team
            for player_id in list_goalkeeper_away_ids:
                df_goalkeeper = df_kinexon_event[
                    df_kinexon_event["league_id"] == player_id.split(":")[-1]
                ]
                if not df_goalkeeper.empty:
                    break
        else:
            # Check goalkeeper in home team
            for player_id in list_goalkeeper_home_ids:
                df_goalkeeper = df_kinexon_event[
                    df_kinexon_event["league_id"] == player_id.split(":")[-1]
                ]
                if not df_goalkeeper.empty:
                    break

        if not df_goalkeeper.empty:
            dict_event["id_goalkeeper"] = df_goalkeeper["league_id"].values[0]
            dict_event["name_goalkeeper"] = df_goalkeeper["full_name"].values[
                0
            ]


def add_scores_to_events(dict_event_id_kinexon_path: dict) -> None:
    """
    Add scores to the events in the timeline.
    """
    home_score = 0
    away_score = 0

    for event_id, dict_event in dict_event_id_kinexon_path.items():
        if "home_score" in dict_event:
            home_score = dict_event.get("home_score", home_score)
            away_score = dict_event.get("away_score", away_score)

        dict_event["home_score"] = home_score
        dict_event["away_score"] = away_score


def calc_attack_direction(dict_event_id_kinexon_path: dict) -> None:
    """
    Calculate the attack direction based on the goalkeeper's position.
    """
    list_goalkeeper_home_ids = []
    list_goalkeeper_positions_home = []

    # First, determine the goalkeeper IDs for the home team
    for event_id, dict_event in dict_event_id_kinexon_path.items():
        if (
            "id_goalkeeper" not in dict_event
            or "competitor" not in dict_event
            or not "match_time" in dict_event
        ):
            # remove events without goalkeeper or competitor from the list
            continue
        if (
            dict_event["competitor"] == "home"
            and dict_event["match_time"] <= 30
        ):
            if dict_event["id_goalkeeper"] not in list_goalkeeper_home_ids:
                # append the goalkeeper ID to the list
                list_goalkeeper_home_ids.append(dict_event["id_goalkeeper"])

    # Now iterate over all events and determine attack direction
    for event_id, dict_event in dict_event_id_kinexon_path.items():
        if not dict_event.get("id_goalkeeper"):
            continue

        kin_event_for_sportradar_event = glob.glob(
            os.path.join(dict_event["path_kinexon"])
        )

        if len(kin_event_for_sportradar_event) == 0:
            print(f"! File for event {event_id} not found.")
            continue

        df_kinexon_event = pd.read_csv(kin_event_for_sportradar_event[0])

        # convert df_kinexon_event["league_id"] to string
        df_kinexon_event["league_id"] = df_kinexon_event["league_id"].astype(
            str
        )

        df_goalkeeper = df_kinexon_event[
            df_kinexon_event["league_id"] == dict_event["id_goalkeeper"]
        ]

        if df_goalkeeper.empty:
            continue

        last_goalkeeper_position = df_goalkeeper.iloc[-1]
        if (
            dict_event["id_goalkeeper"] in list_goalkeeper_home_ids
            and dict_event["match_time"] <= 30
        ):
            list_goalkeeper_positions_home.append(
                last_goalkeeper_position["pos_x"]
            )

    if not list_goalkeeper_positions_home:
        return

    avg_pos_home_goalkeeper = sum(list_goalkeeper_positions_home) / len(
        list_goalkeeper_positions_home
    )
    attack_direction = "right" if avg_pos_home_goalkeeper < 20 else "left"

    for event_id, dict_event in dict_event_id_kinexon_path.items():
        if not "match_time" in dict_event or not "competitor" in dict_event:
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
                "left" if attack_direction == "right" else "right"
            )
        elif (
            dict_event["match_time"] > 30
            and dict_event["competitor"] == "home"
        ):
            dict_event["attack_direction"] = (
                "left" if attack_direction == "right" else "right"
            )
        elif (
            dict_event["match_time"] > 30
            and dict_event["competitor"] == "away"
        ):
            dict_event["attack_direction"] = attack_direction


def insert_names_competitors(
    dict_event_id_kinexon_path: dict, dict_sportradar: dict
) -> None:
    """
    Insert the names of the competitors into the events.
    """
    # name team home from dict_sportradar
    list_competitors = dict_sportradar["sport_event"]["competitors"]

    for competitor in list_competitors:
        if competitor["qualifier"] == "home":
            name_home = competitor["name"]
        else:
            name_away = competitor["name"]

    for event_id, dict_event in dict_event_id_kinexon_path.items():
        dict_event["name_team_home"] = name_home
        dict_event["name_team_away"] = name_away


def insert_player_ids(
    dict_event_id_kinexon_path: dict, dict_sportradar: dict
) -> None:
    """
    Insert the player IDs into the events.
    """
    for event_id, dict_event in dict_event_id_kinexon_path.items():
        if "players" in dict_event:
            for player in dict_event["players"]:
                if player["type"] == "shot":
                    dict_event["name_player"] = player["name"]
                    dict_event["id_player"] = player["id"]
                elif player["type"] == "blocked":
                    dict_event["name_blocker"] = player["name"]
                    dict_event["id_blocker"] = player["id"]
                elif player["type"] == "saved":
                    dict_event["name_goalkeeper"] = player["name"]
                    dict_event["id_goalkeeper"] = player["id"]
                elif player["type"] == "goalkeeper":
                    dict_event["name_goalkeeper"] = player["name"]
                    dict_event["id_goalkeeper"] = player["id"]

        if "scorer" in dict_event:
                dict_event["name_player"] = dict_event["scorer"]["name"]
                dict_event["id_player"] = dict_event["scorer"]["id"]


def calc_team_shot_efficiency(
    dict_event_id_kinexon_path: dict, dict_sportradar: dict
) -> None:
    """
    Calculate the cumulative team shot efficiency up until each event based on shots on target and goals.
    """

    relevant_event_types = [
        "score_change",
        "shot_off_target",
        "shot_blocked",
        "shot_saved",
        "seven_m_missed",
    ]

    # Convert the sportradar "timeline" data to a DataFrame
    df_sportradar = pd.DataFrame(dict_sportradar["timeline"])

    # Filter out irrelevant events
    df_sportradar = df_sportradar[df_sportradar["type"].isin(relevant_event_types)]

    # Add a cumulative count of shots for each team (competitor), excluding the current event
    df_sportradar["cumulative_shots"] = df_sportradar.groupby("competitor").cumcount()

    # Add cumulative goals, but shift the count by one to exclude the current event
    df_sportradar["cumulative_goals"] = df_sportradar.groupby("competitor")["type"].transform(
        lambda x: (x == "score_change").shift(1).cumsum()
    )

    # Calculate cumulative shot efficiency for each event (up to the previous event)
    df_sportradar["shot_efficiency"] = df_sportradar.apply(
        lambda row: row["cumulative_goals"] / row["cumulative_shots"]
        if row["cumulative_shots"] > 0 else 0, axis=1
    )

    # Now assign the efficiency to each event in dict_event_id_kinexon_path
    for event_id, dict_event in dict_event_id_kinexon_path.items():
        if "competitor" not in dict_event:
            continue
        competitor = dict_event["competitor"]

        # Find the row corresponding to this event for the same competitor
        df_competitor = df_sportradar[df_sportradar["competitor"] == competitor]

        # Get the shot efficiency up until this event (latest row)
        latest_event = df_competitor[df_competitor["id"] == event_id]

        # Assign shot efficiency to the current event in the dictionary
        if not latest_event.empty:
            dict_event["shot_efficiency"] = latest_event["shot_efficiency"].values[0]

            if not np.isnan(latest_event["cumulative_shots"].values[0]):
                dict_event["total_shots"] = int(latest_event["cumulative_shots"].values[0])
            else:
                dict_event["total_shots"] = None

            if not np.isnan(latest_event["cumulative_goals"].values[0]):
                dict_event["total_goals"] = int(latest_event["cumulative_goals"].values[0])
            else:
                dict_event["total_goals"] = None
        else:
            dict_event["shot_efficiency"] = None  # Handle cases where event_id is not found
            dict_event["total_shots"] = None
            dict_event["total_goals"] = None

        # Print the result for debugging or logging purposes
        print(
            f"Event ID: {event_id} - Type: {dict_event['type']} - Competitor {competitor} shot efficiency: {dict_event['shot_efficiency']} ({dict_event['total_goals']} of {dict_event['total_shots']} shots)."
        )




def calc_player_shot_efficiency(
    dict_event_id_kinexon_path: dict, dict_sportradar: dict
) -> None:
    """
    Calculate the cumulative shot efficiency for each player up until each event based on their shots on target and goals.
    """

    relevant_event_types = [
        "score_change",
        "shot_off_target",
        "shot_blocked",
        "shot_saved",
        "seven_m_missed",
    ]

    df_sportradar = pd.DataFrame(dict_event_id_kinexon_path.values())

    # Filter out irrelevant events
    df_sportradar = df_sportradar[
        df_sportradar["type"].isin(relevant_event_types)
    ]

    # print unique player names
    print(df_sportradar["name_player"].unique())

    # Make sure the DataFrame index is properly set
    df_sportradar = df_sportradar.reset_index(drop=True)

    # Add a cumulative count of shots and goals for each player
    # Add a cumulative count of shots and goals for each player
    df_sportradar["cumulative_shots_player"] = (
        df_sportradar.groupby("id_player").cumcount()
    )  # No "+1" to exclude the current event
    df_sportradar["cumulative_goals_player"] = df_sportradar.groupby("id_player")[
        "type"
    ].transform(lambda x: (x == "score_change").shift(1).cumsum())

    # Calculate shot efficiency using past data only, but prevent division by zero
    df_sportradar["shot_efficiency_player"] = df_sportradar["cumulative_shots_player"]

    df_sportradar["shot_efficiency_player"] = df_sportradar.apply(
        lambda row: row["cumulative_goals_player"] / row["cumulative_shots_player"]
        if row["cumulative_shots_player"] > 0 else 0, axis=1
    )


    # Now assign the efficiency to each event in dict_event_id_kinexon_path
    for event_id, dict_event in dict_event_id_kinexon_path.items():
        if "id_player" not in dict_event:
            continue
        player_id = dict_event["id_player"]
        player_name = dict_event["name_player"]

        # Find the row corresponding to this event for the same player
        df_player = df_sportradar[df_sportradar["id_player"] == player_id]

        # Get the shot efficiency up until this event (latest row)
        latest_event = df_player[df_player["id"] == event_id]

        # Assign shot efficiency to the current event in the dictionary
        if not latest_event.empty:
            dict_event["shot_efficiency_player"] = latest_event[
                "shot_efficiency_player"
            ].values[0]
            if not np.isnan(latest_event["cumulative_shots_player"].values[0]):
                # also total shots and goals
                dict_event["total_shots_players"] = int(latest_event[
                    "cumulative_shots_player"
                ].values[0])
            else:
                dict_event["total_shots_player"] = None

            if not np.isnan(latest_event["cumulative_goals_player"].values[0]):
                dict_event["total_goals_player"] = int(latest_event[
                    "cumulative_goals_player"
                ].values[0])
            else:
                dict_event["total_goals_player"] = None
        else:
            dict_event["shot_efficiency_player"] = (
                None  # Handle cases where event_id is not found
            )
            dict_event["total_shots_player"] = None
            dict_event["total_goals_player"] = None

        print(
            f"Event ID: {event_id} - Type: {dict_event["type"]} -  Player {player_id} ({player_name}) shot efficiency: {dict_event['shot_efficiency_player']}"
        )  # noqa

def calc_goalkeeper_efficiency(
    dict_event_id_kinexon_path: dict, dict_sportradar: dict
) -> None:
    """
    Calculate the cumulative goalkeeper efficiency for each goalkeeper up until each event based on shots saved and goals conceded.
    """

    relevant_event_types = [
        "score_change",
        "shot_off_target",
        "shot_blocked",
        "shot_saved",
        "seven_m_missed",
    ]

    df_sportradar = pd.DataFrame(dict_event_id_kinexon_path.values())

    # Filter out irrelevant events
    df_sportradar = df_sportradar[
        df_sportradar["type"].isin(relevant_event_types)
    ]

    # print unique goalkeeper names
    print(df_sportradar["name_goalkeeper"].unique())

    # Make sure the DataFrame index is properly set
    df_sportradar = df_sportradar.reset_index(drop=True)

    # Add a cumulative count of shots saved and goals conceded for each goalkeeper
    df_sportradar["cumulative_shots_saved"] = (
        df_sportradar.groupby("id_goalkeeper").cumcount()
    )  # No "+1" to exclude the current event
    df_sportradar["cumulative_goals_conceded"] = df_sportradar.groupby("id_goalkeeper")[
        "type"
    ].transform(lambda x: (x == "score_change").shift(1).cumsum())

    # Calculate goalkeeper efficiency using past data only
    df_sportradar["goalkeeper_efficiency"] = df_sportradar.apply(
        lambda row: row["cumulative_shots_saved"] / (row["cumulative_shots_saved"] + row["cumulative_goals_conceded"])
        if (row["cumulative_shots_saved"] + row["cumulative_goals_conceded"]) > 0 else 0, axis=1
    )


    # Now assign the efficiency to each event in dict_event_id_kinexon_path
    for event_id, dict_event in dict_event_id_kinexon_path.items():
        if "id_goalkeeper" not in dict_event or dict_event["id_goalkeeper"] is None:
            continue
        goalkeeper_id = dict_event["id_goalkeeper"]
        goalkeeper_name = dict_event["name_goalkeeper"]

        # Find the row corresponding to this event for the same goalkeeper
        df_goalkeeper = df_sportradar[
            df_sportradar["id_goalkeeper"] == goalkeeper_id
        ]

        # Get the goalkeeper efficiency up until this event (latest row)
        latest_event = df_goalkeeper[df_goalkeeper["id"] == event_id]

        # Assign goalkeeper efficiency to the current event in the dictionary
        if not latest_event.empty:
            dict_event["goalkeeper_efficiency"] = latest_event[
                "goalkeeper_efficiency"
            ].values[0]
            if not np.isnan(latest_event["cumulative_shots_saved"].values[0]):
                # also total shots saved and goals conceded
                dict_event["total_shots_saved"] = int(latest_event[
                    "cumulative_shots_saved"
                ].values[0])
            else:
                dict_event["total_shots_saved"] = None

            if not np.isnan(latest_event["cumulative_goals_conceded"].values[0]):
                dict_event["total_goals_conceded"] = int(latest_event[
                    "cumulative_goals_conceded"
                ].values[0])
            else:
                dict_event["total_goals_conceded"] = None

        else:
            dict_event["goalkeeper_efficiency"] = (
                None  # Handle cases where event_id is not found
            )
            dict_event["total_shots_saved"] = None
            dict_event["total_goals_conceded"] = None

        print(
            f"Event ID: {event_id} - Type: {dict_event["type"]} -  Goalkeeper {goalkeeper_id} ({goalkeeper_name}) efficiency: {round(dict_event['goalkeeper_efficiency'],2)} with {dict_event['total_shots_saved']} shots saved and {dict_event['total_goals_conceded']} goals conceded."
        )
