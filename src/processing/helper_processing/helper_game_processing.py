import os
import pandas as pd
import glob


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
        if "id_goalkeeper" not in dict_event or "competitor" not in dict_event:
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
