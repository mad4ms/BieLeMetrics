import pandas as pd
from typing import Dict


def initialize_event_metadata(dict_event: Dict) -> Dict:
    """
    Initializes core event metadata from the provided event dictionary.

    Args:
        dict_event (Dict): Dictionary containing event data from Sportradar.

    Returns:
        dict: Dictionary with initialized metadata fields.
    """
    event_time_tagged = pd.to_datetime(dict_event.get("time")).replace(
        tzinfo=None
    ) + pd.to_timedelta(2, unit="h")
    event_time_start = event_time_tagged - pd.Timedelta(seconds=15)

    return {
        "event_id": dict_event.get("id"),
        "event_type": dict_event.get("type"),
        "event_time_tagged": event_time_tagged,
        "event_time_start": event_time_start,
        "match_time": dict_event.get("match_time"),
        "match_clock": dict_event.get("match_clock"),
        "match_clock_in_s": dict_event.get("match_clock_in_s", None),
        "competitor": dict_event.get("competitor"),
        "home_score": dict_event.get("home_score"),
        "away_score": dict_event.get("away_score"),
        "scorer": dict_event.get("scorer"),
        "assists": dict_event.get("assists", []),
        "zone": dict_event.get("zone"),
        "player": (
            dict_event.get("scorer")
            if dict_event.get("type") == "score_change"
            else dict_event.get("player")
        ),
        "players": dict_event.get("players", []),
        "shot_type": dict_event.get("shot_type"),
        "method": dict_event.get("method"),
        "outcome": dict_event.get("outcome"),
        "suspension_minutes": dict_event.get("suspension_minutes"),
        "attack_direction": dict_event.get("attack_direction"),
        "id_goalkeeper": dict_event.get("id_goalkeeper"),
    }


def initialize_position_placeholders() -> Dict:
    """
    Initializes placeholders for various entities (player, ball, goalkeeper, etc.) involved in the event.

    Returns:
        dict: Dictionary with placeholders for positional information.
    """
    return {
        # Player
        "name_player": None,
        "id_player": None,
        "pos_x_player": None,
        "pos_y_player": None,
        # Ball
        "id_ball": None,
        "pos_x_ball": None,
        "pos_y_ball": None,
        # Blocker
        "name_blocker": None,
        "id_blocker": None,
        "pos_x_blocker": None,
        "pos_y_blocker": None,
        # Goalkeeper
        "name_goalkeeper": None,
        "pos_x_goalkeeper": None,
        "pos_y_goalkeeper": None,
        # Assists
        "name_assist": None,
        "id_assist": None,
        "pos_x_assist": None,
        "pos_y_assist": None,
        # Additional placeholders
        "event_time_throw": None,
        "peaks": None,
        "data_kinexon_event_player_ball": None,
    }
