import json
import logging
import os
from typing import Dict, Optional, Tuple, List

from dotenv import load_dotenv
import pandas as pd
import difflib

PLAYER_COLS = ["bib", "name", "position"]


def normalize_match_events(
    df_fixture_events_sportradar_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize match events from Sportradar and Kinexon data sources.

    Args:
        df_fixture_events_sportradar_raw (pd.DataFrame): DataFrame containing fixture events from Sportradar.
    Returns:
        pd.DataFrame: Normalized match events DataFrame.
    """

    # columns in normalized_matches: "season_id", "fixture_id", "session_id", "start_time_local", "start_time_utc", "start_time", "round_number", "team_name_home", "team_name_away", "entity_id_home", "entity_id_away", "score_home", "score_away", "start_session", "description", "match_name", "attendance", "team_id_home_kinexon", "team_id_away_kinexon"

    # columns in df_match_events_sportradar_raw:
    cols_to_keep = [
        # "clientId",
        # "clientType",
        "fixture_id",
        # "organizationId",
        # "received",
        # "sport",
        # "topic",
        # "type",
        "class",
        "entityId",
        "eventId",
        "eventTime",
        "eventType",
        "personId",
        # "status",
        # "timestamp",
        "bib",
        # "captain",
        "name",
        "position",
        # "starter",
        # "active",
        "subType",
        "number",
        "periodId",
        # "playId",
        "scores",
        "sequence",
        "clock",
        "options",
        "success",
        "goalKeeperId",
        "x",
        "y",
        "attackType",
        "failureReason",
        "location",
        "value",
        "emptyNet",
    ]
    df_match_events_sportradar = df_fixture_events_sportradar_raw[
        cols_to_keep
    ].copy()
    df_match_events_sportradar = df_match_events_sportradar.rename(
        columns={
            "entityId": "entity_id",
            "eventId": "event_id",
            "eventTime": "event_time",
            "eventType": "event_type",
            "personId": "person_id",
            "periodId": "period_id",
            "goalKeeperId": "goalkeeper_id",
            "subType": "sub_type",
            "attackType": "attack_type",
            "failureReason": "failure_reason",
            "emptyNet": "empty_net",
        }
    )
    # sort by event_time
    df_match_events_sportradar = df_match_events_sportradar.sort_values(
        "event_time"
    )

    # Only keep player info on person rows
    mask_person = df_match_events_sportradar["event_type"] == "person"
    df_match_events_sportradar.loc[~mask_person, PLAYER_COLS] = pd.NA

    # Group-wise forward fill by person
    df_match_events_sportradar[PLAYER_COLS] = (
        df_match_events_sportradar.groupby("person_id", sort=False)[
            PLAYER_COLS
        ].ffill()
    )
    return df_match_events_sportradar


def normalize_match_events_setup(
    df_match_events_normalized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Additional normalization for match events, specifically for 'setup' events.

    Args:
        df_match_events_normalized (pd.DataFrame): DataFrame containing normalized match events.
    Returns:
        pd.DataFrame: Further normalized match events DataFrame.
    """
    df_setup_events = df_match_events_normalized[
        df_match_events_normalized["class"].isin(["setup", "clock"])
    ].copy()

    # Drop number	period_id	scores	sequence	clock	options	success	goalkeeper_id	x	y	attack_type	failureReason	location	value	emptyNet
    cols_to_drop = [
        "number",
        "period_id",
        "scores",
        "sequence",
        "clock",
        "options",
        "success",
        "goalkeeper_id",
        "x",
        "y",
        "attack_type",
        "failure_reason",
        "location",
        "value",
        "empty_net",
    ]

    df_setup_events = df_setup_events.drop(columns=cols_to_drop)

    return df_setup_events


def normalize_match_events_goals(
    df_match_events_normalized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Additional normalization for match events, specifically for 'goals' events.
    Args:
        df_match_events_normalized (pd.DataFrame): DataFrame containing normalized match events.
    Returns:
        pd.DataFrame: Further normalized match events DataFrame.
    """
    df_goals_events = df_match_events_normalized[
        df_match_events_normalized["event_type"].isin(["goal"])
    ].copy()
    df_goals_events = df_goals_events.drop(columns=["options", "number"])

    return df_goals_events
