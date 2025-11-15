"""
Helper functions to fetch player data from Sportradar
API based on match-events DataFrame.
"""

import logging
from typing import List
import pandas as pd


def extract_person_ids_from_setup(df_match_events: pd.DataFrame) -> List[str]:
    """
    Return unique personIds from setup/person events in a match-events DataFrame.
    """
    df_setup = df_match_events[
        (df_match_events.get("class") == "setup")
        & (df_match_events.get("eventType") == "person")
    ]
    return df_setup["personId"].dropna().unique().tolist()


def fetch_players_by_ids(api, person_ids: List[str]) -> pd.DataFrame:
    """
    Fetch player objects from the API for a list of personIds and return a DataFrame.
    """
    if not person_ids:
        logging.debug("No person IDs provided to fetch_players_by_ids()")
        return pd.DataFrame()

    str_person_ids = ",".join(person_ids)
    players = api.get_players_by_ids(person_ids=str_person_ids)
    if not players:
        logging.warning("API returned no players for ids: %s", str_person_ids)
        return pd.DataFrame()
    try:
        df_players = pd.json_normalize([p.to_dict() for p in players])
    except AttributeError:
        df_players = pd.json_normalize([p for p in players])
    except Exception as e:
        logging.error("Error normalizing player data: %s", e)
        return pd.DataFrame()
    return df_players


def enrich_and_filter_players(
    df_players: pd.DataFrame, df_setup_events: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge team info from setup events into the players DataFrame and keep a
    reduced set of player columns.
    """
    if df_players.empty:
        return pd.DataFrame()

    people_map = (
        df_setup_events[["personId", "entityId", "teamName"]]
        .dropna(subset=["personId"])
        .drop_duplicates(subset=["personId"])
    )

    # validate one-to-one where possible but do not raise in production flow
    try:
        df_players = df_players.merge(
            people_map, on="personId", how="left", validate="one_to_one"
        )
    except Exception:
        # fallback to left merge without validation if shapes don't match
        df_players = df_players.merge(people_map, on="personId", how="left")

    columns_to_keep_player_list = [
        "dob",
        "externalId",
        "images",
        "nameFamilyLatin",
        "nameFamilyLocal",
        "nameFullLatin",
        "nameFullLocal",
        "nameGivenLatin",
        "nameGivenLocal",
        "nationality",
        "personId",
        "additionalDetails.height",
        "additionalDetails.weight",
        "teamName",
        "entityId",
        "league_id",
    ]
    cols = [c for c in columns_to_keep_player_list if c in df_players.columns]
    return df_players[cols].copy()


def get_players_for_fixture(
    api, df_match_events: pd.DataFrame
) -> pd.DataFrame:
    """
    High-level helper: given raw match-events DataFrame, return filtered player DataFrame.
    """
    person_ids = extract_person_ids_from_setup(df_match_events)
    if not person_ids:
        return pd.DataFrame()

    df_players = fetch_players_by_ids(api, person_ids)
    if df_players.empty:
        return pd.DataFrame()

    df_players_filtered = enrich_and_filter_players(
        df_players, df_match_events
    )
    return df_players_filtered
