"""Functions to fetch and process match events from Sportradar API for given fixture IDs."""

import logging
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from fetcher_sportradar.fetch_players import get_players_for_fixture


def fetch_events_for_fixture(api, fixture_id: str) -> list:
    """
    Fetch match events for a single fixture ID.
    """
    logging.debug("Fetching events for fixture ID %s", fixture_id)
    match_events = api.get_fixture_events_by_id(
        fixture_id, setup_only=False, with_scores=True
    )
    if not match_events:
        logging.warning("No events found for fixture ID %s", fixture_id)
        return []

    # Flatten nested dicts under 'data' and 'options'
    for event in match_events:
        if "data" in event and event["data"]:
            event.update(event["data"])
            del event["data"]
        if "options" in event and event["options"]:
            event.update(event["options"])
            del event["options"]
    return match_events


def process_fixture_events(
    fixture_id: str,
    match_events: list,
    df_teams: pd.DataFrame,
    api,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process match events for a fixture to extract event and player data.
    Returns (df_match_events, df_players_for_fixture).
    """
    columns_to_keep_event_map = {
        "class": "class",
        "eventId": "event_id",
        "eventTime": "event_time",
        "eventType": "event_type",
        "subType": "sub_type",
        "attendance": "attendance",
        "entityId": "entity_id",
        "personId": "person_id",
        "bib": "bib",
        "name": "name",
        "position": "position",
        "scores": "scores",
        "periodId": "period_id",
        "playId": "play_id",
        "clock": "clock",
        "success": "success",
        "x": "x",
        "y": "y",
        "attackType": "attack_type",
        "goalKeeperId": "goalkeeper_id",
        "location": "location",
        "failureReason": "failure_reason",
        "emptyNet": "empty_net",
        "session_id": "session_id",
    }
    # Check if example key still camelCase exist in match_events
    if "entityId" not in match_events[0]:
        logging.warning(
            "Expected key 'entityId' not found in match events for fixture %s, but might also be expected in fixture_players_sportradar.",
            fixture_id,
        )
        df_fixture_events = pd.DataFrame(match_events)
    else:
        df_fixture_events = pd.DataFrame(
            match_events, columns=list(columns_to_keep_event_map.keys())
        ).rename(columns=columns_to_keep_event_map)
        df_fixture_events["fixture_id"] = fixture_id

    # Add team_name from df_teams
    if (
        "entity_id" in df_fixture_events.columns
        and "name_full_local" in df_teams.columns
    ):
        # make both columns same type for merging: str
        df_fixture_events["entity_id"] = df_fixture_events["entity_id"].astype(str)
        df_teams["entity_id"] = df_teams["entity_id"].astype(str)
        df_fixture_events = df_fixture_events.merge(
            df_teams[["entity_id", "name_full_local"]],
            on="entity_id",
            how="left",
        ).rename(columns={"name_full_local": "team_name"})
    else:
        df_fixture_events["team_name"] = None

    # Get players for this fixture via players module
    df_players = get_players_for_fixture(api, df_fixture_events)

    # Merge personName
    if (
        not df_players.empty
        and "name_full_local" in df_players.columns
        and "person_id" in df_players.columns
    ):
        df_fixture_events = df_fixture_events.merge(
            df_players[["name_full_local", "person_id"]],
            on="person_id",
            how="left",
        ).rename(columns={"name_full_local": "person_name"})

        # Merge goalkeeper name (join on goalKeeperId -> personId)
        df_fixture_events = df_fixture_events.merge(
            df_players[["name_full_local", "person_id"]],
            left_on="goalkeeper_id",
            right_on="person_id",
            how="left",
            suffixes=("", "_goalkeeper"),
        ).rename(columns={"name_full_local": "goalkeeper_name"})
        # drop helper column if created
        if "person_id_goalkeeper" in df_fixture_events.columns:
            df_fixture_events = df_fixture_events.drop(columns=["person_id_goalkeeper"])
    else:
        df_fixture_events["person_name"] = None
        df_fixture_events["goalkeeper_name"] = None
    # make scores more accessible, compare the key with entityId_home and entityId_away
    # if "scores" in df_fixture_events.columns:
    #     valid_ids = df_fixture_events["entity_id"].notna()
    #     unique_entity_ids = df_fixture_events.loc[
    #         valid_ids, "entity_id"
    #     ].unique()
    #     # remove 'nan' manually
    #     unique_entity_ids = [uid for uid in unique_entity_ids if pd.notna(uid)]
    #     # still nan check
    #     unique_entity_ids = [uid for uid in unique_entity_ids if uid != "nan"]
    #     if len(unique_entity_ids) == 2:
    #         for side in unique_entity_ids:
    #             score_col = f"scores_{side}"
    #             df_fixture_events[score_col] = df_fixture_events[
    #                 "scores"
    #             ].apply(lambda x: x.get(side) if isinstance(x, dict) else None)
    #         #

    # # drop original scores column
    # if "scores" in df_fixture_events.columns:
    #     df_fixture_events = df_fixture_events.drop(columns=["scores"])

    return df_fixture_events, df_players


def fetch_and_process_fixture_events(
    api,
    fixture_ids: List[str],
    df_teams: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches and processes events and players for a list of fixture IDs.
    Returns (df_all_match_events, df_all_players).
    """
    all_match_events = []
    all_players = []

    logging.info("Fetching events for %d fixtures.", len(fixture_ids))
    for fixture_id in tqdm(fixture_ids, desc="Processing fixtures"):
        match_events_raw = fetch_events_for_fixture(api, fixture_id)
        if not match_events_raw:
            continue

        df_fixture_events, df_players = process_fixture_events(
            fixture_id, match_events_raw, df_teams, api
        )

        all_match_events.append(df_fixture_events)
        if not df_players.empty:
            all_players.append(df_players)

    if not all_match_events:
        return pd.DataFrame(), pd.DataFrame()

    df_all_fixture_events = pd.concat(all_match_events, ignore_index=True)
    df_all_players = pd.concat(all_players, ignore_index=True).drop_duplicates(
        subset=["personId"]
    )

    return df_all_fixture_events, df_all_players


def fetch_list_fixture_events_multithreaded(
    api,
    fixture_ids: List[str],
    df_teams: pd.DataFrame,
    max_workers: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches and processes events and players for a list of fixture IDs
    using multithreading.
    Returns (df_all_match_events, df_all_players).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_match_events = []
    all_players = []

    logging.info(
        "Fetching events for %d fixtures using %d threads.",
        len(fixture_ids),
        max_workers,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_fixture = {
            executor.submit(fetch_events_for_fixture, api, fixture_id): fixture_id
            for fixture_id in fixture_ids
        }

        for future in tqdm(
            as_completed(future_to_fixture),
            total=len(future_to_fixture),
            desc="Processing fixtures",
        ):
            fixture_id = future_to_fixture[future]
            try:
                match_events_raw = future.result()
                if not match_events_raw:
                    continue

                df_fixture_events, df_players = process_fixture_events(
                    fixture_id, match_events_raw, df_teams, api
                )

                all_match_events.append(df_fixture_events)
                if not df_players.empty:
                    all_players.append(df_players)

            except Exception as e:
                logging.error("Error processing fixture ID %s: %s", fixture_id, e)

    if not all_match_events:
        return pd.DataFrame(), pd.DataFrame()

    df_all_fixture_events = pd.concat(all_match_events, ignore_index=True)
    df_all_players = pd.concat(all_players, ignore_index=True).drop_duplicates(
        subset=["personId"]
    )

    return df_all_fixture_events, df_all_players


def fetch_and_process_fixture_events_multithreaded(
    api,
    fixture_ids: List[str],
    df_teams: pd.DataFrame,
    max_workers: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches and processes events and players for a list of fixture IDs using
    multithreading where each worker both fetches and processes a single fixture.

    This differs from `fetch_list_fixture_events_multithreaded` by running the
    full work pipeline (fetch -> process) inside worker threads which can reduce
    round-trip overhead when I/O-bound.

    Note: if `api` is not thread-safe, consider protecting calls with a lock or
    using max_workers=1.
    Returns (df_all_match_events, df_all_players).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_match_events = []
    all_players = []

    logging.info(
        "Fetching+processing events for %d fixtures using %d threads.",
        len(fixture_ids),
        max_workers,
    )

    def _fetch_and_process(fixture_id: str):
        try:
            match_events_raw = fetch_events_for_fixture(api, fixture_id)
            if not match_events_raw:
                return None
            return process_fixture_events(fixture_id, match_events_raw, df_teams, api)
        except Exception as e:
            logging.exception("Worker failed for fixture %s: %s", fixture_id, e)
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_fixture = {
            executor.submit(_fetch_and_process, fixture_id): fixture_id
            for fixture_id in fixture_ids
        }

        for future in tqdm(
            as_completed(future_to_fixture),
            total=len(future_to_fixture),
            desc="Processing fixtures",
        ):
            fixture_id = future_to_fixture[future]
            try:
                result = future.result()
                if not result:
                    continue

                df_fixture_events, df_players = result
                all_match_events.append(df_fixture_events)
                if not df_players.empty:
                    all_players.append(df_players)

            except Exception as e:
                logging.error(
                    "Unhandled error while processing fixture ID %s: %s",
                    fixture_id,
                    e,
                )

    if not all_match_events:
        return pd.DataFrame(), pd.DataFrame()

    df_all_fixture_events = pd.concat(all_match_events, ignore_index=True)
    df_all_players = pd.concat(all_players, ignore_index=True).drop_duplicates(
        subset=["personId"]
    )

    return df_all_fixture_events, df_all_players
