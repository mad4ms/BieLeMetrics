"""Functions to fetch and process match events from Sportradar API for given fixture IDs."""

import logging
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm

from src.events_sportradar.fetch_players_from_fixture_events import (
    get_players_for_fixture,
)


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
    columns_to_keep_event_list = [
        "class",
        "eventId",
        "eventTime",
        "eventType",
        "subType",
        "attendance",
        "entityId",
        "personId",
        "bib",
        "name",
        "position",
        "scores",
        "periodId",
        "playId",
        "clock",
        "success",
        "x",
        "y",
        "attackType",
        "goalKeeperId",
        "location",
        "failureReason",
        "emptyNet",
    ]
    df_match_events = pd.DataFrame(
        match_events, columns=columns_to_keep_event_list
    )
    df_match_events["fixtureId"] = fixture_id

    # Add teamName from df_teams
    if (
        "entityId" in df_match_events.columns
        and "nameFullLocal" in df_teams.columns
    ):
        df_match_events = df_match_events.merge(
            df_teams[["entityId", "nameFullLocal"]], on="entityId", how="left"
        ).rename(columns={"nameFullLocal": "teamName"})
    else:
        df_match_events["teamName"] = None

    # Get players for this fixture via players module
    df_players = get_players_for_fixture(api, df_match_events)

    # Merge personName
    if (
        not df_players.empty
        and "nameFullLocal" in df_players.columns
        and "personId" in df_players.columns
    ):
        df_match_events = df_match_events.merge(
            df_players[["nameFullLocal", "personId"]],
            on="personId",
            how="left",
        ).rename(columns={"nameFullLocal": "personName"})

        # Merge goalkeeper name (join on goalKeeperId -> personId)
        df_match_events = df_match_events.merge(
            df_players[["nameFullLocal", "personId"]],
            left_on="goalKeeperId",
            right_on="personId",
            how="left",
            suffixes=("", "_goalkeeper"),
        ).rename(columns={"nameFullLocal": "goalkeeperName"})

        # drop helper column if created
        if "personId_goalkeeper" in df_match_events.columns:
            df_match_events = df_match_events.drop(
                columns=["personId_goalkeeper"]
            )
    else:
        df_match_events["personName"] = None
        df_match_events["goalkeeperName"] = None

    return df_match_events, df_players


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

        df_match_events, df_players = process_fixture_events(
            fixture_id, match_events_raw, df_teams, api
        )

        all_match_events.append(df_match_events)
        if not df_players.empty:
            all_players.append(df_players)

    if not all_match_events:
        return pd.DataFrame(), pd.DataFrame()

    df_all_match_events = pd.concat(all_match_events, ignore_index=True)
    df_all_players = pd.concat(all_players, ignore_index=True).drop_duplicates(
        subset=["personId"]
    )

    return df_all_match_events, df_all_players


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
            executor.submit(
                fetch_events_for_fixture, api, fixture_id
            ): fixture_id
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

                df_match_events, df_players = process_fixture_events(
                    fixture_id, match_events_raw, df_teams, api
                )

                all_match_events.append(df_match_events)
                if not df_players.empty:
                    all_players.append(df_players)

            except Exception as e:
                logging.error(
                    "Error processing fixture ID %s: %s", fixture_id, e
                )

    if not all_match_events:
        return pd.DataFrame(), pd.DataFrame()

    df_all_match_events = pd.concat(all_match_events, ignore_index=True)
    df_all_players = pd.concat(all_players, ignore_index=True).drop_duplicates(
        subset=["personId"]
    )

    return df_all_match_events, df_all_players


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
            return process_fixture_events(
                fixture_id, match_events_raw, df_teams, api
            )
        except Exception as e:
            logging.exception(
                "Worker failed for fixture %s: %s", fixture_id, e
            )
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

                df_match_events, df_players = result
                all_match_events.append(df_match_events)
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

    df_all_match_events = pd.concat(all_match_events, ignore_index=True)
    df_all_players = pd.concat(all_players, ignore_index=True).drop_duplicates(
        subset=["personId"]
    )

    return df_all_match_events, df_all_players
