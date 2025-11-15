from typing import Any, Dict, List
import logging

import pandas as pd
from kinexon_handball_api.handball import HandballAPI


def fetch_detected_events_for_session(
    api: HandballAPI,
    session_id: int,
) -> pd.DataFrame:
    """
    Fetch detected events for a single Kinexon session.

    Args:
        api (HandballAPI): Kinexon Handball API client.
        session_id (int): Session ID.

    Returns:
        pd.DataFrame: DataFrame of events with session_id attached.
    """
    try:
        events = api.get_events_for_session(session_id=session_id) or []
        events_dict = [e.to_dict() for e in events]
        df = pd.DataFrame(events_dict)
        if not df.empty:
            df["session_id"] = session_id
        return df
    except Exception as e:
        logging.error(
            "Error fetching detected events for session_id %s: %s",
            session_id,
            str(e),
        )
        return pd.DataFrame()


def fetch_detected_events_for_sessions_multithreaded(
    api: HandballAPI,
    session_ids: List[int],
    max_workers: int = 9,
) -> pd.DataFrame:
    """
    Fetch detected events for multiple Kinexon sessions using multithreading.

    Args:
        api (HandballAPI): Kinexon Handball API client.
        session_ids (List[int]): List of session IDs.
        max_workers (int): Number of threads for parallel processing.

    Returns:
        pd.DataFrame: Concatenated DataFrame of events for all sessions.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    all_events: List[pd.DataFrame] = []

    logging.info(
        "Fetching detected events for %d sessions using %d threads.",
        len(session_ids),
        max_workers,
    )

    def _fetch(session_id: int) -> pd.DataFrame:
        return fetch_detected_events_for_session(api, session_id)

    if not session_ids:
        logging.info(
            "No session_ids provided to fetch_detected_events_for_sessions_multithreaded."
        )
        return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_session = {
            executor.submit(_fetch, session_id): session_id
            for session_id in session_ids
        }

        for future in tqdm(
            as_completed(future_to_session),
            total=len(future_to_session),
            desc="Fetching detected events",
        ):
            session_id = future_to_session[future]
            try:
                df_events = future.result()
                if not df_events.empty:
                    all_events.append(df_events)
            except Exception as e:
                logging.error(
                    "Unhandled error while processing session_id %s: %s",
                    session_id,
                    str(e),
                )

    if not all_events:
        logging.info("No events fetched for any provided session_ids.")
        return pd.DataFrame()

    df_all_events = pd.concat(all_events, ignore_index=True)
    return df_all_events
