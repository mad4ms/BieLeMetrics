import gzip
import io
import logging
import zipfile
from typing import Any, Dict, List

import pandas as pd
from kinexon_handball_api.handball import HandballAPI


def _is_gzip(b: bytes) -> bool:
    """Check if the byte sequence represents a gzip file."""
    return len(b) >= 2 and b[0] == 0x1F and b[1] == 0x8B


def _is_zip(b: bytes) -> bool:
    """Check if the byte sequence represents a zip file."""
    return len(b) >= 4 and b[:4] == b"PK\x03\x04"


def read_positions_payload(payload: bytes) -> pd.DataFrame:
    """
    Accepts raw bytes from Kinexon API. Detects gzip/zip and returns DataFrame.
    CSV is expected to be UTF-8 with semicolon separator.
    """
    raw = payload
    if _is_gzip(raw):
        raw = gzip.decompress(raw)
        csv_bytes = raw
    elif _is_zip(raw):
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
            if not name:
                raise ValueError("ZIP archive does not contain a CSV file.")
            csv_bytes = zf.read(name)
    else:
        csv_bytes = raw

    return pd.read_csv(io.StringIO(csv_bytes.decode("utf-8-sig")), sep=";")


def fetch_positions_for_fixture(api: HandballAPI, session_id: str) -> pd.DataFrame:
    """
    Fetch player positions for a specific fixture.

    Args:
        api (HandballAPI): The Handball API client.
        fixture_id (str): The ID of the fixture.

    Returns:
        pd.DataFrame: A DataFrame containing player positions.
    """
    try:
        resp = api.download_positions_csv_via_custom(
            session_id=session_id, compress_output=True
        )
        payload = (
            resp
            if isinstance(resp, (bytes, bytearray))
            else getattr(resp, "content", None)
        )
        df = read_positions_payload(payload)
        # attach session_id to positions dataframe
        df["session_id"] = session_id

        return df
    except Exception as e:
        logging.error(
            "Error fetching positions for session_id %s: %s",
            session_id,
            str(e),
        )
        return pd.DataFrame()


def fetch_positions_for_fixtures_multithreaded(
    api: HandballAPI,
    session_ids: List[str],
    max_workers: int = 9,
) -> pd.DataFrame:
    """
    Fetch player positions for multiple fixtures using multithreading.

    Args:
        api (HandballAPI): The Handball API client.
        session_ids (List[str]): A list of session IDs.
        max_workers (int): Number of threads for parallel processing.

    Returns:
        pd.DataFrame: A DataFrame containing player positions for all fixtures.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tqdm import tqdm

    all_positions = []

    logging.info(
        "Fetching positions for %d sessions using %d threads.",
        len(session_ids),
        max_workers,
    )

    def _fetch_positions(session_id: str):
        return fetch_positions_for_fixture(api, session_id)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_session = {
            executor.submit(_fetch_positions, session_id): session_id
            for session_id in session_ids
        }

        for future in tqdm(
            as_completed(future_to_session),
            total=len(future_to_session),
            desc="Processing sessions",
        ):
            session_id = future_to_session[future]
            try:
                df_positions = future.result()
                if not df_positions.empty:
                    all_positions.append(df_positions)
            except Exception as e:
                logging.error(
                    "Unhandled error while processing session ID %s: %s",
                    session_id,
                    e,
                )

    if not all_positions:
        return pd.DataFrame()

    df_all_positions = pd.concat(all_positions, ignore_index=True)
    return df_all_positions
