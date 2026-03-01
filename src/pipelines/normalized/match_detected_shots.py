import logging

import pandas as pd


def normalize_match_detected_shots(
    detected_events_kinexon_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize match detected shots from normalized match events.

    Args:
        detected_events_kinexon_raw (pd.DataFrame): DataFrame containing raw detected events from Kinexon.
    Returns:
        pd.DataFrame: Normalized match detected shots DataFrame.
    """
    df_detected_shots = detected_events_kinexon_raw.copy()

    # remove columns that are not needed
    cols_to_drop = [
        "timezone_id",
    ]
    df_detected_shots = df_detected_shots.drop(columns=cols_to_drop, errors="ignore")

    # remove rows where league_id contains "ball" or "Ball"
    df_detected_shots = df_detected_shots[
        ~df_detected_shots["league_id"].str.contains("ball", case=False, na=False)
    ]

    # set to None where validated is 0
    df_detected_shots.loc[df_detected_shots["validated"] == 0, "validated"] = None

    logging.info("Normalized %d detected shots", len(df_detected_shots))

    return df_detected_shots
