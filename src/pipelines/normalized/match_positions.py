import logging

import pandas as pd


def normalize_match_positions(
    df_positions_kinexon_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize match positions from Kinexon data source.

    Args:
        df_positions_kinexon_raw (pd.DataFrame): DataFrame containing raw positions from Kinexon.
    Returns:
        pd.DataFrame: Normalized match positions DataFrame.
    """
    df_positions = df_positions_kinexon_raw.copy()

    dict_cols_to_keep = {
        "ts in ms": "timestamp_ms",
        "formatted local time": "formatted_local_time",
        "sensor id": "sensor_id",
        "mapped id": "mapped_id",
        "full name": "full_name",
        "league id": "league_id",
        "group id": "group_id",
        "group name": "group_name",
        "x in m": "x_m",
        "y in m": "y_m",
        # "z in m": "z_m", # still not available in Kinexon data
        "speed in m/s": "speed_m_s",
        "direction of movement in deg": "direction",
        "acceleration in m/s2": "acceleration",
        "total distance in m": "total_distance",
        "metabolic power in W/kg": "metabolic_power",
        "acceleration load": "acceleration_load",
    }
    df_positions = df_positions.rename(columns=dict_cols_to_keep)
    # drop unnamed_17
    df_positions = df_positions.drop(columns=["Unnamed: 17"], errors="ignore")

    logging.info("Normalized %d positions", len(df_positions))

    return df_positions
