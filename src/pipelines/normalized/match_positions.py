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

    passthrough_cols = ["fixture_id"]
    source_cols = [
        col
        for col in dict_cols_to_keep.keys()
        if col in df_positions_kinexon_raw.columns
    ]
    extra_cols = [
        col for col in passthrough_cols if col in df_positions_kinexon_raw.columns
    ]

    df_positions = df_positions_kinexon_raw[source_cols + extra_cols].copy()
    df_positions = df_positions.rename(columns=dict_cols_to_keep)

    numeric_int_cols = ["timestamp_ms"]
    numeric_float_cols = [
        "x_m",
        "y_m",
        "speed_m_s",
        "direction",
        "acceleration",
        "total_distance",
        "metabolic_power",
        "acceleration_load",
    ]

    for col in numeric_int_cols:
        if col in df_positions.columns:
            df_positions[col] = pd.to_numeric(df_positions[col], errors="coerce")

    for col in numeric_float_cols:
        if col in df_positions.columns:
            df_positions[col] = pd.to_numeric(
                df_positions[col], errors="coerce", downcast="float"
            )

    logging.info("Normalized %d positions", len(df_positions))

    return df_positions
