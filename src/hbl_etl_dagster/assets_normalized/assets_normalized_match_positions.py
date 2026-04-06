import pandas as pd
from dagster import (
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    MetadataValue,
    asset,
)

from src.pipelines.normalized.match_positions import (
    normalize_match_positions as normalize_match_positions_fn,
)
from src.hbl_etl_dagster.utils.metadata import markdown_table

fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")

# Venue Y-offset corrections keyed by home team name (as stored in matches_normalized).
# Value is the number of meters to subtract from y_m to bring coordinates into the
# standard 0–20 m field frame.
# Flensburg: Kinexon origin at their arena is shifted +12.5 m on the Y-axis.
VENUE_Y_OFFSET_M: dict[str, float] = {
    "SG Flensburg-Handewitt": 12.5,
}


@asset(
    group_name="normalized",
    compute_kind="duckdb",
    partitions_def=fixtures_partition_def,
    description="Normalized match positions for a single fixture (partitioned by fixture_id).",
)
def match_positions_normalized(
    context: AssetExecutionContext,
    positions_kinexon_raw: pd.DataFrame,
    matches_normalized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize match positions.

    :param context: AssetExecutionContext
    :param positions_kinexon_raw: Raw Kinexon positioning data for the current partition.
    :param matches_normalized: Non-partitioned match lookup (filtered by fixture_id below).
    :return: Normalized positions DataFrame.
    """
    fixture_id = context.partition_key

    df_normalized = normalize_match_positions_fn(
        df_positions_kinexon_raw=positions_kinexon_raw,
    )

    # Apply venue-specific Y offset correction for home games at known offset venues.
    fixture_meta = matches_normalized[
        matches_normalized["fixture_id"] == str(fixture_id)
    ]
    if not fixture_meta.empty:
        home_team = fixture_meta["team_name_home"].iloc[0]
        if home_team in VENUE_Y_OFFSET_M:
            y_shift = VENUE_Y_OFFSET_M[home_team]
            df_normalized["y_m"] = df_normalized["y_m"] - y_shift
            context.log.info(
                "Applied venue Y offset correction of -%.1f m for home team '%s' (fixture %s)",
                y_shift,
                home_team,
                fixture_id,
            )

    context.log.info("Normalized %d match positions", len(df_normalized))
    context.add_output_metadata(
        {
            "n_rows": len(df_normalized),
            "n_columns": df_normalized.shape[1],
            "preview": MetadataValue.md(markdown_table(df_normalized, n=5)),
        }
    )

    return df_normalized
