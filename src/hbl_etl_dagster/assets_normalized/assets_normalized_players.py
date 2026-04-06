import pandas as pd
from dagster import (
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    MetadataValue,
    asset,
)

from src.pipelines.normalized.match_players import (
    normalize_match_players as normalize_match_players_fn,
)
from src.hbl_etl_dagster.utils.metadata import markdown_table

fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")


@asset(
    group_name="normalized",
    compute_kind="duckdb",
    partitions_def=fixtures_partition_def,
    description="Normalized match players for a single fixture (partitioned by fixture_id).",
)
def match_players_normalized(
    context: AssetExecutionContext,
    players_sportradar_raw: pd.DataFrame,
) -> pd.DataFrame:
    df_normalized_players = normalize_match_players_fn(players_sportradar_raw)
    context.log.info("Normalized %d match players", len(df_normalized_players))
    context.add_output_metadata(
        {
            "n_rows": len(df_normalized_players),
            "n_unique_players": df_normalized_players["person_id"].nunique(),
            "n_columns": df_normalized_players.shape[1],
            "preview": MetadataValue.md(markdown_table(df_normalized_players, n=100)),
        }
    )
    return df_normalized_players
