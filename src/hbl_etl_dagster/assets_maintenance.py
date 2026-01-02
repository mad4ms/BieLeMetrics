import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset

from .utils.duckdb_helpers import duckdb_conn
from .utils.player_league_mapper import PlayerLeagueMapper


@asset(
    required_resource_keys={"io_manager"},
    group_name="maintenance",
    compute_kind="duckdb",
    description=(
        "Backfills players.league_id by matching Sportradar players to Kinexon "
        "positions (name + team, exact + fuzzy, team-aligned). "
        "This asset mutates the 'players' table in DuckDB."
    ),
)
def backfill_player_league_ids(context: AssetExecutionContext) -> pd.DataFrame:
    """
    Asset wrapping the legacy notebook logic for player ↔ league mapping.

    - Uses the DuckDB IO manager connection.
    - Mutates the `players` table in-place (league_id column).
    - Returns the updated players subset (league_id IS NOT NULL) as an asset table.
    """
    duckdb_io_manager = context.resources.io_manager

    with duckdb_conn(duckdb_io_manager) as con:
        mapper = PlayerLeagueMapper(con=con, logger=context.log)
        df_match_all, df_map_safe, df_players_updated = mapper.run()

    num_unique_players_matched = (
        df_map_safe["personId"].nunique() if not df_map_safe.empty else 0
    )

    context.add_output_metadata(
        {
            "n_match_all": len(df_match_all),
            "n_safe_mappings": len(df_map_safe),
            "n_unique_players_matched": num_unique_players_matched,
            "n_players_updated": len(df_players_updated),
            "preview_matches": MetadataValue.md(
                df_match_all.sort_values("similarity", ascending=False)
                .head(30)
                .to_markdown(index=False)
                if not df_match_all.empty
                else "*(no matches)*"
            ),
            "preview_players_updated": MetadataValue.md(
                df_players_updated.head(20).to_markdown(index=False)
                if not df_players_updated.empty
                else "*(no players updated)*"
            ),
        }
    )

    return df_players_updated
