from dagster import Definitions, mem_io_manager, fs_io_manager
from .assets import (
    competition_id,
    season_id,
    teams,
    fixtures,
    session_ids,
    fixtures_with_sessions,
    fixture_events,
    kinexon_positions,
    backfill_player_league_ids,
    kinexon_events,
    kinexon_events_synced,
    sportradar_goals_synced,
)
from .io_managers import duckdb_io_manager
from .resources import sportradar_api, kinexon_api

defs = Definitions(
    assets=[
        competition_id,
        season_id,
        teams,
        fixtures,
        session_ids,
        fixtures_with_sessions,
        fixture_events,
        kinexon_positions,
        backfill_player_league_ids,
        kinexon_events,
        kinexon_events_synced,
        sportradar_goals_synced,
    ],
    resources={
        "io_manager": duckdb_io_manager.configured(
            {"db_path": "data/hbl.duckdb"}
        ),
        "file_io_manager": fs_io_manager,
        "sportradar_api": sportradar_api,
        "kinexon_api": kinexon_api,
    },
)
