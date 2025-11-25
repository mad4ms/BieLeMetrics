from dagster import (
    Definitions,
    FilesystemIOManager,
    AssetSelection,
    define_asset_job,
    multiprocess_executor,
)

from .assets_ids import competition_id, season_id, fixtures_partition_def
from .assets_sportradar_slow import (
    teams,
    list_fixtures_raw,
    list_fixtures,
)
from .assets_sportradar import (
    fixture_events_raw,
    fixture_events_match,
    fixture_players,
)
from .assets_kinexon import kinexon_positions, kinexon_events
from .assets_sync import (
    players_merged,
    sportradar_goals_synced,
    sportradar_goals_refined,
    positions_for_throw_time,
)


# from .assets_maintenance import backfill_player_league_ids

from .io_managers import duckdb_io_manager
from .resources import sportradar_api, kinexon_api
from .sensors import fixture_sensor


SEASON_DEFAULT_CONFIG = {
    "ops": {
        "season_id": {
            "config": {
                "season_year": 2024,
            }
        }
    }
}

season_refresh_job = define_asset_job(
    name="season_refresh_job",
    selection=AssetSelection.assets(
        competition_id,
        season_id,
        teams,
        list_fixtures_raw,
        list_fixtures,
    ),
    config=SEASON_DEFAULT_CONFIG,
)

fixture_backfill_job = define_asset_job(
    name="fixture_backfill_job",
    selection=AssetSelection.assets(
        fixture_events_raw,
        fixture_events_match,
        fixture_players,
        kinexon_positions,
        kinexon_events,
        players_merged,
        sportradar_goals_synced,
        sportradar_goals_refined,
        positions_for_throw_time,
    ),
    partitions_def=fixtures_partition_def,
    executor_def=multiprocess_executor.configured(
        {"max_concurrent": 4}  # e.g., 4 fixture partitions in flight
    ),
)  # noqa

defs = Definitions(
    assets=[
        competition_id,
        season_id,
        teams,
        list_fixtures_raw,
        list_fixtures,
        fixture_events_raw,
        fixture_events_match,
        fixture_players,
        kinexon_positions,
        kinexon_events,
        sportradar_goals_synced,
        sportradar_goals_refined,
        players_merged,
        positions_for_throw_time,
        # backfill_player_league_ids,
    ],
    jobs=[season_refresh_job, fixture_backfill_job],
    sensors=[fixture_sensor],
    resources={
        "io_manager": duckdb_io_manager.configured(
            {"db_path": "data/hbl.duckdb"}
        ),
        "file_io_manager": FilesystemIOManager(),
        "sportradar_api": sportradar_api,
        "kinexon_api": kinexon_api,
    },
)
