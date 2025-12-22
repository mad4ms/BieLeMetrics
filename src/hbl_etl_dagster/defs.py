from dagster import (
    Definitions,
    FilesystemIOManager,
    AssetSelection,
    define_asset_job,
    multiprocess_executor,
)

from .assets_sportradar_slow import (
    competition_id,
    season_id,
    fixtures_partition_def,
    teams_sportradar,
    fixtures_sportradar,
)

from .assets_sportradar_slow import (
    fixtures_have_unique_ids,
    fixture_session_coverage,
    fixtures_are_utc,
)

from .assets_sportradar import (
    fixture_events_sportradar,
    fixture_players_sportradar,
)
from .assets_kinexon import kinexon_positions, kinexon_events
from .assets_sync import (
    players_merged,
    sportradar_goals_synced,
    sportradar_goals_refined,
    positions_for_throw_time,
    rendered_throw_videos_first5,
)
from .assets_feature import features_at_throw_time
from .assets_machlearn import xg_model_training

# from .assets_maintenance import backfill_player_league_ids

from .io_managers import duckdb_io_manager
from .resources import sportradar_api, kinexon_api
from .sensors import fixture_sensor


SEASON_DEFAULT_CONFIG = {
    "ops": {
        "season_id": {
            "config": {
                "season_year": 2025,
            }
        }
    }
}

season_refresh_job = define_asset_job(
    name="season_refresh_job",
    selection=AssetSelection.assets(
        competition_id,
        season_id,
        teams_sportradar,
        fixtures_sportradar,
    ),
    config=SEASON_DEFAULT_CONFIG,
)

fixture_backfill_job = define_asset_job(
    name="fixture_backfill_job",
    selection=AssetSelection.assets(
        fixture_events_sportradar,
        fixture_players_sportradar,
        kinexon_positions,
        kinexon_events,
        players_merged,
        sportradar_goals_synced,
        sportradar_goals_refined,
        positions_for_throw_time,
        features_at_throw_time,
        rendered_throw_videos_first5,
        xg_model_training,
    ),
    partitions_def=fixtures_partition_def,
    executor_def=multiprocess_executor.configured(
        {"max_concurrent": 4}  # e.g., 8 fixture partitions in flight
    ),
)  # noqa

defs = Definitions(
    assets=[
        competition_id,
        season_id,
        teams_sportradar,
        fixtures_sportradar,
        fixture_events_sportradar,
        fixture_players_sportradar,
        kinexon_positions,
        kinexon_events,
        sportradar_goals_synced,
        sportradar_goals_refined,
        players_merged,
        positions_for_throw_time,
        features_at_throw_time,
        # backfill_player_league_ids,
        rendered_throw_videos_first5,
        xg_model_training,
    ],
    asset_checks=[
        fixtures_have_unique_ids,
        fixture_session_coverage,
        fixtures_are_utc,
    ],
    jobs=[season_refresh_job, fixture_backfill_job],
    sensors=[fixture_sensor],
    resources={
        "io_manager": duckdb_io_manager.configured(
            {"db_path": "data/hbl2526.duckdb"}
        ),
        "file_io_manager": FilesystemIOManager(),
        "sportradar_api": sportradar_api,
        "kinexon_api": kinexon_api,
    },
)
