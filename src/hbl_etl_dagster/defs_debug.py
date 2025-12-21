# defs_debug_sportradar_raw.py

from dagster import (
    Definitions,
    FilesystemIOManager,
    AssetSelection,
    define_asset_job,
)

# Season-level raw assets
from .assets_raw.assets_sportradar_raw_season import (
    competition_id,
    season_id,
    teams_sportradar_raw,
    fixtures_sportradar_raw,
    fixtures_partition_def,
    check_fixtures_have_fixture_id_column,
    check_fixtures_have_unique_ids_when_present,
)
from .assets_normalized.assets_normalized_matches import matches_normalized
from .assets_normalized.assets_normalized_match_events import (
    match_events_normalized,
    match_events_normalized_setup,
    match_events_normalized_goals,
)

# Fixture-level raw assets
from .assets_raw.assets_sportradar_raw_fixture import (
    fixture_events_sportradar_raw,
    players_sportradar_raw,
    events_fixture_id_matches_partition_when_present,
    check_events_have_unique_event_ids_when_present,
    check_players_have_unique_person_ids_when_present,
    players_fixture_id_matches_events_when_present,
)
from .assets_raw.assets_kinexon_raw_fixture import (
    positions_kinexon_raw,
    detected_events_kinexon_raw,
)
from .assets_raw.assets_kinexon_raw_season import (
    teams_kinexon_raw,
    sessions_kinexon_raw,
)
from .assets_normalized.assets_normalized_match_positions import (
    match_positions_normalized,
)
from .assets_normalized.assets_normalized_match_detected_shots import (
    match_detected_shots_normalized,
    check_match_detected_shots_normalized,
)
from .assets_normalized.assets_normalized_players import (
    match_players_normalized,
)
from .assets_synced.assets_shot_events import (
    shot_events,
)
from .assets_features.assets_features_xg import (
    features_xg,
)
from .assets_ml.assets_ml import (
    ml_xg_model,
)
from .assets_synced.assets_players import players
from .io_managers import duckdb_io_manager
from .resources import sportradar_api, kinexon_api
from .sensors import fixture_sensor


SEASON_DEFAULT_CONFIG = {
    "ops": {
        "competition_id": {
            "config": {"competition_name": "1. Handball-Bundesliga"},
        },
        "season_id": {
            "config": {"season_year": 2025},
        },
    }
}

season_raw_refresh_job = define_asset_job(
    name="season_raw_refresh_job",
    selection=AssetSelection.assets(
        competition_id,
        season_id,
        teams_sportradar_raw,
        fixtures_sportradar_raw,
        teams_kinexon_raw,
        sessions_kinexon_raw,
        matches_normalized,
    ),
    config=SEASON_DEFAULT_CONFIG,
)

fixture_raw_backfill_job = define_asset_job(
    name="fixture_raw_backfill_job",
    selection=AssetSelection.assets(
        fixture_events_sportradar_raw,
        players_sportradar_raw,
        detected_events_kinexon_raw,
        positions_kinexon_raw,
        match_events_normalized,
        match_events_normalized_setup,
        match_events_normalized_goals,
        match_positions_normalized,
        match_detected_shots_normalized,
        match_players_normalized,
        players,
        shot_events,
        features_xg,
        ml_xg_model,
    ),
    partitions_def=fixtures_partition_def,
)

defs = Definitions(
    assets=[
        competition_id,
        season_id,
        teams_sportradar_raw,
        fixtures_sportradar_raw,
        fixture_events_sportradar_raw,
        players_sportradar_raw,
        teams_kinexon_raw,
        sessions_kinexon_raw,
        matches_normalized,
        detected_events_kinexon_raw,
        positions_kinexon_raw,
        match_events_normalized,
        match_events_normalized_setup,
        match_events_normalized_goals,
        match_positions_normalized,
        match_detected_shots_normalized,
        match_players_normalized,
        players,
        shot_events,
        features_xg,
        ml_xg_model,
    ],
    asset_checks=[
        check_fixtures_have_fixture_id_column,
        check_fixtures_have_unique_ids_when_present,
        events_fixture_id_matches_partition_when_present,
        check_events_have_unique_event_ids_when_present,
        check_players_have_unique_person_ids_when_present,
        players_fixture_id_matches_events_when_present,
        check_match_detected_shots_normalized,
    ],
    jobs=[
        season_raw_refresh_job,
        fixture_raw_backfill_job,
    ],
    resources={
        "io_manager": duckdb_io_manager.configured(
            {"db_path": "data/hbl_raw.duckdb"}
        ),
        "file_io_manager": FilesystemIOManager(),
        "sportradar_api": sportradar_api,
        "kinexon_api": kinexon_api,
    },
)
