from dagster import AssetSelection, Definitions, FilesystemIOManager, define_asset_job

from .assets_features.features_xg import features_xg
from .assets_features.features_xs import features_xs
from .assets_ml.assets_importance import xg_feature_importance
from .assets_ml.assets_ml import ml_xg_model, ml_xs_model
from .assets_normalized.assets_normalized_match_detected_shots import (
    check_match_detected_shots_normalized,
    match_detected_shots_normalized,
)
from .assets_normalized.assets_normalized_match_events import (
    match_events_normalized,
    match_events_normalized_goals,
    match_events_normalized_setup,
)
from .assets_normalized.assets_normalized_match_positions import (
    match_positions_normalized,
)
from .assets_normalized.assets_normalized_matches import matches_normalized
from .assets_normalized.assets_normalized_players import match_players_normalized
from .assets_raw.assets_kinexon_raw_fixture import (
    check_detected_events_have_unique_event_ids_when_present,
    check_positions_kinexon_raw_notna,
    detected_events_kinexon_raw,
    positions_kinexon_raw,
)
from .assets_raw.assets_kinexon_raw_season import (
    sessions_kinexon_raw,
    teams_kinexon_raw,
)

# Fixture-level raw assets
from .assets_raw.assets_sportradar_raw_fixture import (
    check_events_have_unique_event_ids_when_present,
    check_players_have_unique_person_ids_when_present,
    events_fixture_id_matches_partition_when_present,
    fixture_events_sportradar_raw,
    players_fixture_id_matches_events_when_present,
    players_sportradar_raw,
)

# Season-level raw assets
from .assets_raw.assets_sportradar_raw_season import (
    check_fixtures_have_fixture_id_column,
    check_fixtures_have_unique_ids_when_present,
    competition_id,
    fixtures_partition_def,
    fixtures_sportradar_raw,
    season_id,
    teams_sportradar_raw,
)
from .assets_synced.assets_players import check_players_integrity, players
from .assets_synced.assets_shot_events import (
    check_shot_events_integrity,
    check_shot_events_sync_result,
    shot_events,
)
from .io_managers import duckdb_io_manager, in_memory_io_manager
from .resources import kinexon_api, sportradar_api

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
        features_xs,
        ml_xs_model,
        xg_feature_importance,
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
        features_xs,
        ml_xs_model,
        xg_feature_importance,
    ],
    asset_checks=[
        check_fixtures_have_fixture_id_column,
        check_fixtures_have_unique_ids_when_present,
        events_fixture_id_matches_partition_when_present,
        check_events_have_unique_event_ids_when_present,
        check_players_have_unique_person_ids_when_present,
        players_fixture_id_matches_events_when_present,
        check_match_detected_shots_normalized,
        check_positions_kinexon_raw_notna,
        check_detected_events_have_unique_event_ids_when_present,
        check_players_integrity,
        check_shot_events_integrity,
        check_shot_events_sync_result,
    ],
    jobs=[
        season_raw_refresh_job,
        fixture_raw_backfill_job,
    ],
    resources={
        "io_manager": duckdb_io_manager.configured({"db_path": "data/hbl_raw.duckdb"}),
        "in_memory_io_manager": in_memory_io_manager,
        "file_io_manager": FilesystemIOManager(),
        "sportradar_api": sportradar_api,
        "kinexon_api": kinexon_api,
    },
)
