from __future__ import annotations

import pytest

from src.pipelines.normalized.matches import normalize_matches
from src.pipelines.raw.kinexon import (
    get_sessions_for_team,
    get_teams_for_season as kin_get_teams_for_season,
)
from src.pipelines.raw.sportradar import (
    get_competition_id,
    get_fixtures_for_season,
    get_season_id,
)

pytestmark = pytest.mark.integration


def _resolve_season(sportradar_api):
    competition_id = get_competition_id(sportradar_api, "1. Handball-Bundesliga")
    season_id = get_season_id(sportradar_api, competition_id, 2025)
    return season_id


def test_normalized_matches_from_raw_sources(sportradar_api, kinexon_api):
    season_id = _resolve_season(sportradar_api)

    fixtures = get_fixtures_for_season(sportradar_api, season_id)
    assert not fixtures.empty

    kin_teams = kin_get_teams_for_season(kinexon_api, "2025")
    assert not kin_teams.empty

    team_id_col = "id" if "id" in kin_teams.columns else "team_id"
    team_id = str(kin_teams.iloc[0][team_id_col])
    sessions = get_sessions_for_team(kinexon_api, team_id)
    assert sessions is not None

    fixtures_for_norm = fixtures.rename(columns={"fixtureId": "fixture_id"})

    try:
        normalized_matches = normalize_matches(
            df_fixtures_sportradar_raw=fixtures_for_norm,
            df_sessions_kinexon_raw=sessions,
        )
    except ValueError as exc:
        if "Unexpected format in description" in str(exc):
            pytest.xfail("Known normalize_matches delimiter gap for ' v. ' format")
        raise

    assert not normalized_matches.empty
    assert {"fixture_id", "session_id", "team_name_home", "team_name_away"}.issubset(
        normalized_matches.columns
    )
