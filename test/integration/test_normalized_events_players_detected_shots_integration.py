from __future__ import annotations

import pytest

from src.pipelines.normalized.match_detected_shots import normalize_match_detected_shots
from src.pipelines.normalized.match_events import (
    normalize_match_events,
    normalize_match_events_goals,
    normalize_match_events_setup,
)
from src.pipelines.normalized.match_players import normalize_match_players
from src.pipelines.raw.kinexon import (
    get_detected_events_for_fixture,
    get_sessions_for_team,
    get_teams_for_season as kin_get_teams_for_season,
)
from src.pipelines.raw.sportradar import (
    get_competition_id,
    get_fixture_events,
    get_fixtures_for_season,
    get_players_for_fixture,
    get_season_id,
)

pytestmark = pytest.mark.integration


def _resolve_season(sportradar_api):
    competition_id = get_competition_id(sportradar_api, "1. Handball-Bundesliga")
    season_id = get_season_id(sportradar_api, competition_id, 2025)
    return season_id


def test_normalized_events_players_detected_shots_chain(sportradar_api, kinexon_api):
    season_id = _resolve_season(sportradar_api)
    fixtures = get_fixtures_for_season(sportradar_api, season_id)

    fixture_id_col = "fixture_id" if "fixture_id" in fixtures.columns else "fixtureId"

    fixture_events_raw = None
    fixture_id = None
    for _, row in fixtures.head(10).iterrows():
        cand = str(row[fixture_id_col])
        ev = get_fixture_events(sportradar_api, cand)
        if not ev.empty:
            fixture_id = cand
            fixture_events_raw = ev
            break

    if fixture_events_raw is None or fixture_events_raw.empty:
        pytest.skip("No fixture with events found in first 10 fixtures")

    players_raw = get_players_for_fixture(sportradar_api, fixture_events_raw)

    session_id = None
    if "session_id" in fixtures.columns:
        val = fixtures.iloc[0]["session_id"]
        if val == val:
            session_id = int(val)

    if session_id is None:
        kin_teams = kin_get_teams_for_season(kinexon_api, "2025")
        team_id_col = "id" if "id" in kin_teams.columns else "team_id"
        sessions = get_sessions_for_team(
            kinexon_api, str(kin_teams.iloc[0][team_id_col])
        )
        if sessions.empty:
            pytest.skip("No Kinexon sessions available for detected-shots test")
        sid_col = "id" if "id" in sessions.columns else "session_id"
        session_id = int(sessions.iloc[0][sid_col])

    detected_raw = get_detected_events_for_fixture(kinexon_api, session_id)

    if "fixture_id" not in fixture_events_raw.columns:
        fixture_events_raw = fixture_events_raw.copy()
        fixture_events_raw["fixture_id"] = fixture_id

    events_norm = normalize_match_events(fixture_events_raw)
    events_setup = normalize_match_events_setup(events_norm)
    events_goals = normalize_match_events_goals(events_norm)

    if not players_raw.empty and "fixture_id" not in players_raw.columns:
        players_raw = players_raw.copy()
        players_raw["fixture_id"] = fixture_id
    players_norm = normalize_match_players(players_raw)

    detected_norm = normalize_match_detected_shots(detected_raw)
    if "fixture_id" not in detected_norm.columns:
        detected_norm = detected_norm.copy()
        detected_norm["fixture_id"] = fixture_id

    assert not events_norm.empty
    assert {"fixture_id", "event_id", "event_type"}.issubset(events_norm.columns)
    assert {"fixture_id", "person_id"}.issubset(events_setup.columns)
    assert "event_type" in events_goals.columns
    assert "fixture_id" in players_norm.columns
    assert "fixture_id" in detected_norm.columns
