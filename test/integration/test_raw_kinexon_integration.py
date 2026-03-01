from __future__ import annotations

import pytest

from src.pipelines.raw.kinexon import (
    get_detected_events_for_fixture,
    get_sessions_for_team,
    get_teams_for_season,
)


pytestmark = pytest.mark.integration


def test_kinexon_teams_and_sessions(kinexon_api):
    teams = get_teams_for_season(kinexon_api, "2025")
    assert not teams.empty

    # team id naming can vary depending on API response shape
    team_id_col = "id" if "id" in teams.columns else "team_id"
    assert team_id_col in teams.columns

    team_id = str(teams.iloc[0][team_id_col])
    sessions = get_sessions_for_team(kinexon_api, team_id)
    assert sessions is not None


def _get_first_session_id_for_team(kinexon_api) -> int:
    teams = get_teams_for_season(kinexon_api, "2025")
    team_id_col = "id" if "id" in teams.columns else "team_id"
    team_id = str(teams.iloc[0][team_id_col])

    sessions = get_sessions_for_team(kinexon_api, team_id)
    if sessions.empty:
        pytest.skip("No sessions returned for selected Kinexon team")

    session_id_col = "id" if "id" in sessions.columns else "session_id"
    return int(sessions.iloc[0][session_id_col])


def test_kinexon_detected_events_for_one_session(kinexon_api):
    session_id = _get_first_session_id_for_team(kinexon_api)
    detected = get_detected_events_for_fixture(kinexon_api, session_id)
    assert detected is not None


# def test_kinexon_positions_for_one_session(kinexon_api):
#     session_id = _get_first_session_id_for_team(kinexon_api)
#     positions = get_positions_for_session(kinexon_api, session_id)
#     assert positions is not None
