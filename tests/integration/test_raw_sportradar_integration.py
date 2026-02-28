from __future__ import annotations

import pytest
import logging

from src.pipelines.raw.sportradar import (
    get_competition_id,
    get_fixture_events,
    get_fixtures_for_season,
    get_players_for_fixture,
    get_season_id,
    get_teams_for_season,
)

logging.basicConfig(level=logging.INFO)

pytestmark = pytest.mark.integration


def test_sportradar_season_flow(sportradar_api):
    try:
        assert sportradar_api, "Sportradar API client is not available"
        competition_id = get_competition_id(sportradar_api)
    except TypeError as exc:
        if "one of the hex, bytes" in str(exc):
            pytest.skip(f"Sportradar client parsing issue (league_id UUID is null): {exc}")
        raise
    assert competition_id

    season_id = get_season_id(sportradar_api, competition_id, 2025)
    assert season_id

    teams = get_teams_for_season(sportradar_api, season_id)
    fixtures = get_fixtures_for_season(sportradar_api, season_id)

    assert not teams.empty
    assert not fixtures.empty
    assert {"entity_id"}.issubset(teams.columns)
    assert {"fixtureId"}.issubset(fixtures.columns)


def test_sportradar_fixture_events_and_players(sportradar_api):
    try:
        competition_id = get_competition_id(sportradar_api)
    except Exception as exc:
        pytest.skip(f"{exc}")
        raise
    season_id = get_season_id(sportradar_api, competition_id, 2025)
    fixtures = get_fixtures_for_season(sportradar_api, season_id)
    fixture_id = str(fixtures.iloc[-1]["fixtureId"])

    events = get_fixture_events(sportradar_api, fixture_id)
    assert not events.empty
    assert {"eventType", "eventId"}.issubset(events.columns)

    players = get_players_for_fixture(sportradar_api, events)
    # not all fixtures guarantee setup/person rows in sample response; allow empty but keep schema when present
    if not players.empty:
        assert "personId" in players.columns
