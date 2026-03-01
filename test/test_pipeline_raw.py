from __future__ import annotations

import pandas as pd

from src.pipelines.raw import kinexon, sportradar


def test_normalize_season_year() -> None:
    assert kinexon._normalize_season_year("2025") == "2025-26"
    assert kinexon._normalize_season_year("2025-26") == "2025-26"


def test_sportradar_get_players_for_fixture_normalizes_columns(monkeypatch) -> None:
    monkeypatch.setattr(
        sportradar,
        "extract_person_ids_from_setup",
        lambda df: ["p1", "p2"],
    )

    def fake_fetch_players_by_ids(api, person_ids):
        assert person_ids == ["p1", "p2"]
        return pd.DataFrame([{"person.id": "p1", "name.full": "A"}])

    monkeypatch.setattr(sportradar, "fetch_players_by_ids", fake_fetch_players_by_ids)

    out = sportradar.get_players_for_fixture(
        api=object(), df_match_events=pd.DataFrame()
    )
    assert list(out.columns) == ["person_id", "name_full"]
    assert len(out) == 1


def test_sportradar_get_competition_id_none(monkeypatch) -> None:
    monkeypatch.setattr(
        sportradar, "fetch_competition_id", lambda api, competition_name: None
    )
    out = sportradar.get_competition_id(api=object(), competition_name="X")
    assert out is None
