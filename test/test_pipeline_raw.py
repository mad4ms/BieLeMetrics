from __future__ import annotations

import pandas as pd

from src.pipelines.normalized.match_positions import normalize_match_positions
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


def test_normalize_match_positions_applies_flensburg_home_y_offset() -> None:
    df_raw = pd.DataFrame(
        [
            {
                "ts in ms": 1,
                "sensor id": 10,
                "mapped id": 20,
                "league id": "l1",
                "group id": 1,
                "group name": "SG Flensburg-Handewitt",
                "x in m": 31.0,
                "y in m": 22.5,
                "speed in m/s": 0.0,
                "direction of movement in deg": 0.0,
                "acceleration in m/s2": 0.0,
                "total distance in m": 0.0,
                "metabolic power in W/kg": 0.0,
                "acceleration load": 0.0,
                "fixture_id": "f1",
            }
        ]
    )

    out = normalize_match_positions(
        df_positions_kinexon_raw=df_raw,
        home_team_name="SG Flensburg-Handewitt",
    )

    assert out.loc[0, "y_m"] == 10.0


def test_normalize_match_positions_leaves_other_home_fixtures_unchanged() -> None:
    df_raw = pd.DataFrame(
        [
            {
                "ts in ms": 1,
                "sensor id": 10,
                "mapped id": 20,
                "league id": "l1",
                "group id": 1,
                "group name": "THW Kiel",
                "x in m": 31.0,
                "y in m": 22.5,
                "speed in m/s": 0.0,
                "direction of movement in deg": 0.0,
                "acceleration in m/s2": 0.0,
                "total distance in m": 0.0,
                "metabolic power in W/kg": 0.0,
                "acceleration load": 0.0,
                "fixture_id": "f1",
            }
        ]
    )

    out = normalize_match_positions(
        df_positions_kinexon_raw=df_raw,
        home_team_name="THW Kiel",
    )

    assert out.loc[0, "y_m"] == 22.5
