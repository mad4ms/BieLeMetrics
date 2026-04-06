from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipelines.synced.players import extract_players_for_match
from src.pipelines.synced.shot_events import (
    _sync_goals_to_detected_shots,
    normalize_time,
    sync_shot_events,
)

SAMPLES_DIR = Path("assets/data_samples")


def _sample(name: str) -> pd.DataFrame:
    return pd.read_csv(SAMPLES_DIR / name)


def test_all_sample_csv_have_expected_sample_size() -> None:
    csv_files = sorted(SAMPLES_DIR.glob("*.csv"))
    assert csv_files
    for path in csv_files:
        n_rows = len(pd.read_csv(path))
        # Sample exports are capped at 20 rows, but small source tables can have fewer.
        assert 0 < n_rows <= 20, f"{path.name}: expected 1..20 rows, got {n_rows}"


def test_normalize_time_from_pipeline_creates_event_time_ms() -> None:
    goals = _sample("main.sportradar_goals_synced.csv")[
        ["event_id", "event_time"]
    ].copy()
    out = normalize_time(goals)
    assert "event_time_ms" in out.columns
    assert out["event_time_ms"].notna().all()


def test_sync_goals_to_detected_shots_row_preserving_pipeline() -> None:
    goals = _sample("main.sportradar_goals_synced.csv")
    detected = _sample("main.kinexon_events.csv")

    out = _sync_goals_to_detected_shots(
        df_goals=goals,
        df_match_detected_shots_normalized=detected,
        tol_before_ms=30_000,
        tol_after_ms=3_000,
    )

    assert len(out) == len(goals)
    assert "detected_shot_id" in out.columns
    assert "match_method" in out.columns


def test_extract_players_for_match_pipeline_minimal_contract() -> None:
    df_match_normalized = pd.DataFrame(
        [
            {
                "fixture_id": "f1",
                "entity_id_home": "h1",
                "entity_id_away": "a1",
                "team_name_home": "Home",
                "team_name_away": "Away",
            }
        ]
    )

    df_match_events_normalized_setup = pd.DataFrame(
        [
            {
                "fixture_id": "f1",
                "entity_id": "h1",
                "event_type": "person",
                "person_id": "p1",
                "name": "Max Mustermann",
                "bib": 7,
                "position": "LB",
            },
            {
                "fixture_id": "f1",
                "entity_id": "a1",
                "event_type": "person",
                "person_id": "p2",
                "name": "John Doe",
                "bib": 1,
                "position": "GK",
            },
        ]
    )

    df_match_detected_shots_normalized = pd.DataFrame()

    df_match_positions_normalized = pd.DataFrame(
        [
            {
                "fixture_id": "f1",
                "mapped_id": "m1",
                "league_id": "l1",
                "session_id": "s1",
                "full_name": "Max Mustermann",
                "group_name": "Home",
            },
            {
                "fixture_id": "f1",
                "mapped_id": "m2",
                "league_id": "l2",
                "session_id": "s1",
                "full_name": "John Doe",
                "group_name": "Away",
            },
        ]
    )

    df_match_players_normalized = pd.DataFrame(
        [
            {"fixture_id": "f1", "person_id": "p1", "height": 190},
            {"fixture_id": "f1", "person_id": "p2", "height": 195},
        ]
    )

    out = extract_players_for_match(
        df_match_normalized=df_match_normalized,
        df_match_events_normalized_setup=df_match_events_normalized_setup,
        df_match_detected_shots_normalized=df_match_detected_shots_normalized,
        df_match_positions_normalized=df_match_positions_normalized,
        df_match_players_normalized=df_match_players_normalized,
    )

    assert len(out) == 2
    assert {"fixture_id", "entity_id", "person_id", "team_name", "league_id"}.issubset(
        out.columns
    )


def test_extract_players_for_match_tolerates_missing_names() -> None:
    df_match_normalized = pd.DataFrame(
        [
            {
                "fixture_id": "f1",
                "entity_id_home": "h1",
                "entity_id_away": "a1",
                "team_name_home": "Home",
                "team_name_away": "Away",
            }
        ]
    )

    df_match_events_normalized_setup = pd.DataFrame(
        [
            {
                "fixture_id": "f1",
                "entity_id": "h1",
                "event_type": "person",
                "person_id": "p1",
                "name": pd.NA,
                "bib": 7,
                "position": "LB",
            }
        ]
    )

    df_match_positions_normalized = pd.DataFrame(
        [
            {
                "fixture_id": "f1",
                "mapped_id": "m1",
                "league_id": "l1",
                "session_id": "s1",
                "full_name": pd.NA,
                "group_name": "Home",
            }
        ]
    )

    df_match_players_normalized = pd.DataFrame(
        [{"fixture_id": "f1", "person_id": "p1", "height": 190}]
    )

    out = extract_players_for_match(
        df_match_normalized=df_match_normalized,
        df_match_events_normalized_setup=df_match_events_normalized_setup,
        df_match_detected_shots_normalized=pd.DataFrame(),
        df_match_positions_normalized=df_match_positions_normalized,
        df_match_players_normalized=df_match_players_normalized,
    )

    assert len(out) == 1
    assert out.loc[0, "person_id"] == "p1"
    assert pd.isna(out.loc[0, "mapped_id"])
    assert pd.isna(out.loc[0, "league_id"])
    assert pd.isna(out.loc[0, "session_id"])


def test_extract_players_for_match_empty_positions_keeps_expected_columns() -> None:
    df_match_normalized = pd.DataFrame(
        [
            {
                "fixture_id": "f1",
                "entity_id_home": "h1",
                "entity_id_away": "a1",
                "team_name_home": "Home",
                "team_name_away": "Away",
            }
        ]
    )

    df_match_events_normalized_setup = pd.DataFrame(
        columns=[
            "fixture_id",
            "entity_id",
            "event_type",
            "person_id",
            "name",
            "bib",
            "position",
        ]
    )

    out = extract_players_for_match(
        df_match_normalized=df_match_normalized,
        df_match_events_normalized_setup=df_match_events_normalized_setup,
        df_match_detected_shots_normalized=pd.DataFrame(),
        df_match_positions_normalized=pd.DataFrame(),
        df_match_players_normalized=pd.DataFrame(),
    )

    assert out.empty
    assert "league_id" in out.columns
    assert "mapped_id" in out.columns
    assert "session_id" in out.columns


def test_sync_shot_events_empty_goals_keeps_expected_columns() -> None:
    out = sync_shot_events(
        df_match_normalized=pd.DataFrame([{"fixture_id": "f1"}]),
        df_match_events_normalized_goals=pd.DataFrame(
            columns=["fixture_id", "event_type", "event_id"]
        ),
        df_match_detected_shots_normalized=pd.DataFrame(),
        df_positions_normalized=pd.DataFrame(),
        df_players=pd.DataFrame(),
    )

    assert out.empty
    assert "throw_timestamp_ms" in out.columns
    assert "detected_shot_id" in out.columns
    assert "match_method" in out.columns
