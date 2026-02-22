from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.hbl_etl_dagster.utils.time_sync import sync_goals_with_kinexon


SAMPLES_DIR = Path("assets/data_samples")


def _sample(name: str) -> pd.DataFrame:
    return pd.read_csv(SAMPLES_DIR / name)


def test_all_sample_csv_have_20_rows() -> None:
    csv_files = sorted(SAMPLES_DIR.glob("*.csv"))
    assert csv_files, "No sample CSV files found in assets/data_samples"

    for path in csv_files:
        df = pd.read_csv(path)
        assert len(df) == 20, f"Expected 20 rows in {path.name}, got {len(df)}"


def test_sample_schema_contracts_basic_columns() -> None:
    fixture_events = _sample("main.fixture_events_sportradar.csv")
    kinexon_events = _sample("main.kinexon_events.csv")
    goals_synced = _sample("main.sportradar_goals_synced.csv")

    assert {"event_id", "event_time", "fixture_id", "person_id"}.issubset(
        fixture_events.columns
    )
    assert {"timestamp_ms", "player_id", "fixture_id", "league_id"}.issubset(
        kinexon_events.columns
    )
    assert {"event_time_ms", "person_league_id", "match_mode", "matched"}.issubset(
        goals_synced.columns
    )


def test_time_sync_row_preserving_on_real_samples() -> None:
    # Use already-normalized sample inputs from the DuckDB extracts
    goals = _sample("main.sportradar_goals_synced.csv")
    kin = _sample("main.kinexon_events.csv")

    out = sync_goals_with_kinexon(
        df_events=goals,
        df_kinexon=kin,
        tolerance_ms=30_000,
        goal_time_col="event_time_ms",
        kin_time_col="timestamp_ms",
        goal_player_col="person_league_id",
        kin_player_col="league_id",
    )

    assert len(out) == len(goals)
    assert {"kinexon_match_index", "time_diff_ms", "matched", "match_mode"}.issubset(
        out.columns
    )
    assert set(out["match_mode"].dropna().unique()).issubset(
        {"player+time", "time_only", "time_only_fallback", "no_kinexon"}
    )


def test_synced_goals_fixture_ids_exist_in_kinexon_events_sample() -> None:
    goals = _sample("main.sportradar_goals_synced.csv")
    kin = _sample("main.kinexon_events.csv")

    goals_fixture_ids = set(goals["fixture_id"].astype(str).dropna().unique())
    kin_fixture_ids = set(kin["fixture_id"].astype(str).dropna().unique())

    # For representative samples, we expect at least one overlap.
    assert goals_fixture_ids & kin_fixture_ids
