#!/usr/bin/env python
"""
Minimal local helper to inspect and render goal clips AFTER a Dagster run.

Assumptions:
- DuckDB file contains:
    - table `sportradar_goals_synced`  (asset from sportradar_goals_synced)
    - table `kinexon_positions`        (from kinexon_positions asset)
- Utilities are defined in
  src.hbl_etl_dagster.utils.sportradar_kinexon_event_mapper:
    - refine_throw_time_for_event
    - render_goal_with_multifreeze
    - plot_event_sync (for manual use if needed)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.hbl_etl_dagster.utils.sportradar_kinexon_event_mapper import (
    D_POSSESS,
    V_POSSESS_MAX,
    refine_throw_time_for_event,
    render_goal_with_multifreeze,
)


def main(db_path: str, out_dir: str, limit: int) -> None:
    con = duckdb.connect(db_path, read_only=True)

    # 1) Load synced goals (only fixtures that have Kinexon positions)
    df_goals = con.execute(
        """
        SELECT *
        FROM sportradar_goals_synced
        WHERE fixtureId IN (
            SELECT DISTINCT fixtureId
            FROM kinexon_positions
        )
        ORDER BY fixtureId, eventTime
        LIMIT ?
        """,
        [limit],
    ).fetch_df()

    if df_goals.empty:
        print("No rows in sportradar_goals_synced – nothing to render.")
        return

    game_of_choice = 1
    # in df_goals, choose game_of_choice-th unique fixtureId
    unique_fixtures = df_goals["fixtureId"].dropna().unique()
    if game_of_choice - 1 >= len(unique_fixtures):
        print(f"Not enough unique fixtures to select game {game_of_choice}.")
        return
    selected_fixture_id = unique_fixtures[game_of_choice - 1]
    df_goals = df_goals[df_goals["fixtureId"] == selected_fixture_id].copy()
    selected_session_id = df_goals["session_id"].dropna().unique()
    # should only be one, convert to scalar
    if len(selected_session_id) != 1:
        print(
            f"Expected one session_id for fixture {selected_fixture_id}, "
            f"found {len(selected_session_id)}."
        )
        return
    selected_session_id = selected_session_id[0]
    # 2) Load positions once
    df_positions = con.execute(
        f"SELECT * FROM kinexon_positions WHERE session_id = '{selected_session_id}'"
    ).fetch_df()
    if df_positions.empty:
        print("No rows in kinexon_positions – nothing to render.")
        return

    # Ensure 'ts' column exists (UTC)
    if "ts" not in df_positions.columns:
        df_positions["ts"] = pd.to_datetime(
            df_positions["ts in ms"], unit="ms", utc=True, errors="coerce"
        )

    # Map fixtureId -> session_id (first seen per fixture)
    if "session_id" not in df_positions.columns:
        print("kinexon_positions has no session_id column – aborting.")
        return

    fixture_to_session = (
        df_positions.dropna(subset=["session_id"])[["fixtureId", "session_id"]]
        .drop_duplicates(subset=["fixtureId"])
        .set_index("fixtureId")["session_id"]
        .to_dict()
    )

    out_dir_path = Path(out_dir)

    for _, row in df_goals.iterrows():
        fixture_id = row.get("fixtureId")
        if fixture_id is None:
            continue

        plot_title = (
            f"Event sync — team  {row.get('teamName')}, event {row.get('personName')}"
        )

        # --- 3) Refine throw time for this event (1D heuristic) ---
        diag = refine_throw_time_for_event(
            row=row,
            df_positions_all=df_positions,
            fixture_to_session=fixture_to_session,
            plot=True,  # set True for interactive diagnostic plots
            plot_title=plot_title,
        )

        refined_throw_ts = diag.get("refined_throw_ts")

        # --- 4) Build freeze markers ---
        markers = []

        # Match eventTime (Sportradar)
        if pd.notna(row.get("eventTime")):
            markers.append(
                {
                    "name": "Match event",
                    "ts": pd.to_datetime(row["eventTime"], utc=True),
                    "color": (255, 255, 255),
                    "seconds": 0.5,
                }
            )

        # Kinexon matched timestamp_ms from sportradar_goals_synced
        if "kin_timestamp_ms" in row and pd.notna(row["kin_timestamp_ms"]):
            markers.append(
                {
                    "name": "Kinexon event",
                    "ts": pd.to_datetime(
                        int(row["kin_timestamp_ms"]), unit="ms", utc=True
                    ),
                    "color": (0, 0, 255),
                    "seconds": 2.0,
                }
            )

        # Refined throw from heuristic (if found)
        if refined_throw_ts is not None and pd.notna(refined_throw_ts):
            markers.append(
                {
                    "name": "Refined throw",
                    "ts": pd.to_datetime(refined_throw_ts, utc=True),
                    "color": (0, 255, 255),
                    "seconds": 2.0,
                }
            )

        if not markers:
            print(f"No valid markers for eventId={row.get('eventId')}, skipping.")
            continue

        print(
            f"Rendering fixtureId={fixture_id}, eventId={row.get('eventId')} "
            f"with {len(markers)} markers ..."
        )

        df_pos_fixture = df_positions[df_positions["fixtureId"] == fixture_id].copy()
        if df_pos_fixture.empty:
            print(f"No positions for fixtureId={fixture_id}, skipping.")
            continue

        render_goal_with_multifreeze(
            df_positions=df_pos_fixture,
            row_goal=row,
            freeze_markers=markers,
            out_dir=out_dir_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render goal clips from DuckDB after Dagster run."
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/hbl.duckdb",
        help="Path to DuckDB file produced by Dagster run.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/renders/",
        help="Output directory for mp4/png renders.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of goals to render.",
    )
    args = parser.parse_args()
    main(db_path=args.db, out_dir=args.out, limit=args.limit)
