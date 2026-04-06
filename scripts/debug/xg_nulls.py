"""Debug xG feature geometry nulls without loading all positions into memory.

Usage:
    uv run python scripts/debug/xg_nulls.py
    uv run python scripts/debug/xg_nulls.py --tolerance 150 --top 20
    uv run python scripts/debug/xg_nulls.py --fixture 00ba9627-... --top 5
    uv run python scripts/debug/xg_nulls.py --json-out artifacts/xg_nulls_summary.json

The script compares exact timestamp lookup against nearest-frame lookup under a
bounded tolerance and reports how much null geometry coverage improves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import DB_PATH, WORKSPACE_ROOT

if str(WORKSPACE_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(WORKSPACE_ROOT))

import duckdb

from src.pipelines.features.calc_xg_features import (
    FRAME_TOLERANCE_MS,
    _build_group_name_map,
    _positions_for_timestamp,
)


GEOMETRY_KEYS = [
    "timestamp_lookup_miss",
    "offense_team_empty",
    "defense_team_empty",
    "shooter_row_empty",
    "goalkeeper_row_empty",
    "ball_row_empty",
    "shooter_distance_null",
    "shot_angle_null",
    "shooter_gk_distance_null",
    "gk_goal_distance_null",
    "ball_goal_distance_null",
]


def _empty_counters() -> dict[str, int]:
    return {"n_shots": 0, **{key: 0 for key in GEOMETRY_KEYS}}


def _analyze_fixture(
    con: duckdb.DuckDBPyConnection,
    fixture_id: str,
    *,
    frame_tolerance_ms: int,
) -> dict[str, object] | None:
    shots = con.execute(
        """
        SELECT fixture_id, event_id, throw_timestamp_ms, team_name_offense, team_name_defense,
               team_name_home, person_league_id, goalkeeper_league_id, goal_position
        FROM shot_events
        WHERE fixture_id = ?
        """,
        [fixture_id],
    ).df()
    if shots.empty:
        return None

    positions = con.execute(
        """
        SELECT timestamp_ms, group_name, league_id, x_m, y_m
        FROM match_positions_normalized
        WHERE fixture_id = ?
        """,
        [fixture_id],
    ).df()
    if positions.empty:
        return None

    positions = positions.copy()
    group_name_map = _build_group_name_map(positions, shots)
    if group_name_map:
        positions["group_name"] = positions["group_name"].map(
            lambda group_name: group_name_map.get(group_name, group_name)
        )
    positions["timestamp_ms"] = pd.to_numeric(
        positions["timestamp_ms"], errors="coerce"
    )
    positions_by_timestamp = positions.set_index("timestamp_ms", drop=False)
    available_timestamps = np.array(
        sorted(
            {
                float(timestamp)
                for timestamp in positions["timestamp_ms"].tolist()
                if pd.notna(timestamp)
            }
        ),
        dtype=float,
    )

    exact = _empty_counters()
    nearest = _empty_counters()

    for shot in shots.itertuples(index=False):
        exact["n_shots"] += 1
        nearest["n_shots"] += 1

        for counters, tolerance in ((exact, 0), (nearest, frame_tolerance_ms)):
            df_positions_shot = _positions_for_timestamp(
                positions_by_timestamp,
                available_timestamps,
                shot.throw_timestamp_ms,
                frame_tolerance_ms=tolerance,
            )

            if df_positions_shot.empty:
                counters["timestamp_lookup_miss"] += 1

            df_offense = df_positions_shot[
                df_positions_shot["group_name"] == shot.team_name_offense
            ]
            df_defense = df_positions_shot[
                df_positions_shot["group_name"] == shot.team_name_defense
            ]
            if df_offense.empty:
                counters["offense_team_empty"] += 1
            if df_defense.empty:
                counters["defense_team_empty"] += 1

            goal_x = shot.goal_position

            df_shooter = df_offense[df_offense["league_id"] == shot.person_league_id]
            shooter_present = not df_shooter.empty
            if not shooter_present:
                counters["shooter_row_empty"] += 1
            if (not shooter_present) or pd.isna(goal_x):
                counters["shooter_distance_null"] += 1
                counters["shot_angle_null"] += 1

            df_goalkeeper = df_defense[
                df_defense["league_id"] == shot.goalkeeper_league_id
            ]
            goalkeeper_present = not df_goalkeeper.empty
            if not goalkeeper_present:
                counters["goalkeeper_row_empty"] += 1
            if (not shooter_present) or (not goalkeeper_present):
                counters["shooter_gk_distance_null"] += 1
            if (not goalkeeper_present) or pd.isna(goal_x):
                counters["gk_goal_distance_null"] += 1

            df_ball = df_positions_shot[
                df_positions_shot["league_id"]
                .astype(str)
                .str.contains("ball", case=False, na=False)
            ]
            ball_present = not df_ball.empty
            if not ball_present:
                counters["ball_row_empty"] += 1
            if (not ball_present) or pd.isna(goal_x):
                counters["ball_goal_distance_null"] += 1

    row = {"fixture_id": fixture_id, "n_shots": int(exact["n_shots"])}
    for key in GEOMETRY_KEYS:
        row[f"exact_{key}"] = int(exact[key])
        row[f"nearest_{key}"] = int(nearest[key])
        row[f"delta_{key}"] = int(exact[key] - nearest[key])
    return row


def _format_share(value: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{100 * value / total:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=None, help="Analyze one fixture_id only")
    parser.add_argument(
        "--tolerance",
        type=int,
        default=FRAME_TOLERANCE_MS,
        help=f"Nearest-frame tolerance in ms (default: {FRAME_TOLERANCE_MS})",
    )
    parser.add_argument(
        "--top", type=int, default=15, help="Rows to show in top tables"
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write machine-readable JSON summary",
    )
    args = parser.parse_args()

    with duckdb.connect(DB_PATH, read_only=True) as con:
        if args.fixture:
            fixture_ids = [args.fixture]
        else:
            fixture_ids = [
                row[0]
                for row in con.execute(
                    "SELECT DISTINCT fixture_id FROM shot_events ORDER BY fixture_id"
                ).fetchall()
            ]

        fixture_rows = []
        for fixture_id in fixture_ids:
            row = _analyze_fixture(
                con,
                fixture_id,
                frame_tolerance_ms=args.tolerance,
            )
            if row is not None:
                fixture_rows.append(row)

    if not fixture_rows:
        print("No fixtures could be analyzed.")
        return

    fixture_df = pd.DataFrame(fixture_rows)
    total_shots = int(fixture_df["n_shots"].sum())

    exact_summary = {
        key: int(fixture_df[f"exact_{key}"].sum()) for key in GEOMETRY_KEYS
    }
    nearest_summary = {
        key: int(fixture_df[f"nearest_{key}"].sum()) for key in GEOMETRY_KEYS
    }

    print("\n" + "=" * 72)
    print("XG GEOMETRY NULL TRACE")
    print("=" * 72)
    print(f"Fixtures analyzed: {fixture_df['fixture_id'].nunique()}")
    print(f"Shots analyzed:    {total_shots}")
    print(f"Nearest tolerance: {args.tolerance} ms")

    print("\nGlobal Summary:")
    print(
        f"  {'metric':<28}  {'exact':>8}  {'nearest':>8}  {'improved':>9}  {'exact%':>8}  {'nearest%':>8}"
    )
    for key in GEOMETRY_KEYS:
        exact_value = exact_summary[key]
        nearest_value = nearest_summary[key]
        improved = exact_value - nearest_value
        print(
            f"  {key:<28}  {exact_value:>8}  {nearest_value:>8}  {improved:>9}  {_format_share(exact_value, total_shots):>8}  {_format_share(nearest_value, total_shots):>8}"
        )

    improvement_table = fixture_df[
        [
            "fixture_id",
            "n_shots",
            "exact_timestamp_lookup_miss",
            "nearest_timestamp_lookup_miss",
            "delta_timestamp_lookup_miss",
            "exact_shooter_distance_null",
            "nearest_shooter_distance_null",
            "delta_shooter_distance_null",
            "exact_shooter_gk_distance_null",
            "nearest_shooter_gk_distance_null",
            "delta_shooter_gk_distance_null",
        ]
    ].sort_values(
        [
            "delta_shooter_distance_null",
            "delta_shooter_gk_distance_null",
            "delta_timestamp_lookup_miss",
            "n_shots",
        ],
        ascending=[False, False, False, False],
    )

    print("\nTop Fixtures By Null Reduction:")
    print(improvement_table.head(args.top).to_string(index=False))

    if args.json_out:
        payload = {
            "fixtures_analyzed": int(fixture_df["fixture_id"].nunique()),
            "shots_analyzed": total_shots,
            "nearest_tolerance_ms": args.tolerance,
            "exact_summary": exact_summary,
            "nearest_summary": nearest_summary,
            "top_fixture_rows": improvement_table.head(args.top).to_dict(
                orient="records"
            ),
        }
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote JSON summary to {output_path}")


if __name__ == "__main__":
    main()
