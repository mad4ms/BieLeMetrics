"""
Render debug commands — shot event video rendering.

Usage:
    uv run python scripts/debug/render.py render-shot
    uv run python scripts/debug/render.py render-shot <fixture_id>
    uv run python scripts/debug/render.py render-shot --seed 3
    uv run python scripts/debug/render.py render-shot --max-events 10
    uv run python scripts/debug/render.py render-shot --sync-panel
    uv run python scripts/debug/render.py render-shot --output-dir artifacts/renders
"""

import argparse
from pathlib import Path

import duckdb

from _common import DB_PATH, pick_fixture


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_render_shot(args):
    from src.pipelines.synced.shot_events import render_shot_event

    fixture_id = pick_fixture(getattr(args, "fixture_id", None), args.seed)

    with duckdb.connect(DB_PATH, read_only=True) as con:
        table_names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "shot_events" not in table_names:
            print("shot_events table not found — run the fixture pipeline first.")
            return
        if "match_positions_normalized" not in table_names:
            print(
                "match_positions_normalized not found — run the fixture pipeline first."
            )
            return

        df_shot_events = con.execute(
            "SELECT * FROM shot_events WHERE fixture_id = ?",
            [fixture_id],
        ).df()
        df_positions = con.execute(
            "SELECT * FROM match_positions_normalized WHERE fixture_id = ?",
            [fixture_id],
        ).df()

    if df_shot_events.empty:
        print(f"No shot_events for fixture_id={fixture_id}")
        return
    if df_positions.empty:
        print(f"No positions for fixture_id={fixture_id}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else None
    df_rendered = render_shot_event(
        df_shot_events,
        df_positions,
        max_events=args.max_events,
        include_sync_panel=args.sync_panel,
        sync_panel_width_px=args.sync_panel_width,
        output_dir=output_dir,
    )

    if df_rendered.empty:
        print("No renders produced.")
        return
    print("Rendered videos:")
    for _, row in df_rendered.iterrows():
        print(f"  event_id={row['event_id']}  video={row['video_path']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Render debug commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "render-shot",
        help="Render shot event video(s) for a fixture",
    )
    p.add_argument("fixture_id", nargs="?", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-events", type=int, default=5)
    p.add_argument("--sync-panel", action="store_true")
    p.add_argument("--sync-panel-width", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    {"render-shot": cmd_render_shot}[args.command](args)


if __name__ == "__main__":
    main()
