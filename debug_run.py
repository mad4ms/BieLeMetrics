"""
Debug runner — execute Dagster jobs without the UI.

Usage:
    uv run python debug_run.py season
    uv run python debug_run.py fixture                        # random fixture (seed=0)
    uv run python debug_run.py fixture <fixture_id>
    uv run python debug_run.py fixture --seed 3
    uv run python debug_run.py fixture <fixture_id> --from shot_events
    uv run python debug_run.py backfill                       # season + all fixtures + train
    uv run python debug_run.py backfill --skip-season         # skip season refresh
    uv run python debug_run.py backfill --skip-train          # skip xG training at the end
    uv run python debug_run.py backfill --keep-going          # continue on fixture failure
    uv run python debug_run.py backfill --from shot_events    # start each fixture from asset
    uv run python debug_run.py list-fixtures
    uv run python debug_run.py time-analysis                  # all fixtures with position data
    uv run python debug_run.py time-analysis <fixture_id>     # single fixture detail
    uv run python debug_run.py train                          # train xG model on all fixtures
    uv run python debug_run.py render-shot                     # render shot event video(s)
    uv run python debug_run.py render-shot <fixture_id> --sync-panel
"""

import argparse
import random
import sys
from pathlib import Path

import duckdb

RUN_CONFIG_INFO_ONLY = {
    "loggers": {"console": {"config": {"log_level": "INFO"}}},
}

DB_PATH = "data/hbl_raw.duckdb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_known_fixture_ids() -> list[str]:
    with duckdb.connect(DB_PATH, read_only=True) as con:
        rows = con.execute(
            "SELECT DISTINCT fixture_id FROM fixtures_sportradar_raw ORDER BY fixture_id"
        ).fetchall()
    return [r[0] for r in rows]


def pick_fixture(fixture_id: str | None, seed: int) -> str:
    ids = get_known_fixture_ids()
    if not ids:
        print("No fixture_ids found in fixtures_sportradar_raw. Run 'season' first.")
        sys.exit(1)
    if fixture_id:
        if fixture_id not in ids:
            print(
                f"WARNING: {fixture_id} not in fixtures_sportradar_raw — proceeding anyway."
            )
        return fixture_id
    rng = random.Random(seed)
    chosen = rng.choice(ids)
    print(f"Selected fixture_id={chosen} (seed={seed}, {len(ids)} total)")
    return chosen


def get_job(name: str):
    from hbl_etl_dagster.defs import defs

    return defs.resolve_job_def(name)


def report(result) -> None:
    if result.success:
        successes = sum(
            1 for e in result.all_events if e.event_type_value == "STEP_SUCCESS"
        )
        print(f"\n✓ Job succeeded  ({successes} steps)")
    else:
        print("\n✗ Job FAILED")
        for e in result.all_events:
            if e.event_type_value in (
                "STEP_FAILURE",
                "RUN_FAILURE",
                "PIPELINE_FAILURE",
            ):
                print(f"  [{e.event_type_value}] {e.message}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_season(args):
    print("Running season_raw_refresh_job …")
    result = get_job("season_raw_refresh_job").execute_in_process(
        run_config=RUN_CONFIG_INFO_ONLY
    )
    report(result)


def cmd_fixture(args):
    from dagster import DagsterInstance

    fixture_id = pick_fixture(getattr(args, "fixture_id", None), args.seed)

    op_selection = None
    if args.from_asset:
        op_selection = [f"{args.from_asset}*"]
        print(
            f"Running fixture pipeline for {fixture_id} (from {args.from_asset} onward) …"
        )
    else:
        print(f"Running fixture pipeline for {fixture_id} …")

    job = get_job("fixture_raw_backfill_job")

    with DagsterInstance.ephemeral() as instance:
        instance.add_dynamic_partitions("fixture_partitions", [fixture_id])

        kwargs: dict = {
            "partition_key": fixture_id,
            "instance": instance,
            "run_config": RUN_CONFIG_INFO_ONLY,
        }
        if op_selection:
            kwargs["op_selection"] = op_selection

        result = job.execute_in_process(**kwargs)

    report(result)


def cmd_train(args):
    print("Running xg_training_job …")
    result = get_job("xg_training_job").execute_in_process(
        run_config=RUN_CONFIG_INFO_ONLY
    )
    report(result)


def cmd_backfill(args):
    """Run the full end-to-end pipeline: season → all fixtures → xG training."""
    from dagster import DagsterInstance

    # 1. Season refresh
    if not args.skip_season:
        print("=" * 60)
        print("Step 1/3 — season_raw_refresh_job")
        print("=" * 60)
        result = get_job("season_raw_refresh_job").execute_in_process(
            run_config=RUN_CONFIG_INFO_ONLY
        )
        report(result)
    else:
        print("Skipping season refresh (--skip-season).")

    # 2. Fixture backfill
    fixture_ids = get_known_fixture_ids()
    if not fixture_ids:
        print(
            "No fixture_ids found — run season refresh first (or drop --skip-season)."
        )
        sys.exit(1)

    total = len(fixture_ids)
    print(f"\n{'=' * 60}")
    print(f"Step 2/3 — fixture_raw_backfill_job  ({total} fixtures)")
    print("=" * 60)

    op_selection = [f"{args.from_asset}*"] if args.from_asset else None

    failed: list[str] = []
    job = get_job("fixture_raw_backfill_job")

    with DagsterInstance.ephemeral() as instance:
        instance.add_dynamic_partitions("fixture_partitions", fixture_ids)

        for idx, fixture_id in enumerate(fixture_ids, 1):
            print(f"\n[{idx}/{total}] fixture_id={fixture_id}")
            kwargs: dict = {
                "partition_key": fixture_id,
                "instance": instance,
                "run_config": RUN_CONFIG_INFO_ONLY,
            }
            if op_selection:
                kwargs["op_selection"] = op_selection

            result = job.execute_in_process(**kwargs)

            if result.success:
                successes = sum(
                    1 for e in result.all_events if e.event_type_value == "STEP_SUCCESS"
                )
                print(f"  ✓ {fixture_id} ({successes} steps)")
            else:
                print(f"  ✗ FAILED: {fixture_id}")
                for e in result.all_events:
                    if e.event_type_value in ("STEP_FAILURE", "RUN_FAILURE"):
                        print(f"    [{e.event_type_value}] {e.message}")
                failed.append(fixture_id)
                if not args.keep_going:
                    print(
                        "\nAborting backfill. Use --keep-going to continue on failure."
                    )
                    sys.exit(1)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"Fixture backfill done — {total - len(failed)}/{total} succeeded.")
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
    else:
        print(f"Fixture backfill done — all {total} fixtures succeeded.")

    # 3. xG training
    if not args.skip_train:
        print(f"\n{'=' * 60}")
        print("Step 3/3 — xg_training_job")
        print("=" * 60)
        result = get_job("xg_training_job").execute_in_process(
            run_config=RUN_CONFIG_INFO_ONLY
        )
        report(result)
    else:
        print("\nSkipping xG training (--skip-train).")

    if failed:
        sys.exit(1)


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


def cmd_time_analysis(args):
    """
    Compare Sportradar scheduled start vs. Kinexon first/last timestamp.

    Without a fixture_id: summary table across all fixtures that have position data.
    With a fixture_id:    per-minute position row counts so you can see where data
                          density jumps (warmup → actual game start).
    """
    fixture_id = getattr(args, "fixture_id", None)

    with duckdb.connect(DB_PATH, read_only=True) as con:
        has_positions = con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'match_positions_normalized'"
        ).fetchone()[0]
        if not has_positions:
            print("match_positions_normalized not found — run the pipeline first.")
            return

        if fixture_id:
            _time_analysis_single(con, fixture_id)
        else:
            _time_analysis_all(con)


def _time_analysis_all(con) -> None:
    import pandas as pd

    rows = con.execute("""
        SELECT
            p.fixture_id,
            f.startTimeUTC          AS scheduled_utc,
            f.startTimeLocal        AS scheduled_local,
            f.startTimeActualUTC    AS actual_utc,
            min(p.timestamp_ms)     AS kinexon_first_ms,
            max(p.timestamp_ms)     AS kinexon_last_ms,
            count(*)                AS n_rows,
            count(DISTINCT p.session_id) AS n_sessions
        FROM match_positions_normalized p
        JOIN fixtures_sportradar_raw f USING (fixture_id)
        GROUP BY 1, 2, 3, 4
        ORDER BY f.startTimeUTC
    """).fetchall()

    if not rows:
        print("No position data found.")
        return

    COL = "{:<36}  {:>19}  {:>19}  {:>8}  {:>8}  {:>10}  {:>10}"
    print(
        COL.format(
            "fixture_id",
            "scheduled (UTC)",
            "kinexon_first (UTC)",
            "Δ sched",
            "Δ actual",
            "duration",
            "rows",
        )
    )
    print("-" * 130)

    for (
        fid,
        sched_utc,
        sched_local,
        actual_utc,
        first_ms,
        last_ms,
        n_rows,
        n_sess,
    ) in rows:
        first_dt = pd.Timestamp(first_ms, unit="ms", tz="UTC")
        last_dt = pd.Timestamp(last_ms, unit="ms", tz="UTC")
        sched_dt = pd.to_datetime(sched_utc, utc=True) if sched_utc else None
        actual_dt = pd.to_datetime(actual_utc, utc=True) if actual_utc else None

        delta_sched = (
            f"{(first_dt - sched_dt ).total_seconds()/60:+.1f}m"
            if sched_dt
            else "  N/A  "
        )
        delta_actual = (
            f"{(first_dt - actual_dt).total_seconds()/60:+.1f}m"
            if actual_dt
            else "  N/A  "
        )
        duration_min = (last_dt - first_dt).total_seconds() / 60

        print(
            COL.format(
                fid,
                str(sched_dt)[:19] if sched_dt else "N/A",
                str(first_dt)[:19],
                delta_sched,
                delta_actual,
                f"{duration_min:.0f} min",
                f"{n_rows:,}",
            )
        )

    print()
    diffs = []
    for _, sched_utc, _, _, first_ms, _, _, _ in rows:
        if sched_utc:
            sched_dt = pd.to_datetime(sched_utc, utc=True)
            first_dt = pd.Timestamp(first_ms, unit="ms", tz="UTC")
            diffs.append((first_dt - sched_dt).total_seconds() / 60)
    if diffs:
        s = pd.Series(diffs)
        print(
            f"Δ kinexon_first vs scheduled  —  "
            f"min={s.min():.1f}m  median={s.median():.1f}m  max={s.max():.1f}m  "
            f"(n={len(s)}, {(s >= -2).sum()} within 2 min of scheduled)"
        )


def _time_analysis_single(con, fixture_id: str) -> None:
    import pandas as pd

    # Fixture metadata
    meta = con.execute(
        """
        SELECT startTimeUTC, startTimeLocal, startTimeActualUTC, endTimeActualUTC
        FROM fixtures_sportradar_raw
        WHERE fixture_id = ?
    """,
        [fixture_id],
    ).fetchone()
    if not meta:
        print(f"fixture_id {fixture_id} not found in fixtures_sportradar_raw.")
        return

    sched_utc, sched_local, actual_utc, actual_end_utc = meta
    sched_dt = pd.to_datetime(sched_utc, utc=True) if sched_utc else None

    print(f"\nFixture: {fixture_id}")
    print(f"  Scheduled (UTC):      {sched_utc}")
    print(f"  Scheduled (local):    {sched_local}")
    print(
        f"  Actual start (UTC):   {actual_utc or 'N/A (not yet played or not fetched)'}"
    )
    print(f"  Actual end   (UTC):   {actual_end_utc or 'N/A'}")

    # Per-minute row counts (ball + players combined)
    rows = con.execute(
        """
        SELECT
            time_bucket(INTERVAL '1 minute', epoch_ms(timestamp_ms)) AS minute_utc,
            count(*) AS n_rows,
            count(DISTINCT full_name) AS n_entities,
            -- count ball rows separately
            count(*) FILTER (WHERE lower(full_name) LIKE '%ball%') AS ball_rows
        FROM match_positions_normalized
        WHERE fixture_id = ?
        GROUP BY 1
        ORDER BY 1
    """,
        [fixture_id],
    ).fetchall()

    if not rows:
        print("\n  No position data for this fixture.")
        return

    first_kinexon = pd.Timestamp(
        con.execute(
            "SELECT min(timestamp_ms) FROM match_positions_normalized WHERE fixture_id = ?",
            [fixture_id],
        ).fetchone()[0],
        unit="ms",
        tz="UTC",
    )
    last_kinexon = pd.Timestamp(
        con.execute(
            "SELECT max(timestamp_ms) FROM match_positions_normalized WHERE fixture_id = ?",
            [fixture_id],
        ).fetchone()[0],
        unit="ms",
        tz="UTC",
    )

    delta_sched = (first_kinexon - sched_dt).total_seconds() / 60 if sched_dt else None

    print(
        f"\n  Kinexon first ts:     {first_kinexon} "
        f"({delta_sched:+.1f} min vs scheduled)"
        if delta_sched is not None
        else ""
    )
    print(f"  Kinexon last  ts:     {last_kinexon}")
    print(
        f"  Total duration:       {(last_kinexon - first_kinexon).total_seconds()/60:.1f} min"
    )
    print(f"  Total rows:           {sum(r[1] for r in rows):,}")

    print("\n  Per-minute breakdown (UTC | rows | entities | ball_rows | Δ_sched):")
    print(
        f"  {'minute':<22}  {'rows':>6}  {'entities':>8}  {'ball':>4}  {'Δ sched':>8}"
    )
    print("  " + "-" * 56)
    for minute_utc, n_rows, n_ent, ball_rows in rows:
        minute_dt = (
            pd.Timestamp(minute_utc).tz_localize("UTC")
            if getattr(minute_utc, "tzinfo", None) is None
            else pd.Timestamp(minute_utc)
        )
        delta = f"{(minute_dt - sched_dt).total_seconds()/60:+.0f}m" if sched_dt else ""
        print(
            f"  {str(minute_dt)[:19]}  {n_rows:>6,}  {n_ent:>8}  {ball_rows:>4}  {delta:>8}"
        )


def cmd_list_fixtures(args):
    ids = get_known_fixture_ids()
    if ids:
        print(f"Known fixture_ids ({len(ids)}):")
        for fid in ids:
            print(f"  {fid}")
    else:
        print("No fixtures found. Run 'season' first.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Debug runner for BieLeMetrics Dagster jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("season", help="Run season_raw_refresh_job")

    p = sub.add_parser("fixture", help="Run fixture_raw_backfill_job")
    p.add_argument(
        "fixture_id",
        nargs="?",
        default=None,
        help="Fixture ID to run (omit to pick randomly via --seed)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for fixture selection (default: 0)",
    )
    p.add_argument(
        "--from",
        dest="from_asset",
        metavar="ASSET",
        default=None,
        help="Start from this asset and all downstream (e.g. --from shot_events)",
    )

    sub.add_parser(
        "train", help="Train xG model on all available fixtures (xg_training_job)"
    )

    p = sub.add_parser(
        "backfill",
        help="End-to-end pipeline: season refresh → all fixtures → xG training",
    )
    p.add_argument(
        "--skip-season",
        action="store_true",
        help="Skip season_raw_refresh_job (use existing fixture list)",
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip xg_training_job at the end",
    )
    p.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue processing remaining fixtures even if one fails",
    )
    p.add_argument(
        "--from",
        dest="from_asset",
        metavar="ASSET",
        default=None,
        help="Start each fixture from this asset and all downstream (e.g. --from shot_events)",
    )

    sub.add_parser("list-fixtures", help="List known fixture_ids from the DB")

    p = sub.add_parser(
        "time-analysis", help="Analyse scheduled vs actual Kinexon start times"
    )
    p.add_argument(
        "fixture_id",
        nargs="?",
        default=None,
        help="Fixture ID for per-minute detail (omit for summary across all fixtures)",
    )

    p = sub.add_parser(
        "render-shot",
        help="Render shot event video(s) for a fixture (requires shot_events + positions)",
    )
    p.add_argument(
        "fixture_id",
        nargs="?",
        default=None,
        help="Fixture ID to render (omit to pick randomly via --seed)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for fixture selection (default: 0)",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=5,
        help="Max number of events to render (default: 5)",
    )
    p.add_argument(
        "--sync-panel",
        action="store_true",
        help="Include sync timeline panel with moving time line",
    )
    p.add_argument(
        "--sync-panel-width",
        type=int,
        default=None,
        help="Sync panel width in pixels (default: auto)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for renders (default: data/renders)",
    )

    args = parser.parse_args()
    {
        "season": cmd_season,
        "fixture": cmd_fixture,
        "backfill": cmd_backfill,
        "train": cmd_train,
        "list-fixtures": cmd_list_fixtures,
        "time-analysis": cmd_time_analysis,
        "render-shot": cmd_render_shot,
    }[args.command](args)


if __name__ == "__main__":
    main()
