"""
Pipeline debug commands — ETL / data ingestion.

Usage:
    uv run python scripts/debug/pipeline.py season
    uv run python scripts/debug/pipeline.py fixture                        # random fixture (seed=0)
    uv run python scripts/debug/pipeline.py fixture <fixture_id>
    uv run python scripts/debug/pipeline.py fixture --seed 3
    uv run python scripts/debug/pipeline.py fixture <fixture_id> --from shot_events
    uv run python scripts/debug/pipeline.py backfill                       # season + all fixtures + train
    uv run python scripts/debug/pipeline.py backfill --skip-season
    uv run python scripts/debug/pipeline.py backfill --skip-train
    uv run python scripts/debug/pipeline.py backfill --keep-going
    uv run python scripts/debug/pipeline.py backfill --from shot_events
    uv run python scripts/debug/pipeline.py backfill --parallel 4
    uv run python scripts/debug/pipeline.py download-positions             # all pending, 4 threads
    uv run python scripts/debug/pipeline.py download-positions <fixture_id>
    uv run python scripts/debug/pipeline.py download-positions --parallel 8
    uv run python scripts/debug/pipeline.py download-positions --force
"""

import argparse
import re
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import duckdb

from _common import (
    DB_PATH,
    RUN_CONFIG_INFO_ONLY,
    WORKSPACE_ROOT,
    FixtureRunSummary,
    count_step_successes,
    get_job,
    get_known_fixture_ids,
    pick_fixture,
    print_fixture_failure_output,
    report,
)

SCRIPT_PATH = Path(__file__).resolve()


# ---------------------------------------------------------------------------
# Subprocess helpers (reference this script for parallel backfill)
# ---------------------------------------------------------------------------


def build_fixture_cli_command(fixture_id: str, from_asset: str | None) -> list[str]:
    cmd = [sys.executable, str(SCRIPT_PATH), "fixture", fixture_id]
    if from_asset:
        cmd.extend(["--from", from_asset])
    return cmd


def run_fixture_subprocess(
    fixture_id: str,
    from_asset: str | None,
) -> FixtureRunSummary:
    completed = subprocess.run(
        build_fixture_cli_command(fixture_id=fixture_id, from_asset=from_asset),
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    if completed.stderr:
        output = f"{output}\n{completed.stderr}" if output else completed.stderr

    step_successes = None
    if completed.returncode == 0:
        match = re.search(r"✓ Job succeeded\s+\((\d+) steps\)", output)
        if match:
            step_successes = int(match.group(1))

    return FixtureRunSummary(
        fixture_id=fixture_id,
        success=completed.returncode == 0,
        step_successes=step_successes,
        exit_code=completed.returncode,
        output=output,
    )


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

    if args.parallel < 1:
        print("--parallel must be >= 1")
        sys.exit(2)

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

    if args.skip_no_positions:
        with duckdb.connect(DB_PATH, read_only=True) as con:
            tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
            if "positions_kinexon_raw" in tables:
                have_positions = {
                    r[0]
                    for r in con.execute(
                        "SELECT DISTINCT fixture_id FROM positions_kinexon_raw"
                    ).fetchall()
                }
            else:
                have_positions = set()
        before = len(fixture_ids)
        fixture_ids = [fid for fid in fixture_ids if fid in have_positions]
        print(
            f"--skip-no-positions: {before - len(fixture_ids)} fixture(s) skipped (no positions), {len(fixture_ids)} remaining."
        )

    total = len(fixture_ids)
    print(f"\n{'=' * 60}")
    print(
        f"Step 2/3 — fixture_raw_backfill_job  ({total} fixtures, parallel={args.parallel})"
    )
    print("=" * 60)

    op_selection = [f"{args.from_asset}*"] if args.from_asset else None

    failed: list[str] = []
    if args.parallel == 1:
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
                    successes = count_step_successes(result)
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
    else:
        next_index = 0
        completed_count = 0
        stop_submitting = False
        running: dict = {}

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:

            def submit_until_full() -> None:
                nonlocal next_index
                while (
                    not stop_submitting
                    and len(running) < args.parallel
                    and next_index < total
                ):
                    fixture_id = fixture_ids[next_index]
                    next_index += 1
                    future = executor.submit(
                        run_fixture_subprocess,
                        fixture_id,
                        args.from_asset,
                    )
                    running[future] = fixture_id

            submit_until_full()

            while running:
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    fixture_id = running.pop(future)
                    completed_count += 1
                    print(f"\n[{completed_count}/{total}] fixture_id={fixture_id}")

                    try:
                        summary = future.result()
                    except Exception as exc:
                        summary = FixtureRunSummary(
                            fixture_id=fixture_id,
                            success=False,
                            step_successes=None,
                            exit_code=1,
                            output=str(exc),
                        )

                    if summary.success:
                        step_label = (
                            str(summary.step_successes)
                            if summary.step_successes is not None
                            else "?"
                        )
                        print(f"  ✓ {fixture_id} ({step_label} steps)")
                    else:
                        print(f"  ✗ FAILED: {fixture_id} (exit={summary.exit_code})")
                        print_fixture_failure_output(summary.output)
                        failed.append(fixture_id)
                        if not args.keep_going:
                            stop_submitting = True

                submit_until_full()

                if stop_submitting and running:
                    print(
                        "\nFailure detected; waiting for already-started fixture runs to finish. "
                        "Use --keep-going to continue submitting more fixtures after failures."
                    )
                    for future in list(running):
                        fixture_id = running.pop(future)
                        completed_count += 1
                        print(f"\n[{completed_count}/{total}] fixture_id={fixture_id}")
                        try:
                            summary = future.result()
                        except Exception as exc:
                            summary = FixtureRunSummary(
                                fixture_id=fixture_id,
                                success=False,
                                step_successes=None,
                                exit_code=1,
                                output=str(exc),
                            )

                        if summary.success:
                            step_label = (
                                str(summary.step_successes)
                                if summary.step_successes is not None
                                else "?"
                            )
                            print(f"  ✓ {fixture_id} ({step_label} steps)")
                        else:
                            print(
                                f"  ✗ FAILED: {fixture_id} (exit={summary.exit_code})"
                            )
                            print_fixture_failure_output(summary.output)
                            failed.append(fixture_id)
                    break

        if failed and not args.keep_going:
            print("\nAborting backfill. Use --keep-going to continue on failure.")
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


def cmd_download_positions(args):
    """Download Kinexon raw positions in parallel, writing directly to DuckDB."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import pandas as pd
    from filelock import FileLock

    from src.fetcher_kinexon.fetch_positions_for_fixture import (
        fetch_positions_for_fixture,
    )
    from src.hbl_etl_dagster.utils.api_helper import get_api_kinexon

    db_path = Path(DB_PATH).resolve()

    # 1. Load fixture → session_id mapping and find what's already downloaded
    with duckdb.connect(DB_PATH, read_only=True) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}

        if "matches_normalized" not in tables:
            print(
                "matches_normalized not found — run 'season' + at least one 'fixture' first."
            )
            return

        df_matches = con.execute("SELECT * FROM matches_normalized").df()

        already_downloaded: set[str] = set()
        if "positions_kinexon_raw" in tables:
            already_downloaded = {
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT fixture_id FROM positions_kinexon_raw"
                ).fetchall()
            }

    # Resolve session_id (asset uses id as fallback)
    if "session_id" not in df_matches.columns:
        df_matches["session_id"] = df_matches["id"]

    df_matches = (
        df_matches[["fixture_id", "session_id"]]
        .dropna(subset=["session_id"])
        .drop_duplicates(subset=["fixture_id"])
        .copy()
    )
    df_matches["fixture_id"] = df_matches["fixture_id"].astype(str)
    df_matches["session_id"] = df_matches["session_id"].astype(str)

    # Optional single-fixture filter
    if args.fixture_id:
        df_matches = df_matches[df_matches["fixture_id"] == args.fixture_id]
        if df_matches.empty:
            print(f"fixture_id={args.fixture_id} not found in matches_normalized.")
            return

    pending = (
        df_matches
        if args.force
        else df_matches[~df_matches["fixture_id"].isin(already_downloaded)]
    )
    total = len(pending)

    if total == 0:
        print(
            f"Nothing to download — {len(already_downloaded)} fixture(s) already have positions. "
            "Use --force to re-download."
        )
        return

    print(
        f"Downloading positions for {total} fixture(s) "
        f"({len(already_downloaded)} already done, parallel={args.parallel}) …"
    )

    session_to_fixture = dict(zip(pending["session_id"], pending["fixture_id"]))
    session_ids = list(session_to_fixture.keys())

    # 2. Connect to Kinexon API
    api = get_api_kinexon()
    api.connect()

    # 3. Fetch in parallel, write to DuckDB per fixture (sequential writes, parallel fetches)
    lock = FileLock(str(db_path.with_suffix(db_path.suffix + ".lock")))
    done_count = 0
    failed: list[str] = []

    def _fetch(session_id: str) -> tuple[str, pd.DataFrame]:
        return session_id, fetch_positions_for_fixture(api=api, session_id=session_id)

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(_fetch, sid): sid for sid in session_ids}

        for future in as_completed(futures):
            session_id = futures[future]
            fixture_id = session_to_fixture[session_id]

            try:
                _, df = future.result()
            except Exception as exc:
                print(
                    f"  ✗ [{done_count + len(failed) + 1}/{total}] fixture_id={fixture_id}  ERROR: {exc}"
                )
                failed.append(fixture_id)
                continue

            if df.empty:
                print(
                    f"  ✗ [{done_count + len(failed) + 1}/{total}] fixture_id={fixture_id}  (empty response)"
                )
                failed.append(fixture_id)
                continue

            df["fixture_id"] = fixture_id
            if "fixtureId" in df.columns:
                df = df.drop(columns=["fixtureId"])

            with lock:
                with duckdb.connect(str(db_path)) as con:
                    has_table = con.execute(
                        "SELECT count(*) FROM information_schema.tables "
                        "WHERE table_name = 'positions_kinexon_raw'"
                    ).fetchone()[0]
                    con.register("_new_pos", df)
                    if has_table:
                        con.execute(
                            "DELETE FROM positions_kinexon_raw WHERE fixture_id = ?",
                            [fixture_id],
                        )
                        con.execute(
                            "INSERT INTO positions_kinexon_raw SELECT * FROM _new_pos"
                        )
                    else:
                        con.execute(
                            "CREATE TABLE positions_kinexon_raw AS SELECT * FROM _new_pos"
                        )

            done_count += 1
            print(
                f"  ✓ [{done_count}/{total}] fixture_id={fixture_id}  rows={len(df):,}"
            )

    print(f"\nDone — {done_count}/{total} fixtures downloaded.")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline debug commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("season", help="Run season_raw_refresh_job")

    p = sub.add_parser("fixture", help="Run fixture_raw_backfill_job for one fixture")
    p.add_argument("fixture_id", nargs="?", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--from", dest="from_asset", metavar="ASSET", default=None)

    sub.add_parser("train", help="Train xG model (xg_training_job)")

    p = sub.add_parser(
        "backfill", help="End-to-end: season → all fixtures → xG training"
    )
    p.add_argument("--skip-season", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--from", dest="from_asset", metavar="ASSET", default=None)
    p.add_argument("--parallel", type=int, default=1, metavar="N")
    p.add_argument(
        "--skip-no-positions",
        action="store_true",
        help="Skip fixtures that have no rows in positions_kinexon_raw",
    )

    p = sub.add_parser(
        "download-positions",
        help="Download Kinexon raw positions in parallel, writing directly to DuckDB",
    )
    p.add_argument("fixture_id", nargs="?", default=None)
    p.add_argument("--parallel", type=int, default=4, metavar="N")
    p.add_argument("--force", action="store_true")

    args = parser.parse_args()
    {
        "season": cmd_season,
        "fixture": cmd_fixture,
        "train": cmd_train,
        "backfill": cmd_backfill,
        "download-positions": cmd_download_positions,
    }[args.command](args)


if __name__ == "__main__":
    main()
