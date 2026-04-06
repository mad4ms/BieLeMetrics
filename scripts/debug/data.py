"""
Data inspection commands — data coverage, fixture listing, timing analysis.

Usage:
    uv run python scripts/debug/data.py list-fixtures
    uv run python scripts/debug/data.py time-analysis
    uv run python scripts/debug/data.py time-analysis <fixture_id>
    uv run python scripts/debug/data.py analyze data
"""

import argparse

import duckdb

from _common import DB_PATH, get_known_fixture_ids


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _analyze_data():
    """Summarize data coverage and quality."""
    with duckdb.connect(DB_PATH, read_only=True) as con:
        tables = con.execute("SHOW TABLES").fetchall()
        table_names = [r[0] for r in tables]

        print("\n" + "=" * 70)
        print("DATA COVERAGE & QUALITY SUMMARY")
        print("=" * 70)

        if "fixtures_sportradar_raw" in table_names:
            n_fixtures = con.execute(
                "SELECT count(DISTINCT fixture_id) FROM fixtures_sportradar_raw"
            ).fetchone()[0]
            print(f"\nFixtures (Sportradar): {n_fixtures}")

        if "fixture_events_sportradar" in table_names:
            n_events = con.execute(
                "SELECT count(*) FROM fixture_events_sportradar"
            ).fetchone()[0]
            print(f"Raw events (Sportradar): {n_events:,}")

        if "match_positions_normalized" in table_names:
            n_pos = con.execute(
                "SELECT count(*) FROM match_positions_normalized"
            ).fetchone()[0]
            n_pos_fixtures = con.execute(
                "SELECT count(DISTINCT fixture_id) FROM match_positions_normalized"
            ).fetchone()[0]
            print(
                f"Position rows (Kinexon): {n_pos:,} ({n_pos_fixtures} fixtures with data)"
            )

        if "shot_events" in table_names:
            n_shots = con.execute("SELECT count(*) FROM shot_events").fetchone()[0]
            print(f"Shot events: {n_shots}")

        if "features_xg" in table_names:
            n_features = con.execute("SELECT count(*) FROM features_xg").fetchone()[0]
            print(f"xG features: {n_features}")

        if "players_merged" in table_names:
            n_players = con.execute(
                "SELECT count(DISTINCT person_id) FROM players_merged"
            ).fetchone()[0]
            print(f"Unique players: {n_players}")

        print(f"\nTotal tables in DB: {len(table_names)}")
        print(
            f"Table names: {', '.join(table_names[:5])}{'...' if len(table_names) > 5 else ''}"
        )


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
            f"{(first_dt - sched_dt).total_seconds()/60:+.1f}m"
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

    rows = con.execute(
        """
        SELECT
            time_bucket(INTERVAL '1 minute', epoch_ms(timestamp_ms)) AS minute_utc,
            count(*) AS n_rows,
            count(DISTINCT full_name) AS n_entities,
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


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_list_fixtures(args):
    ids = get_known_fixture_ids()
    if ids:
        print(f"Known fixture_ids ({len(ids)}):")
        for fid in ids:
            print(f"  {fid}")
    else:
        print("No fixtures found. Run 'season' first.")


def cmd_time_analysis(args):
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


def cmd_analyze(args):
    if args.analysis_type == "data":
        _analyze_data()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Inspect debug commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

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

    p = sub.add_parser("analyze", help="Summarize data coverage and quality")
    p.add_argument("analysis_type", choices=["data"])

    args = parser.parse_args()
    {
        "list-fixtures": cmd_list_fixtures,
        "time-analysis": cmd_time_analysis,
        "analyze": cmd_analyze,
    }[args.command](args)


if __name__ == "__main__":
    main()
