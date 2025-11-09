"""
ETL: Kinexon positions + detected events → DuckDB

- Reads fixtures (with session_id) from DuckDB
- Downloads per-session Kinexon positions (gzip/zip/plain auto-detect)
- Writes to kinexon_positions (idempotent per session_id)
- Fetches detected events via Kinexon get_events_for_player
- Writes to kinexon_events_detected (idempotent per session_id + event id)
- Builds kinexon_events_detected_enriched by:
  * filtering validated events
  * nearest-neighbor timestamp match to positions (per session)
  * joining matched position row and adding fixture_id
- Prints summary
"""

from __future__ import annotations

import os
import io
import gzip
import zipfile
import logging
import concurrent.futures
from typing import Any, Dict, Iterable, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

from config import get_duckdb_connection, get_api_kinexon

# ---------------------------------------------------------------------
# Config / Logging
# ---------------------------------------------------------------------

LOG_LEVEL = os.getenv("KINEXON_ETL_LOG_LEVEL", "DEBUG").upper()
MAX_WORKERS = int(os.getenv("KINEXON_ETL_MAX_WORKERS", "8"))

logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("kinexon_etl")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _is_gzip(b: bytes) -> bool:
    return len(b) >= 2 and b[0] == 0x1F and b[1] == 0x8B


def _is_zip(b: bytes) -> bool:
    return len(b) >= 4 and b[:4] == b"PK\x03\x04"


def read_positions_payload(payload: bytes) -> pd.DataFrame:
    """
    Accepts raw bytes from Kinexon API. Detects gzip/zip and returns DataFrame.
    CSV is expected to be UTF-8 with semicolon separator.
    """
    raw = payload
    if _is_gzip(raw):
        raw = gzip.decompress(raw)
        csv_bytes = raw
    elif _is_zip(raw):
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = next(
                (n for n in zf.namelist() if n.lower().endswith(".csv")), None
            )
            if not name:
                raise ValueError("ZIP archive does not contain a CSV file.")
            csv_bytes = zf.read(name)
    else:
        csv_bytes = raw

    return pd.read_csv(io.StringIO(csv_bytes.decode("utf-8-sig")), sep=";")


def _ensure_tables(con: duckdb.DuckDBPyConnection) -> None:
    """
    Do not precreate flexible-schema tables. We only ensure the DB is writable
    and clean up any previous placeholder (1-column) tables created by mistake.
    """
    for tbl in (
        "kinexon_positions",
        "kinexon_events_detected",
        "kinexon_events_detected_enriched",
    ):
        _drop_if_placeholder(con, tbl)


def _drop_if_placeholder(con: duckdb.DuckDBPyConnection, table: str) -> None:
    # drop table if it exists as a 1-column dummy (SELECT 1 WHERE 0 artifact)
    try:
        info = con.execute(f"PRAGMA table_info('{table}')").df()
    except duckdb.CatalogException:
        return  # table doesn't exist
    if len(info) == 1 and str(info.loc[0, "name"]) in {
        "1",
        "column0",
        "column_0",
    }:
        con.execute(f"DROP TABLE {table}")
        log.warning(
            "Dropped placeholder table '%s' (wrong 1-column schema).", table
        )


# --- keep this helper; it will create the table from the incoming DataFrame schema ---
def _df_create_if_needed_and_insert(
    con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame
) -> None:
    if df is None or df.empty:
        return
    # if table is a stale placeholder, remove it first
    _drop_if_placeholder(con, table)
    con.register("df", df)
    con.execute(
        f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM df WHERE 1=0"
    )
    con.execute(f"INSERT INTO {table} SELECT * FROM df")
    con.unregister("df")


def _positions_exist_for_session(
    con: duckdb.DuckDBPyConnection, session_id: int
) -> bool:
    try:
        cnt = con.execute(
            "SELECT COUNT(*) FROM kinexon_positions WHERE session_id = ?",
            [session_id],
        ).fetchone()[0]
        return cnt > 0
    except duckdb.CatalogException:
        return False


def _events_exist_for_session(
    con: duckdb.DuckDBPyConnection, session_id: int
) -> bool:
    try:
        cnt = con.execute(
            "SELECT COUNT(*) FROM kinexon_events_detected WHERE session_id = ?",
            [session_id],
        ).fetchone()[0]
        return cnt > 0
    except duckdb.CatalogException:
        return False
    except duckdb.BinderException:
        return False


# ---------------------------------------------------------------------
# Core: positions download
# ---------------------------------------------------------------------


def _download_positions_for_row(
    api, con: duckdb.DuckDBPyConnection, row: pd.Series
) -> Dict[str, Any]:
    session_id = row.get("session_id")
    fixture_id = row.get("fixtureId")
    if pd.isna(session_id):
        return {
            "session_id": None,
            "fixture_id": fixture_id,
            "records": 0,
            "status": "skip_no_session",
        }

    session_id = int(session_id)

    if _positions_exist_for_session(con, session_id):
        return {
            "session_id": session_id,
            "fixture_id": fixture_id,
            "records": 0,
            "status": "skip_exists",
        }

    try:
        resp = api.download_positions_csv_via_custom(
            session_id=session_id, compress_output=True
        )
        payload = (
            resp
            if isinstance(resp, (bytes, bytearray))
            else getattr(resp, "content", None)
        )
        if payload is None:
            return {
                "session_id": session_id,
                "fixture_id": fixture_id,
                "records": 0,
                "status": "error:no_payload",
            }

        df = read_positions_payload(payload)
        if df.empty:
            return {
                "session_id": session_id,
                "fixture_id": fixture_id,
                "records": 0,
                "status": "empty",
            }

        # attach keys
        df["session_id"] = session_id
        df["fixture_id"] = fixture_id

        # persist (create-if-needed)
        _df_create_if_needed_and_insert(con, "kinexon_positions", df)

        return {
            "session_id": session_id,
            "fixture_id": fixture_id,
            "records": len(df),
            "status": "ok",
        }
    except Exception as e:
        return {
            "session_id": session_id,
            "fixture_id": fixture_id,
            "records": 0,
            "status": f"error:{e}",
        }


def ingest_positions(
    con: duckdb.DuckDBPyConnection, api, max_workers: int = MAX_WORKERS
) -> pd.DataFrame:
    # Fixtures with session_id
    df_fx = con.execute(
        """
        SELECT fixtureId, session_id, name_team_home
        FROM fixtures
        WHERE session_id IS NOT NULL
    """
    ).df()

    log.info("Positions: processing %d sessions…", len(df_fx))
    if df_fx.empty:
        return pd.DataFrame(
            columns=["session_id", "fixture_id", "records", "status"]
        )

    results: List[Dict[str, Any]] = []

    # Single-threaded version
    for _, row in df_fx.iterrows():
        log.info("[positions][%s] fetching…", row.get("session_id"))
        res = _download_positions_for_row(api, con, row)
        results.append(res)
        sid = res.get("session_id")
        status = res.get("status")
        if status == "ok":
            log.info(
                "[positions][%s] ✓ inserted %s rows",
                sid,
                res.get("records"),
            )
        else:
            log.debug("[positions][%s] %s", sid, status)

    df_summary = pd.DataFrame(results)
    ok = int((df_summary["status"] == "ok").sum())
    log.info(
        "Positions summary: ok=%d, total=%d, inserted=%d",
        ok,
        len(df_summary),
        int(df_summary.get("records", 0).sum()),
    )
    return df_summary


# ---------------------------------------------------------------------
# Core: detected events
# ---------------------------------------------------------------------


def _fetch_detected_events_for_session(api, session_id: int) -> pd.DataFrame:
    events = api.get_events_for_player(session_id=session_id) or []
    events_dict = [e.to_dict() for e in events]
    df = pd.DataFrame(events_dict)
    # attach session_id now; fixture_id will be joined from positions later
    df["session_id"] = session_id
    return df


def ingest_detected_events(
    con: duckdb.DuckDBPyConnection, api
) -> pd.DataFrame:
    # Source of truth for available sessions: kinexon_positions
    df_sessions = con.execute(
        "SELECT DISTINCT session_id, ANY_VALUE(fixture_id) AS fixture_id FROM kinexon_positions WHERE session_id IS NOT NULL GROUP BY session_id"
    ).df()
    if df_sessions.empty:
        log.warning(
            "Detected events: no sessions in kinexon_positions. Run positions ingestion first."
        )
        return pd.DataFrame()

    results: List[pd.DataFrame] = []
    for _, r in df_sessions.iterrows():
        session_id = int(r["session_id"])
        if _events_exist_for_session(con, session_id):
            log.debug("[events][%s] skip (already in DB)", session_id)
            continue

        try:
            log.debug("[events][%s] fetching…", session_id)
            df = _fetch_detected_events_for_session(api, session_id)
            if df.empty:
                log.debug("[events][%s] empty", session_id)
                continue

            # Attempt to add fixture_id via available mapping
            df["fixture_id"] = r.get("fixture_id")
            # Persist
            _df_create_if_needed_and_insert(con, "kinexon_events_detected", df)
            log.info("[events][%s] ✓ inserted %d rows", session_id, len(df))
            results.append(df)
        except Exception as e:
            log.error("[events][%s] error: %s", session_id, e)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


# ---------------------------------------------------------------------
# Enrichment: match detected events to nearest positions and build final table
# ---------------------------------------------------------------------


def build_detected_events_enriched(
    con: duckdb.DuckDBPyConnection, tolerance_ms: int = 30_000
) -> pd.DataFrame:
    """
    For each session:
      - take detected events (optionally filter validated != 0 if column exists)
      - for each event, find nearest 'ts in ms' in kinexon_positions within tolerance
      - attach matched position row + fixture_id
    Writes/overwrites kinexon_events_detected_enriched.
    """
    # Load base tables
    try:
        df_ev = con.execute("SELECT * FROM kinexon_events_detected").df()
    except duckdb.CatalogException:
        df_ev = pd.DataFrame()

    try:
        df_pos = con.execute(
            """
            SELECT * FROM kinexon_positions
            WHERE session_id IS NOT NULL
        """
        ).df()
    except duckdb.CatalogException:
        df_pos = pd.DataFrame()

    if df_ev.empty or df_pos.empty:
        log.warning(
            "Enrichment skipped (events=%d, positions=%d)",
            len(df_ev),
            len(df_pos),
        )
        return pd.DataFrame()

    # Normalize numeric timestamps
    if "timestamp_ms" in df_ev.columns:
        df_ev["timestamp_ms"] = pd.to_numeric(
            df_ev["timestamp_ms"], errors="coerce"
        )
    else:
        # best-effort: derive from 'timestamp' if present
        if "timestamp" in df_ev.columns:
            # parse to ns; convert to ms
            df_ev["timestamp_ms"] = (
                pd.to_datetime(
                    df_ev["timestamp"], errors="coerce", utc=True
                ).astype("int64")
                // 10**6
            )
        else:
            df_ev["timestamp_ms"] = pd.NA

    # Optional filter: validated != 0 if field exists
    if "validated" in df_ev.columns:
        df_ev = df_ev[
            (df_ev["validated"].astype("Int64").fillna(0) != 0)
            | df_ev["validated"].isna()
        ].copy()

    # Positions ts in ms
    if "ts in ms" not in df_pos.columns:
        raise ValueError("kinexon_positions is missing 'ts in ms' column.")

    df_pos["ts in ms"] = pd.to_numeric(df_pos["ts in ms"], errors="coerce")

    # Build a per-session nearest join
    enriched_rows: List[pd.DataFrame] = []

    for session_id, ev_grp in df_ev.groupby("session_id"):
        ev_grp = ev_grp.copy()
        pos_grp = df_pos[df_pos["session_id"] == session_id].copy()
        if pos_grp.empty:
            continue

        # pre-sort for fast nearest lookup
        pos_ts = pos_grp["ts in ms"].to_numpy(dtype="float64")
        pos_idx = pos_grp.index.to_numpy()

        def _nearest_ts(ts_ms: float) -> Tuple[Optional[int], Optional[float]]:
            if np.isnan(ts_ms):
                return None, None
            # argmin abs difference
            j = int(np.argmin(np.abs(pos_ts - ts_ms)))
            nearest_idx = pos_idx[j]
            diff = float(pos_ts[j] - ts_ms)
            return nearest_idx, diff

        # compute nearest
        nearest_idx_list = []
        diff_list = []
        for v in ev_grp["timestamp_ms"].to_numpy(dtype="float64"):
            idx, diff = _nearest_ts(v)
            nearest_idx_list.append(idx)
            diff_list.append(diff)

        ev_grp["ts_in_positions_idx"] = nearest_idx_list
        ev_grp["time_diff_ms"] = diff_list

        # drop out-of-tolerance
        keep_mask = ev_grp["time_diff_ms"].abs() <= tolerance_ms
        ev_grp = ev_grp[keep_mask].copy()
        if ev_grp.empty:
            continue

        # attach matched position columns
        join_cols = pos_grp.columns.tolist()
        pos_sub = pos_grp.loc[
            ev_grp["ts_in_positions_idx"].dropna().astype(int).tolist(), :
        ].copy()
        pos_sub = pos_sub.reset_index().rename(columns={"index": "join_index"})
        ev_grp = ev_grp.reset_index(drop=True)
        ev_grp["join_index"] = range(len(ev_grp))

        merged = ev_grp.merge(
            pos_sub, on="join_index", how="left", suffixes=("", "_pos")
        )

        # carry fixture_id from positions table
        if (
            "fixture_id" not in merged.columns
            and "fixture_id_pos" in merged.columns
        ):
            merged["fixture_id"] = merged["fixture_id_pos"]

        enriched_rows.append(
            merged.drop(columns=["join_index"], errors="ignore")
        )

    if not enriched_rows:
        log.warning("No enriched rows within tolerance.")
        return pd.DataFrame()

    df_enriched = pd.concat(enriched_rows, ignore_index=True)

    # Persist (overwrite for determinism)
    con.execute("DROP TABLE IF EXISTS kinexon_events_detected_enriched")
    con.register("df_enriched", df_enriched)
    con.execute(
        "CREATE TABLE kinexon_events_detected_enriched AS SELECT * FROM df_enriched"
    )
    con.unregister("df_enriched")

    log.info("Enriched events written: %d", len(df_enriched))
    return df_enriched


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------


def run_kinexon_positions_and_events_etl(
    db_path: Optional[str] = None,
) -> None:
    con = (
        get_duckdb_connection(db_path) if db_path else get_duckdb_connection()
    )
    api = get_api_kinexon()
    _ensure_tables(con)

    # 1) Positions
    df_pos_summary = ingest_positions(con, api, max_workers=MAX_WORKERS)

    # 2) Detected events
    df_events = ingest_detected_events(con, api)

    # 3) Enrichment (nearest match to positions; validated filter)
    _ = build_detected_events_enriched(con, tolerance_ms=30_000)

    # Log summary
    n_pos = (
        int(df_pos_summary.get("records", pd.Series(dtype=int)).sum())
        if not df_pos_summary.empty
        else 0
    )
    n_ev = len(df_events) if isinstance(df_events, pd.DataFrame) else 0
    n_en = (
        con.execute(
            "SELECT COUNT(*) FROM kinexon_events_detected_enriched"
        ).fetchone()[0]
        if not df_events.empty
        else 0
    )

    log.info("=== Kinexon ETL Summary ===")
    log.info("Positions inserted: %d", n_pos)
    log.info("Detected events inserted: %d", n_ev)
    log.info("Enriched events rows: %d", n_en)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Example: python kinexon_etl.py  (uses default DB path from your config)
    #          KINEXON_ETL_MAX_WORKERS=12 python kinexon_etl.py
    run_kinexon_positions_and_events_etl(db_path="./data/mydb2024-25.duckdb")
