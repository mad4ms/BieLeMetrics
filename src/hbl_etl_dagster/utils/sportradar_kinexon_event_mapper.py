from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import logging
import numpy as np
import pandas as pd

# --- plotting / rendering deps ---
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import cv2


# =============================================================================
# Core mapper: Sportradar goals ↔ Kinexon detected events
# =============================================================================


@dataclass
class SportradarKinexonEventMapper:
    """
    Helper to synchronize Sportradar goal events with Kinexon detected events.

    Responsibilities:
      - Load player → league_id mapping from DuckDB (`players` table).
      - Prepare Sportradar goal events (time in ms, league_id attached).
      - Prepare Kinexon events (validated only, normalized timestamp).
      - Perform a two-pass time-based merge (backward then forward) by league_id.
    """

    con: "duckdb.DuckDBPyConnection"
    logger: "logging.Logger"

    # --------------------------------------------------------------------- #
    # DuckDB helpers
    # --------------------------------------------------------------------- #

    def load_players_core(self) -> pd.DataFrame:
        """
        Load minimal players table used for joining league_id to personId.
        Expects DuckDB table: `players(personId, league_id, ...)`.
        """
        df = self.con.execute(
            """
            SELECT personId, league_id
            FROM players
            """
        ).fetch_df()

        if df.empty:
            self.logger.warning(
                "load_players_core(): players table is empty or missing league_id."
            )
            return df

        df["personId"] = df["personId"].astype(str)
        return df

    # --------------------------------------------------------------------- #
    # Data preparation
    # --------------------------------------------------------------------- #

    def prepare_sportradar_goals(
        self,
        fixture_events: pd.DataFrame,
        players_core: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Filter Sportradar fixture_events to goal rows and enrich with league_id.

        Returns:
            DataFrame sorted by eventTime_ms with:
              - eventTime (datetime64[ns, UTC])
              - eventTime_ms (int64, ms since epoch)
              - personId (str)
              - league_id (may be NaN if unknown)
        """
        if fixture_events.empty:
            self.logger.warning(
                "prepare_sportradar_goals(): fixture_events is empty."
            )
            return pd.DataFrame()

        if players_core is None:
            players_core = self.load_players_core()

        df_goals = fixture_events.copy()
        df_goals = df_goals[df_goals.get("eventType") == "goal"].copy()
        if df_goals.empty:
            self.logger.info("prepare_sportradar_goals(): no goal rows found.")
            return df_goals

        if not players_core.empty:
            df_goals["personId"] = df_goals["personId"].astype(str)
            df_goals = df_goals.merge(
                players_core[["personId", "league_id"]],
                on="personId",
                how="left",
            )

        df_goals["eventTime"] = pd.to_datetime(
            df_goals["eventTime"], errors="coerce", utc=True
        )
        df_goals = df_goals.dropna(subset=["eventTime", "personId"]).copy()
        if df_goals.empty:
            self.logger.warning(
                "prepare_sportradar_goals(): all goal rows lost due to "
                "invalid times/personId."
            )
            return df_goals

        df_goals["eventTime_ms"] = (
            df_goals["eventTime"].astype("int64") // 1_000_000
        )
        df_goals = df_goals.sort_values("eventTime_ms").reset_index(drop=True)

        self.logger.info(
            "Prepared %d goal events with eventTime_ms and league_id.",
            len(df_goals),
        )
        return df_goals

    def prepare_kinexon_events(
        self, kinexon_events: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare Kinexon detected events for time-based merge.

        Transformations:
          - sort by timestamp_ms
          - rename player_id → personId
          - drop rows where validated is NaN
          - drop session_id (can be large / not needed for merge)
          - rename success → kin_success to avoid confusion
        """
        if kinexon_events.empty:
            self.logger.warning(
                "prepare_kinexon_events(): kinexon_events is empty."
            )
            return kinexon_events

        df = kinexon_events.copy()

        if "timestamp_ms" not in df.columns:
            if "timestamp" in df.columns:
                t = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
                df["timestamp_ms"] = t.astype("int64") // 1_000_000
            else:
                raise ValueError(
                    "prepare_kinexon_events(): missing 'timestamp_ms' and "
                    "'timestamp' columns."
                )

        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")

        if "validated" in df.columns:
            before = len(df)
            df = df.dropna(subset=["validated"]).copy()
            self.logger.info(
                "prepare_kinexon_events(): kept %d/%d rows with "
                "non-null validated.",
                len(df),
                before,
            )

        if "player_id" in df.columns:
            df = df.rename(columns={"player_id": "personId"})
        df["personId"] = df["personId"].astype(str)

        if "success" in df.columns:
            df = df.rename(columns={"success": "kin_success"})

        if "session_id" in df.columns:
            df = df.drop(columns=["session_id"])

        df = df.sort_values("timestamp_ms").reset_index(drop=True)

        self.logger.info(
            "Prepared %d Kinexon events for time-based merge.", len(df)
        )
        return df

    # --------------------------------------------------------------------- #
    # Synchronization logic
    # --------------------------------------------------------------------- #

    def sync_goals_with_kinexon(
        self,
        fixture_events: pd.DataFrame,
        kinexon_events: pd.DataFrame,
        *,
        backward_tolerance_ms: int = 15_000,
        forward_tolerance_ms: int = 500,
    ) -> pd.DataFrame:
        """
        Main synchronization routine.

        Strategy:
          1. Backward merge:
               - For each goal, find the nearest Kinexon event *before* the goal
                 within backward_tolerance_ms.
          2. Forward merge for unmatched:
               - For remaining unmatched goals, find the nearest Kinexon event
                 *after* the goal within forward_tolerance_ms (to mitigate
                 clock drift).

        Merge key:
          - time: eventTime_ms ↔ timestamp_ms
          - by: league_id  (player identity already resolved via PlayerLeagueMapper)

        Returns:
            df_synced_goals: goals with attached Kinexon event columns and
            time_diff_ms.
        """
        if fixture_events.empty:
            self.logger.warning(
                "sync_goals_with_kinexon(): fixture_events is empty."
            )
            return pd.DataFrame()

        if kinexon_events.empty:
            self.logger.warning(
                "sync_goals_with_kinexon(): kinexon_events is empty."
            )
            return pd.DataFrame()

        players_core = self.load_players_core()
        df_goals = self.prepare_sportradar_goals(
            fixture_events=fixture_events,
            players_core=players_core,
        )
        if df_goals.empty:
            return df_goals

        df_kx = self.prepare_kinexon_events(kinexon_events)
        if df_kx.empty:
            return df_goals.assign(timestamp_ms=pd.NA, time_diff_ms=pd.NA)

        if "league_id" not in df_goals.columns:
            self.logger.warning(
                "sync_goals_with_kinexon(): goals DataFrame has no league_id "
                "column; merge will be done without player grouping."
            )
            merge_by = None
        else:
            merge_by = ["league_id"]

        self.logger.info(
            "Starting backward merge_asof (tolerance %d ms).",
            backward_tolerance_ms,
        )
        df_synced = pd.merge_asof(
            left=df_goals.sort_values("eventTime_ms"),
            right=df_kx.sort_values("timestamp_ms"),
            left_on="eventTime_ms",
            right_on="timestamp_ms",
            by=merge_by,
            direction="backward",
            tolerance=backward_tolerance_ms,
        )

        unmatched_mask = df_synced["timestamp_ms"].isna()
        num_unmatched = int(unmatched_mask.sum())
        if num_unmatched > 0:
            self.logger.info(
                "Backward merge: %d goals unmatched. Running forward merge "
                "with tolerance %d ms for clock drift.",
                num_unmatched,
                forward_tolerance_ms,
            )

            unmatched_goals = df_goals.loc[unmatched_mask].copy()

            df_fwd = pd.merge_asof(
                left=unmatched_goals.sort_values("eventTime_ms"),
                right=df_kx.sort_values("timestamp_ms"),
                left_on="eventTime_ms",
                right_on="timestamp_ms",
                by=merge_by,
                direction="forward",
                tolerance=forward_tolerance_ms,
            )

            df_synced = df_synced.sort_index()
            df_fwd = df_fwd.set_index(unmatched_goals.index)
            df_synced.update(df_fwd)
            df_synced = df_synced.reset_index(drop=True)

        df_synced["time_diff_ms"] = (
            df_synced["timestamp_ms"] - df_synced["eventTime_ms"]
        )

        n_matched = int(df_synced["timestamp_ms"].notna().sum())
        self.logger.info(
            "sync_goals_with_kinexon(): %d/%d goals matched to Kinexon events.",
            n_matched,
            len(df_synced),
        )

        return df_synced

    def run(
        self,
        fixture_events: pd.DataFrame,
        kinexon_events: pd.DataFrame,
        *,
        backward_tolerance_ms: int = 15_000,
        forward_tolerance_ms: int = 500,
    ) -> pd.DataFrame:
        """
        Convenience wrapper for one-shot pipeline.

        Returns:
            df_synced_goals (full goal rows with Kinexon columns + time_diff_ms)
        """
        return self.sync_goals_with_kinexon(
            fixture_events=fixture_events,
            kinexon_events=kinexon_events,
            backward_tolerance_ms=backward_tolerance_ms,
            forward_tolerance_ms=forward_tolerance_ms,
        )


# =============================================================================
# Throw-point refinement: heuristics and utilities
# =============================================================================

# Window around seed time (seconds)
WINDOW_BEFORE_S: float = 15
WINDOW_AFTER_S: float = 0.5

# Possession thresholds
D_POSSESS: float = 1.5  # meters
V_POSSESS_MAX: float = 12.0  # m/s

# Release thresholds (fallback)
V_RELEASE_MIN: float = 2.0  # m/s
D_RELEASE_MIN: float = 2.0  # meters

# Peak detection
ACC_PEAK_NEIGH: int = 1
MIN_SAMPLES: int = 4


def seed_time_from_row(row: pd.Series) -> pd.Timestamp:
    """
    Choose the seed time for refinement:
      1) If 'kinexon_matched_ts' exists and is not null, use it.
      2) Else, if 'timestamp_ms' exists (Kinexon ms column), use that.
      3) Else, fall back to Sportradar eventTime.
    """
    if "kinexon_matched_ts" in row and pd.notna(row.get("kinexon_matched_ts")):
        return pd.to_datetime(
            row["kinexon_matched_ts"], utc=True, errors="coerce"
        )

    if "kin_timestamp_ms" in row and pd.notna(row.get("kin_timestamp_ms")):
        return pd.to_datetime(
            int(row["kin_timestamp_ms"]), unit="ms", utc=True, errors="coerce"
        )

    return pd.to_datetime(row.get("eventTime"), utc=True, errors="coerce")


def extract_tracks_window(
    df_pos: pd.DataFrame,
    session_id: int,
    center_ts: pd.Timestamp,
    w_before_s: float,
    w_after_s: float,
) -> pd.DataFrame:
    """
    Slice positions for a given session around center_ts with margins.
    Requires df_pos with columns: session_id, ts (UTC Timestamp).
    """
    if pd.isna(center_ts):
        return pd.DataFrame()

    t0 = center_ts - pd.Timedelta(seconds=w_before_s)
    t1 = center_ts + pd.Timedelta(seconds=w_after_s)

    df = df_pos[df_pos["session_id"] == session_id]
    if df.empty:
        return df

    df = df[(df["ts"] >= t0) & (df["ts"] <= t1)].copy()
    df.sort_values("ts", inplace=True)
    return df


def build_joined_player_ball(
    df_scene: pd.DataFrame,
    shooter_league_id: Optional[int],
) -> pd.DataFrame:
    """
    Align shooter track with ball track on timestamps.

    df_scene must contain at least:
      - ts (Timestamp)
      - "ts in ms"
      - "league id"
      - "x in m", "y in m"
      - "speed in m/s", "acceleration in m/s2"

    Shooter is identified via league id == shooter_league_id.
    Ball is identified by 'league id' string containing "ball".
    """
    if df_scene.empty or pd.isna(shooter_league_id):
        return pd.DataFrame()

    # Identify ball rows by league-id string
    ball_mask = (
        df_scene["league id"]
        .astype(str)
        .str.contains("ball", case=False, na=False)
    )
    ball = df_scene[ball_mask].copy()
    df_players = df_scene[~ball_mask]

    shooter = df_players[
        df_players["league id"].astype("Int64") == int(shooter_league_id)
    ].copy()

    if ball.empty or shooter.empty:
        return pd.DataFrame()

    cols_keep = [
        "ts",
        "ts in ms",
        "x in m",
        "y in m",
        "speed in m/s",
        "acceleration in m/s2",
    ]

    ball = ball[cols_keep].rename(
        columns={
            "x in m": "ball_x",
            "y in m": "ball_y",
            "speed in m/s": "ball_speed",
            "acceleration in m/s2": "ball_acc",
        }
    )
    shooter = shooter[cols_keep].rename(
        columns={
            "x in m": "pl_x",
            "y in m": "pl_y",
            "speed in m/s": "pl_speed",
            "acceleration in m/s2": "pl_acc",
        }
    )

    df = pd.merge(ball, shooter, on="ts", how="inner")
    if df.empty:
        df = pd.merge_asof(
            ball.sort_values("ts"),
            shooter.sort_values("ts"),
            on="ts",
            direction="nearest",
            tolerance=pd.Timedelta(milliseconds=0),
        )

    if df.empty:
        return df

    df["dist_pb"] = np.hypot(
        df["ball_x"] - df["pl_x"], df["ball_y"] - df["pl_y"]
    )

    if df["ball_acc"].isna().all():
        df = df.sort_values("ts").copy()
        dt = df["ts"].diff().dt.total_seconds().replace(0, np.nan)
        bs = df["ball_speed"].fillna(method="ffill").fillna(0)
        df["ball_acc"] = (bs - bs.shift(1)) / dt
        df["ball_acc"].replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def first_local_maximum(values: np.ndarray, neigh: int = 2) -> Optional[int]:
    """
    Return index of the first local maximum with ±neigh neighborhood,
    ignoring NaNs. If none found, return None.
    """
    x = np.array(values, dtype=float)
    n = x.size
    if n == 0:
        return None
    for i in range(neigh, n - neigh):
        if np.isnan(x[i]):
            continue
        left = x[i - neigh : i]
        right = x[i + 1 : i + 1 + neigh]
        if np.all(x[i] >= left) and np.all(x[i] > right):
            return i
    return None


def detect_throw_point(
    df_pb: pd.DataFrame,
    d_possess: float,
    v_possess_max: float,
    v_release_min: float,
    d_release_min: float,
    acc_peak_neigh: int,
) -> Dict[str, Optional[object]]:
    """
    Compute possession and throw point:

      1) Identify all possession windows (dist_pb <= d_possess).
      2) For the end of *each* possession window, define a 100ms search span (±50ms).
      3) Determine the highest acceleration peak across all these spans.

    Returns metadata dict (indices, timestamps, method).
    """
    out: Dict[str, Optional[object]] = dict(
        n_rows=int(len(df_pb)),
        last_possession_idx=None,
        throw_idx=None,
        throw_ts=None,
        method=None,
    )

    if df_pb.empty or len(df_pb) < MIN_SAMPLES:
        return out

    # 1. Possession mask
    # We use distance threshold to define possession.
    poss = df_pb["dist_pb"] <= d_possess

    if not poss.any():
        return out

    # 2. Identify ends of ALL possession windows
    # A row is an "end of possession" if it is True, and the next row is False (or it's the last row)
    # We assume df_pb is sorted by time.
    is_end_of_possession = poss & (~poss.shift(-1, fill_value=False))

    # Get the timestamps of these ends
    end_timestamps = df_pb.loc[is_end_of_possession, "ts"]

    if end_timestamps.empty:
        return out

    # Set the "last_possession_idx" to the very last one found, for reference/plotting
    last_ts_global = end_timestamps.iloc[-1]
    last_poss_row = df_pb[df_pb["ts"] == last_ts_global].iloc[0]
    out["last_possession_idx"] = int(
        df_pb.index.get_indexer_for([last_poss_row.name])[0]
    )

    # 3. Build a search mask for ±50ms around EACH end timestamp
    search_mask = pd.Series(False, index=df_pb.index)

    for t_end in end_timestamps:
        t_start_window = t_end - pd.Timedelta(milliseconds=50)
        t_end_window = t_end + pd.Timedelta(milliseconds=50)

        # Update mask to include this window
        window_mask = (df_pb["ts"] >= t_start_window) & (
            df_pb["ts"] <= t_end_window
        )
        search_mask |= window_mask

    # 4. Filter data to these windows
    candidates = df_pb[search_mask]

    if candidates.empty:
        # Fallback to the last possession row if windows are somehow empty
        throw_row = last_poss_row
        method = "last_possession_fallback_empty_window"
    else:
        # 5. Find max acceleration in the candidate windows
        if candidates["ball_acc"].isna().all():
            throw_row = last_poss_row
            method = "last_possession_fallback_no_acc"
        else:
            # Find row with max ball_acc across all candidate windows
            best_idx_label = candidates["ball_acc"].astype(float).idxmax()
            throw_row = candidates.loc[best_idx_label]
            method = "max_acc_in_possession_windows"

    out["throw_idx"] = int(df_pb.index.get_indexer_for([throw_row.name])[0])
    out["throw_ts"] = throw_row["ts"]
    out["method"] = method

    return out


# =============================================================================
# Fancy plotting helper (time axis, peaks, annotations)
# =============================================================================


def _ts_to_ms(series_or_ts):
    s = pd.to_datetime(series_or_ts, utc=True, errors="coerce")
    if isinstance(s, pd.Series):
        return s.astype("int64") // 10**6
    return None if pd.isna(s) else int(s.value // 10**6)


def _fmt_time_ms(ms_val, _pos=None):
    """Format milliseconds since epoch to HH:MM:SS.mmm (UTC)."""
    try:
        ts = pd.to_datetime(int(ms_val), unit="ms", utc=True)
        return ts.strftime("%H:%M:%S.%f")[:-3]
    except Exception:
        return ""


def _find_local_maxima(values: np.ndarray, neigh: int = 2) -> List[int]:
    x = np.asarray(values, dtype=float)
    n = x.size
    peaks: List[int] = []
    if n == 0:
        return peaks
    for i in range(neigh, n - neigh):
        if np.isnan(x[i]):
            continue
        if np.all(x[i] >= x[i - neigh : i]) and np.all(
            x[i] > x[i + 1 : i + 1 + neigh]
        ):
            peaks.append(i)
    return peaks


def _annot(ax, x, y, text, color, dy: float = 0.0):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(0, 12 + dy),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=1, alpha=0.8),
    )


def plot_event_sync(
    df_pb: pd.DataFrame,
    *,
    last_possession_idx: Optional[int],
    event_ms: Optional[int],
    kin_ms: Optional[int],
    throw_ms: Optional[int],
    d_possess: float = D_POSSESS,
    title: Optional[str] = None,
    event_row: Optional[pd.Series] = None,
):
    """
    Fancy twin-axis plot around a player–ball timeline.

    Expects df_pb with at least:
      - ts (timestamp-like)
      - ball_acc (float)
      - dist_pb (float)
      - ball_speed (float)
    """
    if df_pb.empty:
        return

    # --- Color Palette ---
    C_ACC = "#4C72B0"  # Muted blue for acceleration
    C_DIST = "#55A868"  # Muted green for distance
    C_POSS = "#DDDDDD"  # Light grey for possession background
    C_EVENT = "#1f77b4"  # Original blue for Sportradar event
    C_KIN = "#ff7f0e"  # Original orange for Kinexon detection
    C_THROW = "#d62728"  # Original red for refined throw

    x_ms = _ts_to_ms(df_pb["ts"])
    acc = df_pb["ball_acc"].to_numpy(dtype=float)
    dist = df_pb["dist_pb"].to_numpy(dtype=float)

    possess = df_pb["dist_pb"] <= d_possess
    start_idx = (last_possession_idx) if last_possession_idx is not None else 0

    fig, axA = plt.subplots(figsize=(11.5, 5.2))
    axA.xaxis.set_major_formatter(FuncFormatter(_fmt_time_ms))

    axA.plot(x_ms, acc, lw=1.6, label="Ball acceleration (m/s²)", color=C_ACC)
    axA.axhline(0, lw=1, color="0.7")
    axA.set_xlabel("Time (HH:MM:SS.mmm, UTC)")
    axA.set_ylabel("Acceleration (m/s²)")

    axD = axA.twinx()
    axD.plot(
        x_ms,
        dist,
        ls="--",
        lw=1.6,
        label="Player-Ball distance (m)",
        color=C_DIST,
    )

    axD.set_ylabel("Distance / Speed")

    ymax = max(dist.max(), df_pb["ball_speed"].max()) * 1.1
    axD.fill_between(
        x_ms,
        0,
        ymax,
        where=possess.to_numpy(),
        alpha=0.6,
        label="Possession",
        color=C_POSS,
    )

    markers = [
        (event_ms, "Match eventTime", C_EVENT),
        (kin_ms, "Kinexon Detection", C_KIN),
        (throw_ms, "Refined throw", C_THROW),
    ]
    for ms, lab, col in markers:
        if ms is None:
            continue
        axA.axvline(ms, color=col, ls=":", lw=1.8, label=lab)
        idx = int(
            np.clip(np.searchsorted(x_ms.to_numpy(), ms), 0, len(acc) - 1)
        )
        y_here = np.nan_to_num(acc[idx], nan=0.0)
        _annot(axA, ms, y_here, lab, col)

    if title:
        axA.set_title(title)

    lines, labels = axA.get_legend_handles_labels()
    r_lines, r_labels = axD.get_legend_handles_labels()
    seen = set()
    all_lines, all_labels = [], []
    for h, t in list(zip(lines + r_lines, labels + r_labels)):
        if t and t not in seen:
            seen.add(t)
            all_lines.append(h)
            all_labels.append(t)
    axA.legend(all_lines, all_labels, loc="upper right", ncol=2, frameon=False)

    axA.grid(True, alpha=0.28)
    fig.tight_layout()
    # save
    plt.savefig(f"data/renders/event_sync_{event_row.get('eventId')}.png")
    plt.show(block=True)


# =============================================================================
# Multi-freeze video renderer for single goals
# =============================================================================

FIELD_IMAGE: Path = Path("assets/handballfeld.png")
OUT_DIR: Path = Path("data/renders/")
FPS: int = 20
POS_PAD_SEC_BEFORE: float = 2.0
POS_PAD_SEC_AFTER: float = 3.0
TRAIL_FRAMES: int = 8


def render_goal_with_multifreeze(
    df_positions: pd.DataFrame,
    row_goal: pd.Series,
    freeze_markers: List[Dict[str, Any]],  # {"name","ts","color","seconds"}
    field_image_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    fps: Optional[int] = None,
) -> Optional[Path]:
    """
    Render a goal clip and freeze at multiple marker times.

    df_positions must contain at least:
      - "ts in ms" (ms since epoch)
      - "x in m", "y in m"
      - "group id" (3=ball, 2/1=teams)
      - "league id", "full name"
    """
    field_image_path = field_image_path or FIELD_IMAGE
    out_dir = Path(out_dir or OUT_DIR)
    fps = fps or FPS
    frame_interval_ms = int(round(1000 / fps))

    img = cv2.imread(str(field_image_path))
    if img is None:
        print(f"⚠️ Could not read field image at: {field_image_path}")
        return None

    height, width = img.shape[:2]
    scale = width / 40.0

    valid_markers = [m for m in freeze_markers if pd.notna(m.get("ts"))]
    if not valid_markers:
        print("⚠️ No valid freeze markers for this event, skipping.")
        return None

    if "ts" not in df_positions.columns:
        df_positions = df_positions.copy()
        df_positions["ts"] = pd.to_datetime(
            df_positions["ts in ms"], unit="ms", utc=True
        )

    t_min = min(pd.to_datetime(m["ts"]) for m in valid_markers)
    t_max = max(pd.to_datetime(m["ts"]) for m in valid_markers)
    t0 = t_min - pd.Timedelta(seconds=POS_PAD_SEC_BEFORE)
    t1 = t_max + pd.Timedelta(seconds=POS_PAD_SEC_AFTER)

    df_scene = df_positions[
        (df_positions["ts"] >= t0) & (df_positions["ts"] <= t1)
    ].copy()
    if df_scene.empty:
        print(
            "⚠️ No positional data in window for eventId:",
            row_goal.get("eventId"),
        )
        return None
    df_scene = df_scene.sort_values("ts").copy()

    df_scene["frame_idx"] = df_scene.groupby("ts").ngroup()
    df_scene["prev_x"] = df_scene.groupby(["mapped id"])["x in m"].shift(1)
    df_scene["prev_y"] = df_scene.groupby(["mapped id"])["y in m"].shift(1)

    name_event = f'event_{row_goal.get("eventId")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name_event}.mp4"
    out_path_img = out_dir / f"{name_event}.png"

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for m in valid_markers:
        ts_val = pd.to_datetime(m["ts"], utc=True)
        m["ms"] = int(ts_val.value // 10**6)
        m["frames"] = int(round(float(m.get("seconds", 0)) * fps))
        m["done"] = False

    shooter_league_id = row_goal.get("person_league_id", None)
    if shooter_league_id is None:
        shooter_league_id = row_goal.get("kin_league_id", None)
    goalkeeper_league_id = row_goal.get("goalkeeper_league_id", None)

    def color_for_group(group_id: float):
        if group_id == 3:
            return (0, 0, 255), 15
        if group_id == 2:
            return (0, 200, 0), 10
        if group_id == 1:
            return (255, 140, 0), 10
        return (200, 200, 200), 8

    def draw_overlay(img_draw, ts_val: pd.Timestamp):
        hud_lines = [
            f"Clock: {row_goal.get('kin_game_clock', '')}Team: {row_goal.get('teamName', '')}",
            f"EventType: {row_goal.get('subType', '')} AttackType: {row_goal.get('attackType', '')}, failureReason: {row_goal.get('failureReason', '')}, sucess: {row_goal.get('kin_success', '')}",
            f"Shooter Name: {row_goal.get('personName', 'N/A')}  |  "
            f"GK Name: {row_goal.get('goalkeeperName', 'N/A')}",
            f"Frame time (UTC): {ts_val.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}",
            "Markers: " + ", ".join([m["name"] for m in valid_markers]),
        ]
        y0 = 28
        for line in hud_lines:
            cv2.putText(
                img_draw,
                line,
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y0 += 24

    first_png_saved = False

    for ts, group in df_scene.groupby("ts"):
        img_draw = img.copy()

        for _, r in group.iterrows():
            gid = r.get("group id", None)
            color, radius = color_for_group(gid)
            if pd.isna(r["x in m"]) or pd.isna(r["y in m"]):
                continue
            x = int(float(r["x in m"]) * scale)
            y = int(float(r["y in m"]) * scale)
            cv2.circle(
                img_draw, (x, y), radius, color, -1, lineType=cv2.LINE_AA
            )

            league_id = r.get("league id", "N/A")
            name = r.get("full name", "N/A")
            cv2.putText(
                img_draw,
                f"{name}",
                (x - 12, y - radius - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            # cv2.putText(
            #     img_draw,
            #     f"ID:{league_id}",
            #     (x - 12, y - radius - 2),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.40,
            #     (255, 255, 255),
            #     1,
            #     cv2.LINE_AA,
            # )
            try:
                if shooter_league_id is not None and int(
                    r.get("league id", -1)
                ) == int(shooter_league_id):
                    cv2.circle(
                        img_draw,
                        (x, y),
                        radius + 6,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                if goalkeeper_league_id is not None and int(
                    r.get("league id", -1)
                ) == int(goalkeeper_league_id):
                    cv2.circle(
                        img_draw,
                        (x, y),
                        radius + 6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            except Exception:
                pass

        current_idx = group["frame_idx"].iloc[0]
        if TRAIL_FRAMES > 0:
            trail_slice = df_scene[
                (df_scene["frame_idx"] <= current_idx)
                & (df_scene["frame_idx"] > current_idx - TRAIL_FRAMES)
            ]
            for _, r in trail_slice.iterrows():
                if pd.isna(r["x in m"]) or pd.isna(r["y in m"]):
                    continue
                x_t = int(float(r["x in m"]) * scale)
                y_t = int(float(r["y in m"]) * scale)
                gid_t = r.get("group id", None)
                c_t, _ = color_for_group(gid_t)
                cv2.circle(img_draw, (x_t, y_t), 3, c_t, -1, cv2.LINE_AA)

        for _, br in group[group["group id"] == 3].iterrows():
            if not (pd.isna(br["prev_x"]) or pd.isna(br["prev_y"])):
                x0 = int(br["prev_x"] * scale)
                y0 = int(br["prev_y"] * scale)
                x1 = int(br["x in m"] * scale)
                y1 = int(br["y in m"] * scale)
                cv2.arrowedLine(
                    img_draw,
                    (x0, y0),
                    (x1, y1),
                    (50, 50, 255),
                    2,
                    tipLength=0.3,
                )

        draw_overlay(img_draw, ts)

        ts_ms = int(ts.value // 10**6)
        any_frozen = False
        for m in valid_markers:
            if m["done"]:
                continue
            if abs(ts_ms - m["ms"]) <= (frame_interval_ms // 2):
                cv2.rectangle(
                    img_draw,
                    (0, 0),
                    (width - 1, height - 1),
                    m["color"],
                    6,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img_draw,
                    m["name"],
                    (width // 2 - 160, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    m["color"],
                    2,
                    cv2.LINE_AA,
                )

                if not first_png_saved:
                    cv2.imwrite(str(out_path_img), img_draw)
                    first_png_saved = True

                for _ in range(m["frames"]):
                    cv2.imshow("Render", img_draw)
                    cv2.waitKey(2)
                    writer.write(cv2.resize(img_draw, (width, height)))
                m["done"] = True
                any_frozen = True
        if any_frozen:
            continue

        if not first_png_saved:
            cv2.imwrite(str(out_path_img), img_draw)
            first_png_saved = True

        cv2.imshow("Render", img_draw)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit early
            print("⏹️ Rendering aborted by user.")
            break
        writer.write(cv2.resize(img_draw, (width, height)))

    writer.release()
    print(f"🎬 Saved: {out_path}  | 🖼️ Preview: {out_path_img}")
    return out_path


# =============================================================================
# Per-event refinement API (used from viz.py / notebooks)
# =============================================================================


def refine_throw_time_for_event(
    row: pd.Series,
    df_positions_all: pd.DataFrame,
    fixture_to_session: Dict[Any, Any],
    *,
    plot: bool = False,
    plot_title: Optional[str] = None,
) -> pd.Series:
    """
    Refine throw time for a single goal event.

    Inputs:
      - row:         one row from sportradar_goals_synced (or similar)
      - df_positions_all: full kinexon_positions table (with ts + session_id)
      - fixture_to_session: mapping fixtureId -> session_id
      - plot:        if True, renders a plot_event_sync() diagnostic

    Returns:
      pd.Series with diagnostics and refined_throw_ts(_ms), deltas, method, etc.
    """
    fixture_id = row.get("fixtureId")
    session_id = fixture_to_session.get(fixture_id)
    seed_ts = seed_time_from_row(row)

    # shooter league-id: prefer 'person_league_id', else 'kin_league_id'
    shooter_league_id = None
    if "person_league_id" in row and pd.notna(row.get("person_league_id")):
        shooter_league_id = row["person_league_id"]
    elif "kin_league_id" in row and pd.notna(row.get("kin_league_id")):
        shooter_league_id = row["kin_league_id"]

    diag = dict(
        fixtureId=fixture_id,
        eventId=row.get("eventId"),
        seed_ts=seed_ts,
        seed_ms=(seed_ts.value // 10**6) if pd.notna(seed_ts) else None,
        eventTime=row.get("eventTime"),
        eventTime_ms=(
            row.get("eventTime_ms") if "eventTime_ms" in row else None
        ),
        kin_timestamp=None,
        kin_timestamp_ms=None,
        refined_throw_ts=None,
        refined_throw_ts_ms=None,
        refined_delta_vs_event_ms=None,
        refined_delta_vs_seed_ms=None,
        method=None,
        n_rows_window=None,
        last_possession_idx=None,
    )

    if (
        session_id is None
        or pd.isna(shooter_league_id)
        or pd.isna(seed_ts)
        or "session_id" not in df_positions_all.columns
    ):
        return pd.Series(diag)

    df_scene = extract_tracks_window(
        df_positions_all,
        session_id=session_id,
        center_ts=seed_ts,
        w_before_s=WINDOW_BEFORE_S,
        w_after_s=WINDOW_AFTER_S,
    )
    diag["n_rows_window"] = int(len(df_scene))

    # print(
    #     f"Refining throw time for eventId={row.get('teamName')} in fixtureId={row.get('personName')} with {len(df_scene)} position rows"
    # )
    # print if kin_timestamp_ms is available and if it is in the timespan of df_scene
    print_kin_ts = False
    if "kin_timestamp_ms" in row and pd.notna(row.get("kin_timestamp_ms")):
        kin_ts = pd.to_datetime(
            int(row["kin_timestamp_ms"]), unit="ms", utc=True, errors="coerce"
        )
        diag["kin_timestamp"] = kin_ts
        diag["kin_timestamp_ms"] = int(row["kin_timestamp_ms"])
        if not df_scene.empty:
            t_min = df_scene["ts"].min()
            t_max = df_scene["ts"].max()
            if kin_ts >= t_min and kin_ts <= t_max:
                print_kin_ts = True
    if print_kin_ts and False:
        print(
            f"  Kinexon timestamp {diag['kin_timestamp']} is within position data range [{df_scene['ts'].min()} .. {df_scene['ts'].max()}], Possession window is {D_POSSESS} m, max speed {V_POSSESS_MAX} m/s"
        )
    if df_scene.empty:
        return pd.Series(diag)

    df_pb = build_joined_player_ball(df_scene, shooter_league_id)
    if df_pb.empty:
        return pd.Series(diag)

    df_pb = df_pb.reset_index(drop=True)

    result = detect_throw_point(
        df_pb,
        d_possess=D_POSSESS,
        v_possess_max=V_POSSESS_MAX,
        v_release_min=V_RELEASE_MIN,
        d_release_min=D_RELEASE_MIN,
        acc_peak_neigh=ACC_PEAK_NEIGH,
    )

    diag["last_possession_idx"] = result.get("last_possession_idx")
    throw_ts = result.get("throw_ts")
    if throw_ts is not None:
        t = pd.to_datetime(throw_ts, utc=True)
        diag["refined_throw_ts"] = t
        diag["refined_throw_ts_ms"] = int(t.value // 10**6)
        diag["method"] = result.get("method")

        if diag.get("eventTime_ms") is not None:
            try:
                diag["refined_delta_vs_event_ms"] = int(
                    diag["refined_throw_ts_ms"]
                    - int(diag["eventTime_ms"])  # type: ignore[arg-type]
                )
            except Exception:
                pass

        if diag.get("seed_ms") is not None:
            try:
                diag["refined_delta_vs_seed_ms"] = int(
                    diag["refined_throw_ts_ms"]
                    - int(diag["seed_ms"])  # type: ignore[arg-type]
                )
            except Exception:
                pass

    if plot:
        event_ms = None
        if diag.get("eventTime_ms") is not None:
            try:
                event_ms = int(diag["eventTime_ms"])  # type: ignore[arg-type]
            except Exception:
                event_ms = None

        kin_ms = None
        if diag.get("kin_timestamp_ms") is not None:
            try:
                kin_ms = int(diag["kin_timestamp_ms"])  # type: ignore[arg-type]
            except Exception:
                kin_ms = None

        throw_ms = None
        if diag.get("refined_throw_ts_ms") is not None:
            throw_ms = int(diag["refined_throw_ts_ms"])  # type: ignore[arg-type]

        title = (
            plot_title
            or f"Event sync — fixture {fixture_id}, player {row.get('person')}"
        )
        plot_event_sync(
            df_pb=df_pb,
            last_possession_idx=diag["last_possession_idx"],
            event_ms=event_ms,
            kin_ms=kin_ms,
            throw_ms=throw_ms,
            title=title,
            event_row=row,
        )

    return pd.Series(diag)
