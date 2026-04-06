# utils/goal_rendering.py

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency for rendering
    cv2 = None
import numpy as np
import pandas as pd

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
    show: bool = False,
    sync_panel: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Render a goal clip and freeze at multiple marker times.

    df_positions must contain at least:
      - "ts in ms" (ms since epoch)
      - "x in m", "y in m"
      - "group id" (3=ball, 2/1=teams)
      - "league id", "full name"
    """
    if cv2 is None:
        raise ImportError(
            "opencv-python is required for goal rendering; install system libs for cv2"
        )

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
            df_positions["timestamp_ms"], unit="ms", utc=True
        )

    t_min = min(
        pd.to_datetime(m["ts"], utc=True, errors="coerce") for m in valid_markers
    )
    t_max = max(
        pd.to_datetime(m["ts"], utc=True, errors="coerce") for m in valid_markers
    )
    t0 = t_min - pd.Timedelta(seconds=POS_PAD_SEC_BEFORE)
    t1 = t_max + pd.Timedelta(seconds=POS_PAD_SEC_AFTER)

    df_scene = df_positions[
        (df_positions["ts"] >= t0) & (df_positions["ts"] <= t1)
    ].copy()
    event_id_for_print = row_goal.get("event_id")
    if df_scene.empty:
        print(
            "⚠️ No positional data in window for eventId:",
            event_id_for_print,
        )
        return None
    df_scene = df_scene.sort_values("ts").copy()

    df_scene["frame_idx"] = df_scene.groupby("ts").ngroup()
    df_scene["prev_x"] = df_scene.groupby(["mapped_id"])["x_m"].shift(1)
    df_scene["prev_y"] = df_scene.groupby(["mapped_id"])["y_m"].shift(1)

    event_id_for_name = (
        event_id_for_print if event_id_for_print is not None else "unknown"
    )
    name_event = f"event_{event_id_for_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name_event}.mp4"
    out_path_img = out_dir / f"{name_event}.png"

    def _ts_to_ms(series_or_ts: Any) -> Optional[pd.Series | int]:
        s = pd.to_datetime(series_or_ts, utc=True, errors="coerce")
        if isinstance(s, pd.Series):
            return s.astype("datetime64[ms, UTC]").astype("int64")
        return None if pd.isna(s) else int(s.value // 10**6)

    def _init_sync_panel(
        panel_data: Dict[str, Any],
        *,
        panel_height_px: int,
    ) -> Optional[Dict[str, Any]]:
        df_pb = panel_data.get("df_pb")
        if df_pb is None or df_pb.empty:
            return None

        df_pb = df_pb.copy()
        if "timestamp_ms" not in df_pb.columns:
            df_pb["timestamp_ms"] = _ts_to_ms(df_pb.get("ts"))
        df_pb["timestamp_ms"] = pd.to_numeric(df_pb["timestamp_ms"], errors="coerce")
        df_pb = df_pb.dropna(subset=["timestamp_ms"]).copy()
        if df_pb.empty:
            return None

        x_ms_abs = df_pb["timestamp_ms"].astype(np.int64).to_numpy()
        origin_ms = int(panel_data.get("origin_ms") or x_ms_abs[0])
        range_ms = panel_data.get("range_ms")
        x_ms = x_ms_abs - origin_ms
        acc = df_pb.get("ball_acc", pd.Series([np.nan] * len(df_pb))).to_numpy(
            dtype=float
        )
        dist = df_pb.get("dist_pb", pd.Series([np.nan] * len(df_pb))).to_numpy(
            dtype=float
        )
        speed = df_pb.get("ball_speed", pd.Series([np.nan] * len(df_pb))).to_numpy(
            dtype=float
        )

        panel_width_px = panel_data.get("panel_width_px")
        if not isinstance(panel_width_px, int) or panel_width_px <= 0:
            panel_width_px = int(panel_height_px * 1.2)

        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure

        dpi = 100
        fig = Figure(
            figsize=(panel_width_px / dpi, panel_height_px / dpi),
            dpi=dpi,
        )
        canvas = FigureCanvas(fig)
        axA = fig.add_subplot(111)
        axA.plot(x_ms, acc, lw=1.4, label="Ball acc")
        axA.axhline(0, lw=0.8, color="0.7")
        axA.set_xlabel("Time (ms from window start)")
        axA.set_ylabel("Ball acc (m/s^2)")

        axD = axA.twinx()
        axD.plot(x_ms, dist, ls="--", lw=1.2, label="Player-ball dist")
        if np.isfinite(speed).any():
            axD.plot(x_ms, speed, ls=":", lw=0.9, alpha=0.6, label="Ball speed")
        axD.set_ylabel("Distance / Speed")

        d_possess = panel_data.get("d_possess")
        if d_possess is not None:
            poss_mask = dist <= float(d_possess)
            if poss_mask.any():
                ymax = np.nanmax(dist) if np.isfinite(dist).any() else 1.0
                axD.fill_between(
                    x_ms,
                    0,
                    ymax,
                    where=poss_mask,
                    alpha=0.18,
                    color="#6aa84f",
                    label="Possession",
                )

        markers = [
            (panel_data.get("event_ms"), "SR event", "tab:blue"),
            (panel_data.get("kin_ms"), "Detected shot", "tab:orange"),
            (panel_data.get("throw_ms"), "Throw / release", "tab:red"),
        ]
        for ms, label, color in markers:
            if ms is None:
                continue
            axA.axvline(ms - origin_ms, color=color, ls=":", lw=1.1, label=label)

        if isinstance(range_ms, (int, float)) and range_ms > 0:
            axA.set_xlim(0, range_ms)

        time_line = axA.axvline(0, color="#d62728", lw=2.0, alpha=0.9)

        lines, labels = axA.get_legend_handles_labels()
        r_lines, r_labels = axD.get_legend_handles_labels()
        axA.legend(lines + r_lines, labels + r_labels, loc="upper right", frameon=False)
        axA.grid(True, alpha=0.2)
        fig.tight_layout()

        return {
            "canvas": canvas,
            "figure": fig,
            "time_line": time_line,
            "origin_ms": origin_ms,
            "range_ms": range_ms,
            "width": panel_width_px,
            "height": panel_height_px,
        }

    def _render_sync_panel(
        state: Dict[str, Any],
        *,
        ts_ms: int,
    ) -> np.ndarray:
        rel_ms = ts_ms - state.get("origin_ms", 0)
        range_ms = state.get("range_ms")
        if isinstance(range_ms, (int, float)) and range_ms > 0:
            rel_ms = min(max(rel_ms, 0), range_ms)
        state["time_line"].set_xdata([rel_ms, rel_ms])
        canvas = state["canvas"]
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

    panel_state = None
    if sync_panel:
        try:
            panel_state = _init_sync_panel(sync_panel, panel_height_px=height)
        except Exception as exc:
            print(f"⚠️ Sync panel init failed: {exc}")
            panel_state = None

    output_width = width + (panel_state["width"] if panel_state else 0)
    output_height = max(height, panel_state["height"]) if panel_state else height

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )

    for m in valid_markers:
        ts_val = pd.to_datetime(m["ts"], utc=True)
        m["ms"] = int(ts_val.value // 10**6)
        m["frames"] = int(round(float(m.get("seconds", 0)) * fps))
        m["done"] = False

    shooter_league_id = row_goal.get("person_league_id", None)
    assisting_league_id = row_goal.get("assisting_league_id", None)
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
        def ms_delta(now: pd.Timestamp, then: str) -> int:
            """Return non-negative delta in milliseconds."""
            delta_ms = (now - pd.to_datetime(then, utc=True)).total_seconds() * 1000
            return int(delta_ms)

        is_home = row_goal.get("entity_id_home", None) == row_goal.get(
            "entity_id", None
        )
        team_side = "Home" if is_home else "Away"

        team_name = row_goal.get("team_name_offense", "N/A")

        # scores look like '{"fe7bdd16-3952-11ef-b585-af5c55c3771d": 1, "fe9e91c7-3952-11ef-bd62-af5c55c3771d": 0}'

        hud_lines = [
            (
                f"Clock: {row_goal.get('clock', '')} | {score_str} | "
                f"Team: {team_name} (Side: {team_side})"
            ),
            (
                f"EventType: {row_goal.get('sub_type', '')} | "
                f"AttackType: {row_goal.get('attack_type', '')} | "
                f"FailureReason: {row_goal.get('failure_reason', '')} | "
                f"Success: {row_goal.get('success', '')}"
            ),
            (
                f"Shooter: {row_goal.get('person_name', 'N/A')} | "
                f"GK: {row_goal.get('goalkeeper_name', 'N/A')}"
            ),
            f"Frame time (UTC): {ts_val.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}",
            "Markers: " + ", ".join(m["name"] for m in valid_markers),
            "Freeze in: "
            + ", ".join(
                f"{m['name']} ({ms_delta(ts_val, m['ts'])} ms)" for m in valid_markers
            ),
            (
                f"Event end: {row_goal.get('event_time_ms', '')} ms | "
                f"{pd.to_datetime(row_goal.get('event_time_ms', 0), unit='ms', utc=True)}"
            ),
        ]

        y0 = 28
        for line in hud_lines:
            cv2.putText(
                img_draw,
                line,
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y0 += 20

    first_png_saved = False

    scores_dict = {}
    try:
        scores_dict = eval(row_goal.get("scores", "{}"))
    except Exception:
        pass
    score_home = scores_dict.get(row_goal["entity_id_home"], 0)
    score_away = scores_dict.get(row_goal["entity_id_away"], 0)
    score_str = f"Score: {score_home} - {score_away}"

    for ts, group in df_scene.groupby("ts"):
        img_draw = img.copy()

        for _, r in group.iterrows():
            gid = r.get("group_id", None)
            color, radius = color_for_group(gid)
            if pd.isna(r["x_m"]) or pd.isna(r["y_m"]):
                continue
            x = int(float(r["x_m"]) * scale)
            y = int(float(r["y_m"]) * scale)
            cv2.circle(img_draw, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

            name = r.get("full_name", "N/A")
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

            try:
                if shooter_league_id is not None and int(r.get("league_id", -1)) == int(
                    shooter_league_id
                ):
                    cv2.circle(
                        img_draw,
                        (x, y),
                        radius + 6,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                if goalkeeper_league_id is not None and int(
                    r.get("league_id", -1)
                ) == int(goalkeeper_league_id):
                    cv2.circle(
                        img_draw,
                        (x, y),
                        radius + 6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                if assisting_league_id is not None and int(
                    r.get("league_id", -1)
                ) == int(assisting_league_id):
                    cv2.circle(
                        img_draw,
                        (x, y),
                        radius + 6,
                        (0, 255, 0),
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
                if pd.isna(r["x_m"]) or pd.isna(r["y_m"]):
                    continue
                x_t = int(float(r["x_m"]) * scale)
                y_t = int(float(r["y_m"]) * scale)
                gid_t = r.get("group_id", None)
                c_t, _ = color_for_group(gid_t)
                cv2.circle(img_draw, (x_t, y_t), 3, c_t, -1, cv2.LINE_AA)

        for _, br in group[group["group_id"] == 3].iterrows():
            if not (
                pd.isna(br["prev_x"])
                or pd.isna(br["prev_y"])
                or pd.isna(br["x_m"])
                or pd.isna(br["y_m"])
            ):  # ball
                x0 = int(br["prev_x"] * scale)
                y0 = int(br["prev_y"] * scale)
                x1 = int(br["x_m"] * scale)
                y1 = int(br["y_m"] * scale)
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
        panel_frame = (
            _render_sync_panel(panel_state, ts_ms=ts_ms) if panel_state else None
        )
        if panel_frame is not None:
            if img_draw.shape[0] != output_height:
                img_draw = cv2.copyMakeBorder(
                    img_draw,
                    0,
                    output_height - img_draw.shape[0],
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                )
            if panel_frame.shape[0] != output_height:
                panel_frame = cv2.copyMakeBorder(
                    panel_frame,
                    0,
                    output_height - panel_frame.shape[0],
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                )
            frame_out = np.hstack([img_draw, panel_frame])
        else:
            frame_out = img_draw
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
                    cv2.imwrite(str(out_path_img), frame_out)
                    first_png_saved = True

                for _ in range(m["frames"]):
                    if show:
                        cv2.imshow("Render", frame_out)
                        cv2.waitKey(10)
                    writer.write(frame_out)
                m["done"] = True
                any_frozen = True
        if any_frozen:
            continue

        if not first_png_saved:
            cv2.imwrite(str(out_path_img), frame_out)
            first_png_saved = True
        if show:
            cv2.imshow("Render", frame_out)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                print("⏹️ Rendering aborted by user.")
                break
        writer.write(frame_out)

    writer.release()
    print(f"🎬 Saved: {out_path}  | 🖼️ Preview: {out_path_img}")
    return out_path
