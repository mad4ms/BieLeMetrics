# utils/goal_rendering.py

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
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
    event_id_for_print = row_goal.get("event_id")
    if df_scene.empty:
        print(
            "⚠️ No positional data in window for eventId:",
            event_id_for_print,
        )
        return None
    df_scene = df_scene.sort_values("ts").copy()

    df_scene["frame_idx"] = df_scene.groupby("ts").ngroup()
    df_scene["prev_x"] = df_scene.groupby(["mapped id"])["x in m"].shift(1)
    df_scene["prev_y"] = df_scene.groupby(["mapped id"])["y in m"].shift(1)

    event_id_for_name = (
        event_id_for_print if event_id_for_print is not None else "unknown"
    )
    name_event = f"event_{event_id_for_name}"
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
            f"Clock: {row_goal.get('kin_game_clock', '')} Team: {row_goal.get('team_name', '')} (is_home: {row_goal.get('is_team_home', '')})",
            f"EventType: {row_goal.get('sub_type', '')} AttackType: {row_goal.get('attack_type', '')}, failureReason: {row_goal.get('failure_reason', '')}, sucess: {row_goal.get('kin_success', '')}",
            f"Shooter Name: {row_goal.get('person_name', 'N/A')}  |  "
            f"GK Name: {row_goal.get('goalkeeper_name', 'N/A')}",
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
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y0 += 20

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
            if not (pd.isna(br["prev_x"]) or pd.isna(br["prev_y"])):  # ball
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
        if key == 27:  # ESC
            print("⏹️ Rendering aborted by user.")
            break
        writer.write(cv2.resize(img_draw, (width, height)))

    writer.release()
    print(f"🎬 Saved: {out_path}  | 🖼️ Preview: {out_path_img}")
    return out_path
