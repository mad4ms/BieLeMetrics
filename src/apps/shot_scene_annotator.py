from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import duckdb
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.image as mpimg
from tkinter import BOTH, END, HORIZONTAL, LEFT, RIGHT, VERTICAL, W, X, Y
import tkinter as tk
from tkinter import messagebox

import ttkbootstrap as ttk

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.synced.shot_events import _build_player_ball_timeline  # noqa: E402


DB_PATH = REPO_ROOT / "data" / "hbl_raw.duckdb"
ANNOTATION_PATH = (
    REPO_ROOT / "artifacts" / "manual_annotations" / "shot_scene_annotations.csv"
)
COURT_IMAGE_PATH = REPO_ROOT / "assets" / "handballfeld.png"

SCENE_PAD_BEFORE_MS = 12_000
SCENE_PAD_AFTER_MS = 4_000
COURT_LENGTH_M = 40.0
COURT_WIDTH_M = 20.0
GOAL_HALF_WIDTH_M = 1.5
PLAY_TIMER_MS = 80
UI_BG = "#eef4f8"
PANEL_BG = "#ffffff"
PANEL_ALT_BG = "#f6f9fc"
SIDEBAR_BG = "#122736"
SIDEBAR_PANEL_BG = "#173547"
ACCENT = "#2d7ff9"
ACCENT_ALT = "#ff8a3d"
TEXT_DARK = "#12212d"
TEXT_MUTED = "#5b7083"
BORDER = "#dbe5ee"


@dataclass(slots=True)
class SceneEvent:
    fixture_id: str
    event_id: str
    event_time_ms: int | None
    outcome_label: str
    detected_shot_ms: int | None
    throw_timestamp_ms: int | None
    person_name: str
    goalkeeper_name: str
    team_name_offense: str
    team_name_defense: str
    entity_id_home: str | None
    entity_id_away: str | None
    goal_position: float | None
    success: bool | None
    match_method: str
    method: str
    score_text: str

    @property
    def anchor_ms(self) -> int | None:
        for candidate in (
            self.throw_timestamp_ms,
            self.detected_shot_ms,
            self.event_time_ms,
        ):
            if candidate is not None:
                return candidate
        return None


def _to_ms(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return None if np.isnan(value) else int(value)
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return int(timestamp.value // 10**6)


def _fmt_ms(ms_value: int | None) -> str:
    if ms_value is None:
        return "-"
    timestamp = pd.to_datetime(ms_value, unit="ms", utc=True)
    return timestamp.strftime("%H:%M:%S.%f")[:-3]


def _fmt_delta(delta_ms: int | None) -> str:
    if delta_ms is None:
        return "-"
    sign = "+" if delta_ms >= 0 else "-"
    return f"{sign}{abs(delta_ms) / 1000:.3f}s"


def _format_score_text(
    scores_raw: Any,
    entity_id_home: str | None,
    entity_id_away: str | None,
) -> str:
    if (
        scores_raw is None
        or pd.isna(scores_raw)
        or entity_id_home is None
        or entity_id_away is None
    ):
        return "(-:-)"

    try:
        if isinstance(scores_raw, str):
            score_map = json.loads(scores_raw)
        elif isinstance(scores_raw, dict):
            score_map = scores_raw
        else:
            return "(-:-)"
    except Exception:
        return "(-:-)"

    home_score = score_map.get(entity_id_home)
    away_score = score_map.get(entity_id_away)
    if home_score is None or away_score is None:
        return "(-:-)"
    return f"({int(home_score)}:{int(away_score)})"


def _format_outcome_label(success: bool | None, failure_reason: Any) -> str:
    if success is True:
        return "GOAL"
    if failure_reason is None or pd.isna(failure_reason):
        return "UNKNOWN"
    return str(failure_reason)


def _format_shooter_label(person_name: str, team_name_offense: str) -> str:
    if not team_name_offense or team_name_offense == "Unknown":
        return person_name
    return f"{person_name} ({team_name_offense})"


class ShotSceneAnnotator:
    def __init__(self, root: tk.Misc, db_path: Path, annotation_path: Path):
        self.root = root
        self.db_path = db_path
        self.annotation_path = annotation_path
        self.connection = duckdb.connect(str(db_path), read_only=True)

        self.fixture_events: dict[str, list[SceneEvent]] = {}
        self.fixture_options: list[tuple[str, str]] = []
        self.selected_fixture_id: str | None = None
        self.selected_event: SceneEvent | None = None
        self.scene_positions = pd.DataFrame()
        self.scene_timestamps: list[int] = []
        self.current_frame_index = 0
        self.current_scene_start_ms: int | None = None
        self.current_scene_end_ms: int | None = None
        self.play_job: str | None = None
        self.current_shooter_league_id: str | None = None
        self.current_goalkeeper_league_id: str | None = None
        self.player_ball_timeline = pd.DataFrame()
        self.scene_detected_events = pd.DataFrame()
        self.court_image = self._load_court_image()
        self.marker_accel_axis = None
        self.marker_current_line = None
        self.marker_current_line_right = None

        self.annotations = self._load_annotations()

        self.root.title("Shot Scene Annotator")
        self.root.geometry("1760x1040")
        self.root.minsize(1440, 900)
        self.root.configure(bg=UI_BG)  # type: ignore[call-arg]

        self.status_var = tk.StringVar(value="Select a fixture to begin.")
        self.fixture_var = tk.StringVar()
        self.scene_info_var = tk.StringVar(value="No scene loaded.")
        self.frame_info_var = tk.StringVar(value="Frame: -")
        self.manual_throw_var = tk.StringVar(value="Manual throw: -")
        self.annotation_status_var = tk.StringVar(value="unreviewed")

        self._configure_styles()
        self._build_ui()
        self.root.bind("<Left>", self._handle_prev_key)
        self.root.bind("<Right>", self._handle_next_key)
        self.root.bind("<space>", self._handle_toggle_play_key)
        self.root.bind("m", self._handle_mark_manual_key)
        self.root.bind("s", self._handle_save_key)
        self.root.bind("e", self._handle_jump_event_key)
        self.root.bind("d", self._handle_jump_detected_key)
        self.root.bind("a", self._handle_jump_synced_key)
        self.root.bind("c", self._handle_clear_manual_key)
        self._load_fixture_options()

    def _load_court_image(self) -> np.ndarray | None:
        if not COURT_IMAGE_PATH.exists():
            return None
        try:
            return mpimg.imread(COURT_IMAGE_PATH)
        except Exception:
            return None

    def _configure_styles(self) -> None:
        style = ttk.Style("flatly")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("App.TFrame", background=UI_BG)
        style.configure("Hero.TFrame", background=TEXT_DARK)
        style.configure(
            "HeroTitle.TLabel",
            background=TEXT_DARK,
            foreground="#f7fbff",
            font=("Segoe UI Semibold", 21),
        )
        style.configure(
            "HeroSubtitle.TLabel",
            background=TEXT_DARK,
            foreground="#bfd0dd",
            font=("Segoe UI", 10),
        )
        style.configure(
            "HeroChip.TLabel",
            background="#1d394c",
            foreground="#f7fbff",
            font=("Segoe UI Semibold", 9),
            padding=(10, 5),
        )
        style.configure("Sidebar.TFrame", background=SIDEBAR_BG)
        style.configure("SidebarInner.TFrame", background=SIDEBAR_PANEL_BG)
        style.configure(
            "SidebarCard.TLabelframe",
            background=SIDEBAR_PANEL_BG,
            borderwidth=0,
            relief="flat",
        )
        style.configure(
            "SidebarCard.TLabelframe.Label",
            background=SIDEBAR_PANEL_BG,
            foreground="#f7fbff",
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "Card.TLabelframe",
            background=PANEL_BG,
            borderwidth=1,
            relief="solid",
            bordercolor=BORDER,
        )
        style.configure("CardInner.TFrame", background=PANEL_BG)
        style.configure(
            "Card.TLabelframe.Label",
            background=PANEL_BG,
            foreground=TEXT_DARK,
            font=("Segoe UI Semibold", 10),
        )
        style.configure("App.TLabel", background=PANEL_BG, foreground=TEXT_DARK)
        style.configure(
            "Meta.TLabel",
            background=PANEL_BG,
            foreground=TEXT_MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "SidebarTitle.TLabel",
            background=SIDEBAR_BG,
            foreground="#f7fbff",
            font=("Segoe UI Semibold", 14),
        )
        style.configure(
            "SidebarMuted.TLabel",
            background=SIDEBAR_PANEL_BG,
            foreground="#c2d2dd",
            font=("Segoe UI", 9),
        )
        style.configure(
            "Status.TLabel",
            background=TEXT_DARK,
            foreground="#ffffff",
            padding=(12, 8),
            font=("Segoe UI Semibold", 9),
        )
        style.configure(
            "Annotator.Treeview",
            rowheight=34,
            fieldbackground=PANEL_BG,
            background=PANEL_BG,
            foreground=TEXT_DARK,
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        style.map(
            "Annotator.Treeview",
            background=[("selected", ACCENT)],
            foreground=[("selected", "#ffffff")],
        )
        style.configure(
            "Annotator.Treeview.Heading",
            background=PANEL_ALT_BG,
            foreground=TEXT_DARK,
            relief="flat",
            font=("Segoe UI Semibold", 9),
        )
        style.map("Annotator.Treeview.Heading", background=[("active", "#e9f1f8")])
        style.configure("Modern.TCombobox", padding=8)
        style.configure(
            "Timeline.Horizontal.TScale", background=PANEL_BG, troughcolor="#d9e6f2"
        )

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=(20, 18, 20, 16), style="App.TFrame")
        main.pack(fill=BOTH, expand=True)

        hero = ttk.Frame(main, padding=(24, 20), style="Hero.TFrame")
        hero.pack(fill=X, pady=(0, 16))
        hero.columnconfigure(0, weight=1)

        hero_text = ttk.Frame(hero, style="Hero.TFrame")
        hero_text.grid(row=0, column=0, sticky="w")
        ttk.Label(
            hero_text, text="Shot Scene Annotator", style="HeroTitle.TLabel"
        ).pack(anchor=W)
        ttk.Label(
            hero_text,
            text="Manual review of shot scenes, Kinexon detections, and release timing.",
            style="HeroSubtitle.TLabel",
        ).pack(anchor=W, pady=(4, 0))

        hero_chips = ttk.Frame(hero, style="Hero.TFrame")
        hero_chips.grid(row=0, column=1, sticky="e")
        for text in [
            "← →: Frame",
            "Space: Play/Pause",
            "M: Mark",
            "E: → Event",
            "D: → Detected",
            "A: → Auto",
            "C: Clear",
            "S: Save",
        ]:
            ttk.Label(hero_chips, text=text, style="HeroChip.TLabel").pack(
                side=LEFT, padx=(6, 0)
            )

        outer = ttk.Panedwindow(main, orient=HORIZONTAL)
        outer.pack(fill=BOTH, expand=True)

        left_frame = ttk.Frame(outer, padding=(0, 0, 16, 0), style="Sidebar.TFrame")
        right_frame = ttk.Frame(outer, style="App.TFrame")
        outer.add(left_frame, weight=1)
        outer.add(right_frame, weight=3)

        self._build_left_panel(left_frame)
        self._build_right_panel(right_frame)

        status_bar = ttk.Label(
            main,
            textvariable=self.status_var,
            anchor=W,
            style="Status.TLabel",
        )
        status_bar.pack(fill=X, pady=(14, 0))

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        sidebar_shell = ttk.Frame(parent, padding=18, style="Sidebar.TFrame")
        sidebar_shell.pack(fill=BOTH, expand=True)

        ttk.Label(
            sidebar_shell, text="Match Browser", style="SidebarTitle.TLabel"
        ).pack(anchor=W)

        fixture_box = ttk.Labelframe(
            sidebar_shell, text="Fixture", style="SidebarCard.TLabelframe"
        )
        fixture_box.pack(fill=X, pady=(14, 12))
        fixture_inner = ttk.Frame(fixture_box, padding=14, style="SidebarInner.TFrame")
        fixture_inner.pack(fill=X, expand=True)
        ttk.Label(
            fixture_inner,
            text="Select a fixture to inspect synchronized goal scenes.",
            style="SidebarMuted.TLabel",
        ).pack(anchor=W, pady=(0, 10))

        self.fixture_combo = ttk.Combobox(
            fixture_inner,
            textvariable=self.fixture_var,
            state="readonly",
            style="Modern.TCombobox",
            bootstyle="light",
        )
        self.fixture_combo.pack(fill=X)
        self.fixture_combo.bind("<<ComboboxSelected>>", self._on_fixture_selected)

        event_box = ttk.Labelframe(
            sidebar_shell, text="Goal Events", style="SidebarCard.TLabelframe"
        )
        event_box.pack(fill=BOTH, expand=True)
        event_inner = ttk.Frame(event_box, padding=14, style="SidebarInner.TFrame")
        event_inner.pack(fill=BOTH, expand=True)
        ttk.Label(
            event_inner,
            text="Outcome, score, shooter, match-sync and throw timing for each shot event.",
            style="SidebarMuted.TLabel",
        ).pack(anchor=W, pady=(0, 10))

        columns = (
            "event_time",
            "outcome",
            "score",
            "shooter",
            "match_method",
            "throw",
            "manual",
        )
        self.event_tree = ttk.Treeview(
            event_inner,
            columns=columns,
            show="headings",
            height=26,
            style="Annotator.Treeview",
            bootstyle="light",
        )
        self.event_tree.heading("event_time", text="Event")
        self.event_tree.heading("outcome", text="Outcome")
        self.event_tree.heading("score", text="Score")
        self.event_tree.heading("shooter", text="Shooter")
        self.event_tree.heading("match_method", text="Match Sync")
        self.event_tree.heading("throw", text="Auto Throw")
        self.event_tree.heading("manual", text="Manual")

        self.event_tree.column("event_time", width=120, stretch=False)
        self.event_tree.column("outcome", width=120, stretch=False)
        self.event_tree.column("score", width=80, stretch=False)
        self.event_tree.column("shooter", width=240, stretch=True)
        self.event_tree.column("match_method", width=130, stretch=False)
        self.event_tree.column("throw", width=100, stretch=False)
        self.event_tree.column("manual", width=100, stretch=False)

        self.event_tree.tag_configure("goal", foreground="#4ade80")
        self.event_tree.tag_configure("annotated", background="#1a3d2a")
        self.event_tree.tag_configure("needs_followup", background="#3d2a0e")

        tree_shell = ttk.Frame(event_inner, style="SidebarInner.TFrame")
        tree_shell.pack(fill=BOTH, expand=True)
        scroll_y = ttk.Scrollbar(
            tree_shell,
            orient=VERTICAL,
            command=self.event_tree.yview,
            bootstyle="light-round",
        )
        self.event_tree.configure(yscrollcommand=scroll_y.set)
        self.event_tree.pack(side=LEFT, fill=BOTH, expand=True)
        scroll_y.pack(side=RIGHT, fill=Y)
        self.event_tree.bind("<<TreeviewSelect>>", self._on_event_selected)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        content = ttk.Frame(parent, style="App.TFrame")
        content.pack(fill=BOTH, expand=True)

        plot_box = ttk.Labelframe(
            content, text="Rendered Positions", style="Card.TLabelframe"
        )
        plot_box.pack(fill=BOTH, expand=True)
        plot_inner = ttk.Frame(plot_box, padding=14, style="CardInner.TFrame")
        plot_inner.pack(fill=BOTH, expand=True)

        plot_header = ttk.Frame(plot_inner, style="CardInner.TFrame")
        plot_header.pack(fill=X, pady=(0, 10))
        ttk.Label(
            plot_header, text="Court view + diagnostics", style="App.TLabel"
        ).pack(anchor=W)
        ttk.Label(
            plot_header,
            text="Overlay review for event, detected shot, estimated release and manual annotation.",
            style="Meta.TLabel",
        ).pack(anchor=W, pady=(2, 0))

        self.figure = Figure(figsize=(10, 8.2), dpi=100, facecolor=UI_BG)
        grid = self.figure.add_gridspec(
            nrows=2, ncols=1, height_ratios=[9, 3], hspace=0.22
        )
        self.axis = self.figure.add_subplot(grid[0])
        self.marker_axis = self.figure.add_subplot(grid[1])
        self.marker_accel_axis = self.marker_axis.twinx()
        self.marker_accel_axis.set_facecolor("none")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_inner)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self.canvas.get_tk_widget().configure(bg=PANEL_BG, highlightthickness=0, bd=0)

        controls_box = ttk.Labelframe(
            content, text="Timeline", style="Card.TLabelframe"
        )
        controls_box.pack(fill=X, pady=(12, 0))
        controls_inner = ttk.Frame(controls_box, padding=14, style="CardInner.TFrame")
        controls_inner.pack(fill=X, expand=True)

        ttk.Label(
            controls_inner, textvariable=self.scene_info_var, style="App.TLabel"
        ).pack(anchor=W)
        ttk.Label(
            controls_inner, textvariable=self.frame_info_var, style="Meta.TLabel"
        ).pack(anchor=W, pady=(4, 0))
        ttk.Label(
            controls_inner, textvariable=self.manual_throw_var, style="Meta.TLabel"
        ).pack(anchor=W, pady=(0, 8))

        self.timeline_scale = ttk.Scale(
            controls_inner,
            from_=0,
            to=0,
            orient=HORIZONTAL,
            command=self._on_timeline_change,
            style="Timeline.Horizontal.TScale",
            bootstyle="info",
        )
        self.timeline_scale.pack(fill=X)

        button_row = ttk.Frame(controls_inner, style="CardInner.TFrame")
        button_row.pack(fill=X, pady=(6, 0))

        ttk.Button(
            button_row,
            text="◀ Prev",
            command=self._step_prev,
            bootstyle="secondary-outline",
        ).pack(side=LEFT)
        ttk.Button(
            button_row,
            text="Next ▶",
            command=self._step_next,
            bootstyle="secondary-outline",
        ).pack(side=LEFT, padx=(6, 0))
        self.play_btn = ttk.Button(
            button_row,
            text="▶  Play",
            command=self._toggle_play,
            bootstyle="primary",
            width=10,
        )
        self.play_btn.pack(side=LEFT, padx=(6, 0))
        ttk.Separator(button_row, orient=VERTICAL).pack(
            side=LEFT, fill=Y, padx=(14, 14)
        )
        ttk.Button(
            button_row,
            text="→ Event",
            command=self._jump_to_event_time,
            bootstyle="info-outline",
        ).pack(side=LEFT)
        ttk.Button(
            button_row,
            text="→ Detected",
            command=self._jump_to_detected_time,
            bootstyle="info-outline",
        ).pack(side=LEFT, padx=(6, 0))
        ttk.Button(
            button_row,
            text="→ Synced",
            command=self._jump_to_synced_throw_time,
            bootstyle="info-outline",
        ).pack(side=LEFT, padx=(6, 0))
        ttk.Button(
            button_row,
            text="→ Manual",
            command=self._jump_to_manual_throw_time,
            bootstyle="info-outline",
        ).pack(side=LEFT, padx=(6, 0))

        annotation_box = ttk.Labelframe(
            content, text="Manual Annotation", style="Card.TLabelframe"
        )
        annotation_box.pack(fill=X, pady=(12, 0))
        annotation_inner = ttk.Frame(
            annotation_box, padding=14, style="CardInner.TFrame"
        )
        annotation_inner.pack(fill=X, expand=True)

        ttk.Label(
            annotation_inner,
            text="Set the release frame, leave a note, and persist the result to CSV.",
            style="Meta.TLabel",
        ).pack(anchor=W, pady=(0, 10))

        form = ttk.Frame(annotation_inner, style="CardInner.TFrame")
        form.pack(fill=X)
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Status", style="App.TLabel").grid(
            row=0, column=0, sticky=W
        )
        self.status_combo = ttk.Combobox(
            form,
            textvariable=self.annotation_status_var,
            state="readonly",
            values=["unreviewed", "annotated", "needs_followup"],
            style="Modern.TCombobox",
            bootstyle="light",
        )
        self.status_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(form, text="Note", style="App.TLabel").grid(
            row=1, column=0, sticky="nw", pady=(10, 0)
        )
        self.note_text = tk.Text(
            form,
            height=3,
            width=40,
            bg=PANEL_ALT_BG,
            fg=TEXT_DARK,
            insertbackground=TEXT_DARK,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
            font=("Segoe UI", 10),
            padx=12,
            pady=8,
        )
        self.note_text.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        action_row = ttk.Frame(annotation_inner, style="CardInner.TFrame")
        action_row.pack(fill=X, pady=(8, 0))

        ttk.Button(
            action_row,
            text="Set Release Frame  (M)",
            command=self._set_manual_throw_from_current_frame,
            bootstyle="warning",
        ).pack(side=LEFT)
        ttk.Button(
            action_row,
            text="Clear  (C)",
            command=self._clear_manual_throw,
            bootstyle="danger-outline",
        ).pack(side=LEFT, padx=(6, 0))
        ttk.Button(
            action_row,
            text="Save  (S)",
            command=self._save_current_annotation,
            bootstyle="success",
        ).pack(side=LEFT, padx=(14, 0))

    def _load_annotations(self) -> pd.DataFrame:
        if not self.annotation_path.exists():
            return pd.DataFrame(
                columns=[
                    "fixture_id",
                    "event_id",
                    "manual_throw_timestamp_ms",
                    "status",
                    "note",
                    "annotated_at_utc",
                ]
            )
        annotations = pd.read_csv(self.annotation_path)
        if "manual_throw_timestamp_ms" in annotations.columns:
            annotations["manual_throw_timestamp_ms"] = pd.to_numeric(
                annotations["manual_throw_timestamp_ms"], errors="coerce"
            ).astype("Int64")
        return annotations

    def _load_fixture_options(self) -> None:
        rows = self.connection.execute(
            """
            select
                m.fixture_id,
                coalesce(m.match_name, m.team_name_home || ' vs ' || m.team_name_away) as match_label,
                m.start_time_local,
                count(se.event_id) as goal_events
            from matches_normalized m
            left join shot_events se using (fixture_id)
            group by 1, 2, 3
            having count(se.event_id) > 0
            order by m.start_time_local nulls last, match_label
            """
        ).fetchall()

        self.fixture_options = []
        display_values = []
        for fixture_id, match_label, start_time_local, goal_events in rows:
            start_label = (
                pd.to_datetime(start_time_local).strftime("%Y-%m-%d %H:%M")
                if start_time_local is not None
                else "unknown"
            )
            display = (
                f"{start_label} | {match_label} | goals={goal_events} | {fixture_id}"
            )
            self.fixture_options.append((display, str(fixture_id)))
            display_values.append(display)

        self.fixture_combo["values"] = display_values
        if display_values:
            self.fixture_combo.current(0)
            self._on_fixture_selected()

    def _query_fixture_events(self, fixture_id: str) -> list[SceneEvent]:
        rows = self.connection.execute(
            """
            select
                fixture_id,
                event_id,
                event_time_ms,
                success,
                failure_reason,
                detected_events_shot_time,
                throw_timestamp_ms,
                person_name,
                goalkeeper_name,
                team_name_offense,
                team_name_defense,
                entity_id_home,
                entity_id_away,
                goal_position,
                match_method,
                method,
                scores
            from shot_events
            where fixture_id = ?
            order by event_time_ms, event_id
            """,
            [fixture_id],
        ).fetchall()

        events: list[SceneEvent] = []
        for row in rows:
            events.append(
                SceneEvent(
                    fixture_id=str(row[0]),
                    event_id=str(row[1]),
                    event_time_ms=_to_ms(row[2]),
                    outcome_label=_format_outcome_label(
                        None if pd.isna(row[3]) else bool(row[3]),
                        row[4],
                    ),
                    detected_shot_ms=_to_ms(row[5]),
                    throw_timestamp_ms=_to_ms(row[6]),
                    person_name=row[7] or "Unknown",
                    goalkeeper_name=row[8] or "Unknown",
                    team_name_offense=row[9] or "Unknown",
                    team_name_defense=row[10] or "Unknown",
                    entity_id_home=None if pd.isna(row[11]) else str(row[11]),
                    entity_id_away=None if pd.isna(row[12]) else str(row[12]),
                    goal_position=None if pd.isna(row[13]) else float(row[13]),
                    success=None if pd.isna(row[3]) else bool(row[3]),
                    match_method=row[14] or "-",
                    method=row[15] or "-",
                    score_text=_format_score_text(
                        row[16],
                        None if pd.isna(row[11]) else str(row[11]),
                        None if pd.isna(row[12]) else str(row[12]),
                    ),
                )
            )
        return events

    def _on_fixture_selected(self, _event: Any = None) -> None:
        selection = self.fixture_var.get()
        fixture_id = None
        for display, candidate_fixture_id in self.fixture_options:
            if display == selection:
                fixture_id = candidate_fixture_id
                break
        if fixture_id is None:
            return

        self._stop_playback()
        self.selected_fixture_id = fixture_id
        self.fixture_events[fixture_id] = self._query_fixture_events(fixture_id)
        self._populate_event_tree(self.fixture_events[fixture_id])
        self.status_var.set(
            f"Loaded fixture {fixture_id} with {len(self.fixture_events[fixture_id])} goal events."
        )

    def _populate_event_tree(self, events: list[SceneEvent]) -> None:
        self.event_tree.delete(*self.event_tree.get_children())
        for event in events:
            annotation = self._get_annotation_row(event.event_id)
            manual_throw = _fmt_ms(
                _to_ms(annotation.get("manual_throw_timestamp_ms"))
                if annotation is not None
                else None
            )
            ann_status = annotation.get("status") if annotation is not None else None
            tags: list[str] = []
            if event.outcome_label == "GOAL":
                tags.append("goal")
            if ann_status == "annotated":
                tags.append("annotated")
            elif ann_status == "needs_followup":
                tags.append("needs_followup")
            self.event_tree.insert(
                "",
                END,
                iid=event.event_id,
                tags=tags,
                values=(
                    _fmt_ms(event.event_time_ms),
                    event.outcome_label,
                    event.score_text,
                    _format_shooter_label(event.person_name, event.team_name_offense),
                    event.match_method,
                    _fmt_ms(event.throw_timestamp_ms),
                    manual_throw,
                ),
            )

        if events:
            first_event_id = events[0].event_id
            self.event_tree.selection_set(first_event_id)
            self._on_event_selected()

    def _on_event_selected(self, _event: Any = None) -> None:
        selection = self.event_tree.selection()
        if not selection or self.selected_fixture_id is None:
            return

        event_id = selection[0]
        event_map = {
            event.event_id: event
            for event in self.fixture_events[self.selected_fixture_id]
        }
        selected_event = event_map.get(event_id)
        if selected_event is None:
            return

        self._stop_playback()
        self.selected_event = selected_event
        self.current_shooter_league_id = self._event_value("person_league_id")
        self.current_goalkeeper_league_id = self._event_value("goalkeeper_league_id")
        self._load_scene_for_event(selected_event)
        self._load_annotation_into_form(selected_event.event_id)
        self._render_current_frame()

    def _load_scene_for_event(self, event: SceneEvent) -> None:
        anchor_ms = event.anchor_ms
        if anchor_ms is None:
            self.scene_positions = pd.DataFrame()
            self.scene_detected_events = pd.DataFrame()
            self.scene_timestamps = []
            self.current_scene_start_ms = None
            self.current_scene_end_ms = None
            self._refresh_marker_strip()
            self.scene_info_var.set(f"{event.event_id}: no anchor timestamp available")
            self.frame_info_var.set("Frame: -")
            self.status_var.set(
                f"Event {event.event_id} has no event, detected-shot, or synced throw timestamp."
            )
            return

        scene_start_ms = anchor_ms - SCENE_PAD_BEFORE_MS
        scene_end_ms = anchor_ms + SCENE_PAD_AFTER_MS

        self.scene_positions = self.connection.execute(
            """
            select timestamp_ms, full_name, league_id, group_name, x_m, y_m, speed_m_s
            from match_positions_normalized
            where fixture_id = ?
              and timestamp_ms between ? and ?
            order by timestamp_ms
            """,
            [event.fixture_id, scene_start_ms, scene_end_ms],
        ).df()
        self.scene_positions["timestamp_ms"] = pd.to_numeric(
            self.scene_positions["timestamp_ms"], errors="coerce"
        )
        self.scene_positions = self.scene_positions.dropna(
            subset=["timestamp_ms"]
        ).copy()
        self.scene_positions["timestamp_ms"] = self.scene_positions[
            "timestamp_ms"
        ].astype(np.int64)

        self.scene_detected_events = self.connection.execute(
            """
            select timestamp_ms, shot_category, shot_type, success, validated, league_id, id
            from match_detected_shots_normalized
            where fixture_id = ?
              and timestamp_ms between ? and ?
            order by timestamp_ms
            """,
            [event.fixture_id, scene_start_ms, scene_end_ms],
        ).df()
        if not self.scene_detected_events.empty:
            self.scene_detected_events["timestamp_ms"] = pd.to_numeric(
                self.scene_detected_events["timestamp_ms"], errors="coerce"
            )
            self.scene_detected_events = self.scene_detected_events.dropna(
                subset=["timestamp_ms"]
            ).copy()
            self.scene_detected_events["timestamp_ms"] = self.scene_detected_events[
                "timestamp_ms"
            ].astype(np.int64)

        self.player_ball_timeline = pd.DataFrame()

        self.scene_timestamps = [
            int(value)
            for value in self.scene_positions["timestamp_ms"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        ]
        self.current_scene_start_ms = scene_start_ms
        self.current_scene_end_ms = scene_end_ms

        if not self.scene_timestamps:
            self.timeline_scale.configure(from_=0, to=0)
            self.current_frame_index = 0
            self._refresh_marker_strip()
            self.scene_info_var.set(
                f"{event.event_id}: no position rows in scene window"
            )
            self.frame_info_var.set("Frame: -")
            self.status_var.set(
                f"No positions found for event {event.event_id} in the selected scene window."
            )
            return

        if self.current_shooter_league_id is not None:
            self.player_ball_timeline = _build_player_ball_timeline(
                self.scene_positions.copy(),
                self.current_shooter_league_id,
                frame_tol_ms=0,
            )

        self.timeline_scale.configure(from_=0, to=len(self.scene_timestamps) - 1)
        target_index = self._nearest_frame_index(event.anchor_ms)
        self.current_frame_index = target_index
        self.timeline_scale.set(target_index)
        self._refresh_marker_strip()

        self.scene_info_var.set(
            (
                f"Event {event.event_id} | shooter={event.person_name} | defense={event.team_name_defense} | "
                f"outcome={event.outcome_label} | score={event.score_text} | "
                f"scene={_fmt_ms(scene_start_ms)} -> {_fmt_ms(scene_end_ms)} | frames={len(self.scene_timestamps)} | "
                f"kinexon_events={len(self.scene_detected_events)}"
            )
        )
        self.status_var.set(f"Loaded scene for event {event.event_id}.")

    def _nearest_frame_index(self, target_ms: int | None) -> int:
        if target_ms is None or not self.scene_timestamps:
            return 0
        timestamp_array = np.asarray(self.scene_timestamps)
        index = int(np.searchsorted(timestamp_array, target_ms))
        candidates: list[int] = []
        if index < len(timestamp_array):
            candidates.append(index)
        if index > 0:
            candidates.append(index - 1)
        if not candidates:
            return 0
        return min(
            candidates, key=lambda item: abs(int(timestamp_array[item]) - target_ms)
        )

    def _on_timeline_change(self, value: str) -> None:
        if not self.scene_timestamps:
            return
        self.current_frame_index = int(float(value))
        self._render_current_frame()

    def _current_frame_timestamp(self) -> int | None:
        if not self.scene_timestamps:
            return None
        return self.scene_timestamps[self.current_frame_index]

    def _render_current_frame(self) -> None:
        self.axis.clear()
        self.axis.set_xlim(0, COURT_LENGTH_M)
        self.axis.set_ylim(0, COURT_WIDTH_M)
        self.axis.set_aspect("equal", adjustable="box")
        self.axis.set_box_aspect(COURT_WIDTH_M / COURT_LENGTH_M)
        self.axis.set_facecolor(PANEL_BG)
        self.axis.grid(alpha=0.15)
        self.axis.set_xlabel("Court x (m)")
        self.axis.set_ylabel("Court y (m)")

        self._draw_court()
        if self.selected_event is None or not self.scene_timestamps:
            self.axis.set_title("No event selected")
            self._update_marker_frame_indicator()
            self.canvas.draw_idle()
            return

        frame_ms = self._current_frame_timestamp()
        if frame_ms is None:
            self.axis.set_title("No frame available")
            self._update_marker_frame_indicator()
            self.canvas.draw_idle()
            return

        frame_rows = self.scene_positions[
            self.scene_positions["timestamp_ms"] == frame_ms
        ].copy()
        if frame_rows.empty:
            self.axis.set_title(
                f"{self.selected_event.event_id}: no positions at frame"
            )
            self._update_marker_frame_indicator()
            self.canvas.draw_idle()
            return

        group_names = frame_rows["group_name"].fillna("Unknown")
        team_colors = {
            self.selected_event.team_name_offense: "#d62728",
            self.selected_event.team_name_defense: "#1f77b4",
        }

        ball_mask = (
            frame_rows["league_id"]
            .astype(str)
            .str.contains("ball", case=False, na=False)
        )
        player_rows = frame_rows[~ball_mask]
        ball_rows = frame_rows[ball_mask]

        for group_name, group_frame in player_rows.groupby(group_names[~ball_mask]):
            color = team_colors.get(group_name, "#7f7f7f")
            self.axis.scatter(
                group_frame["x_m"],
                group_frame["y_m"],
                s=70,
                c=color,
                label=group_name,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.6,
            )

        if not ball_rows.empty:
            self.axis.scatter(
                ball_rows["x_m"],
                ball_rows["y_m"],
                s=110,
                c="#ffbf00",
                label="Ball",
                edgecolors="black",
                linewidths=0.8,
                marker="o",
                zorder=5,
            )

        self._highlight_special_players(frame_rows)

        annotation = self._get_annotation_row(self.selected_event.event_id)
        manual_throw_ms = (
            _to_ms(annotation.get("manual_throw_timestamp_ms"))
            if annotation is not None
            else None
        )
        delta_to_manual = (
            None if manual_throw_ms is None else frame_ms - manual_throw_ms
        )
        delta_to_auto = (
            None
            if self.selected_event.throw_timestamp_ms is None
            else frame_ms - self.selected_event.throw_timestamp_ms
        )
        delta_to_event = (
            None
            if self.selected_event.event_time_ms is None
            else frame_ms - self.selected_event.event_time_ms
        )

        self.axis.set_title(
            (
                f"{self.selected_event.person_name} vs {self.selected_event.goalkeeper_name} | "
                f"frame={_fmt_ms(frame_ms)} | auto={_fmt_delta(delta_to_auto)} | "
                f"event={_fmt_delta(delta_to_event)} | manual={_fmt_delta(delta_to_manual)}"
            ),
            fontsize=11,
        )
        self.frame_info_var.set(
            (
                f"Frame {self.current_frame_index + 1}/{len(self.scene_timestamps)} | "
                f"timestamp={_fmt_ms(frame_ms)} | score={self.selected_event.score_text} | "
                f"auto_method={self.selected_event.method}"
            )
        )

        handles, labels = self.axis.get_legend_handles_labels()
        dedup_handles: list[Any] = []
        dedup_labels: list[str] = []
        seen: set[str] = set()
        for handle, label in zip(handles, labels):
            if label not in seen:
                seen.add(label)
                dedup_handles.append(handle)
                dedup_labels.append(label)
        if dedup_handles:
            self.axis.legend(
                dedup_handles, dedup_labels, loc="upper left", framealpha=0.9
            )

        self._update_marker_frame_indicator()
        self.canvas.draw_idle()

    def _draw_court(self) -> None:
        if self.court_image is not None:
            self.axis.imshow(
                self.court_image,
                extent=[0, COURT_LENGTH_M, 0, COURT_WIDTH_M],
                origin="lower",
                aspect="auto",
                zorder=0,
            )
        self.axis.plot([0, COURT_LENGTH_M], [0, 0], color="black", linewidth=1.2)
        self.axis.plot(
            [0, COURT_LENGTH_M],
            [COURT_WIDTH_M, COURT_WIDTH_M],
            color="black",
            linewidth=1.2,
        )
        self.axis.plot([0, 0], [0, COURT_WIDTH_M], color="black", linewidth=1.2)
        self.axis.plot(
            [COURT_LENGTH_M, COURT_LENGTH_M],
            [0, COURT_WIDTH_M],
            color="black",
            linewidth=1.2,
        )
        self.axis.axvline(COURT_LENGTH_M / 2, color="black", linewidth=0.8, alpha=0.5)

        if (
            self.selected_event is not None
            and self.selected_event.goal_position is not None
        ):
            goal_x = float(self.selected_event.goal_position)
            self.axis.plot(
                [goal_x, goal_x],
                [
                    COURT_WIDTH_M / 2 - GOAL_HALF_WIDTH_M,
                    COURT_WIDTH_M / 2 + GOAL_HALF_WIDTH_M,
                ],
                color="#2ca02c",
                linewidth=4,
            )

    def _refresh_marker_strip(self) -> None:
        if self.marker_accel_axis is None:
            return

        self.marker_axis.clear()
        self.marker_axis.set_facecolor(PANEL_BG)
        self.marker_axis.grid(axis="x", alpha=0.12)
        self.marker_accel_axis.clear()
        self.marker_accel_axis.set_facecolor("none")

        self.marker_current_line = None
        self.marker_current_line_right = None

        if (
            self.selected_event is None
            or self.current_scene_start_ms is None
            or self.current_scene_end_ms is None
        ):
            self.marker_axis.set_xticks([])
            self.marker_axis.set_yticks([])
            self.marker_accel_axis.set_yticks([])
            self.marker_axis.set_title("Scene diagnostics")
            return

        scene_start_ms = self.current_scene_start_ms
        scene_end_ms = self.current_scene_end_ms
        if scene_end_ms <= scene_start_ms:
            self.marker_axis.set_xticks([])
            self.marker_axis.set_yticks([])
            self.marker_accel_axis.set_yticks([])
            self.marker_axis.set_title("Scene diagnostics")
            return

        self.marker_axis.set_xlim(scene_start_ms, scene_end_ms)
        self.marker_accel_axis.set_xlim(scene_start_ms, scene_end_ms)
        tick_values = np.linspace(scene_start_ms, scene_end_ms, num=5)
        self.marker_axis.set_xticks(tick_values)
        self.marker_axis.set_xticklabels(
            [_fmt_ms(int(value)) for value in tick_values], fontsize=8
        )

        if not self.player_ball_timeline.empty:
            df_pb = self.player_ball_timeline.copy()
            df_pb["timestamp_ms"] = pd.to_numeric(
                df_pb["timestamp_ms"], errors="coerce"
            )
            df_pb = df_pb.dropna(subset=["timestamp_ms"]).copy()
            df_pb["timestamp_ms"] = df_pb["timestamp_ms"].astype(np.int64)

            self.marker_axis.plot(
                df_pb["timestamp_ms"],
                pd.to_numeric(df_pb.get("dist_pb"), errors="coerce"),
                color=ACCENT,
                linewidth=2.2,
                label="dist_pb",
            )
            self.marker_axis.set_ylabel("Player-ball distance (m)", color=ACCENT)
            self.marker_axis.tick_params(axis="y", colors=ACCENT)

            if "ball_acc" in df_pb.columns:
                self.marker_accel_axis.plot(
                    df_pb["timestamp_ms"],
                    pd.to_numeric(df_pb["ball_acc"], errors="coerce"),
                    color=ACCENT_ALT,
                    linewidth=1.6,
                    alpha=0.9,
                    label="ball_acc",
                )
                self.marker_accel_axis.set_ylabel(
                    "Ball acceleration (m/s²)", color=ACCENT_ALT
                )
                self.marker_accel_axis.tick_params(axis="y", colors=ACCENT_ALT)
            else:
                self.marker_accel_axis.set_yticks([])
        else:
            self.marker_axis.set_yticks([])
            self.marker_accel_axis.set_yticks([])

        annotation = self._get_annotation_row(self.selected_event.event_id)
        manual_throw_ms = (
            _to_ms(annotation.get("manual_throw_timestamp_ms"))
            if annotation is not None
            else None
        )

        if not self.scene_detected_events.empty:
            detected_ms_values = (
                self.scene_detected_events["timestamp_ms"]
                .dropna()
                .astype(np.int64)
                .tolist()
            )
            matched_detected_ms = self.selected_event.detected_shot_ms
            marker_offsets = [0.08, 0.13, 0.18]
            kin_marker_x: list[int] = []
            kin_marker_y: list[float] = []
            for marker_index, detected_ms in enumerate(detected_ms_values):
                if matched_detected_ms is not None and int(detected_ms) == int(
                    matched_detected_ms
                ):
                    continue
                kin_marker_x.append(int(detected_ms))
                kin_marker_y.append(marker_offsets[marker_index % len(marker_offsets)])
            if kin_marker_x:
                self.marker_axis.scatter(
                    kin_marker_x,
                    kin_marker_y,
                    marker="v",
                    s=28,
                    color="#ffb366",
                    alpha=0.75,
                    linewidths=0,
                    zorder=4,
                    transform=self.marker_axis.get_xaxis_transform(),
                    label="scene Kinexon shots",
                )

        markers = [
            (self.selected_event.event_time_ms, "event", "#1f77b4"),
            (self.selected_event.detected_shot_ms, "detected", "#ff7f0e"),
            (self.selected_event.throw_timestamp_ms, "synced throw", "#2ca02c"),
            (manual_throw_ms, "manual", "#d62728"),
        ]
        label_y = 0.94
        for timestamp_ms, label, color in markers:
            if timestamp_ms is None:
                continue
            self.marker_axis.axvline(
                timestamp_ms,
                ymin=0.05,
                ymax=0.95,
                color=color,
                linewidth=2.2,
                alpha=0.9,
            )
            self.marker_axis.text(
                timestamp_ms,
                label_y,
                label,
                color=color,
                fontsize=8,
                ha="center",
                va="top",
                transform=self.marker_axis.get_xaxis_transform(),
            )
            label_y -= 0.12

        self.marker_axis.set_title(
            "Player-ball distance and ball acceleration timeline with event and Kinexon shot markers",
            fontsize=9,
            color=TEXT_DARK,
        )

        self.marker_current_line = self.marker_axis.axvline(
            scene_start_ms,
            color="#111111",
            linewidth=1.6,
            alpha=0.85,
            visible=False,
        )
        self.marker_current_line_right = self.marker_accel_axis.axvline(
            scene_start_ms,
            color="#111111",
            linewidth=1.0,
            alpha=0.5,
            visible=False,
        )
        self._update_marker_frame_indicator()

        left_handles, left_labels = self.marker_axis.get_legend_handles_labels()
        right_handles, right_labels = self.marker_accel_axis.get_legend_handles_labels()
        if left_handles or right_handles:
            self.marker_axis.legend(
                left_handles + right_handles,
                left_labels + right_labels,
                loc="upper left",
                framealpha=0.95,
            )

    def _update_marker_frame_indicator(self) -> None:
        frame_ms = self._current_frame_timestamp()
        if self.marker_current_line is not None:
            if frame_ms is None:
                self.marker_current_line.set_visible(False)
            else:
                self.marker_current_line.set_xdata([frame_ms, frame_ms])
                self.marker_current_line.set_visible(True)
        if self.marker_current_line_right is not None:
            if frame_ms is None:
                self.marker_current_line_right.set_visible(False)
            else:
                self.marker_current_line_right.set_xdata([frame_ms, frame_ms])
                self.marker_current_line_right.set_visible(True)

    def _highlight_special_players(self, frame_rows: pd.DataFrame) -> None:
        if self.selected_event is None:
            return

        if self.current_shooter_league_id is not None:
            shooter_rows = frame_rows[
                frame_rows["league_id"] == self.current_shooter_league_id
            ]
            if not shooter_rows.empty:
                self.axis.scatter(
                    shooter_rows["x_m"],
                    shooter_rows["y_m"],
                    s=220,
                    facecolors="none",
                    edgecolors="#8b0000",
                    linewidths=2.2,
                    label="Shooter",
                )

        if self.current_goalkeeper_league_id is not None:
            goalkeeper_rows = frame_rows[
                frame_rows["league_id"] == self.current_goalkeeper_league_id
            ]
            if not goalkeeper_rows.empty:
                self.axis.scatter(
                    goalkeeper_rows["x_m"],
                    goalkeeper_rows["y_m"],
                    s=220,
                    facecolors="none",
                    edgecolors="#0047ab",
                    linewidths=2.2,
                    label="Goalkeeper",
                )

    def _event_value(self, column: str) -> Any:
        if self.selected_fixture_id is None or self.selected_event is None:
            return None
        row = self.connection.execute(
            f"select {column} from shot_events where event_id = ?",
            [self.selected_event.event_id],
        ).fetchone()
        return row[0] if row else None

    def _step_prev(self) -> None:
        if not self.scene_timestamps:
            return
        new_index = max(0, self.current_frame_index - 1)
        self.timeline_scale.set(new_index)

    def _step_next(self) -> None:
        if not self.scene_timestamps:
            return
        new_index = min(len(self.scene_timestamps) - 1, self.current_frame_index + 1)
        self.timeline_scale.set(new_index)

    def _toggle_play(self) -> None:
        if self.play_job is not None:
            self._stop_playback()
            return
        if hasattr(self, "play_btn"):
            self.play_btn.configure(text="⏸  Pause")
        self._play_step()

    def _play_step(self) -> None:
        if not self.scene_timestamps:
            return
        if self.current_frame_index >= len(self.scene_timestamps) - 1:
            self._stop_playback()
            return
        self.timeline_scale.set(self.current_frame_index + 1)
        self.play_job = self.root.after(PLAY_TIMER_MS, self._play_step)

    def _stop_playback(self) -> None:
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
            self.play_job = None
        if hasattr(self, "play_btn"):
            self.play_btn.configure(text="▶  Play")

    def _jump_to_time(self, timestamp_ms: int | None) -> None:
        if timestamp_ms is None or not self.scene_timestamps:
            return
        self.timeline_scale.set(self._nearest_frame_index(timestamp_ms))

    def _jump_to_event_time(self) -> None:
        if self.selected_event is not None:
            self._jump_to_time(self.selected_event.event_time_ms)

    def _jump_to_detected_time(self) -> None:
        if self.selected_event is not None:
            self._jump_to_time(self.selected_event.detected_shot_ms)

    def _jump_to_synced_throw_time(self) -> None:
        if self.selected_event is not None:
            self._jump_to_time(self.selected_event.throw_timestamp_ms)

    def _jump_to_manual_throw_time(self) -> None:
        if self.selected_event is None:
            return
        annotation = self._get_annotation_row(self.selected_event.event_id)
        if annotation is None:
            return
        self._jump_to_time(_to_ms(annotation.get("manual_throw_timestamp_ms")))

    def _get_annotation_row(self, event_id: str) -> dict[str, Any] | None:
        matches = self.annotations[self.annotations["event_id"] == event_id]
        if matches.empty:
            return None
        return cast(dict[str, Any], matches.iloc[-1].to_dict())

    def _load_annotation_into_form(self, event_id: str) -> None:
        annotation = self._get_annotation_row(event_id)
        self.note_text.delete("1.0", END)
        if annotation is None:
            self.annotation_status_var.set("unreviewed")
            self.manual_throw_var.set("Manual throw: -")
            return

        self.annotation_status_var.set(annotation.get("status") or "unreviewed")
        self.note_text.insert("1.0", annotation.get("note") or "")
        self.manual_throw_var.set(
            f"Manual throw: {_fmt_ms(_to_ms(annotation.get('manual_throw_timestamp_ms')))}"
        )

    def _set_manual_throw_from_current_frame(self) -> None:
        if self.selected_event is None:
            return
        current_frame_ms = self._current_frame_timestamp()
        if current_frame_ms is None:
            return
        annotation = self._build_annotation_record(current_frame_ms)
        self._upsert_annotation(annotation)
        self._load_annotation_into_form(self.selected_event.event_id)
        self._refresh_event_row(self.selected_event.event_id)
        self._refresh_marker_strip()
        self.status_var.set(
            f"Staged manual throw timestamp {_fmt_ms(current_frame_ms)} for event {self.selected_event.event_id}."
        )
        self._render_current_frame()

    def _clear_manual_throw(self) -> None:
        if self.selected_event is None:
            return
        annotation = self._build_annotation_record(None)
        self._upsert_annotation(annotation)
        self._load_annotation_into_form(self.selected_event.event_id)
        self._refresh_event_row(self.selected_event.event_id)
        self._refresh_marker_strip()
        self.status_var.set(
            f"Cleared manual throw timestamp for event {self.selected_event.event_id}."
        )
        self._render_current_frame()

    def _build_annotation_record(
        self, manual_throw_timestamp_ms: int | None
    ) -> dict[str, Any]:
        if self.selected_event is None:
            raise RuntimeError("No selected event")
        return {
            "fixture_id": self.selected_event.fixture_id,
            "event_id": self.selected_event.event_id,
            "manual_throw_timestamp_ms": manual_throw_timestamp_ms,
            "status": self.annotation_status_var.get(),
            "note": self.note_text.get("1.0", END).strip(),
            "annotated_at_utc": pd.Timestamp.utcnow().isoformat(),
        }

    def _upsert_annotation(self, record: dict[str, Any]) -> None:
        self.annotations = self.annotations[
            self.annotations["event_id"] != record["event_id"]
        ].copy()
        self.annotations = pd.concat(
            [self.annotations, pd.DataFrame([record])], ignore_index=True
        )

    def _save_current_annotation(self) -> None:
        if self.selected_event is None:
            return
        existing = self._get_annotation_row(self.selected_event.event_id)
        manual_throw_ms = (
            _to_ms(existing.get("manual_throw_timestamp_ms"))
            if existing is not None
            else None
        )
        annotation = self._build_annotation_record(manual_throw_ms)
        self._upsert_annotation(annotation)

        self.annotation_path.parent.mkdir(parents=True, exist_ok=True)
        self.annotations.sort_values(["fixture_id", "event_id"]).to_csv(
            self.annotation_path, index=False
        )
        self._refresh_event_row(self.selected_event.event_id)
        self.status_var.set(f"Saved annotations to {self.annotation_path}.")
        messagebox.showinfo(
            "Annotation saved", f"Saved annotations to\n{self.annotation_path}"
        )

    def _handle_prev_key(self, _event: Any) -> None:
        self._step_prev()

    def _handle_next_key(self, _event: Any) -> None:
        self._step_next()

    def _handle_toggle_play_key(self, _event: Any) -> None:
        self._toggle_play()

    def _handle_mark_manual_key(self, _event: Any) -> None:
        self._set_manual_throw_from_current_frame()

    def _handle_save_key(self, _event: Any) -> None:
        self._save_current_annotation()

    def _handle_jump_event_key(self, _event: Any) -> None:
        self._jump_to_event_time()

    def _handle_jump_detected_key(self, _event: Any) -> None:
        self._jump_to_detected_time()

    def _handle_jump_synced_key(self, _event: Any) -> None:
        self._jump_to_synced_throw_time()

    def _handle_clear_manual_key(self, _event: Any) -> None:
        self._clear_manual_throw()

    def _refresh_event_row(self, event_id: str) -> None:
        item = self.event_tree.item(event_id)
        if not item or not self.selected_fixture_id:
            return
        event_map = {
            event.event_id: event
            for event in self.fixture_events[self.selected_fixture_id]
        }
        event = event_map.get(event_id)
        if event is None:
            return
        annotation = self._get_annotation_row(event_id)
        manual_throw = _fmt_ms(
            _to_ms(annotation.get("manual_throw_timestamp_ms"))
            if annotation is not None
            else None
        )
        ann_status = annotation.get("status") if annotation is not None else None
        tags: list[str] = []
        if event.outcome_label == "GOAL":
            tags.append("goal")
        if ann_status == "annotated":
            tags.append("annotated")
        elif ann_status == "needs_followup":
            tags.append("needs_followup")
        self.event_tree.item(
            event_id,
            tags=tags,
            values=(
                _fmt_ms(event.event_time_ms),
                event.outcome_label,
                event.score_text,
                _format_shooter_label(event.person_name, event.team_name_offense),
                event.match_method,
                _fmt_ms(event.throw_timestamp_ms),
                manual_throw,
            ),
        )

    def close(self) -> None:
        self._stop_playback()
        self.connection.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual shot-scene annotation UI for goal events.",
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH, help="Path to DuckDB database"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ANNOTATION_PATH,
        help="CSV path for manual throw annotations",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = ttk.Window(themename="flatly")
    app = ShotSceneAnnotator(
        root=root, db_path=args.db, annotation_path=args.annotations
    )

    def _on_close() -> None:
        app.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
