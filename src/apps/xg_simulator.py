import sys
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button, RadioButtons, CheckButtons
import joblib
import pickle

# Add project root to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

# Constants
GOAL_WIDTH = 3.0
HALF_GOAL = GOAL_WIDTH / 2.0
GOAL_X = 40.0  # Assuming standard court length, goal at right end
GOAL_Y = 10.0  # Center of width
COURT_LENGTH = 40.0
COURT_WIDTH = 20.0

# --- Geometry Helpers (copied from src/pipelines/features/calc_xg_features.py) ---


def shot_angle_to_goal(
    shooter_x: float, shooter_y: float, goal_x: float, goal_y: float = 10.0
) -> float:
    left_post = (goal_x, goal_y - HALF_GOAL)
    right_post = (goal_x, goal_y + HALF_GOAL)
    angle_left = np.arctan2(left_post[1] - shooter_y, left_post[0] - shooter_x)
    angle_right = np.arctan2(
        right_post[1] - shooter_y, right_post[0] - shooter_x
    )
    return float(abs(angle_right - angle_left))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0:
        return np.nan
    c = np.clip(float(np.dot(v1, v2) / n), -1.0, 1.0)
    return float(np.arccos(c))


def point_to_segment_distance(px, py, ax, ay, bx, by) -> float:
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom == 0:
        return float(np.hypot(apx, apy))
    t = (apx * abx + apy * aby) / denom
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    return float(np.hypot(px - cx, py - cy))


def projection_along_ray(px, py, ax, ay, bx, by) -> float:
    ab = np.array([bx - ax, by - ay], dtype=float)
    ap = np.array([px - ax, py - ay], dtype=float)
    denom = float(np.linalg.norm(ab))
    if denom == 0:
        return np.nan
    return float(np.dot(ap, ab) / denom)


# --- Simulator Class ---


class XGSimulator:
    def __init__(self, model_path, image_path):
        self.model = self.load_model(model_path)
        self.image_path = image_path

        self.shooter = None
        self.goalkeeper = None
        self.ball = None
        self.defenders = []
        self.mode = "Shooter"  # Shooter, GK, Defender, Ball
        self.empty_net = False

        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        plt.subplots_adjust(bottom=0.2)

        self.load_image()

        # Event handling
        self.cid = self.fig.canvas.mpl_connect(
            "button_press_event", self.onclick
        )

        # UI Elements
        ax_radio = plt.axes([0.05, 0.05, 0.15, 0.15], facecolor="#e4e4e4")
        self.radio = RadioButtons(
            ax_radio, ("Shooter", "Goalkeeper", "Defender", "Ball")
        )
        self.radio.on_clicked(self.set_mode)

        # Attack Type
        ax_attack = plt.axes([0.05, 0.25, 0.15, 0.15], facecolor="#e4e4e4")
        self.radio_attack = RadioButtons(
            ax_attack, ("Open Play", "Fast Break", "Pivot", "Breakthrough")
        )
        self.radio_attack.on_clicked(self.update_params)

        # Sub Type
        ax_sub = plt.axes([0.05, 0.45, 0.15, 0.15], facecolor="#e4e4e4")
        self.radio_sub = RadioButtons(ax_sub, ("Standard", "9m", "6m", "Wing"))
        self.radio_sub.on_clicked(self.update_params)

        ax_check = plt.axes([0.25, 0.14, 0.1, 0.05], facecolor="#e4e4e4")
        self.check = CheckButtons(ax_check, ["Empty Net"], [False])
        self.check.on_clicked(self.update_empty_net)

        ax_calc = plt.axes([0.25, 0.05, 0.1, 0.075])
        self.b_calc = Button(ax_calc, "Calculate xG")
        self.b_calc.on_clicked(self.calculate_and_predict)

        ax_clear = plt.axes([0.36, 0.05, 0.1, 0.075])
        self.b_clear = Button(ax_clear, "Clear All")
        self.b_clear.on_clicked(self.clear_all)

        ax_clear_def = plt.axes([0.47, 0.05, 0.1, 0.075])
        self.b_clear_def = Button(ax_clear_def, "Clear Defs")
        self.b_clear_def.on_clicked(self.clear_defenders)

        self.text_output = self.fig.text(
            0.6, 0.05, "xG: -", fontsize=14, fontweight="bold"
        )

        plt.show()

    def load_model(self, path):
        print(f"Loading model from {path}...")
        try:
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)

    def load_image(self):
        if os.path.exists(self.image_path):
            img = mpimg.imread(self.image_path)
            self.ax.imshow(
                img, extent=[0, COURT_LENGTH, 0, COURT_WIDTH], zorder=0
            )
        else:
            print(
                f"Warning: Image not found at {self.image_path}. Using blank court."
            )
            self.ax.set_xlim(0, COURT_LENGTH)
            self.ax.set_ylim(0, COURT_WIDTH)
            self.ax.set_aspect("equal")
            # Draw goal
            self.ax.plot(
                [GOAL_X, GOAL_X],
                [GOAL_Y - HALF_GOAL, GOAL_Y + HALF_GOAL],
                "k-",
                linewidth=3,
            )

    def set_mode(self, label):
        self.mode = label

    def update_params(self, label):
        # Auto-calculate if shooter is placed
        if self.shooter:
            self.calculate_and_predict(None)

    def update_empty_net(self, label):
        self.empty_net = not self.empty_net
        # Auto-calculate if shooter is placed
        if self.shooter:
            self.calculate_and_predict(None)

    def onclick(self, event):
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if self.mode == "Shooter":
            self.shooter = (x, y)
            # Default ball to shooter pos if not set
            if self.ball is None:
                self.ball = (x, y)
        elif self.mode == "Goalkeeper":
            self.goalkeeper = (x, y)
        elif self.mode == "Defender":
            self.defenders.append((x, y))
        elif self.mode == "Ball":
            self.ball = (x, y)

        self.update_plot()

        # Auto-calculate if shooter is placed
        if self.shooter:
            self.calculate_and_predict(None)

    def clear_all(self, event):
        self.shooter = None
        self.goalkeeper = None
        self.ball = None
        self.defenders = []
        self.text_output.set_text("xG: -")
        self.update_plot()

    def clear_defenders(self, event):
        self.defenders = []
        self.update_plot()

    def update_plot(self):
        # Clear points and patches (keep image)
        for artist in (
            self.ax.lines
            + self.ax.collections
            + self.ax.patches
            + self.ax.texts
        ):
            artist.remove()

        # Re-draw goal if no image
        if not os.path.exists(self.image_path):
            self.ax.plot(
                [GOAL_X, GOAL_X],
                [GOAL_Y - HALF_GOAL, GOAL_Y + HALF_GOAL],
                "k-",
                linewidth=3,
            )

        # Draw visuals (Cone & Distance)
        start_pos = self.ball if self.ball else self.shooter
        if start_pos:
            sx, sy = start_pos

            # Cone
            left_post = (GOAL_X, GOAL_Y - HALF_GOAL)
            right_post = (GOAL_X, GOAL_Y + HALF_GOAL)
            triangle = plt.Polygon(
                [[sx, sy], left_post, right_post],
                color="yellow",
                alpha=0.15,
                zorder=1,
            )
            self.ax.add_patch(triangle)

            # Line to center
            self.ax.plot(
                [sx, GOAL_X],
                [sy, GOAL_Y],
                "k--",
                alpha=0.4,
                linewidth=1,
                zorder=2,
            )

            # Distance text on line
            dist = np.hypot(sx - GOAL_X, sy - GOAL_Y)
            mid_x = (sx + GOAL_X) / 2
            mid_y = (sy + GOAL_Y) / 2
            self.ax.text(
                mid_x,
                mid_y,
                f"{dist:.1f}m",
                color="black",
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white", alpha=0.7, edgecolor="none", pad=1
                ),
                zorder=3,
            )

        if self.shooter:
            self.ax.plot(
                self.shooter[0],
                self.shooter[1],
                "ro",
                markersize=10,
                label="Shooter",
                zorder=10,
            )
        if self.goalkeeper:
            self.ax.plot(
                self.goalkeeper[0],
                self.goalkeeper[1],
                "go",
                markersize=10,
                label="GK",
                zorder=10,
            )
        if self.ball:
            self.ax.plot(
                self.ball[0],
                self.ball[1],
                "yo",
                markersize=6,
                label="Ball",
                zorder=11,
            )
        if self.defenders:
            dx, dy = zip(*self.defenders)
            self.ax.plot(
                dx, dy, "bo", markersize=8, label="Defender", zorder=9
            )

        self.fig.canvas.draw()

    def calculate_and_predict(self, event):
        if not self.shooter:
            self.text_output.set_text("Error: Place Shooter!")
            return

        features = self.compute_features()
        df = pd.DataFrame([features])

        # Ensure columns match model expectation (fill missing with defaults if needed)
        # The pipeline should handle selection, but we need to provide all used features.

        try:
            proba = self.model.predict_proba(df)[:, 1][0]
            self.text_output.set_text(f"xG: {proba:.4f}")

            # Update UI Text with detailed features
            dist = features.get(
                "ball_distance_to_goal",
                features.get("shooter_distance_to_goal"),
            )
            angle_rad = features.get(
                "ball_angle_to_goal", features.get("shot_angle_to_goal")
            )
            angle_deg = np.degrees(angle_rad) if pd.notna(angle_rad) else 0

            defs_close = features.get("num_defenders_close", 0)
            def_dist = features.get("closest_defender_distance", 10.0)

            info_text = (
                f"xG: {proba:.4f}\n"
                f"Dist: {dist:.1f}m\n"
                f"Angle: {angle_deg:.1f}°\n"
                f"Defs <2m: {defs_close}\n"
                f"Closest Def: {def_dist:.1f}m"
            )

            self.text_output.set_text(info_text)
            self.text_output.set_y(0.02)  # Adjust Y to fit multiline
            self.text_output.set_fontsize(10)
            self.text_output.set_verticalalignment("bottom")

            print(f"Predicted xG: {proba:.4f}")
            print("Input features:")
            for k, v in features.items():
                print(f"  {k}: {v}")
            self.fig.canvas.draw()
        except Exception as e:
            self.text_output.set_text("Error in prediction")
            print(f"Prediction error: {e}")
            # Print missing columns if any
            # print(df.columns)

    def compute_features(self):
        # Defaults
        goal_x = GOAL_X
        goal_y = GOAL_Y
        cone_half_angle_deg = 20.0
        cone_half_angle = np.deg2rad(cone_half_angle_deg)

        shooter_x, shooter_y = self.shooter

        if self.goalkeeper:
            goalkeeper_x, goalkeeper_y = self.goalkeeper
            gk_tracked = 1
        else:
            goalkeeper_x, goalkeeper_y = np.nan, np.nan
            gk_tracked = 0

        if self.ball:
            ball_x, ball_y = self.ball
        else:
            ball_x, ball_y = shooter_x, shooter_y  # Default to shooter

        # --- Feature Calculation ---

        # Distances
        shooter_distance_to_goal = float(
            np.hypot(shooter_x - goal_x, shooter_y - goal_y)
        )
        shot_angle = shot_angle_to_goal(shooter_x, shooter_y, goal_x, goal_y)
        shooter_lateral_offset = float(abs(shooter_y - goal_y))

        shooter_to_goal_dx = float(goal_x - shooter_x)
        shooter_to_goal_dy = float(goal_y - shooter_y)
        shooter_y_signed = float(shooter_y - goal_y)
        shooter_distance_to_goal_sq = float(shooter_distance_to_goal**2)

        # GK Features
        if np.isfinite(goalkeeper_x):
            shooter_distance_to_goalkeeper = float(
                np.hypot(shooter_x - goalkeeper_x, shooter_y - goalkeeper_y)
            )
            goalkeeper_distance_to_goal = float(
                np.hypot(goalkeeper_x - goal_x, goalkeeper_y - goal_y)
            )
            gk_lateral_offset = float(abs(goalkeeper_y - goal_y))

            v_goal = np.array(
                [goal_x - shooter_x, goal_y - shooter_y], dtype=float
            )
            v_gk = np.array(
                [goalkeeper_x - shooter_x, goalkeeper_y - shooter_y],
                dtype=float,
            )
            gk_angle_from_shooter = angle_between(v_goal, v_gk)
            gk_along_shotline_dist = projection_along_ray(
                goalkeeper_x,
                goalkeeper_y,
                shooter_x,
                shooter_y,
                goal_x,
                goal_y,
            )
        else:
            shooter_distance_to_goalkeeper = np.nan
            goalkeeper_distance_to_goal = np.nan
            gk_lateral_offset = np.nan
            gk_angle_from_shooter = np.nan
            gk_along_shotline_dist = np.nan

        # Ball Features
        ball_distance_to_goal = float(
            np.hypot(ball_x - goal_x, ball_y - goal_y)
        )
        ball_angle = shot_angle_to_goal(ball_x, ball_y, goal_x, goal_y)

        if np.isfinite(goalkeeper_x):
            ball_distance_to_goalkeeper = float(
                np.hypot(ball_x - goalkeeper_x, ball_y - goalkeeper_y)
            )
            angle_ball_gk = angle_between(
                np.array([goal_x - ball_x, goal_y - ball_y], dtype=float),
                np.array(
                    [goalkeeper_x - ball_x, goalkeeper_y - ball_y], dtype=float
                ),
            )
        else:
            ball_distance_to_goalkeeper = np.nan
            angle_ball_gk = np.nan

        # Defender Features
        num_defenders_close = 0
        closest_defender_distance = (
            10.0  # Default to large distance (open shot) if no defenders
        )
        min_defender_dist_to_shotline = 10.0
        min_defender_along_shotline = 10.0
        n_defenders_in_shot_cone = 0
        avg_defense_distance = 10.0
        defense_centroid_dist_to_goal = 10.0
        defense_spread = 5.0  # Assume some spread

        if self.defenders:
            def_x = np.array([d[0] for d in self.defenders])
            def_y = np.array([d[1] for d in self.defenders])

            # Dist to shooter
            d_shooter = np.hypot(def_x - shooter_x, def_y - shooter_y)
            num_defenders_close = int((d_shooter <= 2.0).sum())
            closest_defender_distance = float(d_shooter.min())

            # Dist to goal (avg)
            d_goal = np.hypot(def_x - goal_x, def_y - goal_y)
            avg_defense_distance = float(d_goal.mean())

            # Centroid
            cx = def_x.mean()
            cy = def_y.mean()
            defense_centroid_dist_to_goal = float(
                np.hypot(cx - goal_x, cy - goal_y)
            )
            defense_spread = float(np.hypot(def_x - cx, def_y - cy).mean())

            # Shotline
            dist_to_line = []
            along_line = []
            in_cone = 0
            v_goal = np.array(
                [goal_x - shooter_x, goal_y - shooter_y], dtype=float
            )

            for dx, dy in self.defenders:
                dist = point_to_segment_distance(
                    dx, dy, shooter_x, shooter_y, goal_x, goal_y
                )
                dist_to_line.append(dist)

                along = projection_along_ray(
                    dx, dy, shooter_x, shooter_y, goal_x, goal_y
                )
                along_line.append(along)

                v_def = np.array([dx - shooter_x, dy - shooter_y], dtype=float)
                ang = angle_between(v_goal, v_def)
                if np.isfinite(ang) and ang <= cone_half_angle and along > 0:
                    in_cone += 1

            min_defender_dist_to_shotline = float(np.min(dist_to_line))
            along_pos = [a for a in along_line if np.isfinite(a) and a > 0]
            min_defender_along_shotline = (
                float(np.min(along_pos)) if along_pos else np.nan
            )
            n_defenders_in_shot_cone = int(in_cone)

        # Offense Avg Distance (Just shooter for now, or maybe shooter + ball if separate?)
        # Let's just use shooter dist as avg if no other teammates
        avg_offense_distance = shooter_distance_to_goal

        return {
            "fixture_id": "sim",
            "event_id": "sim",
            "avg_offense_distance_to_goal": avg_offense_distance,
            "avg_defense_distance_to_goal": avg_defense_distance,
            "shooter_distance_to_goal": shooter_distance_to_goal,
            "shooter_distance_to_goalkeeper": shooter_distance_to_goalkeeper,
            "goalkeeper_distance_to_goal": goalkeeper_distance_to_goal,
            "ball_distance_to_goal": ball_distance_to_goal,
            "ball_distance_to_goalkeeper": ball_distance_to_goalkeeper,
            "shot_angle_to_goal": shot_angle,
            "ball_angle_to_goal": ball_angle,
            "angle_ball_to_goalkeeper": angle_ball_gk,
            "num_defenders_close": num_defenders_close,
            "closest_defender_distance": closest_defender_distance,
            "shooter_lateral_offset": shooter_lateral_offset,
            "attack_type": self.get_attack_type(),
            "sub_type": self.get_sub_type(),
            "shooter_to_goal_dx": shooter_to_goal_dx,
            "shooter_to_goal_dy": shooter_to_goal_dy,
            "shooter_y_signed": shooter_y_signed,
            "shooter_distance_to_goal_sq": shooter_distance_to_goal_sq,
            "gk_lateral_offset": gk_lateral_offset,
            "gk_angle_from_shooter": gk_angle_from_shooter,
            "gk_along_shotline_dist": gk_along_shotline_dist,
            "min_defender_dist_to_shotline": min_defender_dist_to_shotline,
            "min_defender_along_shotline": min_defender_along_shotline,
            "n_defenders_in_shot_cone": n_defenders_in_shot_cone,
            "defense_centroid_dist_to_goal": defense_centroid_dist_to_goal,
            "defense_spread": defense_spread,
            "gk_tracked": gk_tracked,
            "empty_net": int(self.empty_net),
            "attendance": 5000,
            "is_home_team": True,
            "target": 0,  # Dummy
        }

    def get_attack_type(self):
        val = self.radio_attack.value_selected
        mapping = {
            "Open Play": "OTHER",
            "Fast Break": "FAST_BREAK",
            "Pivot": "PIVOT",
            "Breakthrough": "BREAK_THROUGH",
        }
        return mapping.get(val, "OTHER")

    def get_sub_type(self):
        val = self.radio_sub.value_selected
        mapping = {
            "Standard": "OTHER",
            "9m": "nineMetre",
            "6m": "sixMetre",
            "Wing": "wing",
        }
        return mapping.get(val, "OTHER")


if __name__ == "__main__":
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))

    model_path = os.path.join(
        project_root, "data/models/xg_model_fixture.joblib"
    )
    image_path = os.path.join(project_root, "assets/handballfeld.png")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    sim = XGSimulator(model_path, image_path)
