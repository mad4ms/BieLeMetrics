"""
classify_shot_context.py — XGBoost xG model with spatiotemporal context

Extends the snapshot xG pipeline (calc_xg_features + train_xg) by adding
2-second pre-shot temporal window features to the standard snapshot geometry.

PIPELINE
  1. Load  shot_events + match_positions_normalized + matches_normalized from DuckDB
  2. Snapshot features: delegates to calculate_xg_features() (same 28+ features as
     the Dagster features_xg asset)
  3. Window features: aggregate player motion over 2s before throw_timestamp_ms —
     shooter speed/acceleration, GK movement, defense shape trends, ball speed,
     active-player fractions (suspension/exclusion proxy)
  4. Train: XGBoost via sklearn Pipeline with fixture-grouped validation split
     (no fixture appears in both train and val — prevents leakage between shots
     from the same game, unlike the row-random split in train_xg.py)
  5. Metrics: AUC, logloss, Brier, accuracy vs constant-predictor baseline
     (same metric set as train_xg.py)

WINDOW FEATURES (14 new scalars added to the 28 snapshot features)
  Shooter:        speed_at_throw, speed_mean_window, accel_at_throw, dist_change_window
  Goalkeeper:     speed_at_throw, dist_change_window, lateral_change_window
  Ball:           speed_at_throw, speed_mean_window
  Defense shape:  spread_mean_window, spread_change_window, centroid_dist_change_window
  Player counts:  n_active_offense_frac, n_active_defense_frac  (suspension detection)

USAGE
  uv run scripts/classify_shot_context.py
  uv run scripts/classify_shot_context.py --fixture <id>
  uv run scripts/classify_shot_context.py --out-model data/models/xg_context.joblib
  uv run scripts/classify_shot_context.py --out-csv data/xg_context_pred.csv
  uv run scripts/classify_shot_context.py --eval-only --model-path data/models/xg_context.joblib
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Pipeline modules — snapshot features + inference helpers
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pipelines.features.calc_xg_features import calculate_xg_features  # noqa: E402
from pipelines.ml.infer_xg import infer_xg, summarize_fixture_xg  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent.parent / "data" / "hbl_raw.duckdb"

WINDOW_MS = 2_000  # 2-second pre-shot window
THROW_TOLERANCE_MS = 150  # tolerance when snapping to the nearest tracked frame
RECENT_MS = 200  # window tail for active-player counting (~5 frames @ 40 ms)

_WINDOW_FEATURE_KEYS = (
    "shooter_speed_at_throw",
    "shooter_speed_mean_window",
    "shooter_accel_at_throw",
    "shooter_dist_change_window",
    "gk_speed_at_throw",
    "gk_dist_change_window",
    "gk_lateral_change_window",
    "ball_speed_at_throw",
    "ball_speed_mean_window",
    "defense_spread_mean_window",
    "defense_spread_change_window",
    "defense_centroid_dist_change_window",
    "n_active_offense_frac",
    "n_active_defense_frac",
)

_NAN_WINDOW = {k: np.nan for k in _WINDOW_FEATURE_KEYS}


# ---------------------------------------------------------------------------
# Group-name fuzzy mapper  (mirrors the remap logic inside calc_xg_features)
# ---------------------------------------------------------------------------


def _build_group_name_map(
    df_pos: pd.DataFrame,
    df_shots: pd.DataFrame,
) -> dict[str, str]:
    """
    Build Kinexon group_name → Sportradar team name mapping using fuzzy matching.

    Mirrors the difflib block inside calculate_xg_features so that team-name
    filtering in window-feature extraction is consistent with snapshot features.
    """
    sportradar_teams: set[str] = set()
    for col in ("team_name_offense", "team_name_defense", "team_name_home"):
        if col in df_shots.columns:
            sportradar_teams.update(df_shots[col].dropna().unique())
    sportradar_teams.discard(None)

    kinexon_groups = [
        g
        for g in df_pos["group_name"].dropna().unique()
        if "ball" not in str(g).lower()
    ]

    group_name_map: dict[str, str] = {}
    for kg in kinexon_groups:
        matches = difflib.get_close_matches(kg, sportradar_teams, n=1, cutoff=0.5)
        group_name_map[kg] = matches[0] if matches else kg

    return group_name_map


# ---------------------------------------------------------------------------
# Per-entity temporal statistics
# ---------------------------------------------------------------------------


def _entity_stats(
    df_entity: pd.DataFrame,
    throw_ts: int,
    goal_x: float,
    goal_y: float,
    *,
    include_lateral: bool = False,
) -> dict[str, float]:
    """
    Compute temporal statistics for a single tracked entity over a 2s window.

    Returns speed_at_throw, speed_mean, accel_at_throw, dist_change, and
    optionally lateral_change (signed y-displacement relative to goal_y).
    All values are np.nan when the entity has insufficient data.
    """
    if df_entity.empty:
        return {}

    df_e = df_entity.sort_values("timestamp_ms").copy()
    speed = pd.to_numeric(df_e["speed_m_s"], errors="coerce").values
    accel = (
        pd.to_numeric(df_e["acceleration"], errors="coerce").values
        if "acceleration" in df_e.columns
        else np.full(len(df_e), np.nan)
    )
    x = pd.to_numeric(df_e["x_m"], errors="coerce").values
    y = pd.to_numeric(df_e["y_m"], errors="coerce").values
    ts = df_e["timestamp_ms"].values.astype(np.int64)

    dist = np.hypot(x - goal_x, y - goal_y)

    # "At throw" = row whose timestamp is closest to throw_ts, within tolerance
    delta = np.abs(ts - throw_ts)
    closest_idx = int(np.argmin(delta))
    within_tol = bool(delta[closest_idx] <= THROW_TOLERANCE_MS)

    out: dict[str, float] = {
        "speed_at_throw": float(speed[closest_idx])
        if within_tol and np.isfinite(speed[closest_idx])
        else np.nan,
        "speed_mean": float(np.nanmean(speed)) if len(speed) > 0 else np.nan,
        "accel_at_throw": float(accel[closest_idx])
        if within_tol and np.isfinite(accel[closest_idx])
        else np.nan,
        "dist_change": float(dist[-1] - dist[0]) if len(dist) > 1 else np.nan,
    }
    if include_lateral:
        out["lateral_change"] = float(y[-1] - y[0]) if len(y) > 1 else np.nan

    return out


def _defense_shape_stats(
    df_defense: pd.DataFrame, goal_x: float, goal_y: float
) -> dict[str, float]:
    """
    Aggregate defense formation statistics over the window.

    Groups by timestamp and computes per-frame spread + centroid distance.
    Returns mean spread, spread change (positive = expanding), centroid movement.
    """
    if df_defense.empty or "timestamp_ms" not in df_defense.columns:
        return {
            "defense_spread_mean_window": np.nan,
            "defense_spread_change_window": np.nan,
            "defense_centroid_dist_change_window": np.nan,
        }

    spread_vals: list[float] = []
    centroid_dists: list[float] = []

    for _, grp in df_defense.groupby("timestamp_ms"):
        gx = pd.to_numeric(grp["x_m"], errors="coerce").values
        gy = pd.to_numeric(grp["y_m"], errors="coerce").values
        valid = np.isfinite(gx) & np.isfinite(gy)
        if valid.sum() < 2:
            continue
        cx, cy = gx[valid].mean(), gy[valid].mean()
        spread_vals.append(float(np.hypot(gx[valid] - cx, gy[valid] - cy).mean()))
        centroid_dists.append(float(np.hypot(cx - goal_x, cy - goal_y)))

    if not spread_vals:
        return {
            "defense_spread_mean_window": np.nan,
            "defense_spread_change_window": np.nan,
            "defense_centroid_dist_change_window": np.nan,
        }

    return {
        "defense_spread_mean_window": float(np.mean(spread_vals)),
        "defense_spread_change_window": float(spread_vals[-1] - spread_vals[0])
        if len(spread_vals) > 1
        else np.nan,
        "defense_centroid_dist_change_window": float(
            centroid_dists[-1] - centroid_dists[0]
        )
        if len(centroid_dists) > 1
        else np.nan,
    }


# ---------------------------------------------------------------------------
# Per-shot window feature calculation
# ---------------------------------------------------------------------------


def _calculate_window_features(
    df_pos_fixture: pd.DataFrame,
    shot: pd.Series,
    group_name_map: dict[str, str],
    goal_y: float = 10.0,
) -> dict[str, object]:
    """
    Extract 14 temporal window scalars for one shot from fixture-level positions.

    df_pos_fixture must already be filtered and sorted for this fixture.
    group_name_map maps Kinexon group_name → Sportradar team name.
    """
    throw_ts_raw = shot.get("throw_timestamp_ms")
    if pd.isna(throw_ts_raw):
        return {"event_id": shot["event_id"], **_NAN_WINDOW}

    throw_ts = int(throw_ts_raw)
    start_ts = throw_ts - WINDOW_MS

    # Slice to 2s window
    win = df_pos_fixture[
        (df_pos_fixture["timestamp_ms"] >= start_ts)
        & (df_pos_fixture["timestamp_ms"] <= throw_ts)
    ].copy()

    if win.empty:
        return {"event_id": shot["event_id"], **_NAN_WINDOW}

    # Map Kinexon group names to Sportradar names for team filtering
    win["group_name_sr"] = win["group_name"].map(lambda g: group_name_map.get(g, g))

    goal_x = float(shot.get("goal_position") or 0.0)
    offense_name = shot.get("team_name_offense")
    defense_name = shot.get("team_name_defense")
    shooter_id = str(shot.get("person_league_id") or "")
    gk_id = str(shot.get("goalkeeper_league_id") or "")

    is_ball = win["league_id"].astype(str).str.contains("ball", case=False, na=False)

    # Shooter
    df_offense = (
        win[(win["group_name_sr"] == offense_name) & ~is_ball]
        if offense_name
        else win.iloc[0:0]
    )
    df_shooter = df_offense[df_offense["league_id"].astype(str) == shooter_id]
    s_stats = _entity_stats(df_shooter, throw_ts, goal_x, goal_y)

    # GK
    df_defense_players = (
        win[(win["group_name_sr"] == defense_name) & ~is_ball]
        if defense_name
        else win.iloc[0:0]
    )
    df_gk = (
        df_defense_players[df_defense_players["league_id"].astype(str) == gk_id]
        if gk_id
        else df_defense_players.iloc[0:0]
    )
    gk_stats = _entity_stats(df_gk, throw_ts, goal_x, goal_y, include_lateral=True)

    # Ball — prefer sensor with row closest to throw_ts
    df_ball_all = win[is_ball]
    if not df_ball_all.empty:
        best_ball_id = (
            df_ball_all.assign(_delta=(df_ball_all["timestamp_ms"] - throw_ts).abs())
            .sort_values("_delta")["league_id"]
            .iloc[0]
        )
        df_ball = df_ball_all[df_ball_all["league_id"] == best_ball_id]
    else:
        df_ball = df_ball_all

    b_stats = _entity_stats(df_ball, throw_ts, goal_x, goal_y)

    # Defense shape
    d_stats = _defense_shape_stats(df_defense_players, goal_x, goal_y)

    # Active player fractions over last RECENT_MS (suspension / exclusion proxy)
    recent = win[win["timestamp_ms"] >= throw_ts - RECENT_MS]
    if offense_name and not recent.empty:
        n_off = recent[
            (recent["group_name_sr"] == offense_name)
            & ~is_ball.reindex(recent.index, fill_value=False)
        ]["league_id"].nunique()
        n_active_offense_frac = float(n_off) / 7.0
    else:
        n_active_offense_frac = np.nan

    if defense_name and not recent.empty:
        n_def = recent[
            (recent["group_name_sr"] == defense_name)
            & ~is_ball.reindex(recent.index, fill_value=False)
        ]["league_id"].nunique()
        n_active_defense_frac = float(n_def) / 7.0
    else:
        n_active_defense_frac = np.nan

    return {
        "event_id": shot["event_id"],
        "shooter_speed_at_throw": s_stats.get("speed_at_throw", np.nan),
        "shooter_speed_mean_window": s_stats.get("speed_mean", np.nan),
        "shooter_accel_at_throw": s_stats.get("accel_at_throw", np.nan),
        "shooter_dist_change_window": s_stats.get("dist_change", np.nan),
        "gk_speed_at_throw": gk_stats.get("speed_at_throw", np.nan),
        "gk_dist_change_window": gk_stats.get("dist_change", np.nan),
        "gk_lateral_change_window": gk_stats.get("lateral_change", np.nan),
        "ball_speed_at_throw": b_stats.get("speed_at_throw", np.nan),
        "ball_speed_mean_window": b_stats.get("speed_mean", np.nan),
        **d_stats,
        "n_active_offense_frac": n_active_offense_frac,
        "n_active_defense_frac": n_active_defense_frac,
    }


# ---------------------------------------------------------------------------
# Feature builder — snapshot + window, all fixtures
# ---------------------------------------------------------------------------


def build_context_features_from_db(
    db_path: str,
    fixture_id: Optional[str] = None,
    window_ms: int = WINDOW_MS,
    goal_y: float = 10.0,
) -> pd.DataFrame:
    """
    Load shot and position data from DuckDB; return combined feature DataFrame.

    For each fixture:
      1. calculate_xg_features() — 28+ snapshot geometry features (same as pipeline)
      2. _calculate_window_features() — 14 temporal window scalars per shot
    Merged on event_id; target = 1 if goal scored, 0 otherwise.
    """
    all_features: list[pd.DataFrame] = []

    with duckdb.connect(str(db_path), read_only=True) as con:
        fx_filter = f"AND fixture_id = '{fixture_id}'" if fixture_id else ""

        df_shots_all = con.execute(f"""
            SELECT *
            FROM shot_events
            WHERE throw_timestamp_ms IS NOT NULL
              {fx_filter}
        """).df()

        if df_shots_all.empty:
            log.warning("No shot events with throw_timestamp_ms in DB.")
            return pd.DataFrame()

        fixture_ids = df_shots_all["fixture_id"].unique().tolist()
        log.info("%d shots across %d fixture(s)", len(df_shots_all), len(fixture_ids))

        for fx_id in fixture_ids:
            df_match = con.execute(
                f"SELECT * FROM matches_normalized WHERE fixture_id = '{fx_id}'"
            ).df()
            df_shots_fx = df_shots_all[df_shots_all["fixture_id"] == fx_id].copy()
            df_pos = con.execute(f"""
                SELECT timestamp_ms, league_id, group_name,
                       x_m, y_m, speed_m_s, direction,
                       acceleration, metabolic_power
                FROM match_positions_normalized
                WHERE fixture_id = '{fx_id}'
                ORDER BY timestamp_ms
            """).df()

            if df_pos.empty:
                log.warning("No positions for fixture %s — skipping.", fx_id)
                continue

            # 1. Snapshot geometry features (delegates to pipeline module)
            df_snapshot = calculate_xg_features(
                df_match_normalized=df_match,
                df_shot_events=df_shots_fx,
                df_positions_normalized=df_pos,
                goal_y=goal_y,
            )
            if df_snapshot.empty:
                continue

            # 2. Temporal window features
            group_name_map = _build_group_name_map(df_pos, df_shots_fx)
            window_rows = [
                _calculate_window_features(df_pos, shot, group_name_map, goal_y=goal_y)
                for _, shot in df_shots_fx.iterrows()
            ]
            df_window = pd.DataFrame(window_rows)

            # 3. Merge on event_id
            df_fx = df_snapshot.merge(
                df_window.drop(columns=["fixture_id"], errors="ignore"),
                on="event_id",
                how="left",
            )
            all_features.append(df_fx)
            log.info(
                "  fixture %s: %d shots, %d features", fx_id, len(df_fx), df_fx.shape[1]
            )

    if not all_features:
        log.error("No features could be built — check DB tables.")
        return pd.DataFrame()

    df = pd.concat(all_features, ignore_index=True)
    log.info("Total: %d shots, %d feature columns", len(df), df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Fixture-grouped train/val split  (prevents same-fixture leakage)
# ---------------------------------------------------------------------------


def _fixture_grouped_split(
    df: pd.DataFrame,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """
    Hold out a random set of whole fixtures for validation.

    Row-random splits (as used in train_xg.py) allow shots from the same
    fixture to appear in both train and val, making the metric optimistic.
    Fixture-grouped splits are more conservative and better reflect
    real generalization across unseen matches.
    """
    fixture_ids = sorted(df["fixture_id"].astype(str).unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(fixture_ids)
    n_val = max(1, round(len(fixture_ids) * val_split))
    val_set = set(fixture_ids[:n_val])
    log.info(
        "Val fixtures (%d/%d): %s",
        n_val,
        len(fixture_ids),
        sorted(val_set),
    )
    train_df = df[~df["fixture_id"].isin(val_set)].copy()
    val_df = df[df["fixture_id"].isin(val_set)].copy()
    return train_df, val_df, val_set


# ---------------------------------------------------------------------------
# Training — mirrors train_xg.py with fixture-grouped split
# ---------------------------------------------------------------------------


def train_xg_context(
    df_features: pd.DataFrame,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[Pipeline, dict, pd.DataFrame, pd.Series]:
    """
    Train XGBoost xG model on combined snapshot + window features.

    Uses fixture-grouped validation split to avoid same-match leakage.
    Mirrors train_xg.py preprocessing and hyperparameters exactly:
      - median impute + standard scale for numeric features
      - OHE for categorical (currently disabled, matching train_xg.py)
      - XGBoost binary:logistic, early stopping on logloss
    Returns: (fitted sklearn Pipeline, metrics_flat dict, X_val, y_val)
    """
    log.info("Training xG context model ...")

    TARGET_COL = "target"
    CATEGORICAL_FEATURES: list[str] = []  # disabled, same as train_xg.py
    EXCLUDE = {TARGET_COL, "fixture_id", "event_id", "attack_type", "sub_type"}
    NUMERIC_FEATURES = [c for c in df_features.columns if c not in EXCLUDE]

    df = df_features.copy()
    df[NUMERIC_FEATURES] = (
        df[NUMERIC_FEATURES].astype("float64").replace({pd.NA: np.nan})
    )

    # Fixture-grouped split
    train_df, val_df, val_fixtures = _fixture_grouped_split(df, val_split, seed)
    if len(train_df) < 2 or len(val_df) < 1:
        raise ValueError(
            f"Insufficient data after fixture split: train={len(train_df)}, val={len(val_df)}. "
            "Reduce --val-split or provide more fixtures."
        )

    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET_COL].astype(int)
    X_val = val_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_val = val_df[TARGET_COL].astype(int)

    log.info(
        "Split: %d train / %d val  |  %d train shots / %d val shots",
        len(train_df["fixture_id"].unique()),
        len(val_fixtures),
        len(X_train),
        len(X_val),
    )

    # Baseline: constant prediction = training class rate
    base_rate = float(y_train.mean())
    baseline_proba = np.full(len(y_val), base_rate, dtype=float)
    baseline_metrics = {
        "baseline_auc": roc_auc_score(y_val, baseline_proba),
        "baseline_logloss": log_loss(y_val, baseline_proba),
        "baseline_brier": brier_score_loss(y_val, baseline_proba),
        "baseline_accuracy": accuracy_score(y_val, (baseline_proba >= 0.5).astype(int)),
    }

    # Preprocessing — mirrors train_xg.py exactly
    num_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )

    # Manual fit to enable early stopping with transformed eval_set
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_val_t = preprocessor.transform(X_val)
    clf.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], verbose=False)

    model = Pipeline([("preprocess", preprocessor), ("clf", clf)])

    y_val_proba = model.predict_proba(X_val)[:, 1]

    metrics_flat: dict = {
        "val_auc": float(roc_auc_score(y_val, y_val_proba)),
        "val_logloss": float(log_loss(y_val, y_val_proba)),
        "val_brier": float(brier_score_loss(y_val, y_val_proba)),
        "val_accuracy": float(accuracy_score(y_val, (y_val_proba >= 0.5).astype(int))),
        **baseline_metrics,
        "n_fixtures": int(df["fixture_id"].nunique()),
        "n_val_fixtures": len(val_fixtures),
        "n_samples_total": len(df),
        "n_samples_train": len(X_train),
        "n_samples_val": len(X_val),
        "class_balance_pos": int((df[TARGET_COL] == 1).sum()),
        "class_balance_neg": int((df[TARGET_COL] == 0).sum()),
        "n_features": len(NUMERIC_FEATURES),
    }
    log.info("xG context model metrics:\n%s", json.dumps(metrics_flat, indent=2))
    return model, metrics_flat, X_val, y_val


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XGBoost xG with spatiotemporal context")
    p.add_argument("--db", default=str(DB_PATH), help="DuckDB file path")
    p.add_argument("--fixture", default=None, help="Single fixture_id (default: all)")
    p.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of fixtures for validation",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-model",
        default=None,
        help="Path to save fitted model (joblib). E.g. data/models/xg_context.joblib",
    )
    p.add_argument(
        "--out-csv", default=None, help="Path to write per-shot predictions as CSV"
    )
    p.add_argument(
        "--out-parquet",
        default=None,
        help="Path to write per-shot predictions as Parquet",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load model from --model-path and run inference only",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Path to a pre-trained model for --eval-only",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Build features ---
    df_features = build_context_features_from_db(
        db_path=args.db,
        fixture_id=args.fixture,
    )
    if df_features.empty:
        log.error("No features built — check DB path and required tables.")
        return

    # --- Train or load model ---
    if args.eval_only:
        if not args.model_path:
            log.error("--eval-only requires --model-path.")
            return
        try:
            import joblib

            model: Pipeline = joblib.load(args.model_path)
        except ImportError:
            import pickle

            with open(args.model_path, "rb") as f:
                model = pickle.load(f)
        log.info("Loaded model from %s", args.model_path)
    else:
        model, metrics, X_val, y_val = train_xg_context(
            df_features, val_split=args.val_split, seed=args.seed
        )

        if args.out_model:
            out_path = Path(args.out_model)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import joblib

                joblib.dump(model, out_path)
            except ImportError:
                import pickle

                with open(out_path, "wb") as f:
                    pickle.dump(model, f)
            log.info("Model saved to %s", out_path)

    # --- Per-shot inference on all features ---
    df_pred = infer_xg(model, df_features, proba_col="xg")
    df_sum = summarize_fixture_xg(df_pred, proba_col="xg")

    log.info(
        "Predictions (first 10):\n%s",
        df_pred[["fixture_id", "event_id", "xg", "target"]]
        .head(10)
        .to_string(index=False),
    )
    log.info("Fixture summary:\n%s", df_sum.to_string(index=False))

    if args.out_csv:
        df_pred.to_csv(args.out_csv, index=False)
        log.info("Wrote CSV: %s", args.out_csv)

    if args.out_parquet:
        df_pred.to_parquet(args.out_parquet, index=False)
        log.info("Wrote Parquet: %s", args.out_parquet)


if __name__ == "__main__":
    main()
