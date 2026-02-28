# train_xs.py
import json
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Feature List (xS):
#     {
#         "fixture_id": fixture_id,
#         "event_id": event_id,
#         "goalkeeper_distance_to_goal": ...,
#         "ball_distance_to_goalkeeper": ...,
#         "ball_distance_to_goal": ...,
#         "target": target,  # 1 = save, 0 = goal
#     }


def train_xs_model(
    df_features_xs: pd.DataFrame,
) -> Tuple[Pipeline, Dict[str, Any]]:

    logging.info("Training xS model...")

    TARGET_COL = "target"

    NUMERIC_FEATURES = [
        c
        for c in df_features_xs.columns
        if c not in {TARGET_COL, "fixture_id", "event_id"}
    ]

    # --- enforce numeric hygiene ---
    df = df_features_xs.copy()
    df[NUMERIC_FEATURES] = (
        df[NUMERIC_FEATURES].astype("float64").replace({pd.NA: np.nan})
    )

    X = df[NUMERIC_FEATURES]
    y = df[TARGET_COL].astype(int)

    # --- stratified split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # --- baseline sanity check ---
    base_rate = y_train.mean()
    baseline_proba = np.full_like(y_val, base_rate, dtype=float)

    baseline_metrics = {
        "baseline_auc": roc_auc_score(y_val, baseline_proba),
        "baseline_logloss": log_loss(y_val, baseline_proba),
        "baseline_brier": brier_score_loss(y_val, baseline_proba),
        "baseline_accuracy": accuracy_score(y_val, (baseline_proba >= 0.5).astype(int)),
    }

    # --- preprocessing ---
    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    # --- class imbalance handling ---
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / max(pos, 1)

    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    # --- fit ---
    model.fit(X_train, y_train)

    # --- validation ---
    y_val_proba = model.predict_proba(X_val)[:, 1]

    metrics_raw: Dict[str, Any] = {
        "n_fixtures": df["fixture_id"].nunique(),
        "n_samples_total": len(df),
        "n_samples_train": len(X_train),
        "n_samples_val": len(X_val),
        "class_balance": {
            "neg": int((y == 0).sum()),
            "pos": int((y == 1).sum()),
        },
        "baseline": baseline_metrics,
        "model": {
            "val_auc": float(roc_auc_score(y_val, y_val_proba)),
            "val_logloss": float(log_loss(y_val, y_val_proba)),
            "val_brier": float(brier_score_loss(y_val, y_val_proba)),
            "val_accuracy": float(
                accuracy_score(y_val, (y_val_proba >= 0.5).astype(int))
            ),
        },
    }

    metrics_flat: Dict[str, float | int] = {
        # model performance
        "val_auc": metrics_raw["model"]["val_auc"],
        "val_logloss": metrics_raw["model"]["val_logloss"],
        "val_brier": metrics_raw["model"]["val_brier"],
        "val_accuracy": metrics_raw["model"]["val_accuracy"],
        # baseline
        "baseline_auc": metrics_raw["baseline"]["baseline_auc"],
        "baseline_logloss": metrics_raw["baseline"]["baseline_logloss"],
        "baseline_brier": metrics_raw["baseline"]["baseline_brier"],
        "baseline_accuracy": metrics_raw["baseline"]["baseline_accuracy"],
        # dataset stats
        "n_fixtures": metrics_raw["n_fixtures"],
        "n_samples_total": metrics_raw["n_samples_total"],
        "n_samples_train": metrics_raw["n_samples_train"],
        "n_samples_val": metrics_raw["n_samples_val"],
        "class_balance_pos": metrics_raw["class_balance"]["pos"],
        "class_balance_neg": metrics_raw["class_balance"]["neg"],
    }

    logging.info("xS Model metrics:\n%s", json.dumps(metrics_flat, indent=2))

    return model, metrics_flat


if __name__ == "__main__":
    import duckdb

    logging.basicConfig(level=logging.INFO)

    con_duckdb = "./data/hbl_raw.duckdb"

    with duckdb.connect(con_duckdb) as conn:
        df_features_xs = conn.execute(
            """
            SELECT *
            FROM features_xs
            """
        ).df()

    model, metrics = train_xs_model(df_features_xs)
    logging.info("Trained xS model.")
