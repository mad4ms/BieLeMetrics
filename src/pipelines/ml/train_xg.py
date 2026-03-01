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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Feature List:
#     {
#         "fixture_id": fixture_id,
#         "event_id": event_id,
#         "avg_offense_distance_to_goal": avg_offense_distance,
#         "avg_defense_distance_to_goal": avg_defense_distance,
#         "shooter_distance_to_goal": shooter_distance_to_goal,
#         "shooter_distance_to_goalkeeper": shooter_distance_to_goalkeeper,
#         "goalkeeper_distance_to_goal": goalkeeper_distance_to_goal,
#         "ball_distance_to_goal": ball_distance_to_goal,
#         "ball_distance_to_goalkeeper": ball_distance_to_goalkeeper,
#         "shot_angle_to_goal": shot_angle,
#         "ball_angle_to_goal": ball_angle,
#         "angle_ball_to_goalkeeper": angle_ball_gk,
#         "num_defenders_close": num_defenders_close,
#         "closest_defender_distance": closest_defender_distance,
#         "attack_type": attack_type,
#         "sub_type": sub_type,
#         "target": target,
#     }


def train_xg_model(
    df_features_xg: pd.DataFrame,
) -> Tuple[object, Dict[str, Any]]:
    logging.info("Training xG model...")

    TARGET_COL = "target"
    CATEGORICAL_FEATURES = ["attack_type", "sub_type"]

    NUMERIC_FEATURES = [
        c
        for c in df_features_xg.columns
        if c not in {TARGET_COL, "fixture_id", "event_id", *CATEGORICAL_FEATURES}
    ]

    EXPERIMENTAL_DISABLED_FEATURES = ["sub_type", "attack_type"]
    for feat in EXPERIMENTAL_DISABLED_FEATURES:
        if feat in CATEGORICAL_FEATURES:
            CATEGORICAL_FEATURES.remove(feat)
        if feat in NUMERIC_FEATURES:
            NUMERIC_FEATURES.remove(feat)

    # --- enforce numeric hygiene ---
    df = df_features_xg.copy()
    df[NUMERIC_FEATURES] = (
        df[NUMERIC_FEATURES].astype("float64").replace({pd.NA: np.nan})
    )

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL].astype(int)

    # --- stratified split (CRITICAL) ---
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
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    # --- class imbalance handling ---
    # For xG (probability estimation), we generally do NOT want to rebalance
    # if the imbalance is mild (e.g. 60/40), as it biases the probabilities.
    # We want the model to learn the true frequency.
    # pos = (y_train == 1).sum()
    # neg = (y_train == 0).sum()
    # scale_pos_weight = neg / max(pos, 1)

    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        # scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )

    # We cannot use Pipeline.fit() directly if we want to pass eval_set with transformed data.
    # So we manually fit preprocessor, transform, then fit classifier.

    # 1. Fit preprocessor
    X_train_trans = preprocessor.fit_transform(X_train, y_train)
    X_val_trans = preprocessor.transform(X_val)

    # 2. Fit classifier with early stopping
    clf.fit(
        X_train_trans,
        y_train,
        eval_set=[(X_val_trans, y_val)],
        verbose=False,
    )

    # 3. Construct pipeline for return (inference)
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

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

    logging.info("xG Model metrics:\n%s", json.dumps(metrics_flat, indent=2))
    return model, metrics_flat, X_val, y_val


if __name__ == "__main__":
    import duckdb

    logging.basicConfig(level=logging.INFO)
    fixture_id = "00ba9627-5ca6-11f0-ac5e-5389986df98b"

    con_duckdb = "./data/hbl_raw.duckdb"
    # Tables to load:
    # features_xg: pd.DataFrame,

    with duckdb.connect(con_duckdb) as conn:
        df_features_xg = conn.execute(
            """
            SELECT *
            FROM features_xg
            """
        ).df()

        # avoid pandas.NA issues

    model, metrics, X_val, y_val = train_xg_model(df_features_xg)
    logging.info("Trained xG model.")
