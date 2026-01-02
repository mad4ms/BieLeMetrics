from typing import Any, Dict, List

import numpy as np
import pandas as pd
from dagster import AssetExecutionContext, asset
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .assets_sportradar_slow import (  # only if you really want partitions here
    fixtures_partition_def,
)

NUMERIC_FEATURES: List[str] = [
    "distance_thrower_goalkeeper",
    "distance_thrower_goal",
    "distance_goalkeeper_goal",
    "angle_player_goal",  # global rad
    "angle_ball_goal",  # global rad
    "angle_player_straight",  # deg
    "angle_ball_straight",  # deg
    "num_players_in_triangle",
    "num_players_near_thrower",
    "speed_thrower",
    "distance_covered_thrower",
    # "kinexon_distance",
]

CATEGORICAL_FEATURES: List[str] = [
    "attack_type",
]

TARGET_COL = "success"


@asset(
    required_resource_keys=set(),  # no DuckDB needed here unless you want to persist
    group_name="models",
    compute_kind="xgboost",
    description="Train XGBoost model to predict shot success from throw-time features.",
)
def xg_model_training(
    context: AssetExecutionContext,
    features_at_throw_time: pd.DataFrame,
) -> Pipeline:
    """
    Train an XGBoost binary classifier on throw-time features.

    Returns a sklearn Pipeline (preprocessing + XGBClassifier).
    """

    df = features_at_throw_time.copy()

    # Basic filtering
    if TARGET_COL not in df.columns:
        context.log.error(f"Target column '{TARGET_COL}' not found.")
        return None

    # We use only rows with non-null target
    df = df.dropna(subset=[TARGET_COL])
    if df.empty:
        context.log.warning("No rows with non-null target. Skipping training.")
        return None

    # Make sure target is binary 0/1
    y = df[TARGET_COL].astype(int)

    missing_features = [
        col for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES if col not in df.columns
    ]
    if missing_features:
        context.log.error(f"Missing required feature columns: {missing_features}")
        return None

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    # Simple imputation for numerics (if needed)
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].astype(float)
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(X[NUMERIC_FEATURES].median())

    # Categorical as string, fill NA
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("string").fillna("unknown")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    # XGBoost model
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    # Train
    model.fit(X_train, y_train)

    # Validation metrics
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)

    metrics: Dict[str, Any] = {
        "n_samples_total": len(df),
        "n_samples_train": len(X_train),
        "n_samples_val": len(X_val),
        "class_balance_total": {
            "neg": int((y == 0).sum()),
            "pos": int((y == 1).sum()),
        },
        "val_auc": float(roc_auc_score(y_val, y_val_proba)),
        "val_logloss": float(log_loss(y_val, y_val_proba)),
        "val_accuracy": float(accuracy_score(y_val, y_val_pred)),
        "val_brier_score": float(brier_score_loss(y_val, y_val_proba)),
    }

    context.log.info(
        "XGBoost validation metrics: AUC=%.3f, logloss=%.3f, acc=%.3f, brier_score=%.3f",
        metrics["val_auc"],
        metrics["val_logloss"],
        metrics["val_accuracy"],
        metrics["val_brier_score"],
    )

    context.add_output_metadata(metrics)

    return model
