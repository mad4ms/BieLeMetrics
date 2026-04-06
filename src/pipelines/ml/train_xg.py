import json
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

XG_RANDOM_STATE = 42
XG_MAX_CV_FOLDS = 5

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


def _split_train_val_indices(
    y: pd.Series,
    fixture_ids: pd.Series,
    random_state: int = XG_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, str]:
    fixture_ids = fixture_ids.astype(str)
    n_unique_fixtures = fixture_ids.nunique()

    if n_unique_fixtures >= 2:
        try:
            splitter = StratifiedGroupKFold(
                n_splits=min(XG_MAX_CV_FOLDS, n_unique_fixtures),
                shuffle=True,
                random_state=random_state,
            )
            train_idx, val_idx = next(
                splitter.split(np.zeros(len(y)), y, groups=fixture_ids)
            )
            return train_idx, val_idx, "fixture_stratified"
        except ValueError as exc:
            logging.warning(
                "Falling back to row-level stratified split because grouped split failed: %s",
                exc,
            )

    train_idx, val_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    return train_idx, val_idx, "row_stratified_fallback"


def _build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def _build_xgb_classifier() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=800,
        max_depth=4,
        min_child_weight=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.75,
        gamma=0.1,
        reg_alpha=0.15,
        reg_lambda=2.5,
        random_state=XG_RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )


def _fit_xgb_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    clf = _build_xgb_classifier()

    X_train_trans = preprocessor.fit_transform(X_train, y_train)
    X_val_trans = preprocessor.transform(X_val)
    clf.fit(
        X_train_trans,
        y_train,
        eval_set=[(X_val_trans, y_val)],
        verbose=False,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )


def _run_grouped_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    fixture_ids: pd.Series,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = XG_RANDOM_STATE,
) -> Dict[str, float | int | str]:
    fixture_ids = fixture_ids.astype(str)
    n_unique_fixtures = fixture_ids.nunique()
    if n_unique_fixtures < 3:
        return {
            "grouped_cv_folds": 0,
            "grouped_cv_strategy": "skipped_insufficient_fixtures",
        }

    try:
        n_splits = min(XG_MAX_CV_FOLDS, n_unique_fixtures)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
    except ValueError as exc:
        logging.warning("Skipping grouped CV because fold setup failed: %s", exc)
        return {
            "grouped_cv_folds": 0,
            "grouped_cv_strategy": "setup_failed",
        }

    fold_rows: list[dict[str, float]] = []
    for fold_index, (fold_train_idx, fold_val_idx) in enumerate(
        splitter.split(np.zeros(len(y)), y, groups=fixture_ids),
        start=1,
    ):
        X_fold_train = X.iloc[fold_train_idx].copy()
        X_fold_val = X.iloc[fold_val_idx].copy()
        y_fold_train = y.iloc[fold_train_idx].copy()
        y_fold_val = y.iloc[fold_val_idx].copy()

        fold_model = _fit_xgb_pipeline(
            X_fold_train,
            y_fold_train,
            X_fold_val,
            y_fold_val,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        y_fold_proba = fold_model.predict_proba(X_fold_val)[:, 1]

        fold_rows.append(
            {
                "fold": float(fold_index),
                "auc": float(roc_auc_score(y_fold_val, y_fold_proba)),
                "logloss": float(log_loss(y_fold_val, y_fold_proba)),
                "brier": float(brier_score_loss(y_fold_val, y_fold_proba)),
                "accuracy": float(
                    accuracy_score(y_fold_val, (y_fold_proba >= 0.5).astype(int))
                ),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    return {
        "grouped_cv_folds": int(len(fold_df)),
        "grouped_cv_strategy": "fixture_stratified_kfold",
        "grouped_cv_auc_mean": float(fold_df["auc"].mean()),
        "grouped_cv_auc_std": float(fold_df["auc"].std(ddof=0)),
        "grouped_cv_logloss_mean": float(fold_df["logloss"].mean()),
        "grouped_cv_logloss_std": float(fold_df["logloss"].std(ddof=0)),
        "grouped_cv_brier_mean": float(fold_df["brier"].mean()),
        "grouped_cv_brier_std": float(fold_df["brier"].std(ddof=0)),
        "grouped_cv_accuracy_mean": float(fold_df["accuracy"].mean()),
        "grouped_cv_accuracy_std": float(fold_df["accuracy"].std(ddof=0)),
    }


def train_xg_model(
    df_features_xg: pd.DataFrame,
) -> Tuple[object, Dict[str, Any], pd.DataFrame, pd.Series]:
    logging.info("Training xG model...")

    TARGET_COL = "target"
    CATEGORICAL_FEATURES = ["attack_type", "sub_type"]

    NUMERIC_FEATURES = [
        c
        for c in df_features_xg.columns
        if c not in {TARGET_COL, "fixture_id", "event_id", *CATEGORICAL_FEATURES}
    ]

    # --- enforce numeric hygiene ---
    df = df_features_xg.copy()
    df[NUMERIC_FEATURES] = (
        df[NUMERIC_FEATURES].astype("float64").replace({pd.NA: np.nan})
    )

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL].astype(int)
    fixture_ids = df["fixture_id"].astype(str)

    # --- grouped split by fixture to avoid match-level leakage ---
    train_idx, val_idx, split_strategy = _split_train_val_indices(
        y=y,
        fixture_ids=fixture_ids,
        random_state=XG_RANDOM_STATE,
    )
    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_val = y.iloc[val_idx].copy()
    fixture_ids_train = fixture_ids.iloc[train_idx]
    fixture_ids_val = fixture_ids.iloc[val_idx]

    # --- baseline sanity check ---
    base_rate = y_train.mean()
    baseline_proba = np.full_like(y_val, base_rate, dtype=float)

    baseline_metrics = {
        "baseline_auc": roc_auc_score(y_val, baseline_proba),
        "baseline_logloss": log_loss(y_val, baseline_proba),
        "baseline_brier": brier_score_loss(y_val, baseline_proba),
        "baseline_accuracy": accuracy_score(y_val, (baseline_proba >= 0.5).astype(int)),
    }

    grouped_cv_metrics = _run_grouped_cross_validation(
        X,
        y,
        fixture_ids,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        random_state=XG_RANDOM_STATE,
    )

    model = _fit_xgb_pipeline(
        X_train,
        y_train,
        X_val,
        y_val,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
    model._bielemetrics_training_metadata = {
        "target_col": TARGET_COL,
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "feature_columns": list(NUMERIC_FEATURES + CATEGORICAL_FEATURES),
        "split_strategy": split_strategy,
        **grouped_cv_metrics,
        "train_fixture_ids": sorted(fixture_ids_train.unique().tolist()),
        "val_fixture_ids": sorted(fixture_ids_val.unique().tolist()),
        "n_unique_fixtures_total": int(fixture_ids.nunique()),
        "n_unique_fixtures_train": int(fixture_ids_train.nunique()),
        "n_unique_fixtures_val": int(fixture_ids_val.nunique()),
    }

    # --- validation ---
    y_val_proba = model.predict_proba(X_val)[:, 1]

    metrics_raw: Dict[str, Any] = {
        "n_fixtures": df["fixture_id"].nunique(),
        "n_fixtures_train": int(fixture_ids_train.nunique()),
        "n_fixtures_val": int(fixture_ids_val.nunique()),
        "n_samples_total": len(df),
        "n_samples_train": len(X_train),
        "n_samples_val": len(X_val),
        "split_strategy": split_strategy,
        "grouped_cv": grouped_cv_metrics,
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
    metrics_flat: Dict[str, Any] = {
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
        "n_fixtures_train": metrics_raw["n_fixtures_train"],
        "n_fixtures_val": metrics_raw["n_fixtures_val"],
        "n_samples_total": metrics_raw["n_samples_total"],
        "n_samples_train": metrics_raw["n_samples_train"],
        "n_samples_val": metrics_raw["n_samples_val"],
        "class_balance_pos": metrics_raw["class_balance"]["pos"],
        "class_balance_neg": metrics_raw["class_balance"]["neg"],
        "split_strategy": metrics_raw["split_strategy"],
        **{
            key: value
            for key, value in grouped_cv_metrics.items()
            if isinstance(value, (int, float, str))
        },
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
