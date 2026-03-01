# src\pipelines\ml\importance_xg.py

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class FeatureImportanceArtifacts:
    perm_importance: pd.DataFrame
    xgb_gain_importance: pd.DataFrame
    perm_plot_path: str
    xgb_plot_path: str
    perm_table_path: str
    xgb_table_path: str


def _plot_barh(
    df_imp: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    top_n: int,
    outpath: Path,
) -> None:
    dfp = df_imp.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]

    plt.figure(figsize=(10, max(5, 0.25 * len(dfp))))
    plt.barh(dfp["feature"], dfp[value_col])
    plt.title(title)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def compute_permutation_importance_sampled(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    scoring: str = "neg_log_loss",
    sample_size: int = 2000,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Column-level permutation importance (operates on original feature columns).
    Works with your Pipeline (preprocess + clf).
    """
    n = len(X_val)
    rs = np.random.RandomState(random_state)
    idx = rs.choice(n, size=min(sample_size, n), replace=False)

    Xs = X_val.iloc[idx]
    ys = y_val.iloc[idx]

    r = permutation_importance(
        model,
        Xs,
        ys,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    return (
        pd.DataFrame(
            {
                "feature": X_val.columns,
                "importance_mean": r.importances_mean,
                "importance_std": r.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def compute_xgb_gain_importance_transformed(
    model: Pipeline,
    *,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    Gain importance on transformed features (after OHE).
    Useful to inspect which one-hot levels matter.
    """
    pre = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    feature_names = pre.get_feature_names_out()
    booster = clf.get_booster()
    score = booster.get_score(importance_type=importance_type)  # keys: f0, f1, ...

    values = np.zeros(len(feature_names), dtype=float)
    for k, v in score.items():
        if k.startswith("f"):
            j = int(k[1:])
            if 0 <= j < len(values):
                values[j] = float(v)

    return (
        pd.DataFrame({"feature": feature_names, "gain": values})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )


def materialize_feature_importance_artifacts(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    artifact_dir: str,
    scoring: str = "neg_log_loss",
    sample_size: int = 2000,
    n_repeats: int = 10,
    random_state: int = 42,
    top_n_perm: int = 25,
    top_n_gain: int = 30,
) -> FeatureImportanceArtifacts:
    """
    Computes permutation + gain importances and writes:
      - perm_importance.parquet + perm_importance.png
      - xgb_gain_importance.parquet + xgb_gain_importance.png
    """
    outdir = Path(artifact_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    perm_df = compute_permutation_importance_sampled(
        model,
        X_val,
        y_val,
        scoring=scoring,
        sample_size=sample_size,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    gain_df = compute_xgb_gain_importance_transformed(model, importance_type="gain")

    perm_table = outdir / "perm_importance.parquet"
    gain_table = outdir / "xgb_gain_importance.parquet"
    perm_plot = outdir / "perm_importance.png"
    gain_plot = outdir / "xgb_gain_importance.png"

    perm_df.to_parquet(perm_table, index=False)
    gain_df.to_parquet(gain_table, index=False)

    _plot_barh(
        perm_df,
        value_col="importance_mean",
        title="Permutation Importance (sampled val) — higher is more important",
        top_n=top_n_perm,
        outpath=perm_plot,
    )
    _plot_barh(
        gain_df,
        value_col="gain",
        title="XGBoost Gain Importance (transformed features)",
        top_n=top_n_gain,
        outpath=gain_plot,
    )

    return FeatureImportanceArtifacts(
        perm_importance=perm_df,
        xgb_gain_importance=gain_df,
        perm_plot_path=str(perm_plot),
        xgb_plot_path=str(gain_plot),
        perm_table_path=str(perm_table),
        xgb_table_path=str(gain_table),
    )


if __name__ == "__main__":
    import duckdb

    from src.pipelines.ml.train_xg import (  # adjust import to your project
        train_xg_model,
    )

    logging.basicConfig(level=logging.INFO)

    con_duckdb = "./data/hbl_raw.duckdb"
    artifact_dir = "./artifacts/xg_feature_importance_local"

    # 1) Load features
    with duckdb.connect(con_duckdb) as conn:
        df_features_xg = conn.execute("SELECT * FROM features_xg").df()

    # 2) Train + get validation split from training API
    model, metrics, X_val, y_val = train_xg_model(df_features_xg)
    logging.info("Loaded validation split from training function.")

    logging.info("Model metrics:\n%s", json.dumps(metrics, indent=2))

    # plot of distribution of goalkeeper_distance_to_goal
    plt.figure(figsize=(8, 6))
    plt.hist(X_val["goalkeeper_distance_to_goal"], bins=30, color="blue", alpha=0.7)
    plt.title("Distribution of goalkeeper_distance_to_goal")
    plt.xlabel("goalkeeper_distance_to_goal")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # 3) Compute + write artifacts
    artifacts = materialize_feature_importance_artifacts(
        model=model,
        X_val=X_val,
        y_val=y_val,
        artifact_dir=artifact_dir,
        scoring="neg_log_loss",
        sample_size=2000,
        n_repeats=10,
        random_state=42,
        top_n_perm=25,
        top_n_gain=30,
    )

    # 4) Print top results
    print("\nTop permutation importances:")
    print(artifacts.perm_importance.head(20).to_string(index=False))

    print("\nTop XGB gain importances (transformed features):")
    print(artifacts.xgb_gain_importance.head(30).to_string(index=False))

    logging.info("Wrote artifacts:")
    logging.info("  perm plot : %s", artifacts.perm_plot_path)
    logging.info("  gain plot : %s", artifacts.xgb_plot_path)
    logging.info("  perm table: %s", artifacts.perm_table_path)
    logging.info("  gain table: %s", artifacts.xgb_table_path)
