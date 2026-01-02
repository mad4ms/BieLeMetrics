# src\hbl_etl_dagster\assets_ml\assets_importance.py
from pathlib import Path

import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset
from sklearn.pipeline import Pipeline

from src.pipelines.ml.importance_xg import (
    materialize_feature_importance_artifacts,
)

# Assumption (matches your pipeline style):
# - You already have an asset (e.g., `xg_model`) that returns a trained sklearn Pipeline.
# - You can load a validation slice (or you already have an asset providing it).
#
# Minimal pragmatic pattern:
# - Use the full `features_xg` table (across fixtures) as input
# - Re-create a deterministic split inside this asset (same as training) OR (better)
#   consume `X_val/y_val` from the training step if you already expose it.
#
# Below follows your style: Dagster asset orchestrates, pure function does work.


@asset(
    group_name="modeling",
    description="Compute sampled permutation importance and XGBoost gain importance for the xG model.",
)
def xg_feature_importance(
    context: AssetExecutionContext,
    features_xg: pd.DataFrame,
    ml_xg_model: Pipeline,  # sklearn Pipeline from your training asset
) -> pd.DataFrame:
    """
    Produces:
      - artifacts/xg_feature_importance/perm_importance.(png|parquet)
      - artifacts/xg_feature_importance/xgb_gain_importance.(png|parquet)

    Returns the permutation-importance table (column-level) as a DataFrame.
    """
    df = features_xg.copy()

    context.log.info("ml_xg_model type: %s", type(ml_xg_model))

    # --- build X/y consistent with training ---
    TARGET_COL = "target"
    CATEGORICAL_FEATURES = ["attack_type", "sub_type"]
    drop_cols = {TARGET_COL, "fixture_id", "event_id", *CATEGORICAL_FEATURES}

    numeric_features = [c for c in df.columns if c not in drop_cols]
    X = df[numeric_features + CATEGORICAL_FEATURES]
    y = df[TARGET_COL].astype(int)

    # --- deterministic sampling/splitting for importance ---
    # For *feature importance only*, you want a stable subset.
    # If you already persist a real validation split from training, use that instead.
    rs = 42
    idx = X.sample(n=min(2000, len(X)), random_state=rs).index
    X_val = X.loc[idx]
    y_val = y.loc[idx]

    artifact_dir = "artifacts/xg_feature_importance"
    artifacts = materialize_feature_importance_artifacts(
        model=ml_xg_model,
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

    context.add_output_metadata(
        {
            "artifact_dir": MetadataValue.path(
                str(Path(artifact_dir).resolve())
            ),
            "perm_plot": MetadataValue.path(artifacts.perm_plot_path),
            "gain_plot": MetadataValue.path(artifacts.xgb_plot_path),
            "perm_table": MetadataValue.path(artifacts.perm_table_path),
            "gain_table": MetadataValue.path(artifacts.xgb_table_path),
            "perm_top10": MetadataValue.md(
                artifacts.perm_importance.head(10).to_markdown(index=False)
            ),
            "gain_top10": MetadataValue.md(
                artifacts.xgb_gain_importance.head(10).to_markdown(index=False)
            ),
        }
    )

    context.log.info("Wrote feature-importance artifacts to %s", artifact_dir)

    # reinsert fixture_id for downstream use if needed
    artifacts.perm_importance["fixture_id"] = df["fixture_id"].mode()[0]

    # Return column-level permutation importances (most actionable)
    return artifacts.perm_importance
