from pathlib import Path
from typing import Any

import duckdb
from dagster import AssetExecutionContext, Config, MetadataValue, asset

from src.hbl_etl_dagster.utils.metadata import markdown_table
from src.pipelines.ml.importance_xg import materialize_feature_importance_artifacts


class XgFeatureImportanceConfig(Config):
    db_path: str = "data/hbl_raw.duckdb"


@asset(
    group_name="modeling",
    deps=["features_xg"],
    description="Compute sampled permutation importance and XGBoost gain importance for the xG model.",
)
def xg_feature_importance(
    context: AssetExecutionContext,
    config: XgFeatureImportanceConfig,
    ml_xg_model: Any,
) -> object:
    """
    Produces:
      - artifacts/xg_feature_importance/perm_importance.(png|parquet)
      - artifacts/xg_feature_importance/xgb_gain_importance.(png|parquet)

    Returns the permutation-importance table (column-level) as a DataFrame.
    """
    with duckdb.connect(config.db_path, read_only=True) as con:
        df = con.execute("SELECT * FROM features_xg").df()

    context.log.info(
        "Loaded %d rows from features_xg across %d fixtures for importance analysis",
        len(df),
        df["fixture_id"].nunique(),
    )
    context.log.info("ml_xg_model type: %s", type(ml_xg_model))

    TARGET_COL = "target"
    CATEGORICAL_FEATURES = ["attack_type", "sub_type"]
    drop_cols = {TARGET_COL, "fixture_id", "event_id", *CATEGORICAL_FEATURES}

    numeric_features = [c for c in df.columns if c not in drop_cols]
    X = df[numeric_features + CATEGORICAL_FEATURES]
    y = df[TARGET_COL].astype(int)

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
            "artifact_dir": MetadataValue.path(str(Path(artifact_dir).resolve())),
            "perm_plot": MetadataValue.path(artifacts.perm_plot_path),
            "gain_plot": MetadataValue.path(artifacts.xgb_plot_path),
            "perm_table": MetadataValue.path(artifacts.perm_table_path),
            "gain_table": MetadataValue.path(artifacts.xgb_table_path),
            "perm_top10": MetadataValue.md(
                markdown_table(artifacts.perm_importance, n=10)
            ),
            "gain_top10": MetadataValue.md(
                markdown_table(artifacts.xgb_gain_importance, n=10)
            ),
        }
    )

    context.log.info("Wrote feature-importance artifacts to %s", artifact_dir)

    return artifacts.perm_importance
