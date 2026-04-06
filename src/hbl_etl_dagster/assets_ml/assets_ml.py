import json
from pathlib import Path
from typing import Any

import duckdb
import joblib
import pandas as pd
from dagster import AssetExecutionContext, Config, TableColumn, TableSchema, asset

from src.pipelines.ml.train_xg import train_xg_model as train_xg_model_fn

MODEL_PATH = Path("data/models/xg_context.joblib")
MODEL_METADATA_PATH = Path("data/models/xg_context_metadata.json")


class MlXgModelConfig(Config):
    db_path: str = "data/hbl_raw.duckdb"


@asset(
    io_manager_key="file_io_manager",
    group_name="ml",
    compute_kind="xgboost",
    deps=["features_xg"],
    description="xG model trained on all available fixtures.",
)
def ml_xg_model(context: AssetExecutionContext, config: MlXgModelConfig) -> Any:
    """
    Train xG model on all features_xg rows across every fixture.
    Reads directly from DuckDB to bypass the per-partition IO manager.
    """
    with duckdb.connect(config.db_path, read_only=True) as con:
        df_features_xg = con.execute("SELECT * FROM features_xg").df()

    context.log.info(
        "Loaded %d rows from features_xg across %d fixtures",
        len(df_features_xg),
        df_features_xg["fixture_id"].nunique(),
    )

    model, metrics, _, _ = train_xg_model_fn(df_features_xg=df_features_xg)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    training_metadata = dict(getattr(model, "_bielemetrics_training_metadata", {}))
    training_metadata["artifact_path"] = str(MODEL_PATH)
    training_metadata["db_path"] = config.db_path
    training_metadata["metrics"] = metrics
    MODEL_METADATA_PATH.write_text(
        json.dumps(training_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    context.log.info("Persisted xG model to %s", MODEL_PATH)

    context.add_output_metadata(
        {
            "dagster/row_count": len(df_features_xg),
            "dagster/column_schema": TableSchema(
                columns=[
                    TableColumn(name=col, type=str(df_features_xg[col].dtype))
                    for col in df_features_xg.columns
                ]
            ),
            "n_unique_fixtures": df_features_xg["fixture_id"].nunique(),
            "model_path": str(MODEL_PATH),
            "model_metadata_path": str(MODEL_METADATA_PATH),
            **metrics,
        }
    )

    return model


@asset(
    io_manager_key="file_io_manager",
    group_name="ml",
    compute_kind="xgboost",
    description="Trained xS model (expected save) over goalkeeper-centric features.",
)
def ml_xs_model(
    context: AssetExecutionContext,
    features_xs: pd.DataFrame,
) -> Any:
    """
    Train xS model for goalkeeper-centric feature table.

    :param context: Dagster execution context
    :param features_xs: xS feature DataFrame
    :return: Trained xS model pipeline
    """
    return False
