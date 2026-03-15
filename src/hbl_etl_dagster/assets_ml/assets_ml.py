import duckdb
import pandas as pd
from dagster import AssetExecutionContext, Config, TableColumn, TableSchema, asset
from sklearn.pipeline import Pipeline

from src.pipelines.ml.train_xg import train_xg_model as train_xg_model_fn


class MlXgModelConfig(Config):
    db_path: str = "data/hbl_raw.duckdb"


@asset(
    io_manager_key="file_io_manager",
    group_name="ml",
    compute_kind="xgboost",
    deps=["features_xg"],
    description="xG model trained on all available fixtures.",
)
def ml_xg_model(context: AssetExecutionContext, config: MlXgModelConfig) -> Pipeline:
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
) -> Pipeline:
    """
    Train xS model for goalkeeper-centric feature table.

    :param context: Dagster execution context
    :param features_xs: xS feature DataFrame
    :return: Trained xS model pipeline
    """
    return False
