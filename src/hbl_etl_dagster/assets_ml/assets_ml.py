import pandas as pd
from dagster import (
    AssetCheckResult,
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    Failure,
    Field,
    MetadataValue,
    TableColumn,
    TableSchema,
    asset,
    asset_check,
)
from sklearn.pipeline import Pipeline

from src.pipelines.ml.train_xg import train_xg_model as train_xg_model_fn


@asset(
    io_manager_key="file_io_manager",
    group_name="ml",
    compute_kind="xgboost",
    description="Trained xG model for a single fixture (partitioned by fixture_id).",
)
def ml_xg_model(
    context: AssetExecutionContext,
    features_xg: pd.DataFrame,
) -> Pipeline:
    """
    Train xG model for feature table.

    :param context: Description
    :type context: AssetExecutionContext

    :param features_xg: Description
    :type features_xg: pd.DataFrame
    :return: Description
    :rtype: object
    """
    df_features_xg = features_xg.copy()

    model, metrics, _, _ = train_xg_model_fn(df_features_xg=df_features_xg)

    context.log.info("Trained xG model.")
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


from src.pipelines.ml.train_xs import train_xs_model as train_xs_model_fn


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

    df_features_xs = features_xs.copy()

    model, metrics = train_xs_model_fn(df_features_xs=df_features_xs)

    context.log.info("Trained xS model.")

    context.add_output_metadata(
        {
            "dagster/row_count": len(df_features_xs),
            "dagster/column_schema": TableSchema(
                columns=[
                    TableColumn(
                        name=col,
                        type=str(df_features_xs[col].dtype),
                    )
                    for col in df_features_xs.columns
                ]
            ),
            "n_unique_fixtures": df_features_xs["fixture_id"].nunique(),
            **metrics,
        }
    )

    return model
