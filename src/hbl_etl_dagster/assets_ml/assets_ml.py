# assets_sportradar_raw.py
from dagster import (
    AssetExecutionContext,
    AssetCheckResult,
    DynamicPartitionsDefinition,
    Failure,
    Field,
    MetadataValue,
    asset,
    asset_check,
)
import pandas as pd

from sklearn.pipeline import Pipeline


from src.pipelines.ml.train_xg import (
    train_xg_model as train_xg_model_fn,
)


@asset(
    required_resource_keys=set(),  # no DuckDB needed here unless you want to persist
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

    model, metrics = train_xg_model_fn(df_features_xg=df_features_xg)

    context.log.info("Trained xG model.")
    context.add_output_metadata(
        {
            "dagster/row_count": len(df_features_xg),
            "n_unique_fixtures": df_features_xg["fixture_id"].nunique(),
            **metrics,
        }
    )

    return model
