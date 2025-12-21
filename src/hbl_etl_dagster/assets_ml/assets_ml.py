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

from src.pipelines.ml.train_xg import (
    train_xg_model as train_xg_model_fn,
)


@asset(
    group_name="ml",
    compute_kind="duckdb",
    description="Trained xG model for a single fixture (partitioned by fixture_id).",
)
def ml_xg_model(
    context: AssetExecutionContext,
    features_xg: pd.DataFrame,
) -> object:
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
            "n_samples": len(df_features_xg),
            "n_unique_fixtures": df_features_xg["fixture_id"].nunique(),
            "metrics": MetadataValue.json(metrics),
        }
    )

    return model
