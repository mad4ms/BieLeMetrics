# src/hbl_etl_dagster/assets_ml/assets_xg_inference.py
import pandas as pd
from dagster import (
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    MetadataValue,
    TableColumn,
    TableSchema,
    asset,
)
from sklearn.pipeline import Pipeline

from src.pipelines.ml.infer_xg import infer_xg, summarize_fixture_xg


fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")


@asset(
    io_manager_key="io_manager",
    group_name="ml",
    compute_kind="xgboost",
    description="Per-shot xG predictions for one fixture partition (fixture_id).",
    partitions_def=fixtures_partition_def,
)
def xg_predictions(
    context: AssetExecutionContext,
    ml_xg_model: Pipeline,
    features_xg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produces event-level xG per shot.

    Notes:
    - Output is partitioned by fixture_id and stored in DuckDB.
    - The DuckDB IO manager uses fixture_id as default partition column. :contentReference[oaicite:1]{index=1}
    """
    df_features = features_xg.copy()
    if df_features.empty:
        context.log.warning(
            "features_xg is empty for partition=%s", context.partition_key
        )
        return pd.DataFrame()

    df_pred = infer_xg(
        model=ml_xg_model,
        df_features_xg=df_features,
        proba_col="xg",
        keep_input_cols=True,
    )

    # Metadata
    md = {
        "dagster/row_count": len(df_pred),
        "dagster/column_schema": TableSchema(
            columns=[
                TableColumn(name=c, type=str(df_pred[c].dtype))
                for c in df_pred.columns
            ]
        ),
        "fixture_id": context.partition_key,
        "xg_sum": float(df_pred["xg"].sum()),
        "xg_mean": float(df_pred["xg"].mean()),
        "preview": MetadataValue.md(df_pred.head(20).to_markdown(index=False)),
    }
    context.add_output_metadata(md)
    return df_pred


@asset(
    io_manager_key="io_manager",
    group_name="ml",
    compute_kind="pandas",
    description="Fixture-level xG summary (sum/mean/max, shots, optional goals).",
    partitions_def=fixtures_partition_def,
)
def xg_fixture_summary(
    context: AssetExecutionContext,
    xg_predictions: pd.DataFrame,
) -> pd.DataFrame:
    df = xg_predictions.copy()
    if df.empty:
        return pd.DataFrame()

    df_sum = summarize_fixture_xg(df, proba_col="xg")

    context.add_output_metadata(
        {
            "dagster/row_count": len(df_sum),
            "fixture_id": context.partition_key,
            "preview": MetadataValue.md(df_sum.to_markdown(index=False)),
        }
    )
    return df_sum
