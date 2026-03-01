# src/pipelines/ml/infer_xg.py
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def _required_columns_from_preprocessor(pre: ColumnTransformer) -> set[str]:
    """
    Derive the required input columns from a fitted ColumnTransformer.
    Works even if you did not explicitly persist the feature lists elsewhere.
    """
    required: set[str] = set()

    # transformers is defined pre-fit; transformers_ exists post-fit
    transformers = getattr(pre, "transformers", None) or getattr(
        pre, "transformers_", None
    )
    if not transformers:
        return required

    for name, trans, cols in transformers:
        if name == "remainder":
            continue
        if cols is None:
            continue
        if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
            required.update([str(c) for c in cols])
        elif isinstance(cols, slice):
            # rare in your setup, but handle generically
            # cannot resolve slice without feature_names_in_, so skip
            pass
        else:
            # single column
            required.add(str(cols))

    return required


def infer_xg(
    model: Pipeline,
    df_features_xg: pd.DataFrame,
    proba_col: str = "xg",
    keep_input_cols: bool = True,
) -> pd.DataFrame:
    """
    Run xG inference with the trained sklearn Pipeline (preprocess + clf).

    Output:
      - fixture_id, event_id (if present)
      - proba_col (default: "xg") = P(goal)
      - optional: keeps all input feature columns if keep_input_cols=True
    """
    if df_features_xg is None or df_features_xg.empty:
        return pd.DataFrame()

    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline, got {type(model)}")

    df = df_features_xg.copy()

    # Avoid leaking label into inference features (but keep it in output if present)
    y_true: Optional[pd.Series] = None
    if "target" in df.columns:
        y_true = df["target"].copy()

    # Validate required columns
    if "preprocess" in model.named_steps and isinstance(
        model.named_steps["preprocess"], ColumnTransformer
    ):
        required = _required_columns_from_preprocessor(model.named_steps["preprocess"])
        missing = sorted([c for c in required if c not in df.columns])
        if missing:
            raise ValueError(
                f"Missing {len(missing)} required feature columns for xG inference. "
                f"Examples: {missing[:10]}"
            )

    # Predict probabilities
    proba = model.predict_proba(df)[:, 1].astype("float64")

    # Assemble output
    base_cols = [c for c in ["fixture_id", "event_id"] if c in df.columns]
    out = df[base_cols].copy() if base_cols else pd.DataFrame(index=df.index)

    out[proba_col] = proba

    if y_true is not None:
        out["target"] = y_true.astype("int64", errors="ignore")

    if keep_input_cols:
        # Join back everything (but avoid duplicating base cols)
        extra_cols = [c for c in df.columns if c not in out.columns]
        out = pd.concat([out, df[extra_cols]], axis=1)

    # Basic hygiene
    if "fixture_id" in out.columns:
        out["fixture_id"] = out["fixture_id"].astype(str)

    logging.info(
        "xG inference done: n=%d, mean=%0.4f, sum=%0.4f",
        len(out),
        float(np.nanmean(out[proba_col])),
        float(np.nansum(out[proba_col])),
    )
    return out


def summarize_fixture_xg(
    df_pred: pd.DataFrame,
    proba_col: str = "xg",
) -> pd.DataFrame:
    """
    Fixture-level summary: sum/mean/max xG and (if present) goals count.
    """
    if df_pred is None or df_pred.empty:
        return pd.DataFrame()

    if "fixture_id" not in df_pred.columns:
        raise ValueError("df_pred must contain fixture_id for fixture summary.")

    g = df_pred.groupby("fixture_id", as_index=False)
    agg = g.agg(
        xg_sum=(proba_col, "sum"),
        xg_mean=(proba_col, "mean"),
        xg_max=(proba_col, "max"),
        n_shots=(proba_col, "count"),
    )

    if "target" in df_pred.columns:
        goals = g["target"].sum().rename(columns={"target": "goals"})
        agg = agg.merge(goals, on="fixture_id", how="left")

    return agg


if __name__ == "__main__":
    import argparse
    import pickle

    import duckdb

    try:
        import joblib  # type: ignore
    except Exception:  # pragma: no cover
        joblib = None  # type: ignore

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run xG inference locally.")
    parser.add_argument(
        "--duckdb-path",
        type=str,
        default="./data/hbl_raw.duckdb",
        help="Path to DuckDB file (default: ./data/hbl_raw.duckdb).",
    )
    parser.add_argument(
        "--features-table",
        type=str,
        default="features_xg",
        help="DuckDB table/view containing xG features (default: features_xg).",
    )
    parser.add_argument(
        "--fixture-id",
        type=str,
        default=None,
        help="Optional fixture_id filter (partition key).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        default="./data/models/xg_model_fixture.joblib",
        help="Path to persisted sklearn Pipeline (joblib or pickle).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional output CSV path.",
    )
    parser.add_argument(
        "--out-parquet",
        type=str,
        default=None,
        help="Optional output Parquet path.",
    )
    parser.add_argument(
        "--proba-col",
        type=str,
        default="xg",
        help='Name of probability column (default: "xg").',
    )
    parser.add_argument(
        "--no-keep-input-cols",
        action="store_true",
        help="If set, do NOT keep input feature columns in output.",
    )

    args = parser.parse_args()

    # --- load model ---
    model: Pipeline
    if joblib is not None:
        try:
            model = joblib.load(args.model_path)
        except Exception:
            with open(args.model_path, "rb") as f:
                model = pickle.load(f)
    else:
        with open(args.model_path, "rb") as f:
            model = pickle.load(f)

    if not isinstance(model, Pipeline):
        raise TypeError(f"Loaded model is not a sklearn Pipeline. Got: {type(model)}")

    # --- load features ---
    where = ""
    if args.fixture_id:
        where = "WHERE fixture_id = ?"

    query = f"SELECT * FROM {args.features_table} {where}"
    with duckdb.connect(args.duckdb_path, read_only=True) as conn:
        if args.fixture_id:
            df_features = conn.execute(query, [args.fixture_id]).df()
        else:
            df_features = conn.execute(query).df()

    if df_features.empty:
        logging.warning(
            "No rows loaded from %s (fixture_id=%s).",
            args.features_table,
            args.fixture_id,
        )
        raise SystemExit(0)

    # --- infer ---
    df_pred = infer_xg(
        model=model,
        df_features_xg=df_features,
        proba_col=args.proba_col,
        keep_input_cols=not args.no_keep_input_cols,
    )

    df_sum = summarize_fixture_xg(df_pred, proba_col=args.proba_col)

    logging.info("Predictions head:\n%s", df_pred.head(10).to_string(index=False))
    logging.info("Fixture summary:\n%s", df_sum.to_string(index=False))

    # --- persist outputs ---
    if args.out_csv:
        df_pred.to_csv(args.out_csv, index=False)
        logging.info("Wrote CSV: %s", args.out_csv)

    if args.out_parquet:
        df_pred.to_parquet(args.out_parquet, index=False)
        logging.info("Wrote Parquet: %s", args.out_parquet)
