"""
ML debug commands — model training, evaluation and feature analysis.

Usage:
    uv run python scripts/debug/ml.py train
    uv run python scripts/debug/ml.py evaluate
    uv run python scripts/debug/ml.py analyze models
    uv run python scripts/debug/ml.py analyze features
"""

import argparse

import duckdb

from _common import DB_PATH, RUN_CONFIG_INFO_ONLY, get_job, report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _subgroup_metric_rows(df_pred, *, group_col: str, display_name: str):
    import pandas as pd
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    subgroup_df = df_pred[[group_col, "target", "xg"]].copy()
    subgroup_df[group_col] = subgroup_df[group_col].fillna("UNKNOWN").astype(str)

    rows = []
    for group_value, group_frame in subgroup_df.groupby(group_col, dropna=False):
        y_true = group_frame["target"].astype(int)
        y_pred = group_frame["xg"].astype(float)
        row = {
            display_name: group_value,
            "shots": int(len(group_frame)),
            "goals": int(y_true.sum()),
            "goal_rate": float(y_true.mean()),
            "xg_rate": float(y_pred.mean()),
            "xg_sum": float(y_pred.sum()),
            "xg_minus_goals": float(y_pred.sum() - y_true.sum()),
            "auc": float("nan"),
            "logloss": float("nan"),
            "brier": float("nan"),
        }
        if y_true.nunique() > 1:
            row["auc"] = float(roc_auc_score(y_true, y_pred))
            row["logloss"] = float(log_loss(y_true, y_pred, labels=[0, 1]))
            row["brier"] = float(brier_score_loss(y_true, y_pred))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["shots", "goals"], ascending=[False, False])


def _distance_band_metric_rows(df_pred):
    import numpy as np
    import pandas as pd

    distance_frame = df_pred[["shooter_distance_to_goal", "target", "xg"]].copy()
    distance_frame["distance_band"] = pd.cut(
        distance_frame["shooter_distance_to_goal"],
        bins=[-np.inf, 6.0, 9.0, 12.0, np.inf],
        labels=["<6m", "6-9m", "9-12m", "12m+"],
        right=False,
    )
    distance_frame["distance_band"] = (
        distance_frame["distance_band"].astype(object).fillna("UNKNOWN")
    )
    return _subgroup_metric_rows(
        distance_frame,
        group_col="distance_band",
        display_name="distance_band",
    )


def _print_subgroup_table(summary_df, *, label_col: str, title: str):
    print(f"\n{title}:")
    print(
        f"  {label_col:<18}  {'shots':>6}  {'goals':>6}  {'goal%':>7}  {'xg%':>7}  {'xg-goals':>9}  {'auc':>7}  {'logloss':>8}  {'brier':>7}"
    )
    for _, row in summary_df.iterrows():
        auc_text = f"{row['auc']:.4f}" if row["auc"] == row["auc"] else "n/a"
        logloss_text = (
            f"{row['logloss']:.4f}" if row["logloss"] == row["logloss"] else "n/a"
        )
        brier_text = f"{row['brier']:.4f}" if row["brier"] == row["brier"] else "n/a"
        print(
            f"  {str(row[label_col]):<18}  {int(row['shots']):>6}  {int(row['goals']):>6}  {100 * row['goal_rate']:>6.1f}%  {100 * row['xg_rate']:>6.1f}%  {row['xg_minus_goals']:>9.2f}  {auc_text:>7}  {logloss_text:>8}  {brier_text:>7}"
        )


def _analyze_models():
    """Show trained model info and performance."""
    import os

    print("\n" + "=" * 70)
    print("TRAINED MODELS")
    print("=" * 70)

    model_path = "data/models/xg_context.joblib"
    if os.path.exists(model_path):
        import joblib

        from src.pipelines.ml.infer_xg import _required_columns_from_preprocessor

        model = joblib.load(model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print("\nxG Context Model (XGBoost):")
        print(f"  Path: {model_path}")
        print(f"  Size: {size_mb:.2f} MB")

        training_metadata = getattr(model, "_bielemetrics_training_metadata", None)
        if training_metadata:
            print(
                f"  Split strategy: {training_metadata.get('split_strategy', 'unknown')}"
            )
            print(
                f"  Train/val fixtures: {training_metadata.get('n_unique_fixtures_train', '?')}/"
                f"{training_metadata.get('n_unique_fixtures_val', '?')}"
            )
            print(
                f"  Declared feature columns: {len(training_metadata.get('feature_columns', []))}"
            )
            grouped_cv_folds = training_metadata.get("grouped_cv_folds", 0)
            if grouped_cv_folds:
                print(
                    f"  Grouped CV ({grouped_cv_folds} folds): "
                    f"AUC {training_metadata.get('grouped_cv_auc_mean', float('nan')):.4f} "
                    f"+/- {training_metadata.get('grouped_cv_auc_std', float('nan')):.4f}, "
                    f"LogLoss {training_metadata.get('grouped_cv_logloss_mean', float('nan')):.4f} "
                    f"+/- {training_metadata.get('grouped_cv_logloss_std', float('nan')):.4f}, "
                    f"Brier {training_metadata.get('grouped_cv_brier_mean', float('nan')):.4f} "
                    f"+/- {training_metadata.get('grouped_cv_brier_std', float('nan')):.4f}"
                )

        if hasattr(model, "named_steps") and "preprocess" in model.named_steps:
            required_columns = sorted(
                _required_columns_from_preprocessor(model.named_steps["preprocess"])
            )
            print(f"  Required feature columns: {len(required_columns)}")

            with duckdb.connect(DB_PATH, read_only=True) as con:
                has_features = con.execute(
                    "SELECT count(*) FROM information_schema.tables WHERE table_name = 'features_xg'"
                ).fetchone()[0]
                if has_features:
                    current_columns = {
                        row[1]
                        for row in con.execute(
                            "PRAGMA table_info('features_xg')"
                        ).fetchall()
                    }
                    missing_columns = sorted(
                        column
                        for column in required_columns
                        if column not in current_columns
                    )
                    extra_columns = sorted(
                        column
                        for column in current_columns
                        if column not in set(required_columns)
                        and column not in {"fixture_id", "event_id", "target"}
                    )
                    if missing_columns:
                        print(
                            f"  Schema drift: current features_xg is missing {len(missing_columns)} model-required columns"
                        )
                        print(f"    Examples: {', '.join(missing_columns[:5])}")
                    else:
                        print("  Schema drift: no missing model-required columns")
                    if extra_columns:
                        print(
                            f"  Extra current feature columns not used by model: {len(extra_columns)}"
                        )
                        print(f"    Examples: {', '.join(extra_columns[:5])}")

    for model_name in ["hstt_v1.pt", "hstt_v1_binary.pt"]:
        model_path = f"artifacts/{model_name}"
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"\nTransformer Model: {model_name}")
            print(f"  Path: {model_path}")
            print(f"  Size: {size_mb:.2f} MB")

    with duckdb.connect(DB_PATH, read_only=True) as con:
        if con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'xg_model_training'"
        ).fetchone()[0]:
            print("\nTraining data: xg_model_training table exists in DuckDB")


def _analyze_features():
    """Show feature statistics and distributions."""
    with duckdb.connect(DB_PATH, read_only=True) as con:
        print("\n" + "=" * 70)
        print("FEATURE ANALYSIS")
        print("=" * 70)

        if con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'features_xg'"
        ).fetchone()[0]:
            stats = con.execute(
                """
                SELECT
                    count(*) as n_records,
                    count(DISTINCT fixture_id) as n_fixtures,
                    min(shooter_distance_to_goal) as min_distance,
                    max(shooter_distance_to_goal) as max_distance,
                    avg(shooter_distance_to_goal) as mean_distance,
                    min(shot_angle_to_goal) as min_angle,
                    max(shot_angle_to_goal) as max_angle,
                    avg(shot_angle_to_goal) as mean_angle
                FROM features_xg
                """
            ).fetchone()

            (
                n_records,
                n_fixtures,
                min_dist,
                max_dist,
                mean_dist,
                min_ang,
                max_ang,
                mean_ang,
            ) = stats

            print("\nxG Features (features_xg):")
            print(f"  Total records: {n_records:,}")
            print(f"  Fixtures: {n_fixtures}")
            print("\n  Shot Distance (m):")
            print(f"    Range: [{min_dist:.1f}, {max_dist:.1f}]")
            print(f"    Mean: {mean_dist:.1f}")
            print("\n  Shot Angle (radians):")
            print(f"    Range: [{min_ang:.1f}, {max_ang:.1f}]")
            print(f"    Mean: {mean_ang:.1f}")

            goal_stats = con.execute(
                "SELECT count(*), sum(coalesce(target, 0)) FROM features_xg"
            ).fetchone()
            n_total, n_goals = goal_stats
            print(
                f"\n  Goals vs non-goals: {n_goals}/{n_total} ({100*n_goals/n_total:.1f}% goal rate)"
            )
        else:
            print("\nfeatures_xg table not found. Run the fixture pipeline first.")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_train(args):
    print("Running xg_training_job …")
    result = get_job("xg_training_job").execute_in_process(
        run_config=RUN_CONFIG_INFO_ONLY
    )
    report(result)


def cmd_analyze(args):
    if args.analysis_type == "models":
        _analyze_models()
    elif args.analysis_type == "features":
        _analyze_features()


def cmd_evaluate(args):
    """Evaluate the trained xG model on features_xg."""
    import os

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        auc,
        brier_score_loss,
        confusion_matrix,
        log_loss,
        precision_recall_curve,
        roc_curve,
    )

    from src.pipelines.ml.infer_xg import _required_columns_from_preprocessor, infer_xg

    model_path = "data/models/xg_context.joblib"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Train the model first.")
        return

    model = joblib.load(model_path)
    training_metadata = getattr(model, "_bielemetrics_training_metadata", {})

    with duckdb.connect(DB_PATH, read_only=True) as con:
        if not con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'features_xg'"
        ).fetchone()[0]:
            print("features_xg table not found. Run the fixture pipeline first.")
            return

        df = con.execute("SELECT * FROM features_xg WHERE target IS NOT NULL").df()

    if df.empty:
        print("No features available for evaluation.")
        return

    evaluation_scope = "full table"
    val_fixture_ids = training_metadata.get("val_fixture_ids") or []
    if val_fixture_ids:
        df = df[df["fixture_id"].astype(str).isin(val_fixture_ids)].copy()
        evaluation_scope = "held-out validation fixtures"

    if df.empty:
        print("No evaluation rows remain after applying validation fixture filtering.")
        return

    if not hasattr(model, "named_steps") or "preprocess" not in model.named_steps:
        print("Loaded model does not expose a preprocess step; cannot evaluate safely.")
        return

    required_columns = sorted(
        _required_columns_from_preprocessor(model.named_steps["preprocess"])
    )
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    for column in missing_columns:
        df[column] = pd.Series(np.nan, index=df.index, dtype="float64")

    df_pred = infer_xg(
        model=model,
        df_features_xg=df,
        proba_col="xg",
        keep_input_cols=True,
    )

    y = df_pred["target"].astype(int).to_numpy()
    y_pred_proba = df_pred["xg"].astype(float).to_numpy()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)

    brier = brier_score_loss(y, y_pred_proba)
    logloss = log_loss(y, y_pred_proba)

    calibration_frame = pd.DataFrame({"y_true": y, "y_pred": y_pred_proba})
    calibration_frame["calibration_bin"] = pd.qcut(
        calibration_frame["y_pred"],
        q=min(10, len(calibration_frame)),
        duplicates="drop",
    )
    calibration_summary = (
        calibration_frame.groupby("calibration_bin", observed=False)
        .agg(
            n=("y_true", "size"),
            mean_pred=("y_pred", "mean"),
            goal_rate=("y_true", "mean"),
        )
        .reset_index(drop=True)
    )
    calibration_summary["abs_gap"] = (
        calibration_summary["mean_pred"] - calibration_summary["goal_rate"]
    ).abs()
    expected_calibration_error = float(
        (calibration_summary["n"] * calibration_summary["abs_gap"]).sum()
        / calibration_summary["n"].sum()
    )
    max_calibration_gap = float(calibration_summary["abs_gap"].max())

    threshold_rows = []
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        y_threshold = (y_pred_proba >= threshold).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y, y_threshold).ravel()
        precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        specificity_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
        f1_t = (
            2 * (precision_t * recall_t) / (precision_t + recall_t)
            if (precision_t + recall_t) > 0
            else 0
        )
        threshold_rows.append(
            {
                "threshold": threshold,
                "precision": precision_t,
                "recall": recall_t,
                "specificity": specificity_t,
                "f1": f1_t,
                "predicted_positive_rate": float(y_threshold.mean()),
            }
        )
    threshold_summary = pd.DataFrame(threshold_rows)
    best_threshold_row = threshold_summary.sort_values(
        ["f1", "precision", "recall"], ascending=False
    ).iloc[0]

    df_pred["pred_goal_0_5"] = y_pred

    with duckdb.connect(DB_PATH, read_only=True) as con:
        sportradar_fixture_summary = con.execute(
            """
            SELECT
                fixture_id,
                count(*) AS sportradar_rows,
                sum(CASE WHEN success THEN 1 ELSE 0 END) AS sportradar_goals
            FROM match_events_normalized_goals
            GROUP BY 1
            """
        ).df()
        shot_events_fixture_summary = con.execute(
            """
            SELECT
                fixture_id,
                count(*) AS shot_event_rows,
                sum(CASE WHEN success THEN 1 ELSE 0 END) AS shot_event_goals
            FROM shot_events
            GROUP BY 1
            """
        ).df()
        features_fixture_summary = con.execute(
            """
            SELECT
                fixture_id,
                count(*) AS feature_rows,
                sum(coalesce(target, 0)) AS feature_goals
            FROM features_xg
            GROUP BY 1
            """
        ).df()

    fixture_eval_summary = (
        df_pred.groupby("fixture_id", as_index=False)
        .agg(
            eval_rows=("target", "size"),
            eval_goals=("target", "sum"),
            predicted_goals_0_5=("pred_goal_0_5", "sum"),
            xg_sum=("xg", "sum"),
            xg_mean=("xg", "mean"),
        )
        .merge(sportradar_fixture_summary, on="fixture_id", how="left")
        .merge(shot_events_fixture_summary, on="fixture_id", how="left")
        .merge(features_fixture_summary, on="fixture_id", how="left")
    )
    fill_zero_cols = [
        "sportradar_rows",
        "sportradar_goals",
        "shot_event_rows",
        "shot_event_goals",
        "feature_rows",
        "feature_goals",
    ]
    fixture_eval_summary[fill_zero_cols] = fixture_eval_summary[fill_zero_cols].fillna(
        0
    )
    for column in fill_zero_cols + ["eval_rows", "eval_goals", "predicted_goals_0_5"]:
        fixture_eval_summary[column] = fixture_eval_summary[column].astype(int)

    fixture_eval_summary["shot_event_row_loss"] = (
        fixture_eval_summary["sportradar_rows"]
        - fixture_eval_summary["shot_event_rows"]
    )
    fixture_eval_summary["feature_row_loss"] = (
        fixture_eval_summary["sportradar_rows"] - fixture_eval_summary["feature_rows"]
    )
    fixture_eval_summary["shot_event_goal_loss"] = (
        fixture_eval_summary["sportradar_goals"]
        - fixture_eval_summary["shot_event_goals"]
    )
    fixture_eval_summary["feature_goal_loss"] = (
        fixture_eval_summary["sportradar_goals"] - fixture_eval_summary["feature_goals"]
    )
    fixture_eval_summary["xg_minus_goals"] = (
        fixture_eval_summary["xg_sum"] - fixture_eval_summary["eval_goals"]
    )

    coverage_totals = fixture_eval_summary[
        [
            "sportradar_rows",
            "sportradar_goals",
            "shot_event_rows",
            "shot_event_goals",
            "feature_rows",
            "feature_goals",
        ]
    ].sum()
    fixture_loss_rows = fixture_eval_summary[
        (fixture_eval_summary["shot_event_row_loss"] != 0)
        | (fixture_eval_summary["feature_row_loss"] != 0)
        | (fixture_eval_summary["shot_event_goal_loss"] != 0)
        | (fixture_eval_summary["feature_goal_loss"] != 0)
    ].copy()
    fixture_loss_rows = fixture_loss_rows.sort_values(
        [
            "feature_goal_loss",
            "feature_row_loss",
            "shot_event_goal_loss",
            "shot_event_row_loss",
            "fixture_id",
        ],
        ascending=[False, False, False, False, True],
    )
    fixture_xg_gap_rows = fixture_eval_summary.reindex(
        fixture_eval_summary["xg_minus_goals"].abs().sort_values(ascending=False).index
    ).head(10)
    attack_type_summary = _subgroup_metric_rows(
        df_pred, group_col="attack_type", display_name="attack_type"
    )
    sub_type_summary = _subgroup_metric_rows(
        df_pred, group_col="sub_type", display_name="sub_type"
    )
    distance_band_summary = _distance_band_metric_rows(df_pred)

    print("\n" + "=" * 70)
    print("xG MODEL EVALUATION")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Evaluation scope: {evaluation_scope}")
    print(f"Test set size: {len(df_pred):,} shots")
    print(f"Goal rate: {y.sum()}/{len(y)} ({100*y.mean():.1f}%)")
    print(f"Model-required feature columns: {len(required_columns)}")
    if missing_columns:
        print(
            f"Injected missing columns as NaN for compatibility: {len(missing_columns)}"
        )
        print(f"  Examples: {', '.join(missing_columns[:5])}")

    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {tn:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  True Positives:  {tp:,}")

    print("\nClassification Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    print("\nRanking Metrics:")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    print(f"  PR AUC:    {pr_auc:.4f}")

    print("\nCalibration Metrics:")
    print(f"  Log Loss:  {logloss:.4f}")
    print(f"  Brier:     {brier:.4f}")
    print(f"  ECE:       {expected_calibration_error:.4f}")
    print(f"  Max Gap:   {max_calibration_gap:.4f}")

    print("\nCalibration Bins:")
    print(
        f"  {'bin':>3}  {'count':>6}  {'mean_pred':>10}  {'goal_rate':>10}  {'abs_gap':>8}"
    )
    for idx, row in calibration_summary.iterrows():
        print(
            f"  {idx + 1:>3}  {int(row['n']):>6}  {row['mean_pred']:>10.4f}  {row['goal_rate']:>10.4f}  {row['abs_gap']:>8.4f}"
        )

    print("\nThreshold Sweep:")
    print(
        f"  {'thr':>4}  {'precision':>9}  {'recall':>8}  {'specificity':>11}  {'f1':>6}  {'pred+ rate':>10}"
    )
    for _, row in threshold_summary.iterrows():
        print(
            f"  {row['threshold']:>4.2f}  {row['precision']:>9.4f}  {row['recall']:>8.4f}  {row['specificity']:>11.4f}  {row['f1']:>6.4f}  {row['predicted_positive_rate']:>10.4f}"
        )
    print(
        f"  Best F1 threshold in sweep: {best_threshold_row['threshold']:.2f} "
        f"(F1={best_threshold_row['f1']:.4f}, Precision={best_threshold_row['precision']:.4f}, Recall={best_threshold_row['recall']:.4f})"
    )

    print("\nFixture Coverage vs Sportradar:")
    print(
        f"  Fixtures in evaluation scope: {fixture_eval_summary['fixture_id'].nunique()}"
    )
    print(
        f"  Sportradar rows/goals: {int(coverage_totals['sportradar_rows']):,} / {int(coverage_totals['sportradar_goals']):,}"
    )
    print(
        f"  Shot events rows/goals: {int(coverage_totals['shot_event_rows']):,} / {int(coverage_totals['shot_event_goals']):,}"
    )
    print(
        f"  features_xg rows/goals: {int(coverage_totals['feature_rows']):,} / {int(coverage_totals['feature_goals']):,}"
    )
    if fixture_loss_rows.empty:
        print(
            "  No row or goal-count loss detected between Sportradar normalized goals, shot_events, and features_xg."
        )
    else:
        print(f"  Fixtures with row/goal-count loss: {len(fixture_loss_rows)}")
        print(
            f"  {'fixture_id':<36}  {'sr_rows':>7}  {'se_rows':>7}  {'fx_rows':>7}  {'sr_goals':>8}  {'se_goals':>8}  {'fx_goals':>8}"
        )
        for _, row in fixture_loss_rows.head(10).iterrows():
            print(
                f"  {row['fixture_id']:<36}  {row['sportradar_rows']:>7}  {row['shot_event_rows']:>7}  {row['feature_rows']:>7}  {row['sportradar_goals']:>8}  {row['shot_event_goals']:>8}  {row['feature_goals']:>8}"
            )

    print("\nTop Fixture xG vs Goals Gaps:")
    print(
        f"  {'fixture_id':<36}  {'eval_rows':>8}  {'goals':>6}  {'pred@0.5':>8}  {'xg_sum':>8}  {'xg-goals':>9}"
    )
    for _, row in fixture_xg_gap_rows.iterrows():
        print(
            f"  {row['fixture_id']:<36}  {row['eval_rows']:>8}  {row['eval_goals']:>6}  {row['predicted_goals_0_5']:>8}  {row['xg_sum']:>8.2f}  {row['xg_minus_goals']:>9.2f}"
        )

    _print_subgroup_table(
        attack_type_summary,
        label_col="attack_type",
        title="Subgroup Evaluation: attack_type",
    )
    _print_subgroup_table(
        sub_type_summary, label_col="sub_type", title="Subgroup Evaluation: sub_type"
    )
    _print_subgroup_table(
        distance_band_summary,
        label_col="distance_band",
        title="Subgroup Evaluation: shooter_distance_to_goal bands",
    )

    perm_importance_path = "artifacts/xg_feature_importance/perm_importance.parquet"
    if os.path.exists(perm_importance_path):
        import pandas as pd

        perm_df = pd.read_parquet(perm_importance_path)
        top_perm = perm_df.head(5)
        print("\nTop 5 Features (Permutation Importance):")
        for _, row in top_perm.iterrows():
            print(f"  {row['feature']:<30} {row['importance_mean']:.4f}")
    else:
        clf = model.named_steps.get("clf")
        if clf is not None and hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            top_k = min(5, len(importances))
            top_indices = np.argsort(importances)[-top_k:][::-1]

            feature_names = None
            preprocessor = model.named_steps["preprocess"]
            if hasattr(preprocessor, "get_feature_names_out"):
                feature_names = list(preprocessor.get_feature_names_out())

            print(f"\nTop {top_k} Features (Transformed Gain Fallback):")
            for idx in top_indices:
                feat_name = (
                    feature_names[idx]
                    if feature_names is not None
                    else f"feature_{idx}"
                )
                importance = importances[idx]
                print(f"  {feat_name:<30} {importance:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="ML debug commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Train xG model (xg_training_job)")
    sub.add_parser("evaluate", help="Evaluate trained xG model on features_xg")

    p = sub.add_parser(
        "analyze", help="Analyze trained models or feature distributions"
    )
    p.add_argument("analysis_type", choices=["models", "features"])

    args = parser.parse_args()
    {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "analyze": cmd_analyze,
    }[args.command](args)


if __name__ == "__main__":
    main()
