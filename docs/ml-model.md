# ML Model — xG (Expected Goals)

## Purpose

The xG model estimates the probability that a given shot attempt results in a goal.
Input: snapshot features extracted at the moment of ball release (`throw_timestamp_ms`).
Output: calibrated probability in [0, 1].

---

## Active Model

| Property | Value |
|----------|-------|
| Algorithm | XGBoost (`XGBClassifier`, `binary:logistic`) |
| Objective | `binary:logistic` |
| Eval metric | `logloss` |
| Early stopping | 50 rounds on validation log-loss |
| Preprocessing | `ColumnTransformer`: median imputation + standard scaling (numeric); one-hot encoding (categorical) |
| Persistence | sklearn `Pipeline` via `FilesystemIOManager` (`file_io_manager`) |
| Training location | `src/pipelines/ml/train_xg.py::train_xg_model` |
| Dagster asset | `ml_xg_model` (global, `xg_training_job`) |

The active production baseline is a single global XGBoost model over all shot domains.
Penalty-vs-rest routed variants were tested and retained only as experiments; they are not the
default because they did not clearly outperform the simpler single-model baseline on held-out
validation.

---

## Features

Features are computed per shot at `throw_timestamp_ms` from positional data.
Feature engineering lives in `src/pipelines/features/calc_xg_features.py`.

### Numeric features

| Feature | Description |
|---------|-------------|
| `shooter_distance_to_goal` | Euclidean distance from shooter to centre of goal |
| `shot_angle_to_goal` | Solid angle (radians) subtended by the goal posts from the shooter's position |
| `goalkeeper_distance_to_goal` | GK displacement from goal line (proxy for GK positioning quality) |
| `shooter_distance_to_goalkeeper` | Shooter–GK distance |
| `ball_distance_to_goal` | Ball distance to goal |
| `ball_distance_to_goalkeeper` | Ball–GK distance |
| `ball_angle_to_goal` | Angle from ball to goal |
| `angle_ball_to_goalkeeper` | Angle between ball→goal vector and ball→GK vector |
| `avg_offense_distance_to_goal` | Mean distance of all attacking players to goal |
| `avg_defense_distance_to_goal` | Mean distance of all defending players to goal |
| `num_defenders_close` | Count of defenders within a forward cone in front of the shooter |
| `closest_defender_distance` | Distance to the nearest defender |

### Categorical features

| Feature | Values | Description |
|---------|--------|-------------|
| `attack_type` | SET_PLAY, FAST_BREAK, BREAK_THROUGH, PIVOT | Sportradar-labelled attack classification |
| `sub_type` | Various | Sportradar sub-classification of attack |

`attack_type` and `sub_type` are currently enabled in the active model. Ablation testing showed
that disabling them hurt held-out performance.

### Label

| Column | Type | Description |
|--------|------|-------------|
| `target` | int (0/1) | 1 = goal scored, 0 = shot saved or missed |

---

## Training Protocol

### Data source

Reads the full `features_xg` table from DuckDB (all fixtures, all partitions).

### Train / validation split

Fixture-level grouped split using `StratifiedGroupKFold`:
- Groups by `fixture_id` to prevent within-match leakage (shots from the same match are highly correlated).
- Stratified to preserve goal rate in both splits.
- Falls back to row-level stratified split (`train_test_split`) if fewer than 2 fixtures are available.
- Split strategy is recorded in model metadata (`split_strategy` field).

### Grouped cross-validation

Training now also records grouped cross-validation metrics across fixture-level folds:
- Up to 5 grouped folds (`StratifiedGroupKFold`) over the full `features_xg` table.
- Metrics tracked: grouped CV AUC, log-loss, Brier, and accuracy mean/std.
- These metrics are persisted in `_bielemetrics_training_metadata` and surfaced by `debug_run.py analyze models`.

This does not replace the primary held-out validation split used for early stopping and day-to-day model checks; it gives a more stable benchmark when comparing feature or hyperparameter changes.

### Class imbalance

Not actively rebalanced (no `scale_pos_weight`). xG is a probability estimation problem —
rebalancing would bias predicted probabilities away from the true goal rate.

### Evaluation metrics

The model reports both model and baseline (constant prediction at training base rate) metrics:

| Metric | Description |
|--------|-------------|
| `val_auc` | ROC-AUC on validation set |
| `val_logloss` | Log-loss on validation set |
| `val_brier` | Brier score (calibration quality) |
| `val_accuracy` | Accuracy at 0.5 threshold |

Baseline equivalents (`baseline_auc`, `baseline_logloss`, etc.) are always reported for comparison.

### Key hyperparameters

```python
XGBClassifier(
  n_estimators=800,
  max_depth=4,
  min_child_weight=6,
  learning_rate=0.03,
  subsample=0.85,
  colsample_bytree=0.75,
  gamma=0.1,
  reg_alpha=0.15,
  reg_lambda=2.5,
    early_stopping_rounds=50,
    tree_method="hist",
)
```

The regularization was tightened to make the model less eager to chase narrow one-hot splits while keeping the single global model structure.

---

## Model Artifact

The trained model is a sklearn `Pipeline`:

```
Pipeline(
  preprocess: ColumnTransformer(
      num: Pipeline(SimpleImputer(median) → StandardScaler)
      cat: OneHotEncoder(handle_unknown="ignore")
  )
  clf: XGBClassifier(...)
)
```

A custom attribute `_bielemetrics_training_metadata` is attached to the pipeline object at
training time, containing feature lists, split strategy, train/val fixture IDs, and fixture counts.
This attribute is persisted with the model artifact and is used by `debug_run.py evaluate` to scope
evaluation to held-out validation fixtures. It now also includes grouped cross-validation summary metrics.

## Feature Importance

Two importance views are produced:

- Permutation importance on original feature columns
- XGBoost gain importance on transformed features (after one-hot encoding)

For human interpretation, prefer permutation importance. The transformed gain view can be useful for
tree-level debugging, but it can over-emphasize individual one-hot encoded category levels such as
specific `sub_type` values.

`debug_run.py evaluate` now reports the top features from permutation importance by default.

It also reports subgroup diagnostics by:
- `attack_type`
- `sub_type`
- `shooter_distance_to_goal` bands

Those subgroup tables make it easier to spot where the model is over- or under-estimating xG by shot domain even when the global metrics look stable.

---

## Inference

Inference asset: `src/hbl_etl_dagster/assets_ml/assets_xg_inference.py`.
Loads the trained pipeline from the filesystem and applies it per-fixture to `features_xg`.

---

## Known Limitations

- **Snapshot-only**: features are computed at a single instant (`throw_timestamp_ms`), not from a
  pre-shot temporal window. This misses trajectory, formation build-up, and movement dynamics.
- **`throw_timestamp_ms` precision**: relies on the Kinexon detected-shot sync; if the sync fails
  (e.g. reversed clock polarity), the positional lookup may use the wrong frame.
- **`attack_type` label noise**: Sportradar labels are applied post-hoc by human scorers.
  FAST_BREAK vs BREAK_THROUGH is inconsistently applied; treat classification accuracy as
  an upper-bounded, noisy signal.
- **Domain specialists not yet justified**: a seven-meter specialist / gated model variant improved
  some calibration diagnostics but did not clearly beat the single global model on held-out ranking
  metrics.
- **Global training in a partitioned job (historical issue)**: `ml_xg_model` was previously inside
  the fixture-partitioned job, causing per-fixture retraining. This has been corrected — `ml_xg_model`
  now runs in the separate `xg_training_job`.

---

## Planned Upgrade

A parallel spatiotemporal xG path using a PyTorch transformer encoder over pre-shot temporal windows
is planned. See [transformers.md](transformers.md) for the full plan. The XGBoost snapshot model
remains active as the control baseline.

---

## Related

- [data-model.md](data-model.md) — `features_xg` table schema
- [architecture.md](architecture.md) — job structure and training job
- [transformers.md](transformers.md) — planned sequence model
- `src/pipelines/ml/train_xg.py` — training implementation
- `src/pipelines/features/calc_xg_features.py` — feature engineering
