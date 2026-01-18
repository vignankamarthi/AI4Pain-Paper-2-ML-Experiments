# Phase 1: Ensemble Exploration Results

**Generated:** 2026-01-14 22:45

---

## Summary

**Best Model:** LightGBM
**Balanced Accuracy:** 75.59%
**Improvement over Paper 1 Baseline:** -3.81% (from 79.4%)

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Embedding Dimension (d) | 7 |
| Time Delay (tau) | 2 |
| Signals | EDA, BVP, RESP, SPO2 |
| Features per Signal | 8 |
| Total Features | 32 |
| Normalization | Per-subject baseline (no-pain reference) |
| Train/Test Split | 80/20 stratified |
| Optuna Trials | 50 per model |
| CV Folds | 5 |

---

## Leaderboard

| Rank | Model | Balanced Acc | Accuracy | F1 | No Pain | Low Pain | High Pain |
|------|-------|--------------|----------|-----|---------|----------|-----------|
| 1 | LightGBM | 75.59% | 81.73% | 0.8173 | 100.0% | 62.2% | 64.6% |
| 2 | Stacked (RF+XGB+LGB) | 72.97% | 79.76% | 0.7975 | 100.0% | 62.2% | 56.7% |
| 3 | XGBoost | 72.70% | 79.57% | 0.7952 | 100.0% | 63.8% | 54.3% |
| 4 | Random Forest | 72.70% | 79.57% | 0.7949 | 100.0% | 65.4% | 52.8% |
| 5 | Stacked (RF+XGB) | 71.39% | 78.59% | 0.7856 | 100.0% | 60.6% | 53.5% |

---

## Comparison to Paper 1 Baseline

| Metric | Paper 1 (catch22) | Paper 2 (Entropy) | Difference |
|--------|-------------------|-------------------|------------|
| Balanced Accuracy | 79.4% | 75.59% | -3.81% |

---

## Best Hyperparameters

### Random Forest
```json
{
  "n_estimators": 500,
  "max_depth": 10,
  "min_samples_split": 2,
  "min_samples_leaf": 4,
  "max_features": null,
  "class_weight": null,
  "random_state": 42,
  "n_jobs": -1
}
```

### XGBoost
```json
{
  "n_estimators": 100,
  "max_depth": 7,
  "learning_rate": 0.01,
  "subsample": 0.8,
  "colsample_bytree": 0.6,
  "gamma": 0,
  "reg_alpha": 1.0,
  "reg_lambda": 0,
  "random_state": 42,
  "n_jobs": -1,
  "use_label_encoder": false,
  "eval_metric": "mlogloss"
}
```

### LightGBM
```json
{
  "n_estimators": 100,
  "max_depth": -1,
  "learning_rate": 0.01,
  "num_leaves": 31,
  "subsample": 0.6,
  "colsample_bytree": 0.6,
  "reg_alpha": 0.5,
  "reg_lambda": 0,
  "random_state": 42,
  "n_jobs": -1,
  "verbose": -1
}
```

---

## Per-Class Analysis

The best model (LightGBM) achieves:
- **No Pain (baseline + rest):** 100.0% accuracy
- **Low Pain:** 62.2% accuracy
- **High Pain:** 64.6% accuracy

---

## Decision Recommendation

**TRIGGER PHASE 2 (NEURAL NET EXPLORATION)**

The best model achieves 75.59% balanced accuracy, below 82%.
Neural net exploration is recommended to improve performance.

---

## Output Files

```
results/phase1_ensembles/
├── leaderboard.csv
├── hyperparameters.json
├── phase1_report.md
├── confusion_matrices/
│   └── [model]_confusion_matrix.png
├── feature_importance_plots/
│   └── [model]_feature_importance.png
└── models/
    └── [model].pkl
```

---

## Next Steps

1. Review leaderboard and confusion matrices
2. Select top 2-5 models for LOSO validation
3. Proceed to Phase 3 for rigorous cross-validation
