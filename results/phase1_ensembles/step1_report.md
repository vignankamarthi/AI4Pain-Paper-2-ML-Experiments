# Phase 6 Step 1: Ensemble Exploration Results (BASELINE-ONLY)

**Generated:** 2026-01-16 01:28

---

## Executive Summary

**Best Model:** Stacked (RF+XGB+LGB)
**Balanced Accuracy:** 73.75%
**Improvement over Paper 1 Baseline:** -5.65% (from 79.4%)

**KEY DIFFERENCE:** Rest segments EXCLUDED - only baseline defines no-pain class.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Embedding Dimension (d) | 7 |
| Time Delay (tau) | 2 |
| Signals | EDA, BVP, RESP, SPO2 |
| Features per Signal | 8 |
| Total Features | 32 |
| Normalization | Per-subject baseline (baseline-only reference) |
| No-Pain Class | BASELINE ONLY (rest excluded) |
| Train/Test Split | 80/20 stratified |
| Optuna Trials | 50 per model |
| CV Folds | 5 |

---

## Leaderboard

| Rank | Model | Balanced Acc | Accuracy | F1 | No Pain | Low Pain | High Pain |
|------|-------|--------------|----------|-----|---------|----------|-----------|
| 1 | Stacked (RF+XGB+LGB) | 73.75% | 62.26% | 0.6226 | 100.0% | 60.6% | 60.6% |
| 2 | Stacked (RF+XGB) | 73.49% | 61.89% | 0.6186 | 100.0% | 63.0% | 57.5% |
| 3 | XGBoost | 72.18% | 60.00% | 0.5994 | 100.0% | 62.2% | 54.3% |
| 4 | Random Forest | 72.18% | 60.00% | 0.5984 | 100.0% | 64.6% | 52.0% |
| 5 | LightGBM | 70.08% | 56.98% | 0.5676 | 100.0% | 62.2% | 48.0% |

---

## Comparison to Paper 1 Baseline

| Metric | Paper 1 (catch22) | Paper 2 (Entropy) | Difference |
|--------|-------------------|-------------------|------------|
| Balanced Accuracy | 79.4% | 73.75% | -5.65% |

---

## Best Hyperparameters

### Random Forest
```json
{
  "n_estimators": 200,
  "max_depth": 10,
  "min_samples_split": 5,
  "min_samples_leaf": 1,
  "max_features": "log2",
  "class_weight": null,
  "random_state": 42,
  "n_jobs": -1
}
```

### XGBoost
```json
{
  "n_estimators": 100,
  "max_depth": 3,
  "learning_rate": 0.05,
  "subsample": 0.6,
  "colsample_bytree": 0.8,
  "gamma": 0.1,
  "reg_alpha": 0.5,
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
  "n_estimators": 200,
  "max_depth": 5,
  "learning_rate": 0.05,
  "num_leaves": 15,
  "subsample": 0.6,
  "colsample_bytree": 1.0,
  "reg_alpha": 1.0,
  "reg_lambda": 0.5,
  "random_state": 42,
  "n_jobs": -1,
  "verbose": -1
}
```

---

## Per-Class Analysis

The best model (Stacked (RF+XGB+LGB)) achieves:
- **No Pain (baseline ONLY):** 100.0% accuracy
- **Low Pain:** 60.6% accuracy
- **High Pain:** 60.6% accuracy

---

## Decision Recommendation

**TRIGGER STEP 2 (NEURAL NET EXPLORATION)**

The best model achieves 73.75% balanced accuracy, below 82%.
Neural net exploration is recommended to improve performance.

---

## Output Files

```
results/phase6_step1_ensembles/
├── leaderboard.csv
├── hyperparameters.json
├── step1_report.md
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
3. Proceed to Step 3 for rigorous cross-validation
