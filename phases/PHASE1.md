# Phase 1: Ensemble Classification (80/20 Split)

**Status:** COMPLETE
**Methodology:** Baseline-only (rest segments excluded)

---

## Objective

Train and evaluate ensemble classifiers for 3-class pain classification using baseline-only no_pain definition.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Features | 24 (EDA + BVP + RESP, 8 each) |
| Train/Test | 80/20 stratified split |
| Class weights | Balanced |
| Optuna trials | 50 per model |
| Normalization | Global z-score |

---

## Models

1. LightGBM
2. XGBoost
3. Random Forest
4. Stacked Ensemble (RF + XGB + LGB)

---

## Label Mapping

```python
CLASS_MAPPING = {
    'baseline': 0,  # no_pain - baseline ONLY
    # 'rest': EXCLUDED from dataset
    'low': 1,       # low_pain
    'high': 2       # high_pain
}
```

---

## Data Filtering

```python
# Exclude rest segments entirely
df = df[df['state'] != 'rest'].copy()
```

---

## Results

| Rank | Model | Balanced Accuracy |
|------|-------|-------------------|
| 1 | Stacked (RF+XGB+LGB) | 73.75% |
| 2 | Stacked (RF+XGB) | 73.49% |
| 3 | XGBoost | 72.18% |
| 4 | RandomForest | 72.18% |
| 5 | LightGBM | 70.08% |

**Outcome:** Best ensemble (73.75%) < 85% target. Proceeded to Phase 2.

---

## Output Files

- `results/phase1_ensembles/leaderboard.csv`
- `results/phase1_ensembles/confusion_matrices/`
- `results/phase1_ensembles/step1_report.md`

---

## Execution

```bash
python src/phase1_ensemble_experiments.py
```

---

*Phase 1 complete. Baseline-only methodology.*
