# Experiment Status Tracker

**Last Updated:** 2026-01-15 16:15

---

## Current Phase
**PHASE4_COMPLETE**

**Status:** All experiments complete. Critical findings discovered.

---

## Phase 4 Progress

| Experiment | Status | Result |
|------------|--------|--------|
| 4.1 Normalization Comparison (5-fold CV) | COMPLETE | per_subject_baseline leads in CV |
| 4.2 Backward Elimination | COMPLETE | Keep all 24 features |
| 4.3+4.4 Full Training + LOSO | COMPLETE | **CRITICAL: per_subject_baseline FAILS in LOSO** |

### Experiment 4.3+4.4: Full Leaderboard (8 Valid Experiments)

| Model | Normalization | CV Accuracy | LOSO Accuracy | Gap |
|-------|--------------|-------------|---------------|-----|
| LightGBM | global_zscore | 72.17% | **71.28%** | 0.89% |
| LightGBM | raw | 70.26% | 70.75% | -0.50% |
| SimpleMLP | global_zscore | 68.76% | 68.61% | 0.16% |
| Stacked | global_zscore | 68.34% | 67.14% | 1.20% |
| Stacked | raw | 69.39% | 67.01% | 2.39% |
| Stacked | per_subject_baseline | 70.07% | 34.85% | 35.22% |
| LightGBM | per_subject_baseline | 73.53% | 34.80% | 38.73% |
| SimpleMLP | per_subject_baseline | 73.93% | 34.17% | 39.76% |

### CRITICAL FINDING

**per_subject_baseline normalization FAILS catastrophically in LOSO validation!**

- In 5-fold CV: 73-74% accuracy (looks good because baseline from same subject is available)
- In LOSO: 34-35% accuracy (near-random because new subject has NO baseline to normalize against)

**Root cause:** When testing on a held-out subject, we don't have their baseline samples to compute per-subject statistics. The test subject's data cannot be properly normalized.

### Correct Approach

**global_zscore normalization is the only valid approach for generalizing to new subjects.**

- Best: LightGBM + global_zscore = **71.28% LOSO**
- Minimal CV-LOSO gap (0.89%) indicates good generalization

---

## Phase Progress (All Phases)

| Phase | Status | Started | Completed |
|-------|--------|---------|-----------|
| Stage 0: Binary Classification | COMPLETE | 2026-01-14 | 2026-01-14 |
| Phase 1: Ensemble Exploration | COMPLETE | 2026-01-14 | 2026-01-14 |
| Phase 2: Neural Net Exploration | COMPLETE | 2026-01-14 | 2026-01-15 |
| Phase 3: LOSO Validation | SUPERSEDED | 2026-01-15 | 2026-01-15 |
| Phase 4: Full Optimization + LOSO | COMPLETE | 2026-01-15 | 2026-01-15 |

**Note:** Phase 3 results are now superseded by Phase 4's more comprehensive analysis which revealed the per_subject_baseline failure mode.

---

## Corrected Final Summary

### Stage 0: Binary Classification
**Result:** 99.97% linear accuracy with per-subject baseline normalization
- This works because we test within the same subjects (not held-out)
- Does NOT generalize to new subjects

### Phase 1-2: Model Exploration (80/20 Split)
These phases used per_subject_baseline with test data from same subjects.
- Results are valid for within-subject prediction
- DO NOT represent generalization to new subjects

### Phase 3: LOSO Validation (NOW SUPERSEDED)
Original 74.48% LOSO result was INVALID because:
- per_subject_baseline normalization was applied incorrectly
- Test subject data was normalized using training set statistics (not their own baseline)

### Phase 4: Corrected LOSO Validation
**Best Model:** LightGBM + global_zscore at **71.28%** LOSO

| Rank | Model | Normalization | LOSO Accuracy |
|------|-------|---------------|---------------|
| 1 | LightGBM | global_zscore | 71.28% +/- 8.90% |
| 2 | LightGBM | raw | 70.75% +/- 8.75% |
| 3 | SimpleMLP | global_zscore | 68.61% +/- 11.76% |
| 4 | Stacked | global_zscore | 67.14% +/- 9.10% |

---

## Key Findings (Corrected)

1. **per_subject_baseline Does NOT Generalize:** Excellent within-subject (99.97% binary) but fails LOSO (34%)
2. **global_zscore is Required for New Subjects:** Only approach that generalizes properly
3. **LightGBM is Best:** Consistently outperforms neural nets and stacked ensembles
4. **Simple is Better:** LightGBM with default hyperparameters beats complex ensembles
5. **Below Paper 1 Baseline:** 71.28% vs 79.4% (-8.12 pp)

---

## Comparison to Paper 1

| Metric | Paper 1 | This Work | Delta |
|--------|---------|-----------|-------|
| Features | catch22 | Entropy-Complexity | - |
| Normalization | Global z-score | Global z-score | Same |
| Best Model | SVM | LightGBM | - |
| LOSO Accuracy | 79.4% | 71.28% | -8.12 pp |
| Classes | 3 | 3 | Same |

---

## Output Files

```
results/phase4/
├── experiment_4.1_normalization/
│   ├── normalization_comparison_results.csv
│   ├── mlp_normalization_results.csv
│   └── normalization_summary.json
├── experiment_4.2_backward_elimination/
│   ├── elimination_history.csv
│   └── optimal_features.json
└── experiment_4.3_full_training/
    ├── tree_based_results.csv
    ├── mlp_results.csv
    └── full_leaderboard.csv
```

---

**Experiment Status:** COMPLETE

*The entropy-complexity features achieve 71.28% LOSO accuracy, which is 8.12 percentage points below Paper 1's catch22 baseline. Future work should explore alternative feature engineering approaches or hybrid feature sets.*
