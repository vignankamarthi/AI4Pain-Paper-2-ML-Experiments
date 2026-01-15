# Experiment Status Tracker

**Last Updated:** 2026-01-15 08:10

---

## Current Phase
**PHASE3_COMPLETE**

**Status:** EXPERIMENT COMPLETED

**Final Result:** LightGBM achieves **74.48% +/- 7.69%** balanced accuracy on LOSO validation.

---

## Phase Progress

| Phase | Status | Started | Completed |
|-------|--------|---------|-----------|
| Stage 0: Binary Classification | COMPLETE | 2026-01-14 | 2026-01-14 |
| Phase 1: Ensemble Exploration | COMPLETE | 2026-01-14 | 2026-01-14 |
| Phase 2: Neural Net Exploration | COMPLETE | 2026-01-14 | 2026-01-15 |
| Phase 3: LOSO Validation | COMPLETE | 2026-01-15 | 2026-01-15 |

---

## Final Summary

### Stage 0: Binary Classification
**Result:** 99.97% linear accuracy with per-subject baseline normalization
- Key discovery: Per-subject normalization is critical for class separability
- Binary (pain vs no-pain) classification nearly perfect

### Phase 1: Ensemble Exploration (80/20 Split)
**Best Model:** LightGBM at 75.59% balanced accuracy

| Rank | Model | Balanced Accuracy |
|------|-------|-------------------|
| 1 | LightGBM | 75.59% |
| 2 | Stacked (RF+XGB+LGB) | 72.97% |
| 3 | XGBoost | 72.70% |
| 4 | Random Forest | 72.70% |

**Decision:** Proceed to Phase 2 (best model < 85% threshold)

### Phase 2: Neural Network Exploration (100 trials per architecture)
**Best Model:** Simple MLP at 76.38% test accuracy

| Architecture | CV Accuracy | Test Accuracy | Gap |
|--------------|-------------|---------------|-----|
| Simple MLP | 78.07% | 76.38% | 1.69% |
| Deep MLP | 78.13% | 75.85% | 2.28% |
| Medium MLP | 78.00% | 75.07% | 2.93% |
| Regularized MLP | 77.81% | 74.80% | 3.01% |

**Finding:** Simple 2-layer architecture generalizes best. Deeper networks show more overfitting.

### Phase 3: LOSO Validation
**Best Model:** LightGBM at 74.48% +/- 7.69%

| Rank | Model | LOSO Balanced Acc | 95% CI |
|------|-------|-------------------|--------|
| 1 | LightGBM | 74.48% +/- 7.69% | [72.43%, 76.53%] |
| 2 | XGBoost | 74.21% +/- 7.01% | [72.35%, 76.08%] |
| 3 | RandomForest | 74.19% +/- 6.71% | [72.40%, 75.98%] |

**Note:** Neural network LOSO excluded due to PyTorch/macOS stability issues.

---

## Key Findings

1. **Per-Subject Baseline Normalization Works:** Binary classification achieves 99.97% accuracy
2. **No-Pain Detection is Perfect:** 100% accuracy in LOSO for detecting no-pain state
3. **Pain Intensity is the Challenge:** Low (66.82%) vs High (56.60%) discrimination is difficult
4. **Ensembles Outperform Neural Nets:** LightGBM generalizes better than MLPs on LOSO
5. **Below Paper 1 Baseline:** 74.48% vs 79.4% (-6.20%)

---

## Comparison to Paper 1

| Metric | Paper 1 | This Work | Delta |
|--------|---------|-----------|-------|
| Features | catch22 | Entropy-Complexity | - |
| Baseline | 79.4% | 74.48% | -4.92 pp |
| Classes | 3 | 3 | Same |
| Validation | LOSO | LOSO | Same |

---

## Output Files

```
results/
├── stage0_binary/
│   ├── STAGE0_FINAL_REPORT.md
│   └── FINAL_ch_plane_binary.png
├── phase1_ensembles/
│   ├── leaderboard.csv
│   ├── confusion_matrices/
│   └── PHASE1_REPORT.md
├── phase2_neuralnets/
│   ├── leaderboard.csv
│   ├── hyperparameters.json
│   └── training_curves/
└── phase3_loso/
    ├── final_report.md
    ├── loso_leaderboard.csv
    ├── per_subject_results.csv
    ├── statistical_tests.csv
    ├── confusion_matrices/
    ├── ch_plane_visualizations/
    └── best_models/
```

---

**Experiment Status:** COMPLETE

*Next steps: Consider additional feature engineering or alternative entropy measures to improve pain intensity discrimination.*
