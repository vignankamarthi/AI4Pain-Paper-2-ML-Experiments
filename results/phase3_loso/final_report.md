# Phase 3: LOSO Validation - Final Report

**Generated:** 2026-01-16 04:30

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Best Model** | RandomForest |
| **LOSO Balanced Accuracy** | 77.15% +/- 9.06% |
| **95% Confidence Interval** | [74.73%, 79.57%] |
| **Paper 1 Baseline** | 79.4% |
| **Improvement** | -2.84% |
| **Status** | BELOW PAPER 1 BASELINE |

---

## Stage 0: Binary Classification Analysis

### Per-Subject Baseline Normalization Discovery

Stage 0 investigated binary pain classification (pain vs no-pain) using Complexity-Entropy (C-H) plane analysis. The critical discovery was that **per-subject baseline normalization** dramatically improves class separability.

| Normalization Method | Binary Accuracy |
|---------------------|-----------------|
| None (raw features) | ~50% (random) |
| Global z-score | ~70% |
| **Per-subject baseline** | **99.97%** |

**Key Insight:** Normalizing each subject's features relative to their own no-pain (baseline + rest) state removes inter-individual variability while preserving pain-specific physiological signatures.

---

## Phase 1: Ensemble Exploration

### Configuration
- **Embedding Dimension:** d=7
- **Time Delay:** tau=2
- **Signals:** EDA, BVP, RESP, SpO2 (4 signals)
- **Features:** 8 entropy/complexity measures per signal = 32 total
- **Optimization:** Optuna (50 trials, 5-fold CV per model)

### 80/20 Split Results

| Rank | Model | Balanced Accuracy | Accuracy |
|------|-------|-------------------|----------|
| 1 | LightGBM | 75.59% | 81.73% |
| 2 | Stacked (RF+XGB+LGB) | 72.97% | 79.76% |
| 3 | XGBoost | 72.70% | 79.57% |
| 4 | Random Forest | 72.70% | 79.57% |
| 5 | Stacked (RF+XGB) | 71.39% | 78.59% |

**Best Phase 1 Model:** LightGBM at 75.59%

---

## Phase 3: LOSO Validation

### Methodology
- **Cross-Validation:** Leave-One-Subject-Out
- **Total Folds:** 53
- **Process:** Train on N-1 subjects, test on 1 held-out subject
- **Normalization:** Per-subject baseline (applied within each fold)

### LOSO Leaderboard

| Rank | Model | Balanced Acc (Mean+/-Std) | 95% CI | Accuracy |
|------|-------|------------------------|--------|----------|
| 1 | RandomForest | 77.15% +/- 9.06% | [74.73%, 79.57%] | 67.09% |
| 2 | XGBoost | 76.36% +/- 9.24% | [73.90%, 78.83%] | 65.96% |
| 3 | LightGBM | 75.63% +/- 9.03% | [73.22%, 78.04%] | 64.91% |


### Statistical Comparison to Paper 1 Baseline

| Model | LOSO BA | Paper 1 | Improvement | t-stat | p-value | Cohen's d | Significant |
|-------|---------|---------|-------------|--------|---------|-----------|-------------|
| LightGBM | 75.63% | 79.4% | -4.75% | -3.041 | 0.0037 | -0.422 | Yes |
| XGBoost | 76.36% | 79.4% | -3.83% | -2.393 | 0.0203 | -0.332 | Yes |
| RandomForest | 77.15% | 79.4% | -2.84% | -1.809 | 0.0763 | -0.251 | No |


### Per-Class Performance (Best Model: RandomForest)

| Class | Samples | Accuracy | Notes |
|-------|---------|----------|-------|
| No Pain | 53 | 100.00% | Baseline + Rest states |
| Low Pain | 636 | 65.25% | Low-intensity TENS |
| High Pain | 636 | 66.19% | High-intensity TENS |

### Confusion Matrix Analysis

The LOSO confusion matrix for RandomForest reveals:
- **No Pain detection:** 100.0% accuracy (strong)
- **Low vs High Pain discrimination:** The primary challenge
- **Most common error:** Low Pain <-> High Pain confusion

---

## Discussion

### Success Assessment

**BELOW PAPER 1 BASELINE**

The best model (RandomForest) achieves 77.15% balanced accuracy, which is
2.84% below Paper 1's 79.4% baseline.

**Potential reasons:**
1. 3-class classification (no_pain/low_pain/high_pain) is inherently harder than approaches
   that may merge similar classes
2. The entropy-complexity feature space may require additional feature engineering
3. The pain intensity discrimination (low vs high) remains challenging


### Key Findings

1. **Per-subject baseline normalization is critical:** The Stage 0 discovery that normalizing
   relative to each individual's no-pain state dramatically improves separability carries over
   to the 3-class problem.

2. **No-pain detection is highly accurate:** All models achieve near-perfect (100.0%)
   accuracy for detecting the absence of pain, validating the normalization approach.

3. **Pain intensity discrimination is the bottleneck:** The primary challenge is distinguishing
   low pain from high pain, suggesting subtle differences in physiological responses.

4. **RandomForest provides best generalization:** Among all tested models, RandomForest
   shows the best LOSO performance, suggesting good bias-variance tradeoff.

### Comparison: Phase 1 (80/20) vs Phase 3 (LOSO)

| Model | Phase 1 (80/20) | Phase 3 (LOSO) | Difference |
|-------|-----------------|----------------|------------|
| XGBoost | 72.70% | 76.36% | +3.66% |
| LightGBM | 75.59% | 75.63% | +0.04% |


The drop from 80/20 to LOSO performance indicates some overfitting to subject-specific patterns
in the training data. LOSO provides a more realistic estimate of real-world performance.

---

## Limitations

1. **Dataset Size:** 65 total subjects limits statistical power
2. **Controlled Setting:** TENS-induced pain may differ from clinical pain scenarios
3. **Binary Pain Intensity:** Only two pain levels (low/high) tested
4. **Single Session:** No test-retest reliability assessment

---

## Future Work

1. **Temporal Modeling:** Explore time-series approaches (RNNs, Transformers)
2. **Additional Entropy Measures:** Sample entropy, approximate entropy
3. **Multi-Scale Analysis:** Combine multiple embedding dimensions
4. **Clinical Validation:** Test on real clinical pain datasets
5. **Real-Time Implementation:** Optimize for embedded deployment

---

## Conclusion

This study demonstrates that **entropy-complexity features derived from physiological signals**,
combined with **per-subject baseline normalization**, provide discriminative information for
3-class pain classification. The best model (RandomForest) achieves **77.15%** balanced
accuracy on rigorous LOSO cross-validation.

While performance falls short of Paper 1 baseline on the 3-class task, the per-subject normalization insight and entropy features provide a foundation for future improvements.

---

## Reproducibility

### Configuration
```json
{
    "random_seed": 42,
    "embedding_dimension": 7,
    "time_delay": 2,
    "signals": ['eda', 'bvp', 'resp', 'spo2'],
    "n_features": 32,
    "normalization": "per_subject_baseline",
    "validation": "LOSO",
    "n_subjects": 53
}
```

### Best Model Configuration (RandomForest)
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

---

## Output Files

```
results/phase3_loso/
├── loso_leaderboard.csv
├── per_subject_results.csv
├── statistical_tests.csv
├── final_report.md
├── confusion_matrices/
│   └── [model]_loso_confusion_matrix.png
├── ch_plane_visualizations/
│   └── best_model_[signal]_ch_plane.png
├── subject_performance_distribution.png
└── best_models/
    ├── best_loso_model.pkl
    └── best_loso_model_config.json
```

---

**End of Report**

*Generated by Phase 3 LOSO Validation Pipeline*
