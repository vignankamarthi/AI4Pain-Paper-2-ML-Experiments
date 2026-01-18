# Phase 4: Full Optimization and LOSO Validation Results

**Generated:** 2026-01-15 16:30

---

## Summary

**Best Model:** LightGBM + global_zscore normalization
**LOSO Balanced Accuracy:** 71.28% +/- 8.90%
**Gap from Paper 1 Baseline:** -8.12% (from 79.4%)

**Critical Discovery:** per_subject_baseline normalization FAILS in LOSO validation (34% accuracy = near-random). Only global_zscore normalization generalizes to new subjects.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Embedding Dimension (d) | 5 |
| Time Delay (tau) | 2 |
| Signals | EDA, BVP, RESP (SpO2 removed) |
| Features per Signal | 8 |
| Total Features | 24 |
| Valid Experiments | 8 (3 models x 3 normalizations - 1 invalid) |
| CV Method | 5-fold Stratified Cross-Validation |
| Final Validation | Leave-One-Subject-Out (LOSO) |
| Subjects | 53 |

---

## Full Leaderboard (8 Valid Experiments)

| Rank | Model | Normalization | CV Accuracy | LOSO Accuracy | CV-LOSO Gap |
|------|-------|---------------|-------------|---------------|-------------|
| 1 | LightGBM | global_zscore | 72.17% | **71.28%** | 0.89% |
| 2 | LightGBM | raw | 70.26% | 70.75% | -0.50% |
| 3 | SimpleMLP | global_zscore | 68.76% | 68.61% | 0.16% |
| 4 | Stacked | global_zscore | 68.34% | 67.14% | 1.20% |
| 5 | Stacked | raw | 69.39% | 67.01% | 2.39% |
| 6 | Stacked | per_subject_baseline | 70.07% | 34.85% | 35.22% |
| 7 | LightGBM | per_subject_baseline | 73.53% | 34.80% | 38.73% |
| 8 | SimpleMLP | per_subject_baseline | 73.93% | 34.17% | 39.76% |

**Note:** SimpleMLP + raw was excluded (38.75% CV = near-random, invalid experiment).

---

## Critical Finding: per_subject_baseline Fails in LOSO

### The Problem

per_subject_baseline normalization normalizes each subject's features relative to their own no-pain (baseline + rest) samples. This works excellently for:
- Within-subject prediction (same subjects in train and test)
- 5-fold CV (subject's baseline samples are in training set)

But FAILS for:
- LOSO validation (test subject has NO samples in training set)
- Real-world deployment (new patients have no baseline data)

### Evidence

| Model | CV (baseline available) | LOSO (no baseline) | Gap |
|-------|------------------------|-------------------|-----|
| LightGBM | 73.53% | 34.80% | 38.73% |
| SimpleMLP | 73.93% | 34.17% | 39.76% |
| Stacked | 70.07% | 34.85% | 35.22% |

### Root Cause

When testing on a held-out subject in LOSO:
1. We don't have their baseline (no-pain) samples
2. Cannot compute per-subject mean/std for normalization
3. Fallback to global statistics doesn't work (distribution mismatch)
4. Model sees out-of-distribution data, predicts near-randomly

### Correct Approach

**global_zscore normalization is the only valid approach for generalizing to new subjects.**

Benefits:
- Consistent normalization across all subjects
- Training statistics apply to test subjects
- Minimal CV-LOSO gap (0.89% for best model)

---

## Comparison to Paper 1

| Metric | Paper 1 (catch22) | Paper 2 (Entropy) | Delta |
|--------|-------------------|-------------------|-------|
| Features | catch22 | Entropy-Complexity | - |
| Best Model | SVM | LightGBM | - |
| Normalization | Global z-score | Global z-score | Same |
| LOSO Accuracy | 79.4% | 71.28% | -8.12 pp |

---

## Experiment 4.1: Normalization Comparison (5-fold CV)

Initial comparison showed per_subject_baseline leading in CV:

| Model | per_subject_baseline | global_zscore | raw |
|-------|---------------------|---------------|-----|
| LightGBM | **74.84%** | 74.16% | 74.11% |
| SimpleMLP | **72.62%** | 69.96% | 38.75% |
| Stacked | **71.25%** | 68.92% | 69.44% |

**Key observations:**
- LightGBM: Essentially normalization-agnostic (<1% difference)
- SimpleMLP: Requires normalization (raw = random guessing)
- Stacked: Modest preference for per_subject_baseline

This CV result was MISLEADING because subjects' baseline data was available in all folds.

---

## Experiment 4.2: Backward Elimination

Feature elimination analysis determined all 24 features should be kept:
- Baseline accuracy: 74.84%
- Removing least important feature (eda_tsallispe): -0.52% accuracy
- Threshold: 0.5% drop triggers stop
- Decision: Keep all 24 features

---

## Per-Class Analysis (Best Model)

LightGBM + global_zscore LOSO performance breakdown:

| Class | Description | Expected Difficulty |
|-------|-------------|---------------------|
| No Pain | baseline + rest | Easiest (distinct physiological state) |
| Low Pain | mild pain stimulus | Moderate |
| High Pain | intense pain stimulus | Hardest (overlaps with low pain) |

The primary challenge remains discriminating between low and high pain intensities, consistent with Phase 1-3 findings.

---

## Implications for Deployment

### Within-Subject Monitoring
If baseline data is available (e.g., clinical setting with calibration period):
- per_subject_baseline normalization is valid
- Expected accuracy: ~74% (based on CV results)

### Cross-Subject Generalization
For new patients without baseline data:
- MUST use global_zscore normalization
- Expected accuracy: ~71% (based on LOSO results)
- This is the realistic deployment scenario

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
├── experiment_4.3_full_training/
│   ├── tree_based_results.csv
│   ├── mlp_results.csv
│   └── full_leaderboard.csv
└── phase4_report.md
```

---

## Conclusions

1. **Normalization matters for generalization:** per_subject_baseline excels in CV but fails in LOSO
2. **global_zscore is required for new subjects:** Only approach that generalizes properly
3. **LightGBM is the best model:** Simple, fast, and consistent across normalizations
4. **Entropy-complexity features underperform catch22:** 71.28% vs 79.4% (-8.12 pp)
5. **Pain intensity discrimination remains challenging:** The core ML problem persists

---

## Recommendations

1. **For Paper 2:** Report 71.28% LOSO accuracy with global_zscore as the honest, generalizable result
2. **Future work:** Explore hybrid feature sets (entropy + catch22), alternative entropy measures, or deep learning on raw signals
3. **Clinical deployment:** Use global_zscore normalization; consider per_subject_baseline only with guaranteed baseline calibration

---

*Report generated by Phase 4 optimization pipeline.*
