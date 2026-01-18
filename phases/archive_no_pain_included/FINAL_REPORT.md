# AI4Pain Paper 2 - Final Experimental Report

**Date:** 2026-01-16
**Objective:** 3-class pain classification using entropy-complexity features

---

## Paper 1 Reference (Boda et al., ICMI 2025)

| Aspect | Paper 1 | This Study |
|--------|---------|------------|
| Features | catch22 (24/signal) | Entropy-Complexity (8/signal) |
| Total Features | 72 | 24 |
| Signals | EDA, BVP, RESP | EDA, BVP, RESP |
| Normalization | z-score | Global z-score |
| No-Pain Class | Baseline only | Baseline only (Phase 6) |
| Optimization | Bayesian (Optuna) | Optuna (50 trials) |

### Paper 1 Three-Class Results

| Model | Optimization | Validation | Balanced Acc |
|-------|--------------|------------|--------------|
| Random Forest | Bayesian | 80/20 | 79.4% |
| XGBoost | Default | LOSO | 78.0% |
| LightGBM | Bayesian | LOSO | 75.8% |

**Note:** Paper 1's best result (79.4%) uses 80/20 split. Best LOSO is 78.0%.

---

## Validation Methodology

### 80/20 Split Limitation

Both this study and Paper 1 use standard `train_test_split` with class stratification. This does **not** ensure subject separation - samples from the same subject may appear in both train and test sets. This can inflate accuracy due to subject-specific pattern leakage.

**LOSO (Leave-One-Subject-Out) is the only valid subject-independent evaluation.**

---

## Results Summary

### Primary Comparison (LOSO - Subject-Independent)

| Study | Model | LOSO Balanced Acc |
|-------|-------|-------------------|
| Paper 1 | XGBoost | **78.0%** |
| This Study (Original) | LightGBM | 74.5% |
| This Study (Baseline-Only) | RandomForest | 77.2% |

**Gap to Paper 1 LOSO:** -0.8 percentage points

### Secondary Comparison (80/20 - May Have Subject Leakage)

| Study | Model | 80/20 Balanced Acc |
|-------|-------|-------------------|
| Paper 1 | Random Forest | 79.4% |
| This Study (Original) | LightGBM | 75.6% |
| This Study (Baseline-Only) | Medium MLP | 80.1% |

---

## Detailed Results by Phase

### Original Pipeline (no_pain = baseline + rest)

| Phase | Validation | Best Model | Balanced Acc |
|-------|------------|------------|--------------|
| 1 | 80/20 | LightGBM | 75.6% |
| 3 | LOSO | LightGBM | 74.5% |
| 4 | LOSO | LightGBM | 71.6% |
| 5 | LOSO (hierarchical) | LightGBM | 71.7% |

### Baseline-Only Pipeline (no_pain = baseline only, matching Paper 1)

| Step | Validation | Best Model | Balanced Acc |
|------|------------|------------|--------------|
| 1 | 80/20 | Stacked | 73.8% |
| 2 | 80/20 | Medium MLP | 80.1% |
| 3 | LOSO | RandomForest | 77.2% |
| 4 | LOSO | Stacked | 65.7% |
| 5 | LOSO (hierarchical) | Stacked | 67.2% |

---

## Key Findings

### 1. Labeling Strategy

| Labeling | Best LOSO |
|----------|-----------|
| Original (baseline + rest) | 74.5% |
| Baseline-only (rest excluded) | 77.2% |
| Improvement | +2.7 pp |

Excluding rest segments from no_pain class improves LOSO accuracy by 2.7 percentage points.

### 2. Per-Subject Normalization Failure

| Normalization | CV Accuracy | LOSO Accuracy | Gap |
|---------------|-------------|---------------|-----|
| Global z-score | 72.9% | 71.6% | 1.3 pp |
| Per-subject baseline | 75.5% | 35.3% | 40.2 pp |

Per-subject baseline normalization fails in LOSO (35% = random). This indicates data leakage.

### 3. Hierarchical Classification

| Stage | Accuracy |
|-------|----------|
| Stage 1 (Pain Detection) | 90-98% |
| Stage 2 (Intensity) | 58-60% |
| Combined 3-Class | 67-72% |

Pain detection is reliable. Intensity discrimination (low vs high) is the bottleneck.

### 4. Model Comparison

| Model Type | Best 80/20 | Best LOSO |
|------------|------------|-----------|
| Tree Ensembles | 75.6% (LightGBM) | 77.2% (RandomForest) |
| Neural Networks | 80.1% (Medium MLP) | Not tested in LOSO |

Neural networks show strong 80/20 performance but were not systematically evaluated in LOSO.

---

## Comparison to Paper 1

| Metric | Paper 1 | This Study | Delta |
|--------|---------|------------|-------|
| Best LOSO | 78.0% (XGBoost) | 77.2% (RandomForest) | -0.8 pp |
| Best 80/20 | 79.4% (RandomForest) | 80.1% (Medium MLP) | +0.7 pp |

**Conclusion:** Entropy-complexity features achieve comparable but slightly lower LOSO performance than catch22 features (-0.8 pp). The 80/20 comparison is not reliable due to potential subject leakage in both studies.

---

## Recommendations

1. **Use LOSO for all comparisons** - 80/20 results are unreliable for subject-independent claims
2. **Use baseline-only labeling** - matches Paper 1 methodology, improves performance
3. **Use global z-score normalization** - per-subject normalization causes data leakage
4. **Focus on intensity discrimination** - this is the primary bottleneck (58-60%)

---

## Next Steps (Phase 7)

- Meta-analysis to select best model configuration
- Nested Optuna-LOSO validation for unbiased hyperparameter optimization
- Target: exceed Paper 1 LOSO baseline (78.0%)

---

## File Structure

```
results/
  phase1_ensembles/           # 80/20 ensembles (original)
  phase3_loso/                # LOSO (original)
  phase4/                     # Full training (original)
  phase5/                     # Hierarchical (original)
  phase6_step1_ensembles/     # 80/20 ensembles (baseline-only)
  phase6_step2_neuralnets/    # 80/20 neural nets (baseline-only)
  phase6_step3_loso/          # LOSO (baseline-only)
  phase6_step4_full_training/ # Full training (baseline-only)
  phase6_step5_hierarchical/  # Hierarchical (baseline-only)
```

---

*Report generated by ML-experiment-loop pipeline*
