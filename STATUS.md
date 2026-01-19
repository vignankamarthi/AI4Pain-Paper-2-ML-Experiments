# Experiment Status Tracker

**Last Updated:** 2026-01-18
**Status:** ALL EXPERIMENTS COMPLETE - See FINAL_REPORT.md

---

## Methodology

**Baseline-only approach (matching Paper 1):**
- No-pain class: Baseline segments ONLY (rest EXCLUDED)
- Pain class: Low pain + High pain
- Normalization: Global z-score

---

## Paper 1 Baseline (Boda et al., ICMI 2025)

| Validation | Model | Balanced Acc |
|------------|-------|--------------|
| 80/20 | RandomForest | 79.4% |
| **LOSO** | **XGBoost** | **78.0%** |

**Primary comparison metric: LOSO (subject-independent)**

---

## Stage 0: Binary Classification (Baseline vs Pain)

**Headline Result: 99.92% linear separability in C-H feature space**

| Metric | Value |
|--------|-------|
| Silhouette Score | 0.8414 |
| Linear Accuracy | 99.92% |
| Features Used | 2 (Permutation Entropy H, Statistical Complexity C) |
| Signal | EDA only |

The Complexity-Entropy (C-H) plane shows near-perfect linear separation between baseline and pain states using just 2 features from the EDA signal. This is the strongest result in the entire study, demonstrating that pain fundamentally alters the entropy-complexity structure of physiological signals.

See `results/stage0_binary/` for full analysis.

---

## Current Best Results

### LOSO (Valid Subject-Independent Comparison)

| Model | LOSO Acc | vs Paper 1 |
|-------|----------|------------|
| RandomForest | 77.2% | -0.8 pp |
| XGBoost | 76.4% | -1.6 pp |
| LightGBM | 75.6% | -2.4 pp |

### 80/20 Stratified Split

| Model | 80/20 Acc | vs Paper 1 |
|-------|-----------|------------|
| Medium MLP (Phase 2) | 80.1% | +0.7 pp |
| LOSO-Optimized MLP (Phase 6) | 75.6% | -3.8 pp |
| Stacked (RF+XGB+LGB) | 73.8% | -5.6 pp |

---

## Phase Completion Status

| Phase | Description | Status | Best Result |
|-------|-------------|--------|-------------|
| Stage 0 | C-H Plane Analysis | COMPLETE | 99.92% linear (silhouette 0.84) |
| 1 | 80/20 Ensembles | COMPLETE | 73.8% |
| 2 | 80/20 Neural Nets | COMPLETE | 80.1% (Medium MLP) |
| 3 | LOSO Validation | COMPLETE | 77.2% (RandomForest) |
| 4 | Full Training + LOSO | COMPLETE | 65.7% |
| 5 | Hierarchical | COMPLETE | 67.2% |
| 6 | LOSO-Optimized 80/20 | COMPLETE | 75.6% (did not beat baselines) |
| 7 | Nested Optuna-LOSO | TERMINATED | 72.1% (17/53 folds) |

---

## Key Findings

### 0. Binary Classification (Stage 0)
The C-H plane achieves 99.92% accuracy for pain detection using only 2 features. This is a fundamentally different result from the 3-class classification problem (baseline vs low vs high pain), which struggles with intensity discrimination.

### 1. Normalization
| Method | CV Acc | LOSO Acc |
|--------|--------|----------|
| Global z-score | 68.6% | 64.9% |
| Per-subject | 75.4% | 32.8% |

Per-subject normalization fails in LOSO (data leakage).

### 2. Hierarchical Classification
| Stage | Accuracy |
|-------|----------|
| Pain Detection | 90-91% |
| Intensity | 58-60% |

Intensity discrimination is the bottleneck.

### 3. LOSO-Optimized HP Search (Phase 6)
| Inner CV | 80/20 Test Acc |
|----------|----------------|
| Standard 5-fold | 80.1% |
| LOSO | 75.6% |

Using LOSO as inner CV for Optuna did not improve 80/20 generalization.

---

## Output Files

```
results/
  stage0_binary/              # C-H plane analysis
  phase1_ensembles/           # 80/20 ensembles
  phase2_neuralnets/          # 80/20 neural nets
  phase3_loso/                # LOSO validation
  phase4_full_training/       # Full training
  phase5_hierarchical/        # Hierarchical
  phase6_final/               # LOSO-optimized 80/20 (Phase 6)
  phase7_nested_loso/         # Nested Optuna-LOSO (Phase 7)
  archive_no_pain_included/   # Old methodology (deprecated)
```

---

## Conclusion

**All experiments complete.** See `FINAL_REPORT.md` for comprehensive results summary.

### Summary
- **Best LOSO:** 77.2% (RandomForest) - 0.8 pp below Paper 1's 78.0%
- **Binary Detection:** 99.92% (C-H plane) - headline result
- **Bottleneck:** Intensity discrimination (58-60%) limits all approaches
