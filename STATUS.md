# Experiment Status Tracker

**Last Updated:** 2026-01-16
**Status:** Phases 1-5 COMPLETE, Stage 0 PENDING, Phase 6 PENDING

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

## Current Best Results

### LOSO (Valid Subject-Independent Comparison)

| Model | LOSO Acc | vs Paper 1 |
|-------|----------|------------|
| RandomForest | 77.2% | -0.8 pp |
| XGBoost | 76.4% | -1.6 pp |
| LightGBM | 75.6% | -2.4 pp |

### 80/20 (Subject Leakage Warning)

| Model | 80/20 Acc |
|-------|-----------|
| Medium MLP | 80.1% |
| Stacked (RF+XGB+LGB) | 73.8% |

**Note:** 80/20 splits do not guarantee subject separation.

---

## Phase Completion Status

| Phase | Description | Status | Best Result |
|-------|-------------|--------|-------------|
| Stage 0 | Binary (Baseline vs Pain) | PENDING | - |
| 1 | 80/20 Ensembles | COMPLETE | 73.8% |
| 2 | 80/20 Neural Nets | COMPLETE | 80.1% (Medium MLP) |
| 3 | LOSO Validation | COMPLETE | 77.2% (RandomForest) |
| 4 | Full Training + LOSO | COMPLETE | 65.7% |
| 5 | Hierarchical | COMPLETE | 67.2% |
| 6 | Final Optimized | PENDING | - |

---

## Key Findings

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

---

## Output Files

```
results/
  phase1_ensembles/           # 80/20 ensembles
  phase2_neuralnets/          # 80/20 neural nets
  phase3_loso/                # LOSO validation
  phase4_full_training/       # Full training
  phase5_hierarchical/        # Hierarchical
  archive_no_pain_included/   # Old methodology (deprecated)
```

---

## Next Steps

1. **Stage 0:** Binary classification (Baseline vs Pain) - entropy-complexity plane
2. **Phase 6:** Nested Optuna-LOSO to exceed Paper 1's 78.0% LOSO

See `phases/STAGE0.md` and `phases/PHASE6.md` for details.
