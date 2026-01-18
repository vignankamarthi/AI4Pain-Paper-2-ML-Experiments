# AI4Pain Paper 2 - Experimental Plan

## Research Objective
Improve pain classification accuracy beyond Paper 1's LOSO baseline using entropy and complexity-based features.

**Paper 1 Reference:**
- Best 80/20: 79.4% (RandomForest)
- Best LOSO: 78.0% (XGBoost)

**Target:** Exceed Paper 1 LOSO (78.0%) with subject-independent validation.

---

## Methodology

**Baseline-only approach (matching Paper 1):**
- No-pain class: Baseline segments ONLY (rest EXCLUDED)
- Pain class: Low pain + High pain
- Normalization: Global z-score
- Validation: LOSO (primary), 80/20 (secondary)

---

## Results Summary

### Paper 1 Comparison (LOSO - Primary Metric)

| Study | Model | LOSO Balanced Acc | vs Paper 1 |
|-------|-------|-------------------|------------|
| Paper 1 | XGBoost | **78.0%** | - |
| This Study | RandomForest | 77.2% | -0.8 pp |

**Status:** 0.8 pp below Paper 1 LOSO baseline.

### 80/20 Results (Subject Leakage Warning)

| Study | Model | 80/20 Balanced Acc |
|-------|-------|-------------------|
| Paper 1 | RandomForest | 79.4% |
| This Study | Medium MLP | 80.1% |

**Note:** 80/20 splits do not ensure subject separation.

---

## Todo Checklist

### Main Pipeline (Baseline-Only)
- [ ] Stage 0: Binary classification (Baseline vs Pain)
- [x] Phase 1: 80/20 ensembles - Stacked 73.8%
- [x] Phase 2: 80/20 neural nets - Medium MLP 80.1%
- [x] Phase 3: LOSO - RandomForest 77.2%
- [x] Phase 4: Full training + LOSO - Stacked 65.7%
- [x] Phase 5: Hierarchical - 67.2%
- [ ] Phase 6: Final experiment (nested Optuna-LOSO)

---

## Phase Results

### Phase 3: LOSO (Primary Metric)
| Model | LOSO Acc |
|-------|----------|
| RandomForest | 77.2% |
| XGBoost | 76.4% |
| LightGBM | 75.6% |

### Phase 2: 80/20 (Secondary)
| Model | 80/20 Acc |
|-------|-----------|
| Medium MLP | 80.1% |
| Regularized MLP | 80.1% |
| Simple MLP | 79.8% |

---

## Key Findings

1. **Normalization:** Per-subject fails in LOSO (32.8% = random). Use global z-score.
2. **Validation:** 80/20 may have subject leakage. LOSO is the valid comparison.
3. **Bottleneck:** Pain intensity discrimination (58-60%) limits 3-class accuracy.
4. **Pain Detection:** Binary pain detection achieves 90%+ accuracy.

---

## Comparison to Paper 1

| Aspect | Paper 1 | This Study |
|--------|---------|------------|
| Features | catch22 (72) | Entropy-Complexity (24) |
| No-Pain Class | Baseline only | Baseline only |
| Normalization | z-score | Global z-score |
| Best LOSO | 78.0% | 77.2% |
| Delta | - | -0.8 pp |

---

## Next Steps

1. **Stage 0:** Binary classification using entropy-complexity plane
2. **Phase 6:** Nested Optuna-LOSO to close the 0.8 pp gap

See `phases/STAGE0.md` and `phases/PHASE6.md` for details.

---

**Status:** Phases 1-5 COMPLETE. Stage 0 and Phase 6 PENDING.
