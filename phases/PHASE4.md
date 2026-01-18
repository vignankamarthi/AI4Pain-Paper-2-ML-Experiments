# Phase 4: Full Training + LOSO Optimization

**Status:** COMPLETE
**Methodology:** Baseline-only (rest segments excluded)

---

## Objective

Comprehensive hyperparameter optimization with LOSO validation.

---

## Experiments

### 4.1 Normalization Comparison

| Method | CV Acc | LOSO Acc |
|--------|--------|----------|
| Global z-score | 68.56% | 64.94% |
| Raw | 67.66% | 64.94% |
| Per-subject baseline | 75.42% | 32.76% |

**Critical Finding:** Per-subject baseline normalization fails catastrophically in LOSO (32.76% = random guessing). This indicates data leakage in the normalization step.

### 4.2 Full Training Results

| Model | Normalization | CV Accuracy | LOSO Accuracy |
|-------|---------------|-------------|---------------|
| Stacked | global_zscore | 66.93% | 65.72% |
| LightGBM | global_zscore | 68.56% | 64.94% |
| LightGBM | raw | 67.66% | 64.94% |

---

## Key Findings

1. **Per-subject normalization fails in LOSO** - 32.76% accuracy = random chance
2. **Global z-score is stable** - consistent CV and LOSO performance
3. **Full training LOSO underperforms Phase 3** - suggests simpler approach is better

---

## Output Files

- `results/phase4_full_training/normalization_results.csv`
- `results/phase4_full_training/full_leaderboard.csv`
- `results/phase4_full_training/step4_report.md`

---

## Execution

```bash
python src/phase4_full_training.py
```

---

*Phase 4 complete. Per-subject normalization failure confirmed.*
