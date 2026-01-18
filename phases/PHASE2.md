# Phase 2: Neural Network Exploration (80/20 Split)

**Status:** COMPLETE
**Methodology:** Baseline-only (rest segments excluded)

---

## Objective

Test MLP architectures for 3-class pain classification. Triggered when Phase 1 best < 85%.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Architectures | Simple, Medium, Deep, Regularized MLP |
| Optuna trials | 100 per architecture |
| Class weights | Balanced (in loss function) |
| Normalization | Global z-score |

---

## Results

| Rank | Model | Balanced Accuracy |
|------|-------|-------------------|
| 1 | Medium MLP | 80.05% |
| 2 | Regularized MLP | 80.05% |
| 3 | Simple MLP | 79.79% |
| 4 | Deep MLP | 79.00% |

**Note:** 80/20 split does not ensure subject separation. LOSO validation required for subject-independent evaluation.

---

## Output Files

- `results/phase2_neuralnets/leaderboard.csv`
- `results/phase2_neuralnets/phase2_report.md`

---

## Execution

```bash
python src/phase2_neural_experiments.py
```

---

*Phase 2 complete. Medium MLP achieved 80.05% balanced accuracy.*
