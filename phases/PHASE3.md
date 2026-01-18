# Phase 3: LOSO Validation

**Status:** COMPLETE
**Methodology:** Baseline-only (rest segments excluded)

---

## Objective

Rigorous Leave-One-Subject-Out validation on top models from Phases 1-2.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Validation | LOSO (53 folds) |
| Subjects | 53 (train + validation combined) |
| Normalization | Global z-score (LOSO-compatible) |

---

## Results

| Rank | Model | LOSO Balanced Acc | 95% CI |
|------|-------|-------------------|--------|
| 1 | RandomForest | 77.15% | [74.73%, 79.57%] |
| 2 | XGBoost | 76.36% | [73.90%, 78.83%] |
| 3 | LightGBM | 75.63% | [73.22%, 78.04%] |

**Comparison to Paper 1:**
- Paper 1 LOSO: 78.0% (XGBoost)
- This Study LOSO: 77.15% (RandomForest)
- Gap: -0.85 pp

---

## Key Finding

LOSO shows more conservative estimates than 80/20. Neural networks were not evaluated in LOSO because they did not exceed ensembles by the required 2% margin in 80/20.

---

## Output Files

- `results/phase3_loso/loso_leaderboard.csv`
- `results/phase3_loso/per_subject_results.csv`
- `results/phase3_loso/step3_report.md`

---

## Execution

```bash
python src/phase3_loso_validation.py
```

---

*Phase 3 complete. RandomForest achieved best LOSO performance (77.15%).*
