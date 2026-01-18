# Phase 5: Hierarchical Classification

**Status:** COMPLETE
**Methodology:** Baseline-only (rest segments excluded)

---

## Objective

Two-stage cascaded classification to separate pain detection from intensity discrimination.

---

## Architecture

```
                    Input Sample
                         |
                         v
              +---------------------+
              |   Stage 1 Classifier |
              |  Baseline vs Pain    |
              +---------------------+
                    /         \
                   /           \
            Baseline            Pain
            (done)               |
                                 v
                      +---------------------+
                      |   Stage 2 Classifier |
                      |   Low Pain vs High   |
                      +---------------------+
                            /         \
                           /           \
                     Low Pain       High Pain
```

---

## Results

| Configuration | Stage 1 (Pain Detection) | Stage 2 (Intensity) | Combined 3-Class |
|---------------|--------------------------|---------------------|------------------|
| Stacked_global_zscore | 90.57% | 60.30% | 67.24% |
| LightGBM_raw | 90.53% | 58.65% | 66.09% |

---

## Key Findings

1. **Pain detection is reliable** - Stage 1 achieves 90.57% accuracy
2. **Intensity discrimination is the bottleneck** - Stage 2 only 60.30%
3. **Hierarchical does not outperform direct 3-class** - 67.24% vs 77.15% (Phase 3)

---

## Output Files

- `results/phase5_hierarchical/stage1_results.csv`
- `results/phase5_hierarchical/stage2_results.csv`
- `results/phase5_hierarchical/combined_results.csv`
- `results/phase5_hierarchical/phase5_report.md`

---

## Execution

```bash
python src/phase5_hierarchical.py
```

---

*Phase 5 complete. Hierarchical approach underperforms direct 3-class classification.*
