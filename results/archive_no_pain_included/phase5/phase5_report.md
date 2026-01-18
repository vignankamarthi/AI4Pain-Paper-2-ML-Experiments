# Phase 5: Hierarchical Binary Classification Results

**Generated:** 2026-01-16 01:20:09

---

## Summary

Phase 5 implements a two-stage hierarchical classifier:
- **Stage 1:** No Pain vs Pain (binary)
- **Stage 2:** Low Pain vs High Pain (binary, pain samples only)

---

## Results by Configuration

| Configuration | Stage 1 Acc | Stage 2 Acc | Combined 3-Class |
|--------------|-------------|-------------|------------------|
| LightGBM_global_zscore | 97.84% +/- 4.95% | 59.59% +/- 12.62% | 71.72% +/- 9.04% |
| Stacked_global_zscore | 97.84% +/- 4.95% | 59.59% +/- 12.62% | 71.72% +/- 9.04% |
| LightGBM_raw | 98.19% +/- 4.15% | 58.65% +/- 11.35% | 71.20% +/- 8.12% |

---

## Best Configuration

**LightGBM_global_zscore**
- Stage 1 (Pain Detection): 97.84% +/- 4.95%
- Stage 2 (Intensity): 59.59% +/- 12.62%
- Combined 3-Class: 71.72% +/- 9.04%

---

## Comparison to Direct 3-Class (Phase 4)

| Approach | Best LOSO Balanced Accuracy |
|----------|----------------------------|
| Direct 3-Class (Phase 4) | (see Phase 4 results) |
| Hierarchical (Phase 5) | 71.72% |
| Paper 1 Baseline | 79.4% |

---

## Analysis

### Stage 1: Pain Detection
Expected to be easier task - distinguishing pain from no-pain states.

### Stage 2: Intensity Discrimination
The harder task - distinguishing low from high pain intensity.

### Error Propagation
Stage 1 errors cascade to final prediction. If Stage 1 misclassifies a pain sample as no-pain, Stage 2 never gets a chance to correct it.

---

*Phase 5 hierarchical classification complete.*
