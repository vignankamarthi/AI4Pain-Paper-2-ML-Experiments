# Stage 0: Complexity-Entropy Plane Analysis

**Status:** COMPLETE
**Methodology:** Baseline-only (rest segments excluded)

---

## Objective

Visualize binary pain classification on the Complexity-Entropy (C-H) plane using only 2 features. No ML model required - just linear separation in 2D.

---

## Classification Task

| Class | Definition |
|-------|------------|
| 0 | Baseline (pre-stimulus) |
| 1 | Pain (low + high combined) |

**Rest segments:** EXCLUDED

---

## Features

Only 2 features from EDA signal:
- **H:** Permutation Entropy (pe)
- **C:** Statistical Complexity (comp)

| Parameter | Value |
|-----------|-------|
| Signal | EDA |
| Dimension (d) | 7 |
| Time Delay (tau) | 2 |

---

## Normalization

Global z-score (valid for LOSO validation).

---

## Output Files

```
results/stage0_binary/
    ch_plane_binary.png       # Main C-H plane with decision boundary
    ch_plane_comparison.png   # Raw vs normalized comparison
```

---

## Execution

```bash
python src/stage0_binary.py
```

---

## Results (2026-01-18)

| Metric | Value |
|--------|-------|
| Silhouette Score | **0.8414** |
| Linear Accuracy | **99.92%** |

### Key Finding

With just 2 features (H, C) and a simple linear boundary, pain states separate almost perfectly from baseline in the entropy-complexity plane. No complex ML model needed.
