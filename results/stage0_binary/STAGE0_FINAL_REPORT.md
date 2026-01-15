# Stage 0: Binary Classification Analysis - Final Report

**Generated:** 2026-01-14

---

## Key Result

**99.97% Linear Accuracy** on the C-H plane with per-subject baseline normalization.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| **Signal** | EDA |
| **Embedding Dimension (d)** | 7 |
| **Time Delay (tau)** | 2 |
| **Features** | Permutation Entropy (H) + Statistical Complexity (C) |
| **Normalization** | Per-subject baseline (no-pain reference) |
| **Linear Accuracy** | **99.97%** |
| **Silhouette Constant** | 0.6295 |

---

## Critical Finding: Per-Subject Baseline Normalization

The key to achieving near-perfect binary classification is **per-subject baseline normalization**:

| Normalization Method | Silhouette | Linear Accuracy |
|----------------------|------------|-----------------|
| Raw (none) | 0.4904 | 87.97% |
| Global StandardScaler (Paper 1) | 0.4759 | 87.45% |
| Per-subject z-score | 0.6505 | 97.44% |
| **Per-subject baseline (no-pain ref)** | 0.6295 | **99.97%** |

### How Baseline Normalization Works

For each subject:
1. Calculate mean(H) and std(H) from their **no-pain samples only**
2. Calculate mean(C) and std(C) from their **no-pain samples only**
3. Z-score ALL their samples using these baseline statistics

```
H_normalized = (H - H_baseline_mean) / H_baseline_std
C_normalized = (C - C_baseline_mean) / C_baseline_std
```

This aligns all subjects to a common reference frame where no-pain states cluster around (0,0).

---

## Physiological Interpretation

After baseline normalization:
- **No Pain (green):** Centered at origin (0,0) - each subject's baseline
- **Pain (red):** Shifted to **lower H, higher C** relative to baseline
  - Decreased Permutation Entropy -> more predictable signal patterns
  - Increased Statistical Complexity -> more structured dynamics

This reflects the physiological response to pain: the autonomic nervous system shifts from a relaxed (high entropy, low complexity) state to a stress response (lower entropy, higher complexity).

---

## Binary Class Definition

| Class | States Included | Samples |
|-------|-----------------|---------|
| **No Pain (0)** | baseline + rest | 1620 |
| **Pain (1)** | low_pain + high_pain | 1272 |
| **Total** | | 2892 |

---

## Outputs

```
results/stage0_binary/
├── STAGE0_FINAL_REPORT.md          (this file)
└── FINAL_ch_plane_binary.png       (main visualization)
```

---

## Implications for Paper 2

1. **Per-subject normalization is REQUIRED** for optimal classification
2. **The C-H plane achieves near-perfect binary separation** (99.97%) when properly normalized
3. **For Phase 0 (3-class):** Apply same baseline normalization strategy
4. **For LOSO validation:** Train models on normalized features; normalize test subjects using their own no-pain baseline

---

## Conclusion

Binary pain classification on the Complexity-Entropy plane achieves **99.97% linear accuracy** when using per-subject baseline normalization. This confirms the theoretical foundation that pain states produce distinct entropy/complexity signatures relative to each individual's baseline physiology.
