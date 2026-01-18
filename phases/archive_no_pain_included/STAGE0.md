# Stage 0: Binary Classification with Per-Subject Baseline Normalization

## Objective
Establish binary pain classification (pain vs no-pain) baseline and discover optimal normalization strategy.

**Status:** COMPLETE

---

## Key Finding

**Per-subject baseline normalization achieves 99.97% linear accuracy** on the C-H plane.

| Normalization Method | Accuracy |
|----------------------|----------|
| Raw (none) | 87.97% |
| Global StandardScaler | 87.45% |
| Per-subject z-score | 97.44% |
| **Per-subject baseline** | **99.97%** |

---

## How Per-Subject Baseline Normalization Works

For each subject:
1. Identify their no-pain samples (baseline + rest states)
2. Compute mean and std of each feature from ONLY those no-pain samples
3. Z-score ALL their samples using those baseline statistics

```python
for each subject:
    baseline_mean = mean(features[no_pain_samples])
    baseline_std = std(features[no_pain_samples])
    normalized = (features - baseline_mean) / baseline_std
```

This aligns all subjects to a common reference frame where no-pain = (0,0).

---

## Best Parameters

| Parameter | Value |
|-----------|-------|
| Signal | EDA |
| Embedding Dimension (d) | 7 |
| Time Delay (tau) | 2 |
| Features | pe (H) + comp (C) |

---

## Outputs

```
results/stage0_binary/
├── STAGE0_FINAL_REPORT.md
└── FINAL_ch_plane_binary.png
```

---

## Implications for Subsequent Phases

1. **Per-subject baseline normalization is REQUIRED** - apply to all phases
2. **EDA d=7 tau=2** is optimal for this dataset
3. **For 3-class:** Apply same normalization to all 8 entropy features
