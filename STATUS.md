# Experiment Status Tracker

**Last Updated:** 2026-01-14 22:15

---

## Current Phase
**STAGE0_COMPLETE**

**Status:** APPROVED, READY_FOR_PHASE1

**Next Action:** Proceed to Phase 1 (3-class ensemble exploration) with per-subject baseline normalization.

---

## Phase Progress

| Phase | Status | Started | Completed |
|-------|--------|---------|-----------|
| Stage 0: Binary Classification | COMPLETE | 2026-01-14 | 2026-01-14 |
| Phase 1: Ensemble Exploration | NOT_STARTED | - | - |
| Phase 2: Neural Net Exploration | NOT_STARTED | - | - |
| Phase 3: LOSO Validation | NOT_STARTED | - | - |

---

## Stage 0 Final Results

### Key Achievement
**99.97% Linear Accuracy** on the C-H plane with per-subject baseline normalization.

### Configuration
| Parameter | Value |
|-----------|-------|
| Signal | EDA |
| Embedding Dimension (d) | 7 |
| Time Delay (tau) | 2 |
| Features | Permutation Entropy (H) + Statistical Complexity (C) |
| Normalization | Per-subject baseline (no-pain reference) |

### Normalization Comparison
| Method | Linear Accuracy |
|--------|-----------------|
| Raw (none) | 87.97% |
| Global StandardScaler | 87.45% |
| Per-subject z-score | 97.44% |
| **Per-subject baseline** | **99.97%** |

### Key Finding
Per-subject baseline normalization is **REQUIRED** for optimal classification. Global normalization (Paper 1 method) doesn't improve performance.

### Outputs
```
results/stage0_binary/
├── STAGE0_FINAL_REPORT.md    (detailed report)
└── FINAL_ch_plane_binary.png (visualization with decision boundary)
```

### Implications for Next Phases
1. Apply per-subject baseline normalization to Phase 1 (3-class)
2. For LOSO: normalize test subjects using their own no-pain baseline
