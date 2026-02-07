# Phase 8: Cluster Experiments

**Status:** READY_TO_RUN
**Environment:** Northeastern Explorer Cluster (SLURM, H100/H200 GPUs)
**Target:** Beat Paper 1's 78.0% 3-class LOSO balanced accuracy

---

## Motivation

Phases 1-7 ran locally and established:
- Binary pain detection is solved (99.92% C-H plane, 2 features)
- 3-class LOSO peaked at 77.2% with 24 entropy-complexity features
- Pain intensity discrimination (low vs high) at 58-60% is the bottleneck
- Phase 7 (nested LOSO) terminated at 17/53 folds due to OOM (exit 137)

Local compute is insufficient for the three approaches most likely to break the ceiling.

---

## Sub-Experiments

| Phase | Experiment | Hypothesis | Compute |
|-------|------------|------------|---------|
| 8.1 | Raw Signal Deep Learning | End-to-end models on raw waveforms bypass feature extraction bottleneck | GPU (H100) |
| 8.2 | Feature Fusion | catch22 + entropy-complexity features are complementary | CPU (high memory) |
| 8.3 | Nested LOSO Completion | Complete Phase 7 with adequate memory (128GB) | CPU (high memory) |

---

## Cluster Resources

| Resource | Phase 8.1 | Phase 8.2 | Phase 8.3 |
|----------|-----------|-----------|-----------|
| Partition | gpu | short | gpu |
| GPU | H100 x1 | None | None |
| CPUs | 8 | 8 | 8 |
| Memory | 64GB | 64GB | 128GB |
| Wall Time | 8h (resubmit) | 4h | 8h (resubmit) |

---

## Execution Order

Experiments are independent. Run in parallel or any order. Recommended:

1. **Phase 8.3** first -- lowest risk, completes existing work
2. **Phase 8.2** next -- feature fusion is straightforward
3. **Phase 8.1** last -- most complex, requires architecture design

---

## Success Criteria

| Metric | Target | Current Best | Paper 1 |
|--------|--------|--------------|---------|
| 3-Class LOSO Balanced Accuracy | > 78.0% | 77.2% | 78.0% |

Any sub-experiment exceeding 78.0% LOSO is a success.

---

## Documentation

- [PHASE8_1.md](PHASE8_1.md) -- Raw Signal Deep Learning
- [PHASE8_2.md](PHASE8_2.md) -- Feature Fusion
- [PHASE8_3.md](PHASE8_3.md) -- Nested LOSO Completion
- [TASK.md](TASK.md) -- Cluster Execution Guide

---

## Methodology (Inherited)

All sub-experiments follow the same constraints as Phases 1-7:

- **No-pain class:** Baseline segments ONLY (rest excluded)
- **Normalization:** Global z-score (never per-subject)
- **Validation:** LOSO (primary), 80/20 (secondary)
- **Metric:** Balanced accuracy
- **Hyperparameters:** Optuna optimization (50 trials, TPE sampler)
