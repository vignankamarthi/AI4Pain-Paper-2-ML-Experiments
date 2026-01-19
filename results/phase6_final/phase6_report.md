# Phase 6 Results: LOSO-Optimized 80/20 Experiment

**Generated:** 2026-01-18 18:52:00

---

## Methodology

**Key Innovation:** LOSO as inner CV for Optuna optimization

- Task: 3-class classification (baseline vs low vs high)
- Model: Medium MLP (3-layer)
- Data: 1325 samples, 53 subjects (train + validation pooled)
- Split: 80/20 stratified random
- HP Optimization: Optuna with 50 trials, LOSO inner CV
- Normalization: Global z-score

---

## Results

### Final Test Accuracy (20% held-out)

| Metric | Value |
|--------|-------|
| Balanced Accuracy | 0.7559 (75.59%) |
| Overall Accuracy | 0.6491 (64.91%) |
| F1 Weighted | 0.6486 |

### Per-Class Accuracy

| Class | Accuracy |
|-------|----------|
| No Pain (baseline) | 1.0000 (100.0%) |
| Low Pain | 0.6693 (66.9%) |
| High Pain | 0.5984 (59.8%) |

---

## Comparison to Baselines

| Baseline | Accuracy | Difference |
|----------|----------|------------|
| Paper 1 (79.4%) | 79.4% | -3.81 pp |
| Phase 2 (80.05%) | 80.05% | -4.46 pp |

---

## Optuna Optimization

- Trials: 50
- Inner CV: LOSO (Leave-One-Subject-Out)
- Best inner LOSO score: 0.7698140433989491

### Best Hyperparameters

```json
{
  "layer1": 64,
  "layer2": 48,
  "layer3": 40,
  "dropout": 0.2371197153065265,
  "learning_rate": 0.009475797225215772,
  "weight_decay": 0.00022662277369012458,
  "batch_size": 32,
  "activation": "leaky_relu"
}
```

---

## Data Split

- Training samples: 1060
- Test samples: 265
- Training subjects: 53
- Test subjects: 53

---

## Conclusion

LOSO-based HP optimization did not improve over Phase 2 baseline.

Did not beat Paper 1's 79.4% benchmark.

---

*Phase 6 complete. Single experiment with LOSO-optimized hyperparameters.*
