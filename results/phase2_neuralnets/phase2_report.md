# Phase 2: Neural Network Exploration Results

**Generated:** 2026-01-15 06:48

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Best Neural Net** | Simple MLP |
| **Best NN Balanced Accuracy** | 76.38% |
| **Best Overall Model** | Simple MLP |
| **Best Overall Balanced Acc** | 76.38% |
| **Paper 1 Baseline** | 79.4% |

---

## Neural Network Architectures

Four MLP architectures were explored:
1. **Simple MLP**: 2 hidden layers
2. **Medium MLP**: 3 hidden layers
3. **Deep MLP**: 4 hidden layers
4. **Regularized MLP**: 3 hidden layers with high dropout and L2 regularization

### Training Configuration
- **Optimization:** Optuna (Bayesian) with 100 trials per architecture
- **Validation:** 5-fold stratified CV
- **Early Stopping:** Patience = 20 epochs
- **Device:** mps

---

## Neural Network Leaderboard

| Rank | Architecture | Balanced Acc | Accuracy | F1 | No Pain | Low Pain | High Pain |
|------|--------------|--------------|----------|-----|---------|----------|-----------|
| 1 | Simple MLP | 76.38% | 82.32% | 0.813 | 100.00% | 87.40% | 41.73% |
| 2 | Deep MLP | 75.85% | 81.93% | 0.819 | 100.00% | 66.93% | 60.63% |
| 3 | Medium MLP | 75.07% | 81.34% | 0.811 | 100.00% | 73.23% | 51.97% |
| 4 | Regularized MLP | 74.80% | 81.14% | 0.809 | 100.00% | 73.23% | 51.18% |


---

## Combined Leaderboard (Phase 1 + Phase 2)

| Rank | Model | Type | Balanced Acc | Accuracy |
|------|-------|------|--------------|----------|
| 1 | Simple MLP | neural_net | 76.38% | 82.32% |
| 2 | Deep MLP | neural_net | 75.85% | 81.93% |
| 3 | LightGBM | ensemble | 75.59% | 81.73% |
| 4 | Medium MLP | neural_net | 75.07% | 81.34% |
| 5 | Regularized MLP | neural_net | 74.80% | 81.14% |
| 6 | Stacked (RF+XGB+LGB) | ensemble | 72.97% | 79.76% |
| 7 | XGBoost | ensemble | 72.70% | 79.57% |
| 8 | Random Forest | ensemble | 72.70% | 79.57% |
| 9 | Stacked (RF+XGB) | ensemble | 71.39% | 78.59% |


---

## Analysis

### Neural Net vs Ensemble Comparison

**Neural networks improved by 0.79% over best ensemble!**

### Key Observations

1. **Architecture Impact:** The Simple MLP architecture performed best, suggesting that simpler models generalize better on this feature space.

2. **Pain Discrimination Challenge:** All models struggle most with distinguishing
   low pain from high pain states, consistent with Phase 1 ensemble findings.

3. **No Pain Detection:** Neural networks maintain high accuracy on no-pain states,
   validating that the per-subject baseline normalization is effective.

---

## Best Hyperparameters

### Simple MLP
```json
{
  "layer1": 512,
  "layer2": 32,
  "dropout": 0.3,
  "lr": 0.005,
  "batch_size": 16,
  "activation": "elu",
  "input_dim": 32
}
```

### Medium MLP
```json
{
  "layer1": 128,
  "layer2": 256,
  "layer3": 128,
  "dropout": 0.2,
  "lr": 0.001,
  "batch_size": 16,
  "activation": "elu",
  "input_dim": 32
}
```

### Deep MLP
```json
{
  "layer1": 256,
  "layer2": 128,
  "layer3": 64,
  "layer4": 64,
  "dropout": 0.5,
  "lr": 0.001,
  "batch_size": 16,
  "activation": "elu",
  "input_dim": 32
}
```

### Regularized MLP
```json
{
  "layer1": 512,
  "layer2": 128,
  "layer3": 64,
  "dropout": 0.4,
  "lr": 0.0001,
  "batch_size": 16,
  "activation": "elu",
  "l2_reg": 0.1,
  "input_dim": 32
}
```

---

## Recommendation for Phase 3 LOSO

Based on combined results, recommend validating these models with LOSO:

1. **Simple MLP** (neural_net): 76.38%
2. **Deep MLP** (neural_net): 75.85%
3. **LightGBM** (ensemble): 75.59%
4. **Medium MLP** (neural_net): 75.07%
5. **Regularized MLP** (neural_net): 74.80%


---

## Output Files

```
results/phase2_neuralnets/
├── leaderboard.csv
├── combined_leaderboard.csv
├── hyperparameters.json
├── phase2_report.md
├── confusion_matrices/
│   └── [architecture]_confusion_matrix.png
├── training_curves/
│   └── [architecture]_training_curves.png
└── models/
    └── [architecture]_best.pth
```

---

**End of Phase 2 Report**
