# Phase 6: LOSO-Optimized 80/20 Experiment

**Status:** COMPLETE
**Task:** 3-class classification (binary is solved at 100%)
**Model:** Medium MLP (best performer from Phase 2)
**Result:** 75.59% balanced accuracy (did not beat baselines)

---

## Objective

Maximize 80/20 accuracy using LOSO-based hyperparameter optimization.

This approach finds hyperparameters that generalize across subjects, then evaluates on a held-out 20% test set matching Paper 1's validation methodology.

---

## Why This Approach?

### The Problem with Standard 5-Fold CV

```
Standard Optuna with 5-fold CV:
- Folds are random splits of samples
- Same subject can appear in multiple folds
- Optimizes for SAMPLE-level generalization
- May overfit to subject-specific patterns
```

### Our Solution: LOSO Inside Optuna

```
LOSO-optimized Optuna:
- Inner CV uses Leave-One-Subject-Out
- Each fold leaves out a different subject
- Optimizes for SUBJECT-level generalization
- Model learns patterns that transfer across individuals
```

---

## Methodology

### Pipeline Overview

```
1325 samples (train + validation pooled, 53 subjects)
                    |
            80/20 STRATIFIED SPLIT
                    |
        +-----------+-----------+
        |                       |
  80% TRAIN (~1060)      20% TEST (~265)
  (~42 subjects)          (~11 subjects)
        |                       |
        v                       |
  +----------------+            |
  | OPTUNA (50)    |            |
  |                |            |
  | Inner: LOSO CV |            |  <- Never touched during training
  | across ~42     |            |
  | subjects       |            |
  +----------------+            |
        |                       |
        v                       v
  Best Hyperparams -----> Train Final Model -----> Evaluate
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Task | 3-class | Binary solved (100% accuracy) |
| Model | Medium MLP | Best 80/20 performer (80.05% in Phase 2) |
| Optuna trials | 50 | Good balance of exploration vs compute |
| Inner CV | LOSO | Optimizes for subject generalization |
| Final eval | 20% held-out | Matches Paper 1 methodology |
| Normalization | Global z-score | Proven in previous phases |

---

## Comparison to Paper 1

| Aspect | Paper 1 | Phase 6 |
|--------|---------|---------|
| Data pool | 53 subjects | 53 subjects |
| No-pain class | Baseline only | Baseline only |
| Normalization | StandardScaler | Global z-score |
| Features | catch22 (72) | Entropy-complexity (24) |
| 80/20 split | Stratified random | Stratified random |
| HP optimization | 5-fold CV | **LOSO CV** |
| Model | RandomForest | Medium MLP |

**Key difference:** We optimize hyperparameters using LOSO, which should find parameters that generalize better across subjects.

---

## Expected Improvement

Phase 2 achieved **80.05%** with standard 5-fold CV inside Optuna.

With LOSO-based optimization:
- Hyperparameters selected for cross-subject generalization
- May reduce overfitting to subject-specific patterns
- Target: **81-82%** (1-2 pp improvement)

---

## Implementation

### Pseudocode

```python
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
import optuna

# Load and prepare data (baseline-only, 3-class)
X, y, subjects = load_data()  # 1325 samples, 53 subjects

# 80/20 stratified split (Paper 1 methodology)
X_train, X_test, y_train, y_test, subj_train, subj_test = train_test_split(
    X, y, subjects, test_size=0.2, stratify=y, random_state=42
)

# Global z-score normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Optuna objective with LOSO inner CV
def objective(trial):
    params = {
        'hidden_dims': trial.suggest_categorical('hidden_dims', [...]),
        'learning_rate': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        # ... other MLP hyperparameters
    }

    # LOSO CV on training subjects (KEY CHANGE)
    logo = LeaveOneGroupOut()
    fold_scores = []

    for train_idx, val_idx in logo.split(X_train, y_train, groups=subj_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = MediumMLP(**params)
        model.fit(X_tr, y_tr)
        score = balanced_accuracy_score(y_val, model.predict(X_val))
        fold_scores.append(score)

    return np.mean(fold_scores)

# Run Optuna optimization
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50)

# Train final model with best hyperparameters
final_model = MediumMLP(**study.best_params)
final_model.fit(X_train, y_train)

# Evaluate on held-out 20% test set
y_pred = final_model.predict(X_test)
final_accuracy = balanced_accuracy_score(y_test, y_pred)
```

---

## Configuration

```python
# Phase 6 Configuration
TASK = '3-class'  # baseline vs low vs high
MODEL = 'Medium MLP'
OPTUNA_TRIALS = 50
INNER_CV = 'LOSO'  # LeaveOneGroupOut by subject
TEST_SIZE = 0.2
NORMALIZATION = 'global_zscore'
RANDOM_SEED = 42
```

---

## Output Files

```
results/phase6_final/
    phase6_report.md          # Final results and analysis
    leaderboard.csv           # Model performance
    confusion_matrix.png      # 3-class confusion matrix
    best_hyperparameters.json # Optuna-selected params
    optuna_study.pkl          # Full optimization history
    checkpoint.json           # Progress tracking
```

---

## Results

### Final Test Accuracy (20% held-out)

| Metric | Value |
|--------|-------|
| Balanced Accuracy | **75.59%** |
| Overall Accuracy | 64.91% |
| F1 Weighted | 0.6486 |

### Per-Class Accuracy

| Class | Accuracy |
|-------|----------|
| No Pain (baseline) | 100.0% |
| Low Pain | 66.9% |
| High Pain | 59.8% |

### Comparison to Baselines

| Baseline | Accuracy | Difference |
|----------|----------|------------|
| Paper 1 (79.4%) | 79.4% | **-3.81 pp** |
| Phase 2 (80.05%) | 80.05% | **-4.46 pp** |

### Best Hyperparameters (from Optuna)

```json
{
  "layer1": 64,
  "layer2": 48,
  "layer3": 40,
  "dropout": 0.237,
  "learning_rate": 0.00948,
  "weight_decay": 0.000227,
  "batch_size": 32,
  "activation": "leaky_relu"
}
```

---

## Conclusion

LOSO-based HP optimization **did not improve** over Phase 2's standard 5-fold CV approach.

The hypothesis that optimizing for subject-level generalization would improve 80/20 test performance was not supported. The model achieved perfect baseline detection (100%) but struggled with pain intensity discrimination (low: 66.9%, high: 59.8%).

---

## Execution

```bash
python src/phase6_final_experiment.py
```

---

*Phase 6 complete. LOSO-optimized hyperparameters did not beat standard CV baselines.*
