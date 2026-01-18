# Phase 6: Final Optimized Experiment

**Status:** PENDING
**Target:** Beat Paper 1 LOSO (78.0%) AND 80/20 (79.4%) with SAME data validity

---

## Validation Approach: Nested Optuna-LOSO

Phase 6 uses **Nested Optuna-LOSO** - the most rigorous validation approach available.

### Why Nested LOSO?

| Approach | HP Tuning | Bias Risk | Paper 1 Used |
|----------|-----------|-----------|--------------|
| Simple LOSO | None (defaults) | Suboptimal HPs | Yes |
| CV-then-LOSO | CV on all data | HP leakage | No |
| **Nested LOSO** | **CV per fold** | **None** | **No** |

**Advantages over Paper 1:**
- Paper 1 used simple LOSO with default hyperparameters
- We optimize hyperparameters WITHIN each fold (no leakage)
- If we beat 78.0% with nested LOSO, it's a stronger result

### Computational Cost

```
53 LOSO folds x 50 Optuna trials = 2,650 model fits
Estimated runtime: 2-4 hours (single model type)
```

Acceptable for ONE model after meta-analysis selection.

---

## Paper 1 Methodology (from source code analysis)

### Data Handling
- Pooled: train + validation = 53 subjects
- Test set: NOT used
- No-pain class: baseline only (no rest)
- Normalization: StandardScaler (global z-score)

### 80/20 Split
```python
train_test_split(X, y, test_size=0.2, stratify=y)  # stratifies by CLASS, not subject
```
- Subject leakage exists in Paper 1 too
- Same issue as our implementation

### LOSO
```python
LeaveOneGroupOut()  # groups = subject IDs from file_name
```
- Proper subject separation
- Uses DEFAULT hyperparameters (no optimization)

---

## Our Methodology Comparison

| Aspect | Paper 1 | Our Study (Phase 1-5) | Phase 6 |
|--------|---------|----------------------|---------|
| Data pool | 53 subjects | 53 subjects | 53 subjects |
| No-pain class | baseline only | baseline only | baseline only |
| Normalization | StandardScaler | Global z-score | Global z-score |
| Features | catch22 (72) | Entropy-complexity (24) | Entropy-complexity (24) |
| LOSO HP tuning | None (defaults) | None (defaults) | **Nested Optuna** |
| 80/20 subject separation | No | No | **Yes (GroupShuffleSplit)** |

**Conclusion:** Phase 6 uses stricter validation than Paper 1.

---

## Current Results vs Paper 1

| Validation | Paper 1 | Our Best (Phases 1-5) | Gap |
|------------|---------|----------------------|-----|
| LOSO | 78.0% (XGBoost) | 77.2% (RandomForest) | -0.8 pp |
| 80/20 | 79.4% (RandomForest) | 80.1% (Medium MLP) | +0.7 pp |

**Note:** MLPs were not evaluated in LOSO (Phase 3) because they did not exceed ensembles by the required 2% margin in 80/20 to justify additional LOSO evaluation. RandomForest had best LOSO generalization.

---

## Phase 6 Experiment Plan

### Step 1: Meta-Analysis
- [ ] Review all phase results (1-5)
- [ ] Identify best configurations per validation type
- [ ] Compare: ensembles vs neural nets, normalization, labeling

### Step 2: Model Selection
- [ ] Select ONE model for final optimization
- [ ] Candidates: RandomForest, XGBoost, LightGBM
- [ ] Selection criteria: Phase 3 LOSO performance

### Step 3: Nested Optuna-LOSO
- [ ] Outer loop: LOSO (53 folds)
- [ ] Inner loop: Optuna (50 trials) with 5-fold CV on training data
- [ ] No hyperparameter leakage - test subject never seen during HP tuning
- [ ] Target: > 78.0% LOSO

### Step 4: Subject-Grouped 80/20
- [ ] Use GroupShuffleSplit (subjects in train OR test, not both)
- [ ] More valid than Paper 1's 80/20 (which has subject leakage)
- [ ] Target: > 79.4% with proper subject separation

### Step 5: Final Evaluation
- [ ] Compare to Paper 1 with stricter validation
- [ ] Statistical significance testing (if applicable)
- [ ] Generate final report

---

## Key Questions to Resolve

1. **Model Selection:** RandomForest (best LOSO) vs XGBoost vs LightGBM?
2. **Feature Set:** All 24 features or subset?
3. **Optuna Trials:** 50 per fold (baseline) vs 100 (thorough)?

---

## Expected Outcomes

| Validation | Paper 1 | Target | Stretch |
|------------|---------|--------|---------|
| LOSO (nested) | 78.0% | > 78.0% | > 80.0% |
| 80/20 (grouped) | 79.4% | > 79.4% | > 82.0% |

---

## Implementation

### Nested Optuna-LOSO
```python
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
import optuna

logo = LeaveOneGroupOut()
fold_results = []

for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Inner Optuna optimization on training data only
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            # ... other hyperparameters
        }
        model = RandomForestClassifier(**params)

        # 5-fold CV on training data (test subject NOT included)
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=inner_cv, scoring='balanced_accuracy')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    # Train final model with best hyperparameters
    best_model = RandomForestClassifier(**study.best_params)
    best_model.fit(X_train, y_train)

    # Evaluate on held-out subject (never seen during HP tuning)
    y_pred = best_model.predict(X_test)
    fold_acc = balanced_accuracy_score(y_test, y_pred)
    fold_results.append(fold_acc)

final_loso_accuracy = np.mean(fold_results)
```

### Subject-Grouped 80/20
```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in gss.split(X, y, groups=subjects):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Verify no subject overlap
    train_subjects = set(subjects[train_idx])
    test_subjects = set(subjects[test_idx])
    assert len(train_subjects & test_subjects) == 0, "Subject leakage detected!"
```

---

## Files to Create

- `src/phase6_final_experiment.py` - Main execution script
- `results/phase6_final/` - Output directory
- `results/phase6_final/meta_analysis.md` - Configuration comparison
- `results/phase6_final/final_report.md` - Final results

---

*Phase 6 ready for execution after meta-analysis discussion.*
