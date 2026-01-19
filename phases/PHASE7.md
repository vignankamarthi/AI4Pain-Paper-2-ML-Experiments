# Phase 7: Nested Optuna-LOSO to Beat Paper 1

**Status:** READY_TO_RUN
**Target:** Beat Paper 1's LOSO baseline of 78.0% (XGBoost with default params)

---

## Objective

Execute a rigorous nested cross-validation experiment where hyperparameters are optimized per LOSO fold using inner LOSO validation. This is the methodologically correct approach for subject-independent evaluation.

---

## Why This Should Work

Paper 1 achieved 78.0% LOSO using **default hyperparameters** on XGBoost. By:
1. Using proper nested Optuna optimization
2. Tuning hyperparameters for subject generalization (not in-sample fit)
3. Using all 65 subjects (matching Paper 1)

We have a legitimate path to beat the baseline.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Data | All 65 subjects (train=41, val=12, test=12 pooled) |
| Labels | 3-class (baseline=0, low=1, high=2) |
| Features | 24 entropy-complexity features (8 per signal x 3 signals) |
| Normalization | Global z-score (fit on train, transform test per fold) |
| Model | RandomForest |
| Outer CV | LOSO (65 folds) |
| Inner CV | LOSO (64 folds per Optuna trial) |
| Optuna trials | 50 per outer fold |
| Metric | Balanced accuracy |

---

## Methodology: Nested Optuna-LOSO

```
For each outer fold s in [1..65]:
    1. Hold out subject s as test set
    2. Train pool = remaining 64 subjects

    3. Run Optuna (50 trials):
        For each trial:
            a. Suggest hyperparameters
            b. Inner LOSO on 64 subjects:
                For each inner fold i in [1..64]:
                    - Hold out subject i as inner test
                    - Train on 63 subjects
                    - Evaluate on subject i
                Inner score = mean(64 fold scores)
            c. Return inner score to Optuna

    4. Get best hyperparameters from Optuna
    5. Train final model on all 64 subjects with best params
    6. Evaluate on held-out subject s
    7. Save fold result and checkpoint

Final LOSO accuracy = mean(65 fold accuracies)
```

---

## Hyperparameter Search Space (RandomForest)

| Parameter | Range | Type |
|-----------|-------|------|
| n_estimators | [50, 500] | int |
| max_depth | [3, 30] or None | int/None |
| min_samples_split | [2, 20] | int |
| min_samples_leaf | [1, 10] | int |
| max_features | ['sqrt', 'log2', None] | categorical |
| class_weight | ['balanced', 'balanced_subsample', None] | categorical |
| criterion | ['gini', 'entropy'] | categorical |

---

## Computational Estimate

- Outer folds: 65
- Optuna trials per fold: 50
- Inner LOSO folds per trial: 64
- **Total model fits:** 65 x 50 x 64 = 208,000
- Estimated time per RF fit: 0.1-0.5 seconds
- **Total estimated runtime:** 10-20 hours

---

## Checkpointing Strategy

The experiment saves state after each outer fold:
- `checkpoint.json`: Current progress, completed folds
- `fold_results/`: Individual fold results with hyperparameters
- Resume capability: If interrupted, restarts from last completed fold

---

## Output Files

```
results/phase7_nested_loso/
    phase7_report.md              # Summary report
    loso_leaderboard.csv          # Final results
    per_subject_results.csv       # Per-fold accuracies
    best_hyperparameters.json     # Best params per fold
    confusion_matrix.png          # Aggregated confusion matrix
    checkpoint.json               # Resume checkpoint
    fold_results/                 # Individual fold data
```

---

## Success Criteria

| Metric | Target | Paper 1 |
|--------|--------|---------|
| LOSO Balanced Accuracy | > 78.0% | 78.0% (XGBoost default) |

---

## Constraints

- DO NOT use per-subject normalization (causes leakage)
- DO NOT include rest segments in baseline class
- DO use global z-score normalization
- DO checkpoint after each outer fold
- DO use balanced accuracy as primary metric

---

## Execution

```bash
python3 src/phase7_nested_loso.py
```

Or to resume from checkpoint:
```bash
python3 src/phase7_nested_loso.py --resume
```
