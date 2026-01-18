# Phase 5: Hierarchical Binary Classification

**Objective:** Test whether a two-stage cascaded classifier outperforms direct 3-class classification.

**Hypothesis:** Separating "pain detection" from "pain intensity discrimination" allows each classifier to specialize, potentially improving overall accuracy.

---

## Experimental Design

### Architecture

```
                    Input Sample
                         |
                         v
              +---------------------+
              |   Stage 1 Classifier |
              |   No Pain vs Pain    |
              +---------------------+
                    /         \
                   /           \
            No Pain             Pain
            (done)               |
                                 v
                      +---------------------+
                      |   Stage 2 Classifier |
                      |   Low Pain vs High   |
                      +---------------------+
                            /         \
                           /           \
                     Low Pain       High Pain
```

### Stage 1: Pain Detection
- **Task:** Binary classification (No Pain vs Pain)
- **Classes:**
  - Class 0: No Pain (baseline + rest samples)
  - Class 1: Pain (low_pain + high_pain samples combined)
- **Expected:** High accuracy (pain vs no-pain is physiologically distinct)

### Stage 2: Pain Intensity Discrimination
- **Task:** Binary classification (Low Pain vs High Pain)
- **Classes:**
  - Class 0: Low Pain
  - Class 1: High Pain
- **Training Data:** Only pain samples (excludes no_pain entirely)
- **Expected:** This is the harder task - discriminating intensity levels

### Combined Prediction
For a test sample:
1. Stage 1 predicts: No Pain or Pain
2. If No Pain: Final prediction = No Pain
3. If Pain: Stage 2 predicts Low Pain or High Pain

### Metrics
- **Stage 1 Accuracy:** Balanced accuracy on pain detection
- **Stage 2 Accuracy:** Balanced accuracy on intensity (pain samples only)
- **Combined 3-Class Accuracy:** Final balanced accuracy across all 3 classes
- **Comparison Target:** 71.28% (Phase 4 best) and 79.4% (Paper 1 catch22)

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Validation | Leave-One-Subject-Out (LOSO) |
| Features | 24 (EDA + BVP + RESP, 8 features each) |
| Subjects | 53 |
| Hyperparameter Optimization | Optuna (50 trials per model) |

### Model Configurations (Top 3 from Phase 4)

**DEPENDENCY:** This phase uses the top 3 model configurations from Phase 4 results.

**Source:** `results/phase4/experiment_4.3_full_training/full_leaderboard.csv`

The Phase 5 script will automatically:
1. Read the Phase 4 leaderboard (sorted by LOSO accuracy)
2. Select the top 3 configurations (model + normalization)
3. Use these for both Stage 1 and Stage 2 classifiers

**Note:** If Phase 4 has not completed, Phase 5 cannot run.

---

## Hyperparameter Search Spaces

### LightGBM (Optuna)
```python
{
    'n_estimators': (50, 500),
    'max_depth': (3, 12),
    'learning_rate': (0.01, 0.3),
    'num_leaves': (15, 127),
    'min_child_samples': (5, 50),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (1e-8, 10.0),
    'reg_lambda': (1e-8, 10.0)
}
```

### SimpleMLP (Optuna)
```python
{
    'hidden_dim': (32, 256),
    'n_layers': (1, 4),
    'dropout': (0.1, 0.5),
    'learning_rate': (1e-4, 1e-2),
    'batch_size': (16, 128),
    'epochs': (50, 200)
}
```

---

## Experiment Steps

### Experiment 5.1: Stage 1 - Pain Detection (LOSO)

**Objective:** Evaluate binary pain detection under LOSO with Optuna-optimized models.

**Steps:**
1. Load all feature data (24 features)
2. Create binary labels:
   - 0 = no_pain (baseline + rest)
   - 1 = pain (low + high)
3. For each of 3 model configurations:
   a. Run nested LOSO with Optuna inner loop for hyperparameter tuning
   b. Outer loop: LOSO (53 folds)
   c. Inner loop: 5-fold CV for Optuna optimization (50 trials)
4. Record per-subject and aggregate metrics for all 3 configurations

**Output:**
- `results/phase5/stage1_pain_detection/loso_results.csv` (all 3 configs)
- `results/phase5/stage1_pain_detection/best_params.json`
- `results/phase5/stage1_pain_detection/confusion_matrices/`

**Success Criteria:** >90% balanced accuracy expected (pain detection should be relatively easy)

---

### Experiment 5.2: Stage 2 - Pain Intensity (LOSO)

**Objective:** Evaluate binary intensity discrimination using only pain samples with Optuna optimization.

**Steps:**
1. Load all feature data
2. Filter to pain samples only (exclude no_pain/baseline/rest)
3. Create binary labels:
   - 0 = low_pain
   - 1 = high_pain
4. For each of 3 model configurations:
   a. Apply appropriate normalization (global_zscore computed on pain samples only, or raw)
   b. Run nested LOSO with Optuna inner loop
   c. Record metrics
5. Compare all 3 configurations

**Note:** Class balance is 50/50 (low_pain and high_pain are equal sized)

**Output:**
- `results/phase5/stage2_intensity/loso_results.csv` (all 3 configs)
- `results/phase5/stage2_intensity/best_params.json`
- `results/phase5/stage2_intensity/confusion_matrices/`

**Success Criteria:** This is the key experiment. Even 65-70% here could yield good combined results.

---

### Experiment 5.3: Combined Hierarchical Evaluation

**Objective:** Compute end-to-end 3-class accuracy using the cascaded approach for all configurations.

**Method:**
For each model configuration (3 total):
  For each LOSO fold (held-out subject):
  1. Optimize Stage 1 hyperparameters on training subjects (Optuna, 50 trials)
  2. Optimize Stage 2 hyperparameters on training subjects' pain samples (Optuna, 50 trials)
  3. Train final Stage 1 model with best params
  4. Train final Stage 2 model with best params
  5. For each test sample:
     - Run through Stage 1
     - If predicted "Pain", run through Stage 2
     - Record final 3-class prediction
  6. Compute 3-class balanced accuracy for this fold

**Output:**
- `results/phase5/combined/hierarchical_loso_results.csv` (all 3 configs)
- `results/phase5/combined/comparison_to_direct.csv`
- `results/phase5/combined/best_configuration.json`
- `results/phase5/phase5_report.md`

---

## Analysis

### Comparison Table (Target Output)

| Approach | Model | Normalization | LOSO Balanced Accuracy |
|----------|-------|---------------|------------------------|
| Direct 3-class | (from Phase 4) | (from Phase 4) | (from Phase 4) |
| Direct 3-class | (from Phase 4) | (from Phase 4) | (from Phase 4) |
| Direct 3-class | (from Phase 4) | (from Phase 4) | (from Phase 4) |
| Hierarchical | (top 3 configs) | (top 3 configs) | ? |
| Hierarchical | (top 3 configs) | (top 3 configs) | ? |
| Hierarchical | (top 3 configs) | (top 3 configs) | ? |
| Paper 1 Baseline | SVM | catch22 | 79.4% |

**Note:** Direct 3-class results will be populated from `results/phase4/experiment_4.3_full_training/full_leaderboard.csv` when the Phase 5 report is generated.

### Error Analysis

Identify where errors occur:
1. **Stage 1 errors:** Pain samples misclassified as No Pain (or vice versa)
2. **Stage 2 errors:** Low/High confusion (expected to be primary error source)
3. **Error propagation:** Stage 1 errors cascade to final prediction
4. **Configuration comparison:** Which model/normalization combination minimizes total error

---

## Output Structure

```
results/phase5/
├── stage1_pain_detection/
│   ├── loso_results.csv           (all 3 configs)
│   ├── per_subject_results.csv
│   ├── best_params.json
│   └── confusion_matrices/
│       ├── lightgbm_global_zscore.csv
│       ├── lightgbm_raw.csv
│       └── simplemlp_global_zscore.csv
├── stage2_intensity/
│   ├── loso_results.csv           (all 3 configs)
│   ├── per_subject_results.csv
│   ├── best_params.json
│   └── confusion_matrices/
│       ├── lightgbm_global_zscore.csv
│       ├── lightgbm_raw.csv
│       └── simplemlp_global_zscore.csv
├── combined/
│   ├── hierarchical_loso_results.csv  (all 3 configs)
│   ├── comparison_to_direct.csv
│   ├── best_configuration.json
│   └── error_analysis.csv
└── phase5_report.md
```

---

## Success Criteria

**Phase 5 succeeds if:**
- Combined hierarchical accuracy > 71.28% (beats Phase 4 direct approach)

**Bonus success:**
- Combined accuracy approaches or exceeds 79.4% (catches up to Paper 1)

**Phase 5 fails if:**
- Hierarchical approach performs worse than direct 3-class
- Error propagation from Stage 1 negates Stage 2 improvements

---

## Execution

Run via: `python src/phase5_hierarchical.py`

Script will execute all three experiments sequentially and generate the final report.

---

*Phase 5 designed to test hierarchical classification as final optimization attempt.*
