# Phase 1: Ensemble Exploration (80/20 Split)

## Objective
Rapid exploration of ensemble machine learning methods using silhouette-filtered features. Establish baseline ML performance and determine if neural nets (Phase 2) are needed.

---

## Input Data

### Extracted Features (Primary Source)
- **Location:** `data/features/results_{split}_{signal}.csv`
- **Feature subset:** Determined from Phase 0 (use only combos meeting silhouette threshold)
- **Signals:** EDA, BVP, RESP (SpO2 excluded based on Phase 0 results)
- **Features per signal** (8 measures):
  - `pe` - Permutation Entropy (H)
  - `comp` - Statistical Complexity (C)
  - `fisher_shannon` - Fisher-Shannon Information
  - `fisher_info` - Fisher Information
  - `renyipe` - Renyi Permutation Entropy
  - `renyicomp` - Renyi Complexity
  - `tsallispe` - Tsallis Permutation Entropy
  - `tsalliscomp` - Tsallis Complexity
- **Labels** (embedded in feature files):
  - `state` - 3-class label (baseline=0, low_pain=1, high_pain=2)
- **Metadata:** `dimension`, `tau` for filtering based on Phase 0 results

---

## Experiment Configuration

### Feature Selection
Use features from (d, tau, signal) combinations that met the Phase 0 silhouette threshold:
- Example: If "top 40%" threshold approved, use features from top 24 combos
- Each combo contributes 8 measures: PE, C, Fisher-Shannon, Fisher, Renyi PE/C, Tsallis PE/C
- Total features: N_combos × 8 measures

### Train/Test Split
- **Training:** 80% of combined train+validation data
- **Testing:** 20% of combined train+validation data
- **Stratified split:** Maintain class distribution
- **Random seed:** 42 (for reproducibility)

### Models to Evaluate
1. **Random Forest** (individual)
2. **XGBoost** (individual)
3. **LightGBM** (individual)
4. **Stacked Ensemble 1:** RF + XGBoost (meta-learner: Logistic Regression)
5. **Stacked Ensemble 2:** RF + XGBoost + LightGBM (meta-learner: Logistic Regression)

### Hyperparameter Optimization
Use Optuna (Bayesian optimization) for each model:
- **Trials per model:** 50
- **Optimization metric:** Balanced accuracy
- **Validation:** 5-fold cross-validation on training set
- **Timeout:** 2 hours per model (safety limit)

---

## Hyperparameter Search Spaces

### Random Forest
- `n_estimators`: [100, 200, 300, 500]
- `max_depth`: [10, 20, 30, 40, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `max_features`: ['sqrt', 'log2', None]
- `class_weight`: ['balanced', 'balanced_subsample', None]

### XGBoost
- `n_estimators`: [100, 200, 300, 500]
- `max_depth`: [3, 5, 7, 9]
- `learning_rate`: [0.01, 0.05, 0.1, 0.2]
- `subsample`: [0.6, 0.8, 1.0]
- `colsample_bytree`: [0.6, 0.8, 1.0]
- `gamma`: [0, 0.1, 0.2, 0.5]
- `reg_alpha`: [0, 0.1, 0.5, 1.0]
- `reg_lambda`: [0, 0.1, 0.5, 1.0]

### LightGBM
- `n_estimators`: [100, 200, 300, 500]
- `max_depth`: [3, 5, 7, 9, -1]
- `learning_rate`: [0.01, 0.05, 0.1, 0.2]
- `num_leaves`: [15, 31, 63, 127]
- `subsample`: [0.6, 0.8, 1.0]
- `colsample_bytree`: [0.6, 0.8, 1.0]
- `reg_alpha`: [0, 0.1, 0.5, 1.0]
- `reg_lambda`: [0, 0.1, 0.5, 1.0]

### Stacking Meta-Learner
- **Logistic Regression:** Default parameters (C=1.0, max_iter=1000)

---

## Execution Steps

### Step 1: Feature Loading and Preprocessing
1. Load entropy/complexity features for all signals
2. Filter features based on Phase 0 approved threshold
3. Combine train and validation sets
4. Create 80/20 stratified split
5. Standardize features using StandardScaler (fit on train, transform train and test)

### Step 2: Individual Model Training
For each model (RF, XGBoost, LightGBM):
1. Define Optuna objective function (5-fold CV balanced accuracy)
2. Run 50 Optuna trials
3. Select best hyperparameters
4. Train final model on full training set with best hyperparameters
5. Predict on test set
6. Compute metrics: accuracy, balanced accuracy, F1 (weighted), per-class accuracy
7. Generate confusion matrix
8. Save model to disk

### Step 3: Stacked Ensemble Training
For each stacking configuration:
1. Use trained individual models as base estimators
2. Train meta-learner (Logistic Regression) on training set predictions
3. Predict on test set using stacking pipeline
4. Compute metrics: accuracy, balanced accuracy, F1 (weighted), per-class accuracy
5. Generate confusion matrix
6. Save stacked model to disk

### Step 4: Feature Importance Analysis
For tree-based models (RF, XGBoost, LightGBM):
1. Extract feature importance scores
2. Map back to (d, tau, signal, measure) combinations
3. Identify top 20 most important features
4. Generate feature importance plots (bar charts)

### Step 5: Leaderboard Generation
Create ranked leaderboard of all models:
- Sort by balanced accuracy (descending)
- Include: model name, accuracy, balanced accuracy, F1, per-class accuracy
- Highlight best model
- Compare to Paper 1 baseline (79.4%)

### Step 6: Report Generation
Create `phase1_report.md` containing:
- Executive summary with best model performance
- Complete leaderboard table
- Comparison to Paper 1 baseline
- Confusion matrices for all models
- Feature importance analysis for top model
- Hyperparameter settings for all models
- Training time statistics
- Recommendation: proceed to LOSO or trigger Phase 2

---

## Output Requirements

### Files to Generate
1. **`results/phase1_ensembles/leaderboard.csv`**
   - Columns: rank, model, accuracy, balanced_accuracy, f1_weighted, acc_baseline, acc_low, acc_high, training_time_sec
   - 5 rows (one per model)
   - Sorted by balanced_accuracy descending

2. **`results/phase1_ensembles/hyperparameters.json`**
   - JSON file with best hyperparameters for each model
   - Include optimization history (Optuna study object serialized)

3. **`results/phase1_ensembles/feature_importance.csv`**
   - Columns: rank, feature_name, signal, dimension, tau, measure, importance_score, model
   - Top 20 features for each tree-based model

4. **`results/phase1_ensembles/confusion_matrices/`**
   - `rf_confusion_matrix.png`
   - `xgboost_confusion_matrix.png`
   - `lightgbm_confusion_matrix.png`
   - `stacked_ensemble1_confusion_matrix.png`
   - `stacked_ensemble2_confusion_matrix.png`

5. **`results/phase1_ensembles/feature_importance_plots/`**
   - `rf_feature_importance.png`
   - `xgboost_feature_importance.png`
   - `lightgbm_feature_importance.png`

6. **`results/phase1_ensembles/models/`**
   - `rf_best.pkl`
   - `xgboost_best.pkl`
   - `lightgbm_best.pkl`
   - `stacked_ensemble1.pkl`
   - `stacked_ensemble2.pkl`

7. **`results/phase1_ensembles/phase1_report.md`**
   - Executive summary
   - Leaderboard table
   - Comparison to Paper 1 (79.4% baseline)
   - Model-by-model performance analysis
   - Confusion matrix interpretation
   - Feature importance insights
   - Training efficiency statistics
   - Decision recommendation: proceed to LOSO or trigger Phase 2

---

## Checkpoint Instructions

After Phase 1 completes, this is a DECISION CHECKPOINT. Human review required to:
1. Review leaderboard results
2. Compare best model to Paper 1 baseline (79.4%)
3. Decide on next action:
   - **If best balanced accuracy greater than or equal to 85%:** Proceed directly to Phase 3 (LOSO)
   - **If best balanced accuracy less than 85%:** Trigger Phase 2 (Neural Nets)
   - **If best balanced accuracy less than 82%:** Consider feature expansion or revisit Phase 0 threshold

**Human must update STATUS.md with:**
- Best model name and balanced accuracy
- Decision: proceed to Phase 3 or trigger Phase 2
- Phase 1 status changed to "APPROVED"

---

## Completion Criteria

- [ ] All 5 models trained and evaluated
- [ ] Optuna optimization complete (50 trials each)
- [ ] Leaderboard CSV generated
- [ ] All confusion matrices generated
- [ ] Feature importance analysis complete
- [ ] All models saved to disk
- [ ] Report markdown file generated with decision recommendation
- [ ] STATUS.md updated to "PHASE1_COMPLETE, AWAITING_APPROVAL"
- [ ] PLAN.md checklist updated (Phase 1 marked complete)

---

## Expected Runtime
Approximately 4-6 hours on M2 Pro MacBook (50 trials × 5 models with 5-fold CV).

---

## Success Indicators
- Best model should exceed Paper 1 baseline (79.4%)
- Stacked ensembles should outperform individual models
- Confusion matrix should show better discrimination than random guessing
- Feature importance should align with Phase 0 silhouette rankings
- If best model greater than or equal to 85%, Phase 2 may not be needed