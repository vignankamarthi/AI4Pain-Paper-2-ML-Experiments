# Phase 3: LOSO Validation

## Objective
Rigorous Leave-One-Subject-Out (LOSO) cross-validation on top 5 models selected from Phase 1 and/or Phase 2. Establish final generalization performance and generate comprehensive final report.

---

## Input Data

### Extracted Features (Primary Source)
- **Location:** `data/features/results_{split}_{signal}.csv`
- **Feature subset:** Per-subject baseline normalized features from Phase 1/2
- **Full dataset:** All subjects from train, validation, and test sets combined (65 subjects)
- **Signals:** EDA, BVP, RESP, SpO2 (all 4 signals)
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
- **Top 5 models:** Selected after Phase 1 (or Phase 2 if triggered)

---

## Experiment Configuration

### LOSO Cross-Validation
- **Method:** Leave-One-Subject-Out
- **Folds:** 41 (one per training subject in original Paper 1 split)
- **Process:** For each fold:
  - Train on 40 subjects
  - Test on 1 held-out subject
  - Compute metrics for that subject
- **Aggregation:** Average metrics across all 41 folds

### Models to Validate
Top 5 models selected from combined Phase 1 + Phase 2 leaderboard:
- Could be 5 ensembles (if Phase 2 not triggered)
- Could be mix of ensembles and neural nets (if Phase 2 triggered)
- Use exact same hyperparameters from Phase 1/2 optimization

---

## Execution Steps

### Step 1: Data Preparation
1. Load full dataset (train + validation + test combined)
2. Apply same feature filtering from Phase 1/2
3. Extract subject IDs from file names
4. Verify 41 unique subjects available for LOSO
5. Standardize features (fit scaler on each fold's training set separately)

### Step 2: Model Loading
1. Load top 5 model configurations from Phase 1/2
2. Extract hyperparameters from saved JSON files
3. Load model architectures (ensemble pipelines or neural net definitions)

### Step 3: LOSO Execution
For each of top 5 models:
1. Initialize result storage (41 folds × metrics)
2. For each subject (fold):
   - Split data: train on other 40 subjects, test on this subject
   - Fit StandardScaler on training fold
   - Transform both training and test fold
   - Train model from scratch with saved hyperparameters
   - Predict on held-out subject
   - Compute fold metrics: accuracy, balanced accuracy, F1, per-class accuracy
   - Store predictions and true labels
3. Aggregate across all 41 folds:
   - Mean and std for each metric
   - Overall confusion matrix (concatenate all predictions)
   - Per-subject performance distribution

### Step 4: Statistical Analysis
For each model:
1. Compute 95% confidence intervals for balanced accuracy
2. Perform paired t-test comparing to Paper 1 baseline (79.4%)
3. Compute effect size (Cohen's d)
4. Identify subjects with consistently poor performance (potential outliers)

### Step 5: Complexity-Entropy Plane Visualization
For best LOSO model:
1. Generate C-H plane plots for each signal (EDA, BVP, RESP, SpO2)
2. Color points by true class (baseline, low, high)
3. Overlay predicted class as marker shape or border
4. Highlight misclassified points
5. Include classification accuracy in titles

### Step 6: Final Model Selection
1. Rank models by LOSO balanced accuracy
2. Identify best model
3. Compare to Paper 1 baseline (79.4%)
4. Calculate improvement percentage
5. Save best model configuration

### Step 7: Comprehensive Final Report
Generate `final_report.md` with complete experimental narrative:
- Research objective recap
- Stage 0: Binary classification and normalization discovery
- Phase 1: Ensemble exploration results
- Phase 2: Neural net results (if triggered)
- Phase 3: LOSO validation results
- Statistical comparison to Paper 1
- Confusion matrix analysis
- C-H plane interpretation
- Success assessment (met 85% or 90% goal?)
- Limitations and future work
- Clinical implications

---

## Output Requirements

### Files to Generate
1. **`results/phase3_loso/loso_leaderboard.csv`**
   - Columns: rank, model, loso_accuracy_mean, loso_accuracy_std, loso_balanced_accuracy_mean, loso_balanced_accuracy_std, loso_f1_mean, loso_f1_std, ci_95_lower, ci_95_upper
   - 5 rows (one per model)
   - Sorted by loso_balanced_accuracy_mean descending

2. **`results/phase3_loso/per_subject_results.csv`**
   - Columns: model, subject_id, accuracy, balanced_accuracy, f1, n_samples
   - 205 rows (41 subjects × 5 models)
   - Shows per-subject performance variability

3. **`results/phase3_loso/statistical_tests.csv`**
   - Columns: model, loso_balanced_acc, paper1_baseline, improvement_pct, t_statistic, p_value, cohens_d, significant
   - 5 rows (one per model)
   - Tests against Paper 1 baseline (79.4%)

4. **`results/phase3_loso/confusion_matrices/`**
   - One confusion matrix per model (aggregated across all 41 folds)
   - `model1_loso_confusion_matrix.png`
   - `model2_loso_confusion_matrix.png`
   - ... (5 total)

5. **`results/phase3_loso/ch_plane_visualizations/`**
   - For best LOSO model only:
   - `best_model_eda_ch_plane.png` (true vs predicted classes)
   - `best_model_bvp_ch_plane.png`
   - `best_model_resp_ch_plane.png`

6. **`results/phase3_loso/best_models/`**
   - `best_loso_model.pkl` (or .pth for neural net)
   - `best_loso_model_config.json` (full configuration for reproducibility)
   - `best_loso_model_scaler.pkl` (fitted StandardScaler)

7. **`results/phase3_loso/final_report.md`**
   - **Executive Summary**
     - Research objective
     - Key findings
     - Best model and performance
     - Comparison to Paper 1 baseline
     - Success assessment
   - **Stage 0: Binary Classification Analysis**
     - Per-subject baseline normalization methodology
     - Normalization comparison results
     - Key insights (99.97% accuracy discovery)
   - **Phase 1: Ensemble Exploration**
     - Models tested
     - 80/20 performance
     - Best ensemble results
     - Feature importance insights
   - **Phase 2: Neural Net Exploration** (if triggered)
     - Architectures tested
     - Performance comparison to ensembles
     - Training dynamics
   - **Phase 3: LOSO Validation**
     - Complete leaderboard
     - Best model: architecture and hyperparameters
     - Statistical comparison to Paper 1
     - Confusion matrix analysis
     - Per-class performance breakdown
     - Subject variability analysis
   - **Complexity-Entropy Plane Analysis**
     - C-H plane visualizations for best model
     - Interpretation of misclassifications
     - Signal-specific insights
   - **Discussion**
     - Success criteria assessment (85% or 90% achieved?)
     - Improvement over Paper 1 (from 79.4% to X%)
     - Feature engineering impact (entropy/complexity vs catch22)
     - Clinical implications
     - Model interpretability
   - **Limitations**
     - Dataset size (65 subjects)
     - Controlled experimental setting (TENS stimulation)
     - Generalization to real-world pain scenarios
   - **Future Work**
     - Temporal dynamics (time-series modeling)
     - Additional entropy measures
     - Transfer learning to clinical datasets
     - Real-time implementation
   - **Conclusion**
     - Summary of contributions
     - Publication readiness statement
   - **Reproducibility**
     - Full configuration files
     - Random seeds
     - Software versions

---

## Completion Criteria

- [ ] All 5 models validated with LOSO (41 folds each)
- [ ] LOSO leaderboard CSV generated
- [ ] Per-subject results CSV generated
- [ ] Statistical tests completed (comparison to Paper 1)
- [ ] All confusion matrices generated
- [ ] C-H plane visualizations generated for best model
- [ ] Best model and configuration saved
- [ ] Comprehensive final report generated
- [ ] STATUS.md updated to "PHASE3_COMPLETE, EXPERIMENTS_FINISHED"
- [ ] PLAN.md all checkboxes marked complete

---

## Expected Runtime
Approximately 4-5 hours on M2 Pro MacBook (41 folds × 5 models, ~5 minutes per fold).

---

## Success Indicators
- Best LOSO balanced accuracy greater than or equal to 85% (minimum success)
- Best LOSO balanced accuracy greater than or equal to 90% (stretch goal)
- Statistically significant improvement over Paper 1 baseline (p less than 0.05)
- Confusion matrix shows balanced performance across all 3 classes
- C-H plane visualizations show clear clustering aligned with predictions
- Small std across folds (consistent performance across subjects)
- Effect size (Cohen's d) greater than 0.5 (medium to large improvement)

---

## Final Deliverable
After Phase 3 completion, the repository contains:
- Complete experimental documentation (all markdown files)
- Full results hierarchy (Stage 0 through Phase 3)
- Best model ready for deployment
- Publication-ready final report
- Reproducibility package (configs, seeds, hyperparameters)

**This marks completion of the ML experiment loop. Results ready for Paper 2 submission.**