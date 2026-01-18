# Phase 2: Neural Net Exploration (If Needed)

## Objective
Explore deep learning architectures to capture complex feature interactions that ensemble methods may miss. Only triggered if Phase 1 best ensemble achieves less than 85% balanced accuracy.

---

## Trigger Condition
This phase is executed ONLY if Phase 1 results show:
- Best ensemble balanced accuracy less than 85%
- Human approval to proceed with neural net exploration

---

## Input Data

### Extracted Features (Primary Source)
- **Location:** `data/features/results_{split}_{signal}.csv`
- **Feature subset:** Same filtered feature set from Phase 1
- **Train/Test Split:** Same 80/20 split from Phase 1 (for direct comparison)
- **Signals:** EDA, BVP, RESP (SpO2 excluded)
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

---

## Experiment Configuration

### Neural Network Architectures
Test multiple MLP (Multi-Layer Perceptron) configurations:

1. **Simple MLP:** 2 hidden layers
2. **Medium MLP:** 3 hidden layers
3. **Deep MLP:** 4 hidden layers
4. **Regularized MLP:** 3 hidden layers with high dropout

### Framework
- **Library:** PyTorch or TensorFlow/Keras
- **Loss:** Categorical cross-entropy
- **Optimizer:** Adam
- **Callbacks:** Early stopping (patience=20), learning rate reduction on plateau

### Hyperparameter Optimization
Use Optuna (Bayesian optimization) for each architecture:
- **Trials per architecture:** 100
- **Optimization metric:** Balanced accuracy
- **Validation:** 5-fold cross-validation on training set
- **Timeout:** 4 hours per architecture (safety limit)

---

## Hyperparameter Search Spaces

### Simple MLP (2 layers)
- `layer1_size`: [64, 128, 256, 512]
- `layer2_size`: [32, 64, 128, 256]
- `dropout_rate`: [0.1, 0.2, 0.3, 0.4]
- `learning_rate`: [0.0001, 0.0005, 0.001, 0.005]
- `batch_size`: [16, 32, 64]
- `activation`: ['relu', 'elu', 'leaky_relu']

### Medium MLP (3 layers)
- `layer1_size`: [128, 256, 512]
- `layer2_size`: [64, 128, 256]
- `layer3_size`: [32, 64, 128]
- `dropout_rate`: [0.2, 0.3, 0.4, 0.5]
- `learning_rate`: [0.0001, 0.0005, 0.001]
- `batch_size`: [16, 32, 64]
- `activation`: ['relu', 'elu', 'leaky_relu']

### Deep MLP (4 layers)
- `layer1_size`: [256, 512]
- `layer2_size`: [128, 256]
- `layer3_size`: [64, 128]
- `layer4_size`: [32, 64]
- `dropout_rate`: [0.3, 0.4, 0.5]
- `learning_rate`: [0.0001, 0.0005, 0.001]
- `batch_size`: [16, 32]
- `activation`: ['relu', 'elu']

### Regularized MLP (3 layers, high dropout)
- `layer1_size`: [256, 512]
- `layer2_size`: [128, 256]
- `layer3_size`: [64, 128]
- `dropout_rate`: [0.4, 0.5, 0.6]
- `learning_rate`: [0.0001, 0.0005]
- `batch_size`: [16, 32]
- `activation`: ['relu', 'elu']
- `l2_regularization`: [0.001, 0.01, 0.1]

### Training Configuration (All Architectures)
- `epochs`: Maximum 500 (early stopping will terminate earlier)
- `early_stopping_patience`: 20 epochs
- `lr_reduction_patience`: 10 epochs
- `lr_reduction_factor`: 0.5
- `class_weights`: Compute from training set to handle imbalance

---

## Execution Steps

### Step 1: Feature Loading and Preprocessing
1. Load same filtered features from Phase 1
2. Use same 80/20 split from Phase 1 (for direct comparison)
3. Standardize features using StandardScaler (fit on train, transform train and test)
4. Convert to PyTorch tensors or TensorFlow format

### Step 2: Class Weight Computation
Calculate class weights to handle imbalance:
- `class_weight = n_samples / (n_classes × n_samples_per_class)`
- Use in loss function during training

### Step 3: Architecture Training
For each architecture (Simple, Medium, Deep, Regularized):
1. Define Optuna objective function:
   - Build model with trial-suggested hyperparameters
   - Train with 5-fold CV on training set
   - Return balanced accuracy
2. Run 100 Optuna trials
3. Select best hyperparameters
4. Train final model on full training set with best hyperparameters
5. Predict on test set
6. Compute metrics: accuracy, balanced accuracy, F1 (weighted), per-class accuracy
7. Generate confusion matrix
8. Save model to disk

### Step 4: Comparison to Ensembles
Compare neural net results to Phase 1 ensemble results:
1. Load Phase 1 leaderboard
2. Add neural net results to comparison table
3. Identify if any neural net beats best ensemble by greater than or equal to 2%
4. Determine overall best model (ensemble or neural net)

### Step 5: Training Curve Analysis
For best neural net model:
1. Plot training vs. validation loss curves
2. Plot training vs. validation accuracy curves
3. Identify if overfitting occurred despite regularization
4. Analyze convergence behavior

### Step 6: Report Generation
Create `phase2_report.md` containing:
- Executive summary with best neural net performance
- Complete neural net leaderboard table
- Comparison to Phase 1 ensemble results
- Confusion matrices for all neural net architectures
- Training curve plots for best model
- Hyperparameter settings for all architectures
- Training time statistics
- Recommendation: which models to include in Phase 3 LOSO

---

## Output Requirements

### Files to Generate
1. **`results/phase2_neuralnets/leaderboard.csv`**
   - Columns: rank, architecture, accuracy, balanced_accuracy, f1_weighted, acc_baseline, acc_low, acc_high, training_time_sec
   - 4 rows (one per architecture)
   - Sorted by balanced_accuracy descending

2. **`results/phase2_neuralnets/combined_leaderboard.csv`**
   - Merged Phase 1 and Phase 2 results
   - All models ranked by balanced accuracy
   - Column indicating model_type (ensemble or neural_net)

3. **`results/phase2_neuralnets/hyperparameters.json`**
   - JSON file with best hyperparameters for each architecture
   - Include optimization history (Optuna study object serialized)

4. **`results/phase2_neuralnets/confusion_matrices/`**
   - `simple_mlp_confusion_matrix.png`
   - `medium_mlp_confusion_matrix.png`
   - `deep_mlp_confusion_matrix.png`
   - `regularized_mlp_confusion_matrix.png`

5. **`results/phase2_neuralnets/training_curves/`**
   - `best_model_loss_curve.png` (train vs. validation)
   - `best_model_accuracy_curve.png` (train vs. validation)

6. **`results/phase2_neuralnets/models/`**
   - `simple_mlp_best.pth` (or .h5 for TensorFlow)
   - `medium_mlp_best.pth`
   - `deep_mlp_best.pth`
   - `regularized_mlp_best.pth`

7. **`results/phase2_neuralnets/phase2_report.md`**
   - Executive summary
   - Neural net leaderboard table
   - Combined leaderboard (Phase 1 + Phase 2)
   - Best overall model identification
   - Architecture-by-architecture performance analysis
   - Confusion matrix interpretation
   - Training curve analysis (overfitting assessment)
   - Training efficiency statistics
   - Top 5 models recommendation for Phase 3 LOSO

---

## Checkpoint Instructions

After Phase 2 completes, this is a SELECTION CHECKPOINT. Human review required to:
1. Review combined leaderboard (Phase 1 + Phase 2)
2. Select top 5 models for Phase 3 LOSO validation
3. Confirm model selection balances performance and diversity (mix of ensemble and neural net if appropriate)

**Human must update STATUS.md with:**
- Best neural net name and balanced accuracy
- Top 5 models selected for Phase 3 (by rank or name)
- Phase 2 status changed to "APPROVED"

---

## Completion Criteria

- [ ] All 4 neural net architectures trained and evaluated
- [ ] Optuna optimization complete (100 trials each)
- [ ] Neural net leaderboard CSV generated
- [ ] Combined leaderboard (Phase 1 + Phase 2) generated
- [ ] All confusion matrices generated
- [ ] Training curves plotted for best model
- [ ] All models saved to disk
- [ ] Report markdown file generated with top 5 recommendation
- [ ] STATUS.md updated to "PHASE2_COMPLETE, AWAITING_APPROVAL"
- [ ] PLAN.md checklist updated (Phase 2 marked complete)

---

## Expected Runtime
Approximately 6-8 hours on M2 Pro MacBook (100 trials × 4 architectures with 5-fold CV).

---

## Success Indicators
- At least one neural net should exceed best Phase 1 ensemble
- Best neural net should show minimal overfitting (train/val curves close)
- Confusion matrix should show improvement over Phase 1
- Combined top 5 should include mix of ensembles and neural nets
- Training curves should show convergence (not stuck in poor local minimum)