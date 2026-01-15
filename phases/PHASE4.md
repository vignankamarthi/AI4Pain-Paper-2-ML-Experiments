# Phase 4: Experiment Redesign

**Created:** 2026-01-15
**Status:** READY FOR IMPLEMENTATION

**Decisions Finalized:** 2026-01-15

---

## Objective

Redesign the 3-class pain classification experiment based on learnings from Phases 1-3 to improve performance beyond the current 74.48% LOSO balanced accuracy.

**Target:** Exceed Paper 1 baseline (79.4%) with rigorous LOSO validation.

---

## Summary of What We Learned (Phases 1-3)

### Key Wins

1. **Per-Subject Baseline Normalization is Powerful**
   - Binary classification: 99.97% accuracy
   - 3-class no-pain detection: 100% LOSO accuracy
   - Completely eliminates inter-subject variability for pain presence

2. **Simple Models Generalize Better**
   - Simple MLP (2 layers) > Deep MLP (4 layers) on test set
   - LightGBM outperforms complex stacking on LOSO
   - Dataset size (53 subjects) limits deep learning benefits

3. **Pain Intensity is the Bottleneck**
   - No-pain: 100% accurate across all models
   - Low pain: 62-87% depending on model
   - High pain: 41-65% depending on model
   - Main errors: Low <-> High pain confusion

### Key Insights from Results Analysis

#### Per-Class Accuracy Breakdown (Phase 2)

| Model | No Pain | Low Pain | High Pain | Pattern |
|-------|---------|----------|-----------|---------|
| Simple MLP | 100% | **87.4%** | 41.7% | Biased toward Low Pain |
| Deep MLP | 100% | 66.9% | **60.6%** | More balanced |
| LightGBM | 100% | 62.2% | 64.6% | More balanced |
| Regularized MLP | 100% | 73.2% | 51.2% | Middle ground |

**Insight:** Simple MLP has highest Low Pain accuracy but worst High Pain - it may be overfitting to Low Pain patterns. Deep MLP and LightGBM are more balanced.

#### Generalization Gap Analysis (CV to Test)

| Model | CV Accuracy | Test Accuracy | Gap | Interpretation |
|-------|-------------|---------------|-----|----------------|
| Simple MLP | 78.07% | 76.38% | **1.69%** | Best generalization |
| Deep MLP | 78.13% | 75.85% | 2.28% | Moderate overfitting |
| Medium MLP | 78.00% | 75.07% | 2.93% | More overfitting |
| Regularized MLP | 77.81% | 74.80% | **3.01%** | Most overfitting despite regularization |

**Insight:** More parameters and even explicit regularization led to WORSE generalization. Simple architecture wins.

#### LOSO Subject Variability (LightGBM)

From per-subject results:
- Best subject: 89.6% balanced accuracy
- Worst subject: 61.1% balanced accuracy
- Standard deviation: 7.69%

**Insight:** High subject variability suggests some subjects have clearer pain signatures than others.

---

## Finalized Decisions

### Decision 1: Keep Deep MLP [CONFIRMED]

**Decision:** KEEP Deep MLP with modified architecture bounds

**Rationale:**
- Deep MLP shows more balanced per-class accuracy (Low 66.9%, High 60.6% - only 6.3% gap)
- Simple MLP is biased toward Low Pain (87.4% Low vs 41.7% High - 45.7% gap)
- Deep MLP's balanced performance addresses the core bottleneck (Low-High discrimination)
- Different error patterns provide ensemble diversity potential

**Architecture Adjustment for 24 Features:**
- Original bounds (32 features): Layer1 128-512, Layer2 64-256, Layers3-4 32-128
- New bounds (24 features): Layer1 64-192, Layer2 32-128, Layers3-4 16-64
- Optuna will find optimal architecture within these smaller bounds
- Rule of thumb: First hidden layer should be <= 8x input features (24 * 8 = 192 max)

---

### Decision 2: Normalization Strategy [CONFIRMED]

**Decision:** Test all three normalization strategies using HYBRID approach

**The Three Strategies:**

1. **Per-Subject Baseline Normalization (Current)**
   ```
   For each subject:
       baseline_mean = mean(subject's no-pain samples)
       baseline_std = std(subject's no-pain samples)
       normalized_feature = (feature - baseline_mean) / baseline_std
   ```

2. **Global Z-Score Normalization (Paper 1 Style)**
   ```
   For each feature across ALL subjects:
       global_mean = mean(feature across all subjects)
       global_std = std(feature across all subjects)
       normalized_feature = (feature - global_mean) / global_std
   ```

3. **Raw Features (No Normalization)**
   - Use original feature values without any transformation

**Hybrid Approach:**
- Run normalization comparison on 3 representative models (not just LightGBM)
- Models: LightGBM (tree-based), Simple MLP (neural), Stacked (ensemble)
- This catches model-specific preferences without exploding compute cost
- If all 3 agree on best normalization: use it universally
- If they disagree: use model-specific normalization

**Hypothesis:** Global z-score may work better for 3-class by preserving intensity differences, while per-subject excels at binary pain detection.

---

### Decision 3: Feature Space Optimization [CONFIRMED]

**Change 1: Remove SpO2 (PI Decision)**
- Current: 4 signals x 8 features = 32 features
- New: 3 signals x 8 features = 24 features
- Signals to keep: EDA, BVP, RESP
- Signal to remove: SpO2

**Change 2: Signal-Agnostic Backward Elimination**

**Approach:** Remove individual features based on importance, not entire feature types.

**Process:**
1. Start with 24 features (after SpO2 removal)
2. Train LightGBM, get feature importance ranking
3. Remove the LEAST important feature
4. Evaluate using 5-fold CV balanced accuracy
5. Repeat steps 2-4
6. Stop when CV accuracy drops by more than 0.5%
7. Record optimal feature subset

**Note:** If certain feature types consistently rank low across all signals (e.g., all 3 tsalliscomp features in bottom 6), this will emerge naturally and can inform interpretation.

---

### Decision 4: Stacked Ensemble Configuration [CONFIRMED]

**Decision:** Reduce internal CV from 5 to 3 folds

**Rationale:**
- Stacked ensemble uses nested cross-validation:
  - Outer loop: LOSO (53 folds, fixed)
  - Inner loop: Stacking CV (was 5 folds, now 3)
- Model fits: 53 x 5 x 3 = 795 (old) vs 53 x 3 x 3 = 477 (new)
- 40% reduction in training time
- Reduces memory pressure and I/O overhead that caused hanging in Phase 3
- 3 folds is sufficient for generating stacking predictions

---

### Decision 5: Two-Stage Classification [DEFERRED]

**Decision:** Defer to Phase 5 if Phase 4 experiments fail to exceed baseline

**Concept:**
- Stage 1: Binary classification (pain vs no-pain) - leverages 99.97% accuracy
- Stage 2: Intensity classification (Low vs High) - only on predicted pain samples

**Why Defer:**
- Adds architectural complexity
- Current 3-class direct approach is cleaner
- Two-stage becomes valuable fallback if normalization changes don't improve intensity discrimination

**Note:** This approach is documented here as a fallback strategy for Phase 5.

---

## Models for Phase 4

### Selected Models (5)

| Model | Type | Rationale |
|-------|------|-----------|
| **Simple MLP** | Neural Net | Best overall performer, good generalization |
| **Deep MLP** | Neural Net | More balanced per-class, robust architecture |
| **LightGBM** | Ensemble | Best LOSO performer, feature importance available |
| **Regularized MLP** | Neural Net | Loss function regularization may help with reduced features |
| **Stacked (RF+XGB+LGB)** | Ensemble | Ensemble diversity may capture different patterns |

### Dropped Models

| Model | Reason |
|-------|--------|
| Medium MLP | Worst generalization gap (2.93%), no unique advantage |
| XGBoost | Similar to LightGBM but worse LOSO (74.21% vs 74.48%) |
| Random Forest | Worst LOSO among ensembles (74.19%) |
| Stacked (RF+XGB) | Incomplete stacking, no advantage over full stack |

---

## Feature Understanding

### Feature Calculation Process

For each patient's raw signal data (e.g., EDA):
```
Raw EDA signal (time series)
    -> Segment into windows
    -> For each window, calculate 8 entropy features:
        1. pe (Permutation Entropy)
        2. comp (Statistical Complexity)
        3. fisher_shannon (Fisher-Shannon Information)
        4. fisher_info (Fisher Information)
        5. renyipe (Renyi Permutation Entropy)
        6. renyicomp (Renyi Complexity)
        7. tsallispe (Tsallis Permutation Entropy)
        8. tsalliscomp (Tsallis Complexity)
```

**After SpO2 removal:**
- EDA: 8 features
- BVP: 8 features
- RESP: 8 features
- **Total: 24 features**

### Feature Meaning Summary

| Feature | What It Captures | Potentially Useful For |
|---------|-----------------|------------------------|
| pe | Signal randomness/predictability | Pain disrupts normal patterns |
| comp | Balance of order vs chaos | Pain creates different complexity |
| fisher_shannon | Combined localization + disorder | Overall signal structure |
| fisher_info | Signal smoothness/gradients | Sudden changes from pain |
| renyipe | Emphasizes rare events | Unusual pain responses |
| renyicomp | Complexity via Renyi entropy | Non-standard complexity |
| tsallispe | Long-range correlations | Pain's systemic effects |
| tsalliscomp | Tsallis-based complexity | Alternative complexity view |

---

## Existing Results to Leverage

### 1. Confusion Matrix Patterns

**Location:** `results/phase1_ensembles/confusion_matrices/`, `results/phase2_neuralnets/confusion_matrices/`

**How to use:**
- Identify which class pairs are most confused
- Confirmed: Low <-> High is main error
- Models that minimize Low-High confusion should be prioritized

### 2. Training Curves

**Location:** `results/phase2_neuralnets/training_curves/`

**How to use:**
- Analyze overfitting patterns (train vs val divergence)
- Identify optimal early stopping points
- Simple MLP likely shows less divergence

### 3. Per-Subject Results

**Location:** `results/phase3_loso/per_subject_results.csv`

**How to use:**
- Identify "hard" subjects (low accuracy)
- Analyze if certain subjects benefit from specific normalization
- Could inform subject-adaptive approaches

### 4. Statistical Tests

**Location:** `results/phase3_loso/statistical_tests.csv`

**How to use:**
- All models significantly below Paper 1 (p < 0.05)
- Cohen's d ~ -0.65 to -0.78 (medium effect size)
- Need substantial improvement to approach baseline

---

## Phase 4 Experimental Design

### Experiment Structure

```
Phase 4
├── Experiment 4.1: Normalization Comparison (HYBRID)
│   ├── 3 normalization methods x 3 representative models = 9 runs
│   │   ├── Per-subject baseline
│   │   │   ├── LightGBM (tree-based)
│   │   │   ├── Simple MLP (neural)
│   │   │   └── Stacked (ensemble, 3-fold internal CV)
│   │   ├── Global z-score
│   │   │   ├── LightGBM
│   │   │   ├── Simple MLP
│   │   │   └── Stacked
│   │   └── Raw features
│   │       ├── LightGBM
│   │       ├── Simple MLP
│   │       └── Stacked
│   └── Output: Best normalization (universal or model-specific)
│
├── Experiment 4.2: Feature Selection (Backward Elimination)
│   ├── Start: 24 features (EDA, BVP, RESP x 8 entropy measures)
│   ├── Method: Signal-agnostic using LightGBM importance + 5-fold CV
│   ├── Stopping criterion: CV accuracy drops > 0.5%
│   └── Output: Optimal feature subset
│
├── Experiment 4.3: Model Training (5 models)
│   ├── Simple MLP (Optuna, 100 trials)
│   ├── Deep MLP (Optuna, 100 trials, reduced bounds: L1 64-192, L2 32-128, L3-4 16-64)
│   ├── LightGBM (Optuna, 100 trials)
│   ├── Regularized MLP (Optuna, 100 trials)
│   └── Stacked (RF+XGB+LGB, 3-fold internal CV)
│
└── Experiment 4.4: LOSO Validation
    ├── Top 2-3 performers from 4.3
    ├── 53 folds (one per subject from train+validation)
    └── Compare against 74.48% Phase 3 baseline
```

### Validation Strategy

1. **Normalization Comparison (4.1):** 5-fold CV on train+validation, compare balanced accuracy
2. **Feature Selection (4.2):** 5-fold CV on train+validation, use winning normalization
3. **Model Training (4.3):** 80/20 stratified split with Optuna optimization
4. **Final Validation (4.4):** Full LOSO on top performers

### Success Criteria

- Primary: LOSO balanced accuracy > 79.4% (Paper 1 baseline)
- Secondary: Improved Low vs High pain discrimination (reduce per-class gap)
- Tertiary: Lower subject variability (std < 6%)

---

## Resolved Questions Summary

All open questions have been resolved. See "Finalized Decisions" section above for details.

| Question | Resolution |
|----------|------------|
| Normalization strategy | Test all 3 with hybrid approach (3 models x 3 methods) |
| Feature selection method | Signal-agnostic backward elimination |
| Deep MLP sizing | Optuna with bounded search (L1: 64-192, L2: 32-128, L3-4: 16-64) |
| Stacked ensemble speed | Reduce internal CV from 5 to 3 folds |
| Two-stage classification | Deferred to Phase 5 fallback |

---

## Next Steps

1. [x] Finalize decisions on open questions (COMPLETE - 2026-01-15)
2. [ ] Create Phase 4 execution scripts
   - [ ] `src/phase4_normalization_comparison.py`
   - [ ] `src/phase4_backward_elimination.py`
   - [ ] `src/phase4_model_training.py`
   - [ ] `src/phase4_loso_validation.py`
3. [ ] Run Experiment 4.1 (normalization comparison)
4. [ ] Run Experiment 4.2 (backward elimination)
5. [ ] Run Experiment 4.3 (model training)
6. [ ] Run Experiment 4.4 (LOSO validation)
7. [ ] Generate final report

---

## Appendix: Key Results Reference

### Combined Leaderboard (Phase 1 + Phase 2)

| Rank | Model | Type | Balanced Acc | Accuracy |
|------|-------|------|--------------|----------|
| 1 | Simple MLP | neural_net | 76.38% | 82.32% |
| 2 | Deep MLP | neural_net | 75.85% | 81.93% |
| 3 | LightGBM | ensemble | 75.59% | 81.73% |
| 4 | Medium MLP | neural_net | 75.07% | 81.34% |
| 5 | Regularized MLP | neural_net | 74.80% | 81.14% |
| 6 | Stacked (RF+XGB+LGB) | ensemble | 72.97% | 79.76% |

### LOSO Results (Phase 3)

| Model | LOSO Balanced Acc | 95% CI |
|-------|-------------------|--------|
| LightGBM | 74.48% +/- 7.69% | [72.43%, 76.53%] |
| XGBoost | 74.21% +/- 7.01% | [72.35%, 76.08%] |
| RandomForest | 74.19% +/- 6.71% | [72.40%, 75.98%] |

### Best Hyperparameters

**Simple MLP:**
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

**Deep MLP:**
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

**LightGBM:**
```json
{
  "n_estimators": 100,
  "max_depth": -1,
  "learning_rate": 0.01,
  "num_leaves": 31,
  "subsample": 0.6,
  "colsample_bytree": 0.6,
  "reg_alpha": 0.5,
  "reg_lambda": 0,
  "random_state": 42
}
```

---

## Phase 5 Fallback: Two-Stage Classification

If Phase 4 experiments fail to exceed the Paper 1 baseline (79.4%), consider a two-stage hierarchical approach:

**Concept:**
```
Input Sample
    │
    ▼
┌─────────────────────────────────┐
│ Stage 1: Binary Classification  │
│ Pain vs No-Pain                 │
│ (Leverages 99.97% accuracy)     │
└─────────────────────────────────┘
    │
    ├── Predicted: No Pain → Output: "No Pain"
    │
    └── Predicted: Pain ──┐
                          ▼
              ┌─────────────────────────────────┐
              │ Stage 2: Intensity Classification│
              │ Low Pain vs High Pain           │
              │ (Simpler binary problem)        │
              └─────────────────────────────────┘
                          │
                          ├── Output: "Low Pain"
                          └── Output: "High Pain"
```

**Potential Benefits:**
- Decouples two fundamentally different problems
- Stage 1 (deviation detection) may prefer per-subject normalization
- Stage 2 (intensity discrimination) may prefer global z-score
- Error propagation minimal due to near-perfect Stage 1

**Implementation Notes:**
- Train Stage 1 with per-subject baseline normalization (proven 99.97%)
- Train Stage 2 only on pain samples (Low + High) with potentially different normalization
- Combine predictions hierarchically

**When to Trigger:**
- If best Phase 4 LOSO accuracy < 77% (no meaningful improvement)
- If Low-High confusion remains dominant error pattern despite normalization changes

---

**Document Status:** READY FOR IMPLEMENTATION

**All decisions finalized on 2026-01-15.**
