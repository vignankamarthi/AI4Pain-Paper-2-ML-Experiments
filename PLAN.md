# AI4Pain Paper 2 - Experimental Plan

## Research Objective
Improve 3-class pain classification accuracy beyond Paper 1's **79.4% balanced accuracy** baseline using entropy and complexity-based features extracted from multimodal physiological signals.

**Target:** Greater than or equal to 90% balanced accuracy (stretch goal)
**Minimum Success:** Greater than or equal to 85% balanced accuracy with robust LOSO validation

**Final Result:** 74.48% LOSO balanced accuracy (below baseline)

---

## Key Discovery: Per-Subject Baseline Normalization

From Stage 0, we discovered that **per-subject baseline normalization** is critical:
- Normalize each subject's entropy features using ONLY their no-pain samples as reference
- This accounts for individual physiological differences
- Binary classification: 99.97% accuracy (vs 87% with global normalization)
- 3-class no-pain detection: 100% accuracy on LOSO

This normalization strategy was applied to all subsequent phases.

---

## Experimental Waterfall

### Stage 0: Binary Classification (COMPLETE)
**Status:** COMPLETE

**Goal:** Establish binary pain classification baseline on C-H plane.

**Key Finding:** Per-subject baseline normalization achieves **99.97% linear accuracy** for binary classification (pain vs no-pain).

**Best Parameters:** EDA, d=7, tau=2

**Results:**
| Normalization | Accuracy |
|---------------|----------|
| Raw | 87.97% |
| Global StandardScaler | 87.45% |
| **Per-subject baseline** | **99.97%** |

---

### Phase 1: 3-Class Classification (80/20 Split) (COMPLETE)
**Status:** COMPLETE

**Goal:** Train and evaluate classifiers for 3-class pain classification (baseline, low_pain, high_pain) using per-subject baseline normalization.

**Features:** All 8 entropy measures normalized per-subject (x4 signals = 32 features)
- pe, comp, fisher_shannon, fisher_info, renyipe, renyicomp, tsallispe, tsalliscomp

**Results:**
| Model | Balanced Accuracy |
|-------|-------------------|
| LightGBM | 75.59% |
| Stacked (RF+XGB+LGB) | 72.97% |
| XGBoost | 72.70% |
| Random Forest | 72.70% |

**Success Criteria:** Best model achieves >= 85% balanced accuracy
**Outcome:** NOT MET (75.59% < 85%) - Triggered Phase 2

---

### Phase 2: Neural Net Exploration (COMPLETE)
**Status:** COMPLETE

**Trigger:** Phase 1 best model < 85% balanced accuracy

**Goal:** Explore deep learning architectures to capture complex feature interactions.

**Models:** 4 MLP architectures with Optuna optimization (100 trials each)

**Results:**
| Architecture | CV Acc | Test Acc | Generalization Gap |
|--------------|--------|----------|-------------------|
| Simple MLP | 78.07% | 76.38% | 1.69% |
| Deep MLP | 78.13% | 75.85% | 2.28% |
| Medium MLP | 78.00% | 75.07% | 2.93% |
| Regularized MLP | 77.81% | 74.80% | 3.01% |

**Success Criteria:** Neural net beats best ensemble by >= 2% balanced accuracy
**Outcome:** NOT MET - Simple MLP (76.38%) vs LightGBM (75.59%) = +0.79%

**Key Finding:** Simpler architectures generalize better; deeper networks overfit.

---

### Phase 3: LOSO Validation (COMPLETE)
**Status:** COMPLETE

**Goal:** Rigorous cross-validation on top models using Leave-One-Subject-Out.

**Validation:** 53 folds (one per subject from train+validation sets)

**Results:**
| Model | LOSO Balanced Acc | 95% CI |
|-------|-------------------|--------|
| LightGBM | 74.48% +/- 7.69% | [72.43%, 76.53%] |
| XGBoost | 74.21% +/- 7.01% | [72.35%, 76.08%] |
| RandomForest | 74.19% +/- 6.71% | [72.40%, 75.98%] |

**Note:** Neural network LOSO excluded due to PyTorch/macOS stability issues during LOSO training.

**Success Criteria:** Top model achieves >= 85% LOSO balanced accuracy (>= 90% = exceptional)
**Outcome:** NOT MET (74.48% < 85%)

---

## Todo Checklist

- [x] Stage 0: Binary classification complete (99.97% with baseline normalization)
- [x] Phase 1: 3-class classification experiments (80/20)
- [x] Phase 1: Review results - decide if Phase 2 needed (YES - triggered)
- [x] Phase 2: Neural net experiments (4 architectures, 100 trials each)
- [x] Phase 3: LOSO validation (ensemble models)
- [x] Final report generated

---

## Success Metrics

**Primary:** Balanced accuracy (handles class imbalance)
**Secondary:** F1-score (weighted), per-class accuracy, confusion matrix analysis

**Baseline to Beat:** 79.4% (Paper 1, catch22 features)
**Our Best:** 74.48% (LightGBM, LOSO)
**Gap:** -4.92 percentage points

---

## Key Insights

1. **Per-subject baseline normalization is transformative:**
   - Enables near-perfect binary classification (99.97%)
   - Enables perfect no-pain detection in 3-class (100%)
   - Critical for handling inter-individual physiological variability

2. **Pain intensity discrimination is the bottleneck:**
   - No-pain detection: 100% accurate
   - Low pain: 66.82% accurate
   - High pain: 56.60% accurate
   - The main errors are low<->high confusion

3. **Entropy features are information-limited:**
   - Capture broad complexity patterns well
   - May lack fine-grained temporal detail for intensity discrimination
   - Consider: sample entropy, multiscale entropy, wavelet features

4. **Simpler models generalize better:**
   - LightGBM outperforms deeper neural networks on LOSO
   - 2-layer MLP better than 4-layer on test set
   - Data size (53 subjects) may limit deep learning benefit

---

## Recommendations for Future Work

1. **Feature Engineering:**
   - Add sample entropy, approximate entropy
   - Multi-scale analysis (vary embedding dimension)
   - Time-frequency features (wavelets)

2. **Alternative Approaches:**
   - Sequence modeling (LSTM, Transformer) on raw signals
   - Subject adaptation / transfer learning
   - Semi-supervised learning with test subjects

3. **Class Strategy:**
   - Consider binary (pain vs no-pain) as primary task
   - Hierarchical: pain detection then intensity estimation
   - Regression: predict pain intensity as continuous

---

**Experiment Status:** COMPLETE

*The entropy-complexity feature space with per-subject normalization provides a solid foundation but falls short of Paper 1 baseline for 3-class intensity discrimination.*
