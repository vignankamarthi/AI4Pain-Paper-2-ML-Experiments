# AI4Pain Paper 2 - Experimental Plan

## Research Objective
Improve 3-class pain classification accuracy beyond Paper 1's **79.4% balanced accuracy** baseline using entropy and complexity-based features extracted from multimodal physiological signals.

**Target:** Greater than or equal to 90% balanced accuracy (stretch goal)
**Minimum Success:** Greater than or equal to 85% balanced accuracy with robust LOSO validation

---

## Key Discovery: Per-Subject Baseline Normalization

From Stage 0, we discovered that **per-subject baseline normalization** is critical:
- Normalize each subject's entropy features using ONLY their no-pain samples as reference
- This accounts for individual physiological differences
- Binary classification: 99.97% accuracy (vs 87% with global normalization)

This normalization strategy will be applied to all subsequent phases.

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

### Phase 1: 3-Class Classification (80/20 Split)
**Status:** Not started

**Goal:** Train and evaluate classifiers for 3-class pain classification (baseline, low_pain, high_pain) using per-subject baseline normalization.

**Features:** All 8 entropy measures normalized per-subject
- pe, comp, fisher_shannon, fisher_info, renyipe, renyicomp, tsallispe, tsalliscomp

**Signals to test:** EDA, BVP, RESP, SpO2 (all 4)

**Models:** Random Forest, XGBoost, LightGBM, SVM, Logistic Regression

**Deliverable:** Leaderboard of model performance, confusion matrices, feature importance.

**Success Criteria:** Best model achieves >= 85% balanced accuracy

---

### Phase 2: Neural Net Exploration (If Needed)
**Status:** Not started

**Trigger:** Phase 1 best model < 85% balanced accuracy

**Goal:** Explore deep learning architectures to capture complex feature interactions.

**Models:** MLPs with varying depth, dropout, regularization

**Deliverable:** Neural net leaderboard, comparison to ensembles.

**Success Criteria:** Neural net beats best ensemble by >= 2% balanced accuracy.

---

### Phase 3: LOSO Validation
**Status:** Not started

**Goal:** Rigorous cross-validation on top models.

**Validation:** Leave-One-Subject-Out

**Deliverable:** Final report with LOSO results, confusion matrices, best model file.

**Success Criteria:** Top model achieves >= 85% LOSO balanced accuracy (>= 90% = exceptional).

---

## Todo Checklist

- [x] Stage 0: Binary classification complete (99.97% with baseline normalization)
- [ ] Phase 1: 3-class classification experiments (80/20)
- [ ] Phase 1: Review results - decide if Phase 2 needed
- [ ] Phase 2: Neural net experiments (if triggered)
- [ ] Phase 3: LOSO validation
- [ ] Final report generated and submitted

---

## Success Metrics

**Primary:** Balanced accuracy (handles class imbalance)
**Secondary:** F1-score (weighted), per-class accuracy, confusion matrix analysis

**Baseline to Beat:** 79.4% (Paper 1, catch22 features)
