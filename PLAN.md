# AI4Pain Paper 2 - Experimental Plan

## Research Objective
Improve 3-class pain classification accuracy beyond Paper 1's **79.4% balanced accuracy** baseline using entropy and complexity-based features extracted from multimodal physiological signals.

**Target:** Greater than or equal to 90% balanced accuracy (stretch goal)  
**Minimum Success:** Greater than or equal to 85% balanced accuracy with robust LOSO validation

---

## Experimental Waterfall

### Stage 0: Binary Classification Silhouette (Quick Win)
**Status:** COMPLETE - Awaiting Approval

**Goal:** Quantify Paper 1's binary classification (pain vs. no-pain) separation on C-H plane.

**Deliverable:** Silhouette constants for all (d, tau, signal) combos, ranked report.

**Success Criteria:** Clear documentation of which parameter combos yield best binary separation.

**Results:** Best combination EDA d=7 tau=2 (silhouette: **0.490** with pe+comp). Strong binary separation for EDA/BVP at high dimensions. RESP weakest (mean: 0.008).

**Multimodal Extension:** Combined best (d, tau) per signal → EDA+BVP achieves **88.95% balanced accuracy** with SVM. Adding RESP/SpO2 reduces performance.

**FINAL RESULT: 99.97% Linear Accuracy with Per-Subject Baseline Normalization**
- **Method:** Normalize H and C using each subject's no-pain samples as baseline reference
- **Comparison:** Raw=87.97%, Global StandardScaler=87.45%, **Baseline Norm=99.97%**
- **Key insight:** Per-subject normalization aligns all subjects to common reference frame
- **Physiological interpretation:** Pain causes ↓H (more predictable) and ↑C (more structured) relative to baseline

---

### Phase 0: 3-Class Silhouette Analysis
**Status:** Not started

**Goal:** Identify discriminative (d, tau, signal) combinations for 3-class classification (baseline, low, high).

**Deliverable:** Ranked silhouette constants, feature selection justification, SpO2 exclusion rationale.

**Success Criteria:** Clear threshold decision (e.g., "use top 40% of combos") with PI approval.

---

### Phase 1: Ensemble Exploration (80/20 Split)
**Status:** Not started

**Goal:** Rapid exploration of ensemble methods using silhouette-filtered features.

**Models:** Random Forest, XGBoost, LightGBM, Stacked Ensembles

**Deliverable:** Leaderboard of model performance, confusion matrices, feature importance.

**Success Criteria:** Best ensemble achieves greater than or equal to 82% balanced accuracy (if greater than or equal to 85%, proceed to LOSO; if less than 85%, consider Phase 2).

---

### Phase 2: Neural Net Exploration (If Needed)
**Status:** Not started

**Trigger:** Phase 1 best ensemble less than 85% balanced accuracy

**Goal:** Explore deep learning architectures to capture complex feature interactions.

**Models:** MLPs with varying depth, dropout, regularization

**Deliverable:** Neural net leaderboard, comparison to ensembles.

**Success Criteria:** Neural net beats best ensemble by greater than or equal to 2% balanced accuracy.

---

### Phase 3: LOSO Validation
**Status:** Not started

**Goal:** Rigorous cross-validation on top 5 models (ensemble and/or neural net).

**Validation:** Leave-One-Subject-Out (41 folds)

**Deliverable:** Final report with LOSO results, confusion matrices, C-H plane visualizations, best model file.

**Success Criteria:** Top model achieves greater than or equal to 85% LOSO balanced accuracy (greater than or equal to 90% equals exceptional).

---

## Todo Checklist

- [x] Stage 0: Binary silhouette analysis complete
- [ ] Stage 0: Results reviewed and approved by PI
- [ ] Phase 0: 3-class silhouette analysis complete
- [ ] Phase 0: Feature threshold decision made and documented
- [ ] Phase 1: Ensemble experiments complete (80/20)
- [ ] Phase 1: Review results - decide if Phase 2 needed
- [ ] Phase 2: Neural net experiments complete (if triggered)
- [ ] Phase 3: LOSO validation complete
- [ ] Final report generated and submitted

---

## Key Milestones

**Week 1-2:** Silhouette analyses (Stage 0 and Phase 0)  
**Week 3-4:** Ensemble exploration (Phase 1)  
**Week 4-5:** Neural nets if needed (Phase 2)  
**Week 5-6:** LOSO validation and paper writing (Phase 3)

---

## Success Metrics

**Primary:** Balanced accuracy (handles class imbalance)  
**Secondary:** F1-score (weighted), per-class accuracy, confusion matrix analysis

**Baseline to Beat:** 79.4% (Paper 1, catch22 features)