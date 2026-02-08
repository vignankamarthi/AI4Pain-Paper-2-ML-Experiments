# Phase 8.2: Feature Fusion (catch22 + Entropy-Complexity)

**Status:** READY_TO_RUN
**Compute:** CPU (short partition), 4h wall time, 64GB memory
**Hypothesis:** catch22 and entropy-complexity features capture complementary aspects of pain physiology. Combined, they exceed either set alone.

---

## Objective

Merge catch22 features (88: 22 features x 4 signals) with entropy-complexity features (24, from this study) into a 112-feature representation. Note: Paper 1 reported 72 catch22 features from a different extraction configuration. Run LOSO validation with ensemble models and Optuna. Apply feature selection to manage the increased dimensionality relative to 53 labeled subjects.

---

## Feature Sets

| Feature Set | Count | Source | Status |
|-------------|-------|--------|--------|
| Entropy-Complexity | 24 | data/features/*.csv | Pre-extracted (Rust pipeline) |
| catch22 | 88 (22 per signal x 4 signals) | Extract from raw signals | Requires pycatch22 |
| **Combined** | **112** | Merged by subject + segment | To be created |

### catch22 Extraction

```python
import pycatch22

# Extract catch22 from same signal windows used for entropy-complexity
features = pycatch22.catch22_all(signal_window)
# Returns dict with 22 feature names and values
```

Extract from the same raw signal windows used by the Rust feature extraction pipeline. Match segmentation parameters exactly to ensure feature alignment across the two sets.

---

## Experiment Design

| Experiment | Features | Purpose |
|------------|----------|---------|
| A | catch22 only (88) | Reproduce Paper 1 baseline with our validation |
| B | Entropy-complexity only (24) | Confirm Phase 3 result (77.2%) |
| C | Combined (112) | Test complementarity hypothesis |
| D | Selected subset (top-K by MI) | Reduce overfitting from high dimensionality |

All four experiments use the same models, Optuna config, and LOSO folds for direct comparison.

---

## Models

| Model | Rationale |
|-------|-----------|
| RandomForest | Best LOSO performer in this study (77.2%) |
| XGBoost | Paper 1 baseline model (78.0%) |
| LightGBM | Fast, handles high dimensionality well |

All optimized with Optuna (50 trials, TPE sampler, 5-fold stratified inner CV).

---

## Feature Selection Strategy

With 112 features on 53 subjects, overfitting is a real risk. Two strategies:

1. **Mutual Information filter:** Pre-rank features by MI with target. Evaluate top-K subsets (K = 10, 20, 30, 50, 88, 112).
2. **Optuna feature selection:** Include `feature_set` and `top_k` as Optuna hyperparameters so the optimizer jointly selects features and model params.

---

## Hyperparameter Search Space

Standard ensemble search spaces (same as Phase 1/3), plus:

| Parameter | Range | Type |
|-----------|-------|------|
| feature_set | {catch22, entropy, combined, selected} | categorical |
| top_k_features | {10, 20, 30, 50, 88, 112} | int (when selected) |

---

## Validation

| Level | Method |
|-------|--------|
| HP Selection | 5-fold stratified CV |
| Final Evaluation | LOSO (53 folds) |

---

## Checkpointing

- Save after each (model, feature_set) combination completes LOSO
- Resume from last completed experiment on restart
- checkpoint.json tracks completed experiments

---

## Output Files

```
results/phase8_2_feature_fusion/
    loso_leaderboard.csv
    per_subject_results.csv
    feature_importance.csv
    ablation_results.csv
    best_hyperparameters.json
    confusion_matrix.png
    checkpoint.json
    phase8_2_report.md
```

---

## Success Criteria

| Metric | Target | Baseline |
|--------|--------|----------|
| Combined LOSO (any model) | > 78.0% | 77.2% (entropy-only) |
| catch22-only LOSO (88 features) | ~78.0% | Paper 1 reference |
| Selected subset LOSO | > 78.0% | Best of any subset |

---

## Constraints

- DO NOT use per-subject normalization
- DO NOT include rest segments in no-pain class
- DO use global z-score normalization (fit on training subjects per fold)
- DO match catch22 extraction windows to entropy-complexity windows exactly
- DO report ALL feature set ablations, not just the best
- DO document feature alignment procedure

---

## Execution

```bash
sbatch cluster/phase8_2.sbatch
```

Script: `src/phase8_2_feature_fusion.py`
