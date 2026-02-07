# Entropy-Complexity Pain Classification

A systematic comparison of entropy-complexity features against catch22 baseline features for pain classification from physiological signals.

**Paper**: AI4Pain Paper 2 (in preparation)
**Baseline**: Boda et al., ICMI Companion '25 -- [doi:10.1145/3747327.3764784](https://doi.org/10.1145/3747327.3764784)
**Feature Extraction**: [Feature-Extraction-Rust](https://github.com/vignankamarthi/Feature-Extraction-Rust)

All credit for the AI4Pain dataset and baseline methodology belongs to the original authors.

## What This Repository Does

Paper 1 classifies pain from physiological signals (EDA, BVP, Resp, SpO2) using 72 catch22 time-series features with XGBoost, achieving 78.0% LOSO balanced accuracy for 3-class classification (no-pain / low-pain / high-pain).

**This repository investigates an alternative feature space** -- entropy and complexity measures extracted via ordinal pattern analysis -- and extends to GPU cluster experiments:

| Task | This Study | Paper 1 | Delta |
|------|------------|---------|-------|
| Binary Pain Detection | **99.92%** (2 features) | ~98% | +2 pp |
| 3-Class LOSO | 77.2% (24 features) | **78.0%** (72 features) | -0.8 pp |
| 3-Class 80/20 | **80.1%** | 79.4% | +0.7 pp |

## The Problem: Pain Intensity Discrimination

Binary pain detection (pain vs no-pain) is effectively solved. The challenge is classifying pain intensity:

| Stage | Accuracy | Status |
|-------|----------|--------|
| Pain Detection (binary) | 99.92% | Solved |
| No-Pain vs Pain (in 3-class) | ~100% | Solved |
| Low Pain vs High Pain | 58-60% | Bottleneck |
| **Overall 3-Class LOSO** | **77.2%** | **Target: >78.0%** |

Phase 8 targets this bottleneck with three cluster-scale experiments: raw signal deep learning, feature fusion (catch22 + entropy-complexity), and completed nested cross-validation.

## Repository Structure

```
ML-experiment-loop/
├── src/                              # Experiment scripts
│   ├── stage0_binary.py              # C-H plane binary classification
│   ├── phase1_ensemble_experiments.py
│   ├── phase2_neural_experiments.py
│   ├── phase3_loso_validation.py     # Primary LOSO results
│   ├── phase4_full_training.py
│   ├── phase5_hierarchical.py
│   ├── phase6_final_experiment.py
│   ├── phase7_nested_loso.py
│   ├── phase8_1_raw_signal_dl.py     # [Cluster] Raw signal DL
│   ├── phase8_2_feature_fusion.py    # [Cluster] Feature fusion
│   └── phase8_3_nested_loso.py       # [Cluster] Nested LOSO completion
├── cluster/                          # SLURM batch scripts
│   ├── setup.sh
│   ├── phase8_1.sbatch
│   ├── phase8_2.sbatch
│   └── phase8_3.sbatch
├── phases/                           # Phase documentation
│   ├── STAGE0.md ... PHASE7.md
│   └── phase8/                       # Cluster experiments
│       ├── PHASE8.md                 # Overview
│       ├── PHASE8_1.md ... PHASE8_3.md
│       └── TASK.md                   # Execution guide
├── results/                          # Output (large files gitignored)
├── data/                             # Feature + raw signal data (gitignored)
├── FINAL_REPORT.md                   # Phases 1-7 results
├── STATUS.md
├── PLAN.md
└── requirements.txt
```

## Training

Phases 0-7 ran locally. Phase 8 runs on Northeastern's GPU cluster (1x H100).

```bash
# Local (Phases 0-7, completed)
python src/stage0_binary.py
python src/phase1_ensemble_experiments.py
python src/phase2_neural_experiments.py
python src/phase3_loso_validation.py
python src/phase4_full_training.py
python src/phase5_hierarchical.py
python src/phase6_final_experiment.py
python src/phase7_nested_loso.py --resume

# Cluster (Phase 8)
sbatch cluster/phase8_1.sbatch    # Raw signal deep learning (GPU)
sbatch cluster/phase8_2.sbatch    # Feature fusion (CPU)
sbatch cluster/phase8_3.sbatch    # Nested LOSO completion (CPU)
```

All scripts include checkpoint recovery. Resubmit sbatch on wall-time kill to auto-resume.

## Data

Features extracted from AI4Pain multimodal physiological signals using [Feature-Extraction-Rust](https://github.com/vignankamarthi/Feature-Extraction-Rust):

- **Signals:** EDA, BVP, Resp, SpO2
- **Features per signal:** PE, Complexity, Fisher-Shannon, Fisher Info, Renyi PE, Renyi Complexity, Tsallis PE, Tsallis Complexity
- **Embedding parameters:** d=7, tau=2
- **Subjects:** 65 total (train=41, validation=12, test=12)

Feature CSVs expected in `data/features/`. Raw signals in `data/{train,validation,test}/`.

## Requirements

```
numpy
pandas
scikit-learn
xgboost
lightgbm
optuna
torch
matplotlib
seaborn
scipy
pycatch22
```

## Methodology

- **Validation:** Leave-One-Subject-Out (LOSO) cross-validation
- **Normalization:** Global z-score (per-subject causes LOSO leakage)
- **Classes:** Baseline (no pain), Low Pain, High Pain
- **No-Pain Class:** Baseline segments only (rest segments excluded)
- **Metric:** Balanced accuracy
- **Hyperparameters:** Optuna optimization (50 trials, TPE sampler)

## Key Findings

1. Binary pain detection is solved -- 99.92% with 2 features on the C-H plane
2. 3-class LOSO is competitive -- 77.2% vs 78.0% with 1/3 the feature dimensionality
3. Pain intensity discrimination (low vs high) is the field-wide bottleneck at 58-60%
4. Per-subject normalization causes catastrophic data leakage in LOSO (32.8%)

## Reference

Boda SRR, Kamarthi VS, Ozek B, et al (2025) Canonical time series features for pain classification. In: ICMI Companion '25. ACM, pp 1-6. [doi:10.1145/3747327.3764784](https://doi.org/10.1145/3747327.3764784)

## License

Research use only.

---

## Experiment Report

See [FINAL_REPORT.md](FINAL_REPORT.md) for the full experiment report including training configuration, results, and analysis across all phases.
