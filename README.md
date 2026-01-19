# Entropy-Complexity Pain Classification

Machine learning pipeline for pain classification using entropy-complexity features extracted from physiological signals.

## Results

| Task | Accuracy | Features |
|------|----------|----------|
| Binary Pain Detection | 99.92% | 2 (H, C) |
| 3-Class LOSO | 77.2% | 24 |
| Paper 1 Baseline | 78.0% | 72 (catch22) |

Binary pain detection achieves near-perfect accuracy using permutation entropy and statistical complexity on the C-H plane. 3-class performance is competitive with catch22 features using 1/3 the dimensionality.

## Repository Structure

```
ML-experiment-loop/
├── src/                      # Experiment scripts
│   ├── stage0_binary.py      # C-H plane binary classification
│   ├── phase1_ensemble_experiments.py
│   ├── phase2_neural_experiments.py
│   ├── phase3_loso_validation.py
│   ├── phase4_full_training.py
│   ├── phase5_hierarchical.py
│   └── phase7_nested_loso.py
├── results/                  # Output files
│   ├── stage0_binary/        # C-H plane visualizations
│   ├── phase1_ensembles/     # 80/20 ensemble results
│   ├── phase2_neuralnets/    # 80/20 neural net results
│   ├── phase3_loso/          # LOSO validation (primary)
│   ├── phase4_full_training/
│   ├── phase5_hierarchical/
│   └── phase7_nested_loso/
├── phases/                   # Phase documentation
├── data/                     # Feature data (not tracked)
├── FINAL_REPORT.md           # Results summary
├── STATUS.md                 # Experiment status
└── PLAN.md                   # Experiment plan
```

## Data

Features extracted from AI4Pain dataset:
- Signals: EDA, BVP, RESP, SpO2
- Features per signal: PE, Complexity, Fisher-Shannon, Fisher Info, Renyi PE, Renyi Complexity, Tsallis PE, Tsallis Complexity
- Embedding: d=7, tau=2
- Subjects: 65 total (train=41, validation=12, test=12)

Data files expected in `data/features/`:
```
results_train_eda.csv
results_train_bvp.csv
results_train_resp.csv
results_train_spo2.csv
results_validation_*.csv
results_test_*.csv
```

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
```

## Usage

Run experiments in order:

```bash
python src/stage0_binary.py           # Binary C-H plane analysis
python src/phase1_ensemble_experiments.py
python src/phase2_neural_experiments.py
python src/phase3_loso_validation.py  # Primary LOSO results
python src/phase5_hierarchical.py
```

All scripts include checkpoint recovery. Use `--resume` flag to continue interrupted runs.

## Methodology

- Validation: Leave-One-Subject-Out (LOSO) cross-validation
- Normalization: Global z-score (per-subject causes LOSO leakage)
- Classes: Baseline (no pain), Low Pain, High Pain
- Metric: Balanced accuracy

## Key Findings

1. Binary pain detection is solved (99.92% with 2 features)
2. 3-class LOSO is competitive with catch22 (77.2% vs 78.0%)
3. Pain intensity discrimination (low vs high) is the bottleneck (58-60%)
4. Entropy-complexity features are 3x more compact than catch22

## Reference

Baseline comparison: Boda et al., ICMI 2025 (Paper 1)

## License

Research use only.
