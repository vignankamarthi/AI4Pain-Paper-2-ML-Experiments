# Phase 8.1: Raw Signal Deep Learning

**Status:** READY_TO_RUN
**Compute:** GPU partition (H100), 8h wall time, 64GB memory
**Hypothesis:** End-to-end deep learning on raw physiological waveforms captures intensity-discriminative temporal patterns that entropy-complexity features compress away.

---

## Objective

Train temporal deep learning models directly on raw BVP, EDA, Resp, and SpO2 signals. The 58-60% intensity discrimination bottleneck may be a feature engineering ceiling, not a signal ceiling. Raw waveforms preserve fine-grained temporal structure that summary statistics discard.

---

## Data

Raw physiological signals from the AI4Pain dataset:

| Signal | Location | Format |
|--------|----------|--------|
| BVP | data/{split}/Bvp/{subject_id}.csv | Time-series CSV |
| EDA | data/{split}/Eda/{subject_id}.csv | Time-series CSV |
| Resp | data/{split}/Resp/{subject_id}.csv | Time-series CSV |
| SpO2 | data/{split}/Spo2/{subject_id}.csv | Time-series CSV |

- Splits: train (41), validation (12) -- pooled for LOSO
- Test set (12) excluded (unknown labels)
- Total: 53 subjects
- Labels: baseline (0), low pain (1), high pain (2)
- Rest segments EXCLUDED from no-pain class

### Preprocessing

1. Segment raw signals into fixed-length windows (match Paper 1's segmentation parameters)
2. Z-score normalize using global statistics (fit on training subjects per LOSO fold)
3. Multi-signal stack: 4 channels (BVP, EDA, Resp, SpO2) per window

---

## Architecture Candidates

| Architecture | Rationale | Parameters (est.) |
|--------------|-----------|-------------------|
| 1D-CNN | Learns local temporal patterns, fast to train, strong baseline | ~500K |
| BiLSTM | Captures long-range temporal dependencies | ~1M |
| 1D Temporal Transformer | Self-attention over signal positions, state-of-the-art | ~2M |

Start with 1D-CNN. Escalate to BiLSTM and Transformer only if CNN underperforms.

### 1D-CNN Reference Architecture

```
Input: (batch, 4_channels, window_length)
-> Conv1d(4, 32, kernel=7, stride=2) + BatchNorm + ReLU
-> Conv1d(32, 64, kernel=5, stride=2) + BatchNorm + ReLU
-> Conv1d(64, 128, kernel=3, stride=2) + BatchNorm + ReLU
-> AdaptiveAvgPool1d(1)
-> Flatten
-> Linear(128, 64) + ReLU + Dropout(0.3)
-> Linear(64, 3)
```

---

## Hyperparameter Search (Optuna)

| Parameter | Range | Type |
|-----------|-------|------|
| learning_rate | [1e-5, 1e-2] | log-uniform |
| weight_decay | [1e-6, 1e-2] | log-uniform |
| dropout | [0.1, 0.5] | uniform |
| batch_size | {16, 32, 64} | categorical |
| n_conv_layers | {2, 3, 4} | categorical |
| hidden_dim | {32, 64, 128, 256} | categorical |
| kernel_size | {3, 5, 7} | categorical |

- Trials: 50 per architecture
- Inner CV: 5-fold stratified (LOSO too expensive for DL inner loop)
- Outer CV: LOSO (53 folds)

---

## Validation

| Level | Method | Purpose |
|-------|--------|---------|
| HP Selection | 5-fold stratified CV | Fast inner loop for Optuna |
| Final Evaluation | LOSO (53 folds) | Subject-independent comparison to Paper 1 |

---

## Checkpointing

- Save model checkpoint after each LOSO fold completes
- Save Optuna study after each trial
- Resume from last completed fold on wall-time kill
- Individual fold results saved to `fold_results/`

---

## Output Files

```
results/phase8_1_raw_signal_dl/
    loso_leaderboard.csv
    per_subject_results.csv
    best_hyperparameters.json
    confusion_matrix.png
    training_curves/
    fold_results/
    checkpoint.json
    phase8_1_report.md
```

---

## Success Criteria

| Metric | Target | Current Best |
|--------|--------|--------------|
| 3-Class LOSO | > 78.0% | 77.2% (RF, entropy-complexity) |
| Intensity Discrimination | > 65% | 58-60% |

---

## Constraints

- DO NOT use per-subject normalization
- DO NOT include rest segments in no-pain class
- DO use global z-score normalization (fit on training subjects per fold)
- DO checkpoint after each LOSO fold
- DO use balanced accuracy as primary metric
- DO use mixed precision (AMP) for training speed on H100
- DO set random seeds for reproducibility

---

## Execution

```bash
sbatch cluster/phase8_1.sbatch
```

Script: `src/phase8_1_raw_signal_dl.py`
