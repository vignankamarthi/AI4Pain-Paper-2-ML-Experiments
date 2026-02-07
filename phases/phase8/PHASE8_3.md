# Phase 8.3: Nested LOSO Completion

**Status:** READY_TO_RUN
**Compute:** CPU (gpu partition for wall time), 8h limit, 128GB memory
**Goal:** Run Phase 7's nested Optuna-LOSO to completion on all folds.

---

## Objective

Phase 7 (Nested Optuna-LOSO) was terminated at 17/53 folds (72.1% mean accuracy) after three OOM kills (exit code 137) on local hardware. The cluster provides 128GB memory and SLURM-managed resources. Run the same experiment to completion for a definitive nested validation result.

---

## Configuration

Identical to Phase 7:

| Parameter | Value |
|-----------|-------|
| Data | All 65 subjects pooled |
| Labels | 3-class (baseline=0, low=1, high=2) |
| Features | 24 entropy-complexity features |
| Normalization | Global z-score |
| Model | RandomForest |
| Outer CV | LOSO (65 folds) |
| Inner CV | LOSO (64 folds per Optuna trial) |
| Optuna trials | 50 per outer fold |
| Total model fits | 65 x 50 x 64 = 208,000 |

---

## Why the Cluster

| Issue | Local | Cluster |
|-------|-------|---------|
| Memory | OOM at fold 17 (exit 137) | 128GB allocated |
| Runtime | ~20h estimated, killed 3x | 8h wall, resubmit from checkpoint |
| Stability | Process killed by OS | SLURM manages resource limits |

---

## Methodology

```
For each outer fold s in [1..65]:
    1. Hold out subject s as test set
    2. Train pool = remaining 64 subjects
    3. Run Optuna (50 trials):
        For each trial:
            a. Suggest hyperparameters
            b. Inner LOSO on 64 subjects
            c. Return mean inner score to Optuna
    4. Train final model on 64 subjects with best params
    5. Evaluate on held-out subject s
    6. Save fold result and checkpoint

Final LOSO accuracy = mean(65 fold accuracies)
```

---

## Checkpointing

The existing `phase7_nested_loso.py` supports `--resume`:
- Saves checkpoint.json after each outer fold
- Resumes from last completed fold
- Individual fold results saved to `fold_results/`

On wall-time kill, resubmit the same sbatch script. Auto-resumes.

---

## Implementation

Adapt `phase7_nested_loso.py` with minimal changes:
1. Change output directory from `results/phase7_nested_loso/` to `results/phase8_3_nested_loso/`
2. Start fresh (no carry-over from partial Phase 7 run)
3. All other parameters unchanged

---

## Output Files

```
results/phase8_3_nested_loso/
    loso_leaderboard.csv
    per_subject_results.csv
    best_hyperparameters.json
    confusion_matrix.png
    checkpoint.json
    fold_results/
    phase8_3_report.md
```

---

## Expected Outcome

Based on Phase 7 partial results (17/53 folds, 72.1% and declining trend), the final number will likely be 70-73%. This is below the 78.0% target but provides a methodologically rigorous nested validation number for the paper -- the most defensible accuracy estimate in the entire study.

---

## Success Criteria

| Metric | Target | Expected |
|--------|--------|----------|
| Completion | All 65 folds | Phase 7 stopped at 17 |
| 3-Class LOSO | > 78.0% (stretch) | 70-73% (realistic) |

Even if below 78.0%, completion is valuable: it gives a proper nested CV estimate and demonstrates the cost of rigorous validation vs optimistic single-split results.

---

## Constraints

- Use adapted phase7_nested_loso.py with modified output directory
- DO NOT change the methodology -- same code, more memory
- DO use --resume to recover from wall-time kills
- DO start fresh (no checkpoint from local Phase 7 run)

---

## Execution

```bash
sbatch cluster/phase8_3.sbatch
```

Script: `src/phase8_3_nested_loso.py`
