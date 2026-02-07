# Phase 8 Cluster Execution Guide

Step-by-step instructions for running Phase 8 experiments on Northeastern's Explorer cluster.

**Cluster:** Northeastern University Research Computing (Explorer)
**Access:** Open OnDemand > Clusters > Explorer Shell Access
**Compute:** SLURM batch jobs (gpu + short partitions)

---

## Prerequisites

- NEU Explorer cluster access
- Git configured on the cluster
- Familiarity with SLURM (sbatch, squeue, scancel)

---

## Step 1: Get the Data

The original physiological signal data comes from the feature extraction pipeline:

**Repository:** [Feature-Extraction-Rust](https://github.com/vignankamarthi/Feature-Extraction-Rust)

You know how to get the data from this repo. You need:
- **Feature CSVs** -> place in `data/features/` (12 files: `results_{split}_{signal}.csv`)
- **Raw signals** -> place in `data/{train,validation,test}/{Bvp,Eda,Resp,SpO2}/` (Phase 8.1 needs these)

---

## Step 2: Clone the Repository

```bash
cd ~/ondemand
git clone <repo-url> ai4pain-ml-loop
cd ai4pain-ml-loop
```

---

## Step 3: Set Up Environment

```bash
bash cluster/setup.sh
```

This installs all Python dependencies via `pip --user` and verifies imports. If it fails on any package, install manually:

```bash
export PATH="$HOME/.local/bin:$PATH"
pip3 install --user <package>
```

Verify:
```bash
python3 -c "import sklearn, optuna, xgboost, lightgbm, pycatch22; print('All OK')"
```

Do NOT run `import torch` on the login node -- it hangs. Torch is verified at job runtime.

---

## Step 4: Upload Data

From your local machine:

```bash
scp -r data/features/ <user>@explorer.rc.northeastern.edu:~/ondemand/ai4pain-ml-loop/data/features/
scp -r data/train/ <user>@explorer.rc.northeastern.edu:~/ondemand/ai4pain-ml-loop/data/train/
scp -r data/validation/ <user>@explorer.rc.northeastern.edu:~/ondemand/ai4pain-ml-loop/data/validation/
scp -r data/test/ <user>@explorer.rc.northeastern.edu:~/ondemand/ai4pain-ml-loop/data/test/
```

If data already exists on the cluster from a prior project, symlink instead:

```bash
ln -s /path/to/existing/data data
```

---

## Step 5: Create Log Directory

```bash
mkdir -p logs
```

---

## Step 6: Run Experiments

All three experiments are independent. Run in any order or simultaneously.

### Phase 8.3: Nested LOSO Completion (Start Here -- Lowest Risk)

```bash
sbatch cluster/phase8_3.sbatch
```

Completes the nested Optuna-LOSO experiment to all 65 folds. CPU-only, 128GB memory. Resubmit on wall-time kill -- auto-resumes from checkpoint.

### Phase 8.2: Feature Fusion

```bash
sbatch cluster/phase8_2.sbatch
```

Extracts catch22 features from raw signals, merges with entropy-complexity features, runs LOSO validation. CPU-only, 64GB.

### Phase 8.1: Raw Signal Deep Learning

```bash
sbatch cluster/phase8_1.sbatch
```

Trains 1D-CNN / BiLSTM / Transformer directly on raw waveforms. Requires H100 GPU.

---

## Step 7: Monitor

```bash
# Check running/pending jobs
squeue -u $USER

# Watch live output (replace JOB_ID from squeue output)
tail -f logs/phase8_1_JOB_ID.out
tail -f logs/phase8_2_JOB_ID.out
tail -f logs/phase8_3_JOB_ID.out

# Check completed jobs today
sacct -u $USER --starttime=today

# Cancel a job
scancel <job_id>
```

---

## Step 8: Handle Wall-Time Kills

GPU partition has an 8-hour limit. If a job is killed before completion:

```bash
# Check if experiment finished
grep "complete" logs/phase8_*_*.out

# If not, resubmit (auto-resumes from checkpoint)
sbatch cluster/phase8_1.sbatch
sbatch cluster/phase8_2.sbatch
sbatch cluster/phase8_3.sbatch
```

All scripts checkpoint after each major unit of work (LOSO fold, model, etc.) and resume from the last completed unit.

---

## Step 9: Push Results

```bash
cd ~/ondemand/ai4pain-ml-loop
git add results/phase8_1_raw_signal_dl/ results/phase8_2_feature_fusion/ results/phase8_3_nested_loso/
git add -u  # stage modifications to tracked files
git commit -m "Phase 8 cluster experiment results"
git push origin main
```

---

## Quick Reference

| Phase | Script | Sbatch | Partition | GPU | Memory | Est. Runtime |
|-------|--------|--------|-----------|-----|--------|--------------|
| 8.1 | src/phase8_1_raw_signal_dl.py | cluster/phase8_1.sbatch | gpu | H100 x1 | 64GB | 8-24h |
| 8.2 | src/phase8_2_feature_fusion.py | cluster/phase8_2.sbatch | short | None | 64GB | 2-4h |
| 8.3 | src/phase8_3_nested_loso.py | cluster/phase8_3.sbatch | gpu | None | 128GB | 8-20h |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| OOM (exit 137) | Increase `--mem` in sbatch. Reduce batch size for Phase 8.1. |
| CUDA error | Verify `--gres=gpu:h100:1` in sbatch. Check `torch.cuda.is_available()` in job log. |
| Module not found | Run `cluster/setup.sh`. Check `export PATH="$HOME/.local/bin:$PATH"`. |
| Job pending forever | GPU queue is shared. Check depth: `squeue -p gpu \| wc -l` |
| Checkpoint missing | Verify `results/phase8_*/checkpoint.json` exists before resubmitting. |
| torch hangs on login | Normal. Only import torch inside sbatch jobs with GPU allocation. |
