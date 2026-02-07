#!/bin/bash
#SBATCH --job-name=p8.2-fusion
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/phase8_2_%j.out
#SBATCH --error=logs/phase8_2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Phase 8.2: Feature Fusion (catch22 + Entropy-Complexity)
# CPU-only. Extracts catch22, merges features, runs LOSO.
# Resubmit on wall-time kill -- auto-resumes from checkpoint.

export PATH="$HOME/.local/bin:$PATH"
cd ~/ondemand/ai4pain-ml-loop
mkdir -p logs

echo "=== Phase 8.2: Feature Fusion ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

CHECKPOINT="results/phase8_2_feature_fusion/checkpoint.json"

if [ -f "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint..."
    python3 src/phase8_2_feature_fusion.py --resume
else
    echo "Starting fresh..."
    python3 src/phase8_2_feature_fusion.py
fi

echo "End: $(date)"
