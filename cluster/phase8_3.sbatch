#!/bin/bash
#SBATCH --job-name=p8.3-nested
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/phase8_3_%j.out
#SBATCH --error=logs/phase8_3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Phase 8.3: Nested LOSO Completion
# CPU-only (RandomForest), but uses gpu partition for 8h wall time and 128GB memory.
# Completes the nested Optuna-LOSO from Phase 7 to all 65 folds.
# Resubmit on wall-time kill -- auto-resumes from checkpoint.

export PATH="$HOME/.local/bin:$PATH"
cd ~/ondemand/ai4pain-ml-loop
mkdir -p logs

echo "=== Phase 8.3: Nested LOSO Completion ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"

CHECKPOINT="results/phase8_3_nested_loso/checkpoint.json"

if [ -f "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint ($(python3 -c "import json; d=json.load(open('$CHECKPOINT')); print(len(d.get('completed_folds', [])))" 2>/dev/null || echo '?') folds complete)..."
    python3 src/phase8_3_nested_loso.py --resume
else
    echo "Starting fresh (53 folds, ~138K model fits)..."
    python3 src/phase8_3_nested_loso.py
fi

echo "End: $(date)"
