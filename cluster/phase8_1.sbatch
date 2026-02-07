#!/bin/bash
#SBATCH --job-name=p8.1-raw-dl
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/phase8_1_%j.out
#SBATCH --error=logs/phase8_1_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Phase 8.1: Raw Signal Deep Learning
# Trains 1D-CNN/BiLSTM/Transformer on raw physiological waveforms.
# Resubmit on wall-time kill -- auto-resumes from checkpoint.

export PATH="$HOME/.local/bin:$PATH"
cd ~/ondemand/ai4pain-ml-loop
mkdir -p logs

echo "=== Phase 8.1: Raw Signal Deep Learning ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

CHECKPOINT="results/phase8_1_raw_signal_dl/checkpoint.json"

if [ -f "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint..."
    python3 src/phase8_1_raw_signal_dl.py --resume
else
    echo "Starting fresh..."
    python3 src/phase8_1_raw_signal_dl.py
fi

echo "End: $(date)"
