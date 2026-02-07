#!/bin/bash
# Phase 8 Cluster Environment Setup
# Run once on Northeastern Explorer cluster login node.
# Usage: bash cluster/setup.sh

set -e

echo "=== Phase 8 Cluster Setup ==="

# Ensure pip --user packages are on PATH
export PATH="$HOME/.local/bin:$PATH"

# Persist PATH across sessions
if ! grep -q '.local/bin' ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    echo "Added ~/.local/bin to PATH in .bashrc"
fi

# Install PyTorch with CUDA 12.1
echo "Installing PyTorch (CUDA 12.1)..."
pip3 install --user torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install ML stack
echo "Installing ML dependencies..."
pip3 install --user scikit-learn pandas numpy matplotlib seaborn
pip3 install --user xgboost lightgbm optuna
pip3 install --user scipy pycatch22

# Verify imports (skip torch -- hangs on login node)
echo "Verifying imports..."
python3 -c "import sklearn; print('  scikit-learn OK')"
python3 -c "import pandas; print('  pandas OK')"
python3 -c "import numpy; print('  numpy OK')"
python3 -c "import xgboost; print('  xgboost OK')"
python3 -c "import lightgbm; print('  lightgbm OK')"
python3 -c "import optuna; print('  optuna OK')"
python3 -c "import matplotlib; print('  matplotlib OK')"
python3 -c "import seaborn; print('  seaborn OK')"
python3 -c "import scipy; print('  scipy OK')"
python3 -c "import pycatch22; print('  pycatch22 OK')"
echo "  torch: skipped (verify at job runtime)"

# Create required directories
echo "Creating directories..."
mkdir -p logs
mkdir -p results/phase8_1_raw_signal_dl
mkdir -p results/phase8_2_feature_fusion
mkdir -p results/phase8_3_nested_loso

# Check data
echo "Checking data..."
if [ -d "data/features" ]; then
    FEATURE_COUNT=$(ls data/features/results_*.csv 2>/dev/null | wc -l)
    echo "  Feature CSVs found: $FEATURE_COUNT (expected: 12)"
else
    echo "  WARNING: data/features/ not found. Upload feature data before running experiments."
fi

if [ -d "data/train" ]; then
    echo "  Raw signal data found (data/train/)"
else
    echo "  WARNING: data/train/ not found. Phase 8.1 (raw signal DL) requires raw signals."
fi

echo ""
echo "=== Setup Complete ==="
echo "Next: Upload data if not present, then submit jobs with sbatch cluster/phase8_*.sbatch"
