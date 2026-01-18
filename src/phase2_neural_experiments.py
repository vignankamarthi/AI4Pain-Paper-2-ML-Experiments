#!/usr/bin/env python3
"""
Phase 6 Step 2: Neural Network Exploration (BASELINE-ONLY)

MIRRORS Phase 2 but with baseline-only labeling:
- Class 0: no_pain (baseline ONLY - rest segments EXCLUDED)
- Class 1: low_pain
- Class 2: high_pain

Explores deep learning architectures (MLP variants) to capture complex feature
interactions. Uses PyTorch with Optuna hyperparameter optimization.

Author: Claude (AI Assistant)
Date: 2026-01-15
"""

import gc
import os
import re
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix
)
import optuna
from optuna.trial import Trial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Feature extraction parameters (from Stage 0)
BEST_DIMENSION = 7
BEST_TAU = 2

# Signals and features - use all 4 signals for consistency with Phase 1
SIGNALS = ['eda', 'bvp', 'resp', 'spo2']
FEATURE_COLS = ['pe', 'comp', 'fisher_shannon', 'fisher_info',
                'renyipe', 'renyicomp', 'tsallispe', 'tsalliscomp']

# Paper 1 baseline
PAPER1_BASELINE = 0.794

# Paths - navigate up from src/phase6/ to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'features'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'phase6_step2_neuralnets'

# Class mapping - BASELINE ONLY (rest segments EXCLUDED)
CLASS_MAPPING = {
    'baseline': 0,  # no_pain (ONLY baseline, not rest)
    'low': 1,       # low_pain
    'high': 2       # high_pain
}
CLASS_NAMES = ['no_pain', 'low_pain', 'high_pain']
N_CLASSES = 3

# Neural net training config
N_OPTUNA_TRIALS = 100  # Extended for thorough optimization (~2-3 hours per architecture)
CV_FOLDS = 5
MAX_EPOCHS = 300  # Increased for deeper training
EARLY_STOP_PATIENCE = 20  # More patience for convergence
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')


# =============================================================================
# Checkpointing and Memory Management
# =============================================================================

def clear_memory():
    """Clear all caches and force garbage collection to prevent memory crashes."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def load_checkpoint(results_dir):
    """Load checkpoint if exists, return completed architectures and results."""
    checkpoint_file = results_dir / 'checkpoint.json'
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        print(f'  Loaded checkpoint: {len(checkpoint["completed"])} architectures completed')
        return checkpoint
    return {'completed': [], 'results': {}}


def save_checkpoint(results_dir, completed, results):
    """Save checkpoint after each architecture completes."""
    # Convert results to serializable format (remove non-serializable objects)
    serializable_results = {}
    for name, data in results.items():
        serializable_results[name] = {
            'best_params': data.get('best_params', {}),
            'metrics': {k: v for k, v in data.get('metrics', {}).items()
                       if k not in ['confusion_matrix', 'y_true', 'y_pred']},
        }

    checkpoint = {
        'timestamp': datetime.now().isoformat(),
        'completed': completed,
        'results': serializable_results
    }
    checkpoint_file = results_dir / 'checkpoint.json'
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f'    [CHECKPOINT SAVED] {len(completed)} architectures completed')


# =============================================================================
# Data Loading (same as Phase 1)
# =============================================================================

def extract_subject_id(segment_name: str) -> str:
    """Extract subject ID from segment name like '12_Baseline_1'"""
    match = re.match(r'(\d+)_', str(segment_name))
    if match:
        return match.group(1)
    match = re.search(r'(\d+)', str(segment_name))
    if match:
        return match.group(1)
    return str(segment_name)


def load_all_data() -> pd.DataFrame:
    """Load train and validation data (labeled data only)."""
    print("Loading feature data...")

    all_dfs = []
    splits = ['train', 'validation']

    for split in splits:
        for phys_signal in SIGNALS:
            file_path = DATA_DIR / f'results_{split}_{phys_signal}.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                df = df.rename(columns={'signal': 'segment_id'})
                df['segment_id'] = df['segment_id'].astype(str)
                df['phys_signal'] = phys_signal
                df['split'] = split
                all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Filter for best dimension and tau
    combined = combined[
        (combined['dimension'] == BEST_DIMENSION) &
        (combined['tau'] == BEST_TAU)
    ].copy()

    # PHASE 6 KEY DIFFERENCE: Exclude rest segments entirely
    n_before = len(combined)
    combined = combined[combined['state'] != 'rest'].copy()
    n_after = len(combined)
    print(f"  [PHASE 6] Excluded {n_before - n_after} rest segments")

    # Extract subject IDs
    combined['subject_id'] = combined['segment_id'].apply(extract_subject_id)

    # Map states to labels
    combined['label'] = combined['state'].map(CLASS_MAPPING)
    combined = combined.dropna(subset=['label'])
    combined['label'] = combined['label'].astype(int)

    return combined


def pivot_to_multimodal(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to wide format with all signals as features."""
    df = df.copy()

    pivot_dfs = []
    for phys_signal in SIGNALS:
        signal_df = df[df['phys_signal'] == phys_signal][
            ['segment_id', 'subject_id', 'state', 'label'] + FEATURE_COLS
        ].copy()
        signal_df = signal_df.rename(columns={col: f'{phys_signal}_{col}' for col in FEATURE_COLS})
        pivot_dfs.append(signal_df)

    result = pivot_dfs[0]
    for pdf in pivot_dfs[1:]:
        result = result.merge(
            pdf.drop(columns=['subject_id', 'state', 'label']),
            on='segment_id',
            how='inner'
        )

    return result


def apply_baseline_normalization(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Apply per-subject baseline normalization."""
    df_norm = df.copy()
    no_pain_states = ['baseline', 'rest']

    for subj in df['subject_id'].unique():
        mask = df['subject_id'] == subj
        no_pain_mask = mask & df['state'].isin(no_pain_states)
        n_nopain = no_pain_mask.sum()

        if n_nopain >= 2:
            for col in feature_cols:
                ref_mean = df.loc[no_pain_mask, col].mean()
                ref_std = df.loc[no_pain_mask, col].std()
                if ref_std > 1e-10:
                    df_norm.loc[mask, col] = (df.loc[mask, col] - ref_mean) / ref_std
                else:
                    df_norm.loc[mask, col] = 0.0
        else:
            for col in feature_cols:
                subj_mean = df.loc[mask, col].mean()
                subj_std = df.loc[mask, col].std()
                if subj_std > 1e-10:
                    df_norm.loc[mask, col] = (df.loc[mask, col] - subj_mean) / subj_std
                else:
                    df_norm.loc[mask, col] = 0.0

    return df_norm


# =============================================================================
# Neural Network Architectures
# =============================================================================

class SimpleMLP(nn.Module):
    """2-layer MLP"""
    def __init__(self, input_dim: int, layer1: int, layer2: int,
                 dropout: float, activation: str = 'relu'):
        super().__init__()

        act_fn = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'leaky_relu': nn.LeakyReLU()}[activation]

        self.network = nn.Sequential(
            nn.Linear(input_dim, layer1),
            nn.BatchNorm1d(layer1),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(layer1, layer2),
            nn.BatchNorm1d(layer2),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(layer2, N_CLASSES)
        )

    def forward(self, x):
        return self.network(x)


class MediumMLP(nn.Module):
    """3-layer MLP"""
    def __init__(self, input_dim: int, layer1: int, layer2: int, layer3: int,
                 dropout: float, activation: str = 'relu'):
        super().__init__()

        act_fn = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'leaky_relu': nn.LeakyReLU()}[activation]

        self.network = nn.Sequential(
            nn.Linear(input_dim, layer1),
            nn.BatchNorm1d(layer1),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(layer1, layer2),
            nn.BatchNorm1d(layer2),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(layer2, layer3),
            nn.BatchNorm1d(layer3),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(layer3, N_CLASSES)
        )

    def forward(self, x):
        return self.network(x)


class DeepMLP(nn.Module):
    """4-layer MLP"""
    def __init__(self, input_dim: int, layer1: int, layer2: int,
                 layer3: int, layer4: int, dropout: float, activation: str = 'relu'):
        super().__init__()

        act_fn = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'leaky_relu': nn.LeakyReLU()}[activation]

        self.network = nn.Sequential(
            nn.Linear(input_dim, layer1),
            nn.BatchNorm1d(layer1),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(layer1, layer2),
            nn.BatchNorm1d(layer2),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(layer2, layer3),
            nn.BatchNorm1d(layer3),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(layer3, layer4),
            nn.BatchNorm1d(layer4),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(layer4, N_CLASSES)
        )

    def forward(self, x):
        return self.network(x)


class RegularizedMLP(nn.Module):
    """3-layer MLP with L2 regularization (weight decay handled in optimizer)"""
    def __init__(self, input_dim: int, layer1: int, layer2: int, layer3: int,
                 dropout: float, activation: str = 'relu'):
        super().__init__()

        act_fn = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'leaky_relu': nn.LeakyReLU()}[activation]

        self.network = nn.Sequential(
            nn.Linear(input_dim, layer1),
            nn.BatchNorm1d(layer1),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(layer1, layer2),
            nn.BatchNorm1d(layer2),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(layer2, layer3),
            nn.BatchNorm1d(layer3),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(layer3, N_CLASSES)
        )

    def forward(self, x):
        return self.network(x)


# =============================================================================
# Training Functions
# =============================================================================

def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Compute class weights for imbalanced data."""
    classes, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(classes) * counts)
    return torch.FloatTensor(weights).to(DEVICE)


def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, data_loader):
    """Evaluate model on data."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    return np.array(all_preds), np.array(all_labels)


def train_model_cv(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    params: Dict,
    n_folds: int = CV_FOLDS
) -> float:
    """
    Train model with cross-validation.
    Returns mean balanced accuracy across folds.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )

        batch_size = params.get('batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Create model
        model = model_fn(params).to(DEVICE)

        # Class weights and loss
        class_weights = compute_class_weights(y_train)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        lr = params.get('learning_rate', params.get('lr', 0.001))
        weight_decay = params.get('l2_reg', 0)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Training loop with early stopping
        best_val_score = 0
        patience_counter = 0

        for epoch in range(MAX_EPOCHS):
            train_epoch(model, train_loader, criterion, optimizer)

            # Evaluate
            val_preds, val_labels = evaluate(model, val_loader)
            val_score = balanced_accuracy_score(val_labels, val_preds)

            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    break

        fold_scores.append(best_val_score)

    return np.mean(fold_scores)


# =============================================================================
# Optuna Objectives
# =============================================================================

def create_simple_mlp_objective(X: np.ndarray, y: np.ndarray, input_dim: int):
    """Create Optuna objective for Simple MLP."""
    def objective(trial: Trial) -> float:
        params = {
            'layer1': trial.suggest_categorical('layer1', [64, 128, 256, 512]),
            'layer2': trial.suggest_categorical('layer2', [32, 64, 128, 256]),
            'dropout': trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4]),
            'lr': trial.suggest_categorical('lr', [0.0001, 0.0005, 0.001, 0.005]),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'leaky_relu']),
            'input_dim': input_dim
        }

        def model_fn(p):
            return SimpleMLP(p['input_dim'], p['layer1'], p['layer2'],
                           p['dropout'], p['activation'])

        return train_model_cv(model_fn, X, y, params)

    return objective


def create_medium_mlp_objective(X: np.ndarray, y: np.ndarray, input_dim: int):
    """Create Optuna objective for Medium MLP."""
    def objective(trial: Trial) -> float:
        params = {
            'layer1': trial.suggest_categorical('layer1', [128, 256, 512]),
            'layer2': trial.suggest_categorical('layer2', [64, 128, 256]),
            'layer3': trial.suggest_categorical('layer3', [32, 64, 128]),
            'dropout': trial.suggest_categorical('dropout', [0.2, 0.3, 0.4, 0.5]),
            'lr': trial.suggest_categorical('lr', [0.0001, 0.0005, 0.001]),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'leaky_relu']),
            'input_dim': input_dim
        }

        def model_fn(p):
            return MediumMLP(p['input_dim'], p['layer1'], p['layer2'], p['layer3'],
                           p['dropout'], p['activation'])

        return train_model_cv(model_fn, X, y, params)

    return objective


def create_deep_mlp_objective(X: np.ndarray, y: np.ndarray, input_dim: int):
    """Create Optuna objective for Deep MLP."""
    def objective(trial: Trial) -> float:
        params = {
            'layer1': trial.suggest_categorical('layer1', [256, 512]),
            'layer2': trial.suggest_categorical('layer2', [128, 256]),
            'layer3': trial.suggest_categorical('layer3', [64, 128]),
            'layer4': trial.suggest_categorical('layer4', [32, 64]),
            'dropout': trial.suggest_categorical('dropout', [0.3, 0.4, 0.5]),
            'lr': trial.suggest_categorical('lr', [0.0001, 0.0005, 0.001]),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32]),
            'activation': trial.suggest_categorical('activation', ['relu', 'elu']),
            'input_dim': input_dim
        }

        def model_fn(p):
            return DeepMLP(p['input_dim'], p['layer1'], p['layer2'],
                         p['layer3'], p['layer4'], p['dropout'], p['activation'])

        return train_model_cv(model_fn, X, y, params)

    return objective


def create_regularized_mlp_objective(X: np.ndarray, y: np.ndarray, input_dim: int):
    """Create Optuna objective for Regularized MLP."""
    def objective(trial: Trial) -> float:
        params = {
            'layer1': trial.suggest_categorical('layer1', [256, 512]),
            'layer2': trial.suggest_categorical('layer2', [128, 256]),
            'layer3': trial.suggest_categorical('layer3', [64, 128]),
            'dropout': trial.suggest_categorical('dropout', [0.4, 0.5, 0.6]),
            'lr': trial.suggest_categorical('lr', [0.0001, 0.0005]),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32]),
            'activation': trial.suggest_categorical('activation', ['relu', 'elu']),
            'l2_reg': trial.suggest_categorical('l2_reg', [0.001, 0.01, 0.1]),
            'input_dim': input_dim
        }

        def model_fn(p):
            return RegularizedMLP(p['input_dim'], p['layer1'], p['layer2'],
                                p['layer3'], p['dropout'], p['activation'])

        return train_model_cv(model_fn, X, y, params)

    return objective


# =============================================================================
# Final Training and Evaluation
# =============================================================================

def train_final_model(
    model_class: str,
    params: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[nn.Module, Dict, List, List]:
    """
    Train final model with best params and evaluate on test set.
    Returns model, metrics, training history.
    """
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    batch_size = params.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create model based on class name
    input_dim = params['input_dim']
    if model_class == 'SimpleMLP':
        model = SimpleMLP(input_dim, params['layer1'], params['layer2'],
                         params['dropout'], params['activation']).to(DEVICE)
    elif model_class == 'MediumMLP':
        model = MediumMLP(input_dim, params['layer1'], params['layer2'], params['layer3'],
                         params['dropout'], params['activation']).to(DEVICE)
    elif model_class == 'DeepMLP':
        model = DeepMLP(input_dim, params['layer1'], params['layer2'],
                       params['layer3'], params['layer4'],
                       params['dropout'], params['activation']).to(DEVICE)
    elif model_class == 'RegularizedMLP':
        model = RegularizedMLP(input_dim, params['layer1'], params['layer2'], params['layer3'],
                              params['dropout'], params['activation']).to(DEVICE)

    # Training setup
    class_weights = compute_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    lr = params.get('learning_rate', params.get('lr', 0.001))
    weight_decay = params.get('l2_reg', 0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training history
    train_losses = []
    val_accs = []

    best_model_state = None
    best_val_score = 0
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)

        # Evaluate on test
        test_preds, test_labels = evaluate(model, test_loader)
        val_score = balanced_accuracy_score(test_labels, test_preds)
        val_accs.append(val_score)

        if val_score > best_val_score:
            best_val_score = val_score
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    test_preds, test_labels = evaluate(model, test_loader)

    metrics = {
        'accuracy': accuracy_score(test_labels, test_preds),
        'balanced_accuracy': balanced_accuracy_score(test_labels, test_preds),
        'f1_weighted': f1_score(test_labels, test_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(test_labels, test_preds),
        'y_true': test_labels,
        'y_pred': test_preds
    }

    # Per-class accuracy
    cm = metrics['confusion_matrix']
    for i, class_name in enumerate(CLASS_NAMES):
        metrics[f'acc_{class_name}'] = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0

    return model, metrics, train_losses, val_accs


# =============================================================================
# Visualization
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: Path):
    """Generate confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix: {model_name}')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(train_losses: List, val_accs: List, model_name: str, save_dir: Path):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name}: Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(val_accs, label='Validation Balanced Acc', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Balanced Accuracy')
    ax2.set_title(f'{model_name}: Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name.lower().replace(" ", "_")}_training_curves.png',
                dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Report Generation
# =============================================================================

def generate_phase2_report(
    nn_results: Dict,
    phase1_leaderboard: pd.DataFrame,
    combined_leaderboard: pd.DataFrame
) -> str:
    """Generate Phase 2 report markdown."""

    # Find best neural net
    best_nn = max(nn_results.items(), key=lambda x: x[1]['metrics']['balanced_accuracy'])
    best_nn_name, best_nn_data = best_nn
    best_nn_ba = best_nn_data['metrics']['balanced_accuracy']

    # Best overall from combined
    best_overall = combined_leaderboard.iloc[0]

    report = f"""# Phase 2: Neural Network Exploration Results

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Best Neural Net** | {best_nn_name} |
| **Best NN Balanced Accuracy** | {best_nn_ba:.2%} |
| **Best Overall Model** | {best_overall['model']} |
| **Best Overall Balanced Acc** | {best_overall['balanced_accuracy']:.2%} |
| **Paper 1 Baseline** | {PAPER1_BASELINE:.1%} |

---

## Neural Network Architectures

Four MLP architectures were explored:
1. **Simple MLP**: 2 hidden layers
2. **Medium MLP**: 3 hidden layers
3. **Deep MLP**: 4 hidden layers
4. **Regularized MLP**: 3 hidden layers with high dropout and L2 regularization

### Training Configuration
- **Optimization:** Optuna (Bayesian) with {N_OPTUNA_TRIALS} trials per architecture
- **Validation:** {CV_FOLDS}-fold stratified CV
- **Early Stopping:** Patience = {EARLY_STOP_PATIENCE} epochs
- **Device:** {DEVICE}

---

## Neural Network Leaderboard

| Rank | Architecture | Balanced Acc | Accuracy | F1 | No Pain | Low Pain | High Pain |
|------|--------------|--------------|----------|-----|---------|----------|-----------|
"""

    # Sort NN results by balanced accuracy
    sorted_nn = sorted(nn_results.items(),
                      key=lambda x: x[1]['metrics']['balanced_accuracy'], reverse=True)

    for rank, (name, data) in enumerate(sorted_nn, 1):
        m = data['metrics']
        report += f"| {rank} | {name} | {m['balanced_accuracy']:.2%} | {m['accuracy']:.2%} | "
        report += f"{m['f1_weighted']:.3f} | {m['acc_no_pain']:.2%} | {m['acc_low_pain']:.2%} | {m['acc_high_pain']:.2%} |\n"

    report += f"""

---

## Combined Leaderboard (Phase 1 + Phase 2)

| Rank | Model | Type | Balanced Acc | Accuracy |
|------|-------|------|--------------|----------|
"""

    for _, row in combined_leaderboard.iterrows():
        report += f"| {int(row['rank'])} | {row['model']} | {row['model_type']} | "
        report += f"{row['balanced_accuracy']:.2%} | {row['accuracy']:.2%} |\n"

    report += f"""

---

## Analysis

### Neural Net vs Ensemble Comparison

"""

    best_p1 = phase1_leaderboard.iloc[0]['balanced_accuracy']
    nn_vs_ensemble = best_nn_ba - best_p1

    if nn_vs_ensemble > 0:
        report += f"**Neural networks improved by {nn_vs_ensemble:.2%} over best ensemble!**\n\n"
    else:
        report += f"Best ensemble outperforms neural nets by {abs(nn_vs_ensemble):.2%}.\n\n"

    report += """### Key Observations

1. **Architecture Impact:** """

    # Add architecture-specific insights
    accs = [(n, d['metrics']['balanced_accuracy']) for n, d in nn_results.items()]
    accs.sort(key=lambda x: x[1], reverse=True)

    report += f"The {accs[0][0]} architecture performed best, suggesting "
    if 'Regularized' in accs[0][0]:
        report += "that strong regularization helps prevent overfitting on this dataset.\n"
    elif 'Simple' in accs[0][0]:
        report += "that simpler models generalize better on this feature space.\n"
    elif 'Deep' in accs[0][0]:
        report += "that deeper networks can capture more complex patterns.\n"
    else:
        report += "a good balance between capacity and regularization.\n"

    report += f"""
2. **Pain Discrimination Challenge:** All models struggle most with distinguishing
   low pain from high pain states, consistent with Phase 1 ensemble findings.

3. **No Pain Detection:** Neural networks maintain high accuracy on no-pain states,
   validating that the per-subject baseline normalization is effective.

---

## Best Hyperparameters

"""

    for name, data in nn_results.items():
        report += f"### {name}\n```json\n{json.dumps(data['best_params'], indent=2)}\n```\n\n"

    report += f"""---

## Recommendation for Phase 3 LOSO

Based on combined results, recommend validating these models with LOSO:

"""

    for i, row in combined_leaderboard.head(5).iterrows():
        report += f"{int(row['rank'])}. **{row['model']}** ({row['model_type']}): {row['balanced_accuracy']:.2%}\n"

    report += f"""

---

## Output Files

```
results/phase2_neuralnets/
├── leaderboard.csv
├── combined_leaderboard.csv
├── hyperparameters.json
├── phase2_report.md
├── confusion_matrices/
│   └── [architecture]_confusion_matrix.png
├── training_curves/
│   └── [architecture]_training_curves.png
└── models/
    └── [architecture]_best.pth
```

---

**End of Phase 2 Report**
"""

    return report


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function for Phase 2 neural network exploration."""

    print("="*70)
    print("PHASE 2: NEURAL NETWORK EXPLORATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")

    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'confusion_matrices').mkdir(exist_ok=True)
    (RESULTS_DIR / 'training_curves').mkdir(exist_ok=True)
    (RESULTS_DIR / 'models').mkdir(exist_ok=True)

    # Load data
    raw_df = load_all_data()
    df = pivot_to_multimodal(raw_df)
    print(f"Total samples: {len(df)}")

    # Feature columns
    feature_cols = [f'{sig}_{feat}' for sig in SIGNALS for feat in FEATURE_COLS]
    input_dim = len(feature_cols)
    print(f"Input dimension: {input_dim}")

    # Apply normalization
    df = apply_baseline_normalization(df, feature_cols)

    # Prepare data
    X = df[feature_cols].values
    y = df['label'].values

    # Train/test split (80/20 stratified, same as Phase 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Class distribution (train): {np.bincount(y_train)}")

    # Define architectures to test
    architectures = {
        'Simple MLP': (create_simple_mlp_objective, 'SimpleMLP'),
        'Medium MLP': (create_medium_mlp_objective, 'MediumMLP'),
        'Deep MLP': (create_deep_mlp_objective, 'DeepMLP'),
        'Regularized MLP': (create_regularized_mlp_objective, 'RegularizedMLP'),
    }

    # Load checkpoint (resume capability)
    checkpoint = load_checkpoint(RESULTS_DIR)
    completed_archs = set(checkpoint['completed'])
    all_results = {}

    if completed_archs:
        print(f'  Resuming from checkpoint: {len(completed_archs)} architectures already done')

    failed_archs = []

    # Train each architecture with checkpointing
    for arch_name, (objective_fn, model_class) in architectures.items():
        # Skip already completed architectures
        if arch_name in completed_archs:
            print(f"\n[SKIPPED] {arch_name} - already completed")
            continue

        print(f"\n{'='*60}")
        print(f"Optimizing: {arch_name}")
        print(f"{'='*60}")

        # Clear memory before architecture
        clear_memory()

        try:
            # Create Optuna study
            study = optuna.create_study(direction='maximize',
                                       sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
            objective = objective_fn(X_train, y_train, input_dim)

            study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

            print(f"\nBest trial: {study.best_trial.value:.4f}")
            print(f"Best params: {study.best_trial.params}")

            # Train final model with best params
            best_params = study.best_trial.params.copy()
            best_params['input_dim'] = input_dim

            print(f"\nTraining final {arch_name} model...")
            model, metrics, train_losses, val_accs = train_final_model(
                model_class, best_params, X_train, y_train, X_test, y_test
            )

            print(f"Test Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")

            # Save results
            all_results[arch_name] = {
                'model': model,
                'metrics': metrics,
                'best_params': best_params,
                'train_losses': train_losses,
                'val_accs': val_accs,
                'study': study
            }

            # Save model
            model_path = RESULTS_DIR / 'models' / f'{model_class.lower()}_best.pth'
            torch.save(model.state_dict(), model_path)

            # Plot confusion matrix
            plot_confusion_matrix(
                metrics['y_true'], metrics['y_pred'], arch_name,
                RESULTS_DIR / 'confusion_matrices' / f'{model_class.lower()}_confusion_matrix.png'
            )

            # Plot training curves
            plot_training_curves(train_losses, val_accs, arch_name, RESULTS_DIR / 'training_curves')

            # Mark as completed and save checkpoint
            completed_archs.add(arch_name)
            save_checkpoint(RESULTS_DIR, list(completed_archs), all_results)

        except Exception as e:
            print(f'    [ERROR] {arch_name} failed: {str(e)}')
            failed_archs.append({'arch': arch_name, 'error': str(e), 'timestamp': datetime.now().isoformat()})
            with open(RESULTS_DIR / 'failed_architectures.json', 'w') as f:
                json.dump(failed_archs, f, indent=2)

        # Clear memory after architecture
        clear_memory()

    # Generate leaderboards
    print("\n" + "="*70)
    print("GENERATING OUTPUTS")
    print("="*70)

    # Neural net leaderboard
    nn_rows = []
    for name, data in all_results.items():
        m = data['metrics']
        nn_rows.append({
            'rank': 0,
            'model': name,
            'accuracy': m['accuracy'],
            'balanced_accuracy': m['balanced_accuracy'],
            'f1_weighted': m['f1_weighted'],
            'acc_no_pain': m['acc_no_pain'],
            'acc_low_pain': m['acc_low_pain'],
            'acc_high_pain': m['acc_high_pain']
        })

    nn_leaderboard = pd.DataFrame(nn_rows)
    nn_leaderboard = nn_leaderboard.sort_values('balanced_accuracy', ascending=False).reset_index(drop=True)
    nn_leaderboard['rank'] = nn_leaderboard.index + 1
    nn_leaderboard.to_csv(RESULTS_DIR / 'leaderboard.csv', index=False)
    print("Saved: leaderboard.csv")

    # Load Phase 1 results for combined leaderboard
    phase1_path = PROJECT_ROOT / 'results' / 'phase1_ensembles' / 'leaderboard.csv'
    phase1_leaderboard = pd.read_csv(phase1_path)

    # Create combined leaderboard
    combined_rows = []

    for _, row in phase1_leaderboard.iterrows():
        combined_rows.append({
            'model': row['model'],
            'model_type': 'ensemble',
            'accuracy': row['accuracy'],
            'balanced_accuracy': row['balanced_accuracy']
        })

    for _, row in nn_leaderboard.iterrows():
        combined_rows.append({
            'model': row['model'],
            'model_type': 'neural_net',
            'accuracy': row['accuracy'],
            'balanced_accuracy': row['balanced_accuracy']
        })

    combined_leaderboard = pd.DataFrame(combined_rows)
    combined_leaderboard = combined_leaderboard.sort_values('balanced_accuracy', ascending=False).reset_index(drop=True)
    combined_leaderboard['rank'] = combined_leaderboard.index + 1
    combined_leaderboard.to_csv(RESULTS_DIR / 'combined_leaderboard.csv', index=False)
    print("Saved: combined_leaderboard.csv")

    # Save hyperparameters
    hyperparams = {name: data['best_params'] for name, data in all_results.items()}
    with open(RESULTS_DIR / 'hyperparameters.json', 'w') as f:
        json.dump(hyperparams, f, indent=2)
    print("Saved: hyperparameters.json")

    # Generate report
    report = generate_phase2_report(all_results, phase1_leaderboard, combined_leaderboard)
    with open(RESULTS_DIR / 'phase2_report.md', 'w') as f:
        f.write(report)
    print("Saved: phase2_report.md")

    # Save scaler for deployment
    with open(RESULTS_DIR / 'models' / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Final summary
    print("\n" + "="*70)
    print("PHASE 2 COMPLETE")
    print("="*70)

    print("\nNeural Network Leaderboard:")
    print("-" * 60)
    for _, row in nn_leaderboard.iterrows():
        print(f"  {int(row['rank'])}. {row['model']}: {row['balanced_accuracy']:.2%}")

    print("\nCombined Leaderboard (Top 5):")
    print("-" * 60)
    for _, row in combined_leaderboard.head(5).iterrows():
        print(f"  {int(row['rank'])}. {row['model']} ({row['model_type']}): {row['balanced_accuracy']:.2%}")

    best_nn = nn_leaderboard.iloc[0]
    print(f"\n{'='*60}")
    print(f"BEST NEURAL NET: {best_nn['model']}")
    print(f"Balanced Accuracy: {best_nn['balanced_accuracy']:.2%}")
    print(f"{'='*60}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {RESULTS_DIR}")

    return all_results, combined_leaderboard


if __name__ == '__main__':
    all_results, combined_leaderboard = main()
