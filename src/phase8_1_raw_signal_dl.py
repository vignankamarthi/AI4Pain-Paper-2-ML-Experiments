#!/usr/bin/env python3
"""
Phase 8.1: Raw Signal Deep Learning for Pain Classification

Trains temporal deep learning models (1D-CNN, BiLSTM) directly on raw
physiological waveforms for 3-class pain classification.

Target: Leverage temporal dynamics in raw signals that may be lost in
        entropy-complexity feature extraction.

Methodology:
- Raw signal loading from CSV files
- Multi-channel input (BVP, EDA, Resp, SpO2 stacked)
- Fixed-length windows (pad/truncate to MAX_SIGNAL_LENGTH)
- Global z-score normalization per channel
- 1D-CNN and BiLSTM architectures
- Optuna HP optimization (5-fold stratified CV)
- LOSO cross-validation for final evaluation
- REST segments excluded, baseline-only for no-pain

Author: Claude (AI Assistant)
Date: 2026-02-07
"""

import gc
import os
import re
import sys
import json
import warnings
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report
)
import optuna
from optuna.samplers import TPESampler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("Using MPS (Apple Silicon)")
else:
    DEVICE = torch.device('cpu')
    print("Using CPU")

# Signals and mappings
SIGNALS = ['eda', 'bvp', 'resp', 'spo2']
SIGNAL_DIR_MAP = {
    'eda': 'Eda',
    'bvp': 'Bvp',
    'resp': 'Resp',
    'spo2': 'SpO2'
}

# Class mapping - BASELINE ONLY (rest segments EXCLUDED)
CLASS_MAPPING = {
    'baseline': 0,
    'low': 1,
    'high': 2
}
CLASS_NAMES = ['no_pain', 'low_pain', 'high_pain']
N_CLASSES = len(CLASS_NAMES)
N_CHANNELS = len(SIGNALS)

# Dimension and tau for feature CSVs
BEST_DIMENSION = 7
BEST_TAU = 2

# Raw signal configuration
MAX_SIGNAL_LENGTH = 1000  # Pad/truncate to this length

# Training configuration
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
GRAD_CLIP_MAX_NORM = 5.0

# Optuna configuration
N_OPTUNA_TRIALS = 50
ARCHITECTURES = ['Conv1D', 'BiLSTM']

# Paper 1 baseline
PAPER1_LOSO_BASELINE = 0.780

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
FEATURES_DIR = DATA_DIR / 'features'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'phase8_1_raw_signal_dl'


# =============================================================================
# Utility Functions
# =============================================================================

def clear_memory():
    """Clear caches and force garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy/torch types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    else:
        return obj


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def extract_subject_id(segment_name: str) -> str:
    """Extract subject ID from segment name like '12_Baseline_1'."""
    match = re.match(r'(\d+)_', segment_name)
    if match:
        return match.group(1)
    match = re.search(r'(\d+)', segment_name)
    if match:
        return match.group(1)
    return segment_name


# =============================================================================
# Checkpointing
# =============================================================================

def load_checkpoint(results_dir: Path) -> Dict:
    """Load checkpoint if exists."""
    checkpoint_file = results_dir / 'checkpoint.json'
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            n_completed_archs = len(checkpoint.get('completed_architectures', []))
            current_arch = checkpoint.get('current_architecture')
            n_completed_folds = len(checkpoint.get('fold_results', {}).get(current_arch, {})) if current_arch else 0
            print(f"  [CHECKPOINT] Loaded: {n_completed_archs} architectures complete, "
                  f"{n_completed_folds} folds in current architecture")
            return checkpoint
        except json.JSONDecodeError:
            print("  [WARNING] Corrupted checkpoint, starting fresh")
            return {'completed_architectures': [], 'current_architecture': None, 'fold_results': {}}
    return {'completed_architectures': [], 'current_architecture': None, 'fold_results': {}}


def save_checkpoint(results_dir: Path, checkpoint: Dict):
    """Save checkpoint after each fold."""
    checkpoint['last_updated'] = datetime.now().isoformat()
    checkpoint_file = results_dir / 'checkpoint.json'

    # Convert to serializable format
    serializable = convert_to_serializable(checkpoint)

    with open(checkpoint_file, 'w') as f:
        json.dump(serializable, f, indent=2)

    current_arch = checkpoint.get('current_architecture')
    n_completed_folds = len(checkpoint.get('fold_results', {}).get(current_arch, {})) if current_arch else 0
    print(f"    [CHECKPOINT SAVED] Architecture: {current_arch}, Folds: {n_completed_folds}")


def save_fold_result(results_dir: Path, architecture: str, subject_id: str, fold_data: Dict):
    """Save individual fold result to disk."""
    fold_dir = results_dir / 'fold_results' / architecture
    fold_dir.mkdir(parents=True, exist_ok=True)

    fold_file = fold_dir / f'fold_{subject_id}.json'
    with open(fold_file, 'w') as f:
        json.dump(convert_to_serializable(fold_data), f, indent=2)


# =============================================================================
# Data Loading
# =============================================================================

def load_segment_metadata() -> pd.DataFrame:
    """
    Load segment metadata from feature CSVs.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: segment_id, subject_id, state, label,
        signal_type, file_path, signallength
    """
    print("Loading segment metadata from feature CSVs...")

    all_metadata = []
    splits = ['train', 'validation']

    for split in splits:
        for signal_type in SIGNALS:
            file_path = FEATURES_DIR / f'results_{split}_{signal_type}.csv'
            if not file_path.exists():
                print(f"  [WARNING] Missing: {file_path}")
                continue

            df = pd.read_csv(file_path)

            # Filter for best dimension and tau
            df = df[(df['dimension'] == BEST_DIMENSION) & (df['tau'] == BEST_TAU)].copy()

            # Rename 'signal' column to 'segment_id'
            df = df.rename(columns={'signal': 'segment_id'})

            # Extract subject ID
            df['subject_id'] = df['segment_id'].apply(extract_subject_id)

            # Keep only needed columns
            metadata = df[['segment_id', 'subject_id', 'state', 'file_name', 'signallength']].copy()
            metadata['signal_type'] = signal_type
            metadata['split'] = split

            all_metadata.append(metadata)
            print(f"  Loaded {file_path.name}: {len(metadata)} segments")

    combined = pd.concat(all_metadata, ignore_index=True)

    # EXCLUDE rest segments (baseline-only methodology)
    n_before = len(combined)
    combined = combined[combined['state'] != 'rest'].copy()
    n_after = len(combined)
    print(f"  Excluded {n_before - n_after} rest segments")

    # Map states to labels
    combined['label'] = combined['state'].map(CLASS_MAPPING)
    combined = combined.dropna(subset=['label'])
    combined['label'] = combined['label'].astype(int)

    print(f"\nTotal segments: {len(combined)}")
    print(f"Unique subjects: {combined['subject_id'].nunique()}")
    print(f"Class distribution: {combined['label'].value_counts().sort_index().to_dict()}")

    return combined


def load_raw_signal(file_path: Path, segment_id: str, max_length: int = MAX_SIGNAL_LENGTH) -> np.ndarray:
    """
    Load a raw signal segment from CSV file.

    Parameters
    ----------
    file_path : Path
        Path to raw signal CSV file
    segment_id : str
        Segment ID (column name in CSV)
    max_length : int
        Maximum signal length (pad or truncate)

    Returns
    -------
    np.ndarray
        Raw signal array of shape (max_length,)
    """
    try:
        # Read CSV
        df = pd.read_csv(file_path)

        # Get segment column
        if segment_id not in df.columns:
            # Try to find similar column
            possible = [col for col in df.columns if segment_id in col]
            if not possible:
                raise ValueError(f"Segment {segment_id} not found in {file_path}")
            segment_id = possible[0]

        # Extract signal values
        signal = df[segment_id].values

        # Remove NaNs
        signal = signal[~np.isnan(signal)]

        # Pad or truncate
        if len(signal) < max_length:
            # Pad with zeros
            padded = np.zeros(max_length, dtype=np.float32)
            padded[:len(signal)] = signal
            signal = padded
        else:
            # Truncate
            signal = signal[:max_length]

        return signal.astype(np.float32)

    except Exception as e:
        print(f"  [ERROR] Loading {segment_id} from {file_path}: {e}")
        # Return zero signal on error
        return np.zeros(max_length, dtype=np.float32)


def create_multimodal_dataset(metadata: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create multi-channel dataset by loading raw signals.

    Parameters
    ----------
    metadata : pd.DataFrame
        Segment metadata

    Returns
    -------
    X : np.ndarray
        Multi-channel signals of shape (n_samples, n_channels, max_length)
    y : np.ndarray
        Labels of shape (n_samples,)
    subject_ids : np.ndarray
        Subject IDs of shape (n_samples,)
    """
    print("\nCreating multi-channel dataset from raw signals...")

    # Group by segment to get all 4 channels per sample
    segments_grouped = metadata.groupby('segment_id')

    valid_segments = []
    for segment_id, group in segments_grouped:
        # Check if all 4 signals present
        if len(group) == N_CHANNELS:
            valid_segments.append(segment_id)

    print(f"  Valid segments with all {N_CHANNELS} channels: {len(valid_segments)}")

    # Allocate arrays
    n_samples = len(valid_segments)
    X = np.zeros((n_samples, N_CHANNELS, MAX_SIGNAL_LENGTH), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    subject_ids = []

    # Load each segment
    for idx, segment_id in enumerate(valid_segments):
        if idx % 100 == 0:
            print(f"  Loading segment {idx + 1}/{n_samples}...")

        segment_data = metadata[metadata['segment_id'] == segment_id]

        # Get label and subject (same for all channels)
        y[idx] = segment_data['label'].iloc[0]
        subject_ids.append(segment_data['subject_id'].iloc[0])

        # Load each channel
        for channel_idx, signal_type in enumerate(SIGNALS):
            row = segment_data[segment_data['signal_type'] == signal_type].iloc[0]
            file_path = Path(row['file_name'])

            # Load raw signal
            signal = load_raw_signal(file_path, segment_id)
            X[idx, channel_idx, :] = signal

    subject_ids = np.array(subject_ids)

    print(f"\nDataset created:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Unique subjects: {len(np.unique(subject_ids))}")
    print(f"  Class distribution: {np.bincount(y)}")

    return X, y, subject_ids


# =============================================================================
# PyTorch Dataset
# =============================================================================

class RawSignalDataset(Dataset):
    """PyTorch Dataset for raw physiological signals."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Parameters
        ----------
        X : np.ndarray
            Multi-channel signals of shape (n_samples, n_channels, length)
        y : np.ndarray
            Labels of shape (n_samples,)
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# =============================================================================
# Neural Network Architectures
# =============================================================================

class Conv1DClassifier(nn.Module):
    """1D Convolutional Neural Network for time series classification."""

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_classes: int = N_CLASSES,
        hidden_dim: int = 64,
        n_conv_layers: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.3
    ):
        """
        Parameters
        ----------
        n_channels : int
            Number of input channels (4 for our signals)
        n_classes : int
            Number of output classes (3 for our task)
        hidden_dim : int
            Base hidden dimension for convolutional layers
        n_conv_layers : int
            Number of convolutional blocks
        kernel_size : int
            Kernel size for convolutions
        dropout : float
            Dropout probability
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        # Build convolutional blocks
        layers = []
        in_channels = n_channels

        for i in range(n_conv_layers):
            out_channels = hidden_dim * (2 ** i)

            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])

            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, n_channels, length)

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, n_classes)
        """
        x = self.conv_layers(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for time series classification."""

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_classes: int = N_CLASSES,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Parameters
        ----------
        n_channels : int
            Number of input channels (4 for our signals)
        n_classes : int
            Number of output classes (3 for our task)
        hidden_dim : int
            Hidden dimension for LSTM
        n_layers : int
            Number of LSTM layers
        dropout : float
            Dropout probability
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # LSTM expects (batch, seq_len, features)
        # Our input is (batch, n_channels, seq_len)
        # We'll transpose in forward pass

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )

        # Classifier (hidden_dim * 2 because bidirectional)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, n_channels, length)

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, n_classes)
        """
        # Transpose to (batch, length, n_channels)
        x = x.transpose(1, 2)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take last hidden state from both directions
        # h_n shape: (n_layers * 2, batch, hidden_dim)
        forward_hidden = h_n[-2, :, :]  # Last layer, forward direction
        backward_hidden = h_n[-1, :, :]  # Last layer, backward direction

        # Concatenate
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)

        # Classify
        logits = self.classifier(hidden)

        return logits


# =============================================================================
# Training and Evaluation
# =============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Compute class weights for imbalanced dataset."""
    counts = np.bincount(y)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.from_numpy(weights).float().to(DEVICE)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    use_amp: bool = True
) -> float:
    """
    Train for one epoch.

    Returns
    -------
    float
        Average training loss
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        if use_amp and DEVICE.type == 'cuda':
            with autocast():
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model.

    Returns
    -------
    loss : float
        Average loss
    balanced_acc : float
        Balanced accuracy
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    avg_loss = total_loss / len(loader)

    return avg_loss, balanced_acc, y_true, y_pred


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOP_PATIENCE
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Train a model with early stopping.

    Returns
    -------
    model : nn.Module
        Trained model
    train_losses : List[float]
        Training losses per epoch
    val_losses : List[float]
        Validation losses per epoch
    """
    # Class weights
    # Extract labels from train loader
    train_labels = []
    for _, y_batch in train_loader:
        train_labels.append(y_batch.numpy())
    train_labels = np.concatenate(train_labels)
    class_weights = compute_class_weights(train_labels)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler() if DEVICE.type == 'cuda' else None
    early_stopping = EarlyStopping(patience=patience)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(max_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            use_amp=(DEVICE.type == 'cuda')
        )

        # Validate
        val_loss, val_ba, _, _ = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


# =============================================================================
# Hyperparameter Search Spaces
# =============================================================================

def get_conv1d_search_space(trial: optuna.Trial) -> Dict:
    """Define Conv1D hyperparameter search space."""
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'n_conv_layers': trial.suggest_int('n_conv_layers', 2, 4),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
        'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7])
    }
    return params


def get_bilstm_search_space(trial: optuna.Trial) -> Dict:
    """Define BiLSTM hyperparameter search space."""
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
        'n_layers': trial.suggest_int('n_layers', 1, 3)
    }
    return params


# =============================================================================
# Optuna Optimization
# =============================================================================

def optimize_hyperparameters(
    architecture: str,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    train_subjects: List[str]
) -> Dict:
    """
    Optimize hyperparameters using 5-fold stratified CV.

    Parameters
    ----------
    architecture : str
        'Conv1D' or 'BiLSTM'
    X : np.ndarray
        Full feature matrix
    y : np.ndarray
        Full labels
    subject_ids : np.ndarray
        Subject IDs for each sample
    train_subjects : List[str]
        List of training subjects (for masking)

    Returns
    -------
    Dict
        Best hyperparameters and best score
    """
    print(f"\n  Optimizing {architecture} hyperparameters ({N_OPTUNA_TRIALS} trials)...")

    # Mask to training subjects only
    train_mask = np.isin(subject_ids, train_subjects)
    X_train = X[train_mask]
    y_train = y[train_mask]

    # Normalize globally (fit on train)
    scaler = StandardScaler()
    # Reshape to (n_samples, n_features) for StandardScaler
    n_samples, n_channels, seq_len = X_train.shape
    X_train_flat = X_train.reshape(n_samples, -1)
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    X_train_scaled = X_train_scaled_flat.reshape(n_samples, n_channels, seq_len)

    def objective(trial):
        # Get hyperparameters
        if architecture == 'Conv1D':
            params = get_conv1d_search_space(trial)
        else:
            params = get_bilstm_search_space(trial)

        # 5-fold stratified CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        fold_scores = []

        for train_idx, val_idx in skf.split(X_train_scaled, y_train):
            X_fold_train = X_train_scaled[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train_scaled[val_idx]
            y_fold_val = y_train[val_idx]

            # Create datasets
            train_dataset = RawSignalDataset(X_fold_train, y_fold_train)
            val_dataset = RawSignalDataset(X_fold_val, y_fold_val)

            train_loader = DataLoader(
                train_dataset, batch_size=params['batch_size'],
                shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, batch_size=params['batch_size'],
                shuffle=False, num_workers=0
            )

            # Create model
            if architecture == 'Conv1D':
                model = Conv1DClassifier(
                    n_channels=N_CHANNELS,
                    n_classes=N_CLASSES,
                    hidden_dim=params['hidden_dim'],
                    n_conv_layers=params['n_conv_layers'],
                    kernel_size=params['kernel_size'],
                    dropout=params['dropout']
                ).to(DEVICE)
            else:
                model = BiLSTMClassifier(
                    n_channels=N_CHANNELS,
                    n_classes=N_CLASSES,
                    hidden_dim=params['hidden_dim'],
                    n_layers=params['n_layers'],
                    dropout=params['dropout']
                ).to(DEVICE)

            # Train
            model, _, _ = train_model(
                model, train_loader, val_loader,
                learning_rate=params['learning_rate'],
                weight_decay=params['weight_decay'],
                max_epochs=MAX_EPOCHS,
                patience=EARLY_STOP_PATIENCE
            )

            # Evaluate
            criterion = nn.CrossEntropyLoss()
            _, val_ba, _, _ = evaluate(model, val_loader, criterion)
            fold_scores.append(val_ba)

            # Clear memory
            del model, train_dataset, val_dataset, train_loader, val_loader
            clear_memory()

        return np.mean(fold_scores)

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_SEED)
    )

    # Optimize
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False, n_jobs=1)

    print(f"  Best inner CV score: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'n_trials': len(study.trials)
    }


# =============================================================================
# LOSO Cross-Validation
# =============================================================================

def run_loso_fold(
    architecture: str,
    fold_idx: int,
    test_subject: str,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    all_subjects: List[str],
    best_params: Dict
) -> Dict:
    """
    Run a single LOSO fold.

    Parameters
    ----------
    architecture : str
        'Conv1D' or 'BiLSTM'
    fold_idx : int
        Fold index
    test_subject : str
        Subject to hold out
    X, y, subject_ids : np.ndarray
        Full dataset
    all_subjects : List[str]
        All subject IDs
    best_params : Dict
        Best hyperparameters from Optuna

    Returns
    -------
    Dict
        Fold results
    """
    n_subjects = len(all_subjects)
    fold_start = datetime.now()

    print(f"\n{'='*60}")
    print(f"LOSO FOLD {fold_idx + 1}/{n_subjects}: {architecture} - Test Subject = {test_subject}")
    print(f"{'='*60}")

    # Split data
    train_mask = subject_ids != test_subject
    test_mask = subject_ids == test_subject

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"  Train: {len(X_train)} samples from {len(np.unique(subject_ids[train_mask]))} subjects")
    print(f"  Test: {len(X_test)} samples from subject {test_subject}")

    # Global z-score normalization
    scaler = StandardScaler()
    n_train, n_ch, seq_len = X_train.shape
    X_train_flat = X_train.reshape(n_train, -1)
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    X_train_scaled = X_train_scaled_flat.reshape(n_train, n_ch, seq_len)

    n_test = len(X_test)
    X_test_flat = X_test.reshape(n_test, -1)
    X_test_scaled_flat = scaler.transform(X_test_flat)
    X_test_scaled = X_test_scaled_flat.reshape(n_test, n_ch, seq_len)

    # Create datasets
    train_dataset = RawSignalDataset(X_train_scaled, y_train)
    test_dataset = RawSignalDataset(X_test_scaled, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=best_params['batch_size'],
        shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=best_params['batch_size'],
        shuffle=False, num_workers=0
    )

    # Create model
    if architecture == 'Conv1D':
        model = Conv1DClassifier(
            n_channels=N_CHANNELS,
            n_classes=N_CLASSES,
            hidden_dim=best_params['hidden_dim'],
            n_conv_layers=best_params['n_conv_layers'],
            kernel_size=best_params['kernel_size'],
            dropout=best_params['dropout']
        ).to(DEVICE)
    else:
        model = BiLSTMClassifier(
            n_channels=N_CHANNELS,
            n_classes=N_CLASSES,
            hidden_dim=best_params['hidden_dim'],
            n_layers=best_params['n_layers'],
            dropout=best_params['dropout']
        ).to(DEVICE)

    print(f"  Training {architecture} with best hyperparameters...")

    # Train
    model, train_losses, val_losses = train_model(
        model, train_loader, test_loader,
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        max_epochs=MAX_EPOCHS,
        patience=EARLY_STOP_PATIENCE
    )

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    test_loss, test_ba, y_true, y_pred = evaluate(model, test_loader, criterion)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    fold_duration = (datetime.now() - fold_start).total_seconds()

    print(f"\n  FOLD RESULTS:")
    print(f"    Balanced Accuracy: {test_ba:.4f}")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    F1 (weighted): {f1:.4f}")
    print(f"    Fold duration: {format_duration(fold_duration)}")

    # Compile results
    fold_result = {
        'architecture': architecture,
        'fold_idx': fold_idx,
        'test_subject': test_subject,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'best_params': best_params,
        'metrics': {
            'accuracy': accuracy,
            'balanced_accuracy': test_ba,
            'f1_weighted': f1
        },
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'training_epochs': len(train_losses),
        'fold_duration_seconds': fold_duration
    }

    # Clean up
    del model, train_dataset, test_dataset, train_loader, test_loader
    clear_memory()

    return fold_result


def run_architecture_experiment(
    architecture: str,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    resume: bool = False
) -> Dict:
    """
    Run full experiment for one architecture.

    Returns
    -------
    Dict
        Architecture experiment results
    """
    print(f"\n{'='*70}")
    print(f"ARCHITECTURE: {architecture}")
    print(f"{'='*70}")

    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'fold_results' / architecture).mkdir(parents=True, exist_ok=True)

    # Get all subjects
    all_subjects = sorted(np.unique(subject_ids))
    n_subjects = len(all_subjects)

    print(f"Total subjects: {n_subjects}")

    # Load checkpoint if resuming
    if resume:
        checkpoint = load_checkpoint(RESULTS_DIR)
        current_arch = checkpoint.get('current_architecture')

        if current_arch == architecture:
            completed_folds = set(checkpoint.get('fold_results', {}).get(architecture, {}).keys())
            fold_results = checkpoint.get('fold_results', {}).get(architecture, {})
            best_hp_results = checkpoint.get('best_hyperparameters', {}).get(architecture)
            print(f"  Resuming: {len(completed_folds)} folds completed")
        else:
            completed_folds = set()
            fold_results = {}
            best_hp_results = None
    else:
        completed_folds = set()
        fold_results = {}
        best_hp_results = None

    # Step 1: Optimize hyperparameters (if not already done)
    if best_hp_results is None:
        print("\n" + "="*60)
        print("STEP 1: HYPERPARAMETER OPTIMIZATION")
        print("="*60)

        best_hp_results = optimize_hyperparameters(
            architecture, X, y, subject_ids, all_subjects
        )

        # Save checkpoint with HP results
        checkpoint = load_checkpoint(RESULTS_DIR) if resume else {}
        checkpoint['current_architecture'] = architecture
        if 'best_hyperparameters' not in checkpoint:
            checkpoint['best_hyperparameters'] = {}
        checkpoint['best_hyperparameters'][architecture] = best_hp_results
        save_checkpoint(RESULTS_DIR, checkpoint)

    best_params = best_hp_results['best_params']

    # Step 2: LOSO cross-validation
    print("\n" + "="*60)
    print("STEP 2: LOSO CROSS-VALIDATION")
    print("="*60)

    experiment_start = datetime.now()

    for fold_idx, test_subject in enumerate(all_subjects):
        # Skip if already completed
        if test_subject in completed_folds:
            print(f"\n[SKIP] Fold {fold_idx + 1}/{n_subjects} (subject {test_subject}) - already completed")
            continue

        try:
            # Run fold
            fold_result = run_loso_fold(
                architecture, fold_idx, test_subject,
                X, y, subject_ids, all_subjects, best_params
            )

            # Save fold result
            fold_results[test_subject] = fold_result
            completed_folds.add(test_subject)
            save_fold_result(RESULTS_DIR, architecture, test_subject, fold_result)

            # Save checkpoint
            checkpoint = load_checkpoint(RESULTS_DIR)
            checkpoint['current_architecture'] = architecture
            if 'fold_results' not in checkpoint:
                checkpoint['fold_results'] = {}
            checkpoint['fold_results'][architecture] = fold_results
            save_checkpoint(RESULTS_DIR, checkpoint)

            # Progress estimate
            elapsed = (datetime.now() - experiment_start).total_seconds()
            avg_fold_time = elapsed / len(completed_folds)
            remaining_folds = n_subjects - len(completed_folds)
            eta_seconds = avg_fold_time * remaining_folds

            print(f"\n  PROGRESS: {len(completed_folds)}/{n_subjects} folds complete")
            print(f"  ETA: {format_duration(eta_seconds)} remaining")

        except Exception as e:
            print(f"\n[ERROR] Fold {fold_idx + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        clear_memory()

    # Compile results
    total_duration = (datetime.now() - experiment_start).total_seconds()

    results = {
        'architecture': architecture,
        'best_hyperparameters': best_hp_results,
        'n_subjects': n_subjects,
        'fold_results': fold_results,
        'total_duration_seconds': total_duration
    }

    return results


# =============================================================================
# Results Aggregation and Reporting
# =============================================================================

def aggregate_architecture_results(results: Dict) -> Dict:
    """Aggregate fold results for one architecture."""
    fold_results = results['fold_results']

    # Extract metrics
    balanced_accs = []
    accuracies = []
    f1_scores = []
    all_y_true = []
    all_y_pred = []

    for subject_id, fold_data in fold_results.items():
        metrics = fold_data['metrics']
        balanced_accs.append(metrics['balanced_accuracy'])
        accuracies.append(metrics['accuracy'])
        f1_scores.append(metrics['f1_weighted'])
        all_y_true.extend(fold_data['y_true'])
        all_y_pred.extend(fold_data['y_pred'])

    # Summary statistics
    ba_mean = np.mean(balanced_accs)
    ba_std = np.std(balanced_accs)
    ba_median = np.median(balanced_accs)

    # 95% CI
    n = len(balanced_accs)
    ci_lower = ba_mean - 1.96 * ba_std / np.sqrt(n) if n > 0 else 0
    ci_upper = ba_mean + 1.96 * ba_std / np.sqrt(n) if n > 0 else 0

    # Statistical test vs Paper 1
    if n > 1:
        t_stat, p_value = stats.ttest_1samp(balanced_accs, PAPER1_LOSO_BASELINE)
        cohens_d = (ba_mean - PAPER1_LOSO_BASELINE) / ba_std if ba_std > 0 else 0
    else:
        t_stat, p_value, cohens_d = 0, 1.0, 0

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)

    summary = {
        'n_folds': n,
        'balanced_accuracy': {
            'mean': ba_mean,
            'std': ba_std,
            'median': ba_median,
            'min': np.min(balanced_accs) if n > 0 else 0,
            'max': np.max(balanced_accs) if n > 0 else 0,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper
        },
        'accuracy': {
            'mean': np.mean(accuracies) if n > 0 else 0,
            'std': np.std(accuracies) if n > 0 else 0
        },
        'f1_weighted': {
            'mean': np.mean(f1_scores) if n > 0 else 0,
            'std': np.std(f1_scores) if n > 0 else 0
        },
        'vs_paper1': {
            'paper1_baseline': PAPER1_LOSO_BASELINE,
            'improvement': ba_mean - PAPER1_LOSO_BASELINE,
            'improvement_pct': (ba_mean - PAPER1_LOSO_BASELINE) / PAPER1_LOSO_BASELINE * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'beats_baseline': ba_mean > PAPER1_LOSO_BASELINE
        },
        'confusion_matrix': cm.tolist(),
        'per_fold_balanced_accs': balanced_accs
    }

    return summary


def generate_report(all_results: Dict, all_summaries: Dict) -> str:
    """Generate markdown report."""

    # Find best architecture
    best_arch = max(all_summaries.keys(),
                   key=lambda k: all_summaries[k]['balanced_accuracy']['mean'])
    best_ba = all_summaries[best_arch]['balanced_accuracy']
    best_vs_p1 = all_summaries[best_arch]['vs_paper1']

    report = f"""# Phase 8.1: Raw Signal Deep Learning - Final Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Best Architecture** | {best_arch} |
| **LOSO Balanced Accuracy** | {best_ba['mean']:.4f} +/- {best_ba['std']:.4f} |
| **95% CI** | [{best_ba['ci_95_lower']:.4f}, {best_ba['ci_95_upper']:.4f}] |
| **Paper 1 Baseline** | {best_vs_p1['paper1_baseline']:.4f} |
| **Improvement** | {best_vs_p1['improvement']:+.4f} ({best_vs_p1['improvement_pct']:+.2f}%) |
| **Statistical Significance** | p = {best_vs_p1['p_value']:.4f} ({'Yes' if best_vs_p1['significant'] else 'No'}) |

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Input Signals | {', '.join(SIGNALS)} |
| Input Channels | {N_CHANNELS} |
| Max Signal Length | {MAX_SIGNAL_LENGTH} |
| Architectures Tested | {', '.join(ARCHITECTURES)} |
| Optuna Trials | {N_OPTUNA_TRIALS} |
| Max Epochs | {MAX_EPOCHS} |
| Early Stop Patience | {EARLY_STOP_PATIENCE} |
| Device | {DEVICE.type} |

---

## Results by Architecture

"""

    for arch in ARCHITECTURES:
        if arch not in all_summaries:
            continue

        summary = all_summaries[arch]
        ba = summary['balanced_accuracy']
        vs_p1 = summary['vs_paper1']

        report += f"""### {arch}

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Balanced Accuracy | {ba['mean']:.4f} | {ba['std']:.4f} | {ba['min']:.4f} | {ba['max']:.4f} |
| Accuracy | {summary['accuracy']['mean']:.4f} | {summary['accuracy']['std']:.4f} | - | - |
| F1 (weighted) | {summary['f1_weighted']['mean']:.4f} | {summary['f1_weighted']['std']:.4f} | - | - |

**vs Paper 1:** {vs_p1['improvement']:+.4f} ({vs_p1['improvement_pct']:+.2f}%)
**p-value:** {vs_p1['p_value']:.4f} ({'significant' if vs_p1['significant'] else 'not significant'})

"""

    report += f"""---

## Best Model: {best_arch}

### Hyperparameters

```json
{json.dumps(all_results[best_arch]['best_hyperparameters']['best_params'], indent=2)}
```

### Confusion Matrix

```
              Predicted
              no_pain  low_pain  high_pain
Actual
no_pain         {all_summaries[best_arch]['confusion_matrix'][0][0]:4d}      {all_summaries[best_arch]['confusion_matrix'][0][1]:4d}       {all_summaries[best_arch]['confusion_matrix'][0][2]:4d}
low_pain        {all_summaries[best_arch]['confusion_matrix'][1][0]:4d}      {all_summaries[best_arch]['confusion_matrix'][1][1]:4d}       {all_summaries[best_arch]['confusion_matrix'][1][2]:4d}
high_pain       {all_summaries[best_arch]['confusion_matrix'][2][0]:4d}      {all_summaries[best_arch]['confusion_matrix'][2][1]:4d}       {all_summaries[best_arch]['confusion_matrix'][2][2]:4d}
```

---

## Per-Subject Results ({best_arch})

| Subject | Balanced Acc | vs Baseline |
|---------|--------------|-------------|
"""

    # Sort by balanced accuracy
    sorted_folds = sorted(
        all_results[best_arch]['fold_results'].items(),
        key=lambda x: x[1]['metrics']['balanced_accuracy'],
        reverse=True
    )

    for subject_id, fold_data in sorted_folds:
        ba_fold = fold_data['metrics']['balanced_accuracy']
        diff = ba_fold - PAPER1_LOSO_BASELINE
        report += f"| {subject_id} | {ba_fold:.4f} | {diff:+.4f} |\n"

    report += """

---

## Conclusion

"""

    if best_vs_p1['beats_baseline']:
        report += f"""**Raw signal deep learning achieved {best_ba['mean']:.2%} balanced accuracy**
with the {best_arch} architecture, which is {best_vs_p1['improvement']:+.2%} percentage points
above Paper 1's baseline of {best_vs_p1['paper1_baseline']:.2%}.

This demonstrates that temporal patterns in raw physiological signals can be effectively
captured by deep learning models for pain classification.

The result is {'statistically significant' if best_vs_p1['significant'] else 'not statistically significant'}
(p = {best_vs_p1['p_value']:.4f}).
"""
    else:
        report += f"""Raw signal deep learning achieved {best_ba['mean']:.2%} balanced accuracy
with the {best_arch} architecture, which is {abs(best_vs_p1['improvement']):.2%} percentage points
below Paper 1's baseline of {best_vs_p1['paper1_baseline']:.2%}.

This suggests that entropy-complexity features may capture more discriminative information
than raw temporal patterns for this pain classification task.
"""

    report += """

---

**End of Report**

*Generated by Phase 8.1 Raw Signal Deep Learning Pipeline*
"""

    return report


def generate_outputs(all_results: Dict, all_summaries: Dict):
    """Generate all output files."""
    print("\n" + "="*70)
    print("GENERATING OUTPUTS")
    print("="*70)

    # 1. Leaderboard CSV
    print("\nSaving leaderboard...")
    leaderboard_data = []
    for rank, (arch, summary) in enumerate(
        sorted(all_summaries.items(),
               key=lambda x: x[1]['balanced_accuracy']['mean'],
               reverse=True), 1
    ):
        leaderboard_data.append({
            'rank': rank,
            'architecture': arch,
            'loso_balanced_accuracy_mean': summary['balanced_accuracy']['mean'],
            'loso_balanced_accuracy_std': summary['balanced_accuracy']['std'],
            'ci_95_lower': summary['balanced_accuracy']['ci_95_lower'],
            'ci_95_upper': summary['balanced_accuracy']['ci_95_upper'],
            'vs_paper1': summary['vs_paper1']['improvement'],
            'p_value': summary['vs_paper1']['p_value']
        })
    pd.DataFrame(leaderboard_data).to_csv(RESULTS_DIR / 'loso_leaderboard.csv', index=False)
    print("  Saved: loso_leaderboard.csv")

    # 2. Per-subject results CSV
    print("Saving per-subject results...")
    per_subject_data = []
    for arch, results in all_results.items():
        for subject_id, fold_data in results['fold_results'].items():
            per_subject_data.append({
                'architecture': arch,
                'subject_id': subject_id,
                'balanced_accuracy': fold_data['metrics']['balanced_accuracy'],
                'accuracy': fold_data['metrics']['accuracy'],
                'f1_weighted': fold_data['metrics']['f1_weighted'],
                'n_samples': fold_data['n_test_samples'],
                'training_epochs': fold_data['training_epochs']
            })
    pd.DataFrame(per_subject_data).to_csv(RESULTS_DIR / 'per_subject_results.csv', index=False)
    print("  Saved: per_subject_results.csv")

    # 3. Best hyperparameters JSON
    print("Saving hyperparameters...")
    all_hp = {
        arch: results['best_hyperparameters']
        for arch, results in all_results.items()
    }
    with open(RESULTS_DIR / 'best_hyperparameters.json', 'w') as f:
        json.dump(convert_to_serializable(all_hp), f, indent=2)
    print("  Saved: best_hyperparameters.json")

    # 4. Confusion matrix plot (for best architecture)
    print("Generating confusion matrix plot...")
    best_arch = max(all_summaries.keys(),
                   key=lambda k: all_summaries[k]['balanced_accuracy']['mean'])
    cm = np.array(all_summaries[best_arch]['confusion_matrix'])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ba_mean = all_summaries[best_arch]['balanced_accuracy']['mean']
    ax.set_title(f'{best_arch} LOSO Confusion Matrix\nBalanced Accuracy: {ba_mean:.2%}',
                fontsize=14)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j + 0.5, i + 0.75, f'n={cm[i,j]}',
                   ha='center', va='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: confusion_matrix.png")

    # 5. Report
    print("Generating report...")
    report = generate_report(all_results, all_summaries)
    with open(RESULTS_DIR / 'phase8_1_report.md', 'w') as f:
        f.write(report)
    print("  Saved: phase8_1_report.md")

    # 6. Full results JSON
    print("Saving full results...")
    with open(RESULTS_DIR / 'full_results.json', 'w') as f:
        json.dump(convert_to_serializable({
            'all_results': all_results,
            'all_summaries': all_summaries
        }), f, indent=2)
    print("  Saved: full_results.json")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Phase 8.1: Raw Signal Deep Learning')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    print("="*70)
    print("PHASE 8.1: RAW SIGNAL DEEP LEARNING")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Target: Beat Paper 1 LOSO baseline of {PAPER1_LOSO_BASELINE:.1%}")

    # Load segment metadata
    metadata = load_segment_metadata()

    # Create multi-channel dataset
    X, y, subject_ids = create_multimodal_dataset(metadata)

    print(f"\nDataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Subjects: {len(np.unique(subject_ids))}")

    # Run experiments for each architecture
    all_results = {}

    # Load checkpoint to determine which architectures are complete
    if args.resume:
        checkpoint = load_checkpoint(RESULTS_DIR)
        completed_architectures = set(checkpoint.get('completed_architectures', []))
    else:
        completed_architectures = set()

    for architecture in ARCHITECTURES:
        # Skip if already completed
        if architecture in completed_architectures:
            print(f"\n[SKIP] {architecture} - already completed")
            # Load results from checkpoint
            checkpoint = load_checkpoint(RESULTS_DIR)
            all_results[architecture] = {
                'architecture': architecture,
                'best_hyperparameters': checkpoint.get('best_hyperparameters', {}).get(architecture, {}),
                'fold_results': checkpoint.get('fold_results', {}).get(architecture, {}),
                'n_subjects': len(np.unique(subject_ids))
            }
            continue

        # Run architecture experiment
        results = run_architecture_experiment(
            architecture, X, y, subject_ids, resume=args.resume
        )
        all_results[architecture] = results

        # Mark as completed
        checkpoint = load_checkpoint(RESULTS_DIR)
        if 'completed_architectures' not in checkpoint:
            checkpoint['completed_architectures'] = []
        checkpoint['completed_architectures'].append(architecture)
        save_checkpoint(RESULTS_DIR, checkpoint)

        clear_memory()

    # Aggregate results
    print("\n" + "="*70)
    print("AGGREGATING RESULTS")
    print("="*70)

    all_summaries = {}
    for arch, results in all_results.items():
        all_summaries[arch] = aggregate_architecture_results(results)

    # Generate outputs
    generate_outputs(all_results, all_summaries)

    # Final summary
    best_arch = max(all_summaries.keys(),
                   key=lambda k: all_summaries[k]['balanced_accuracy']['mean'])
    best_ba = all_summaries[best_arch]['balanced_accuracy']
    best_vs_p1 = all_summaries[best_arch]['vs_paper1']

    print("\n" + "="*70)
    print("PHASE 8.1 COMPLETE")
    print("="*70)

    print(f"\nBEST ARCHITECTURE: {best_arch}")
    print(f"  LOSO Balanced Accuracy: {best_ba['mean']:.4f} +/- {best_ba['std']:.4f}")
    print(f"  95% CI: [{best_ba['ci_95_lower']:.4f}, {best_ba['ci_95_upper']:.4f}]")
    print(f"  Paper 1 Baseline: {best_vs_p1['paper1_baseline']:.4f}")
    print(f"  Improvement: {best_vs_p1['improvement']:+.4f} ({best_vs_p1['improvement_pct']:+.2f}%)")
    print(f"  p-value: {best_vs_p1['p_value']:.6f}")

    if best_vs_p1['beats_baseline']:
        print(f"\n  SUCCESS: BEAT PAPER 1 BASELINE!")
    else:
        print(f"\n  Result below Paper 1 baseline")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
