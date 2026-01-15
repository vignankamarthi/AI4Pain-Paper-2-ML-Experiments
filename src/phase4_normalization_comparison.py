#!/usr/bin/env python3
"""
Phase 4 Experiment 4.1: Normalization Comparison

Tests three normalization strategies on three representative models:
- Normalization: Per-subject baseline, Global z-score, Raw features
- Models: LightGBM (tree-based), Simple MLP (neural), Stacked (ensemble)

Uses 5-fold cross-validation on combined train+validation data (53 subjects).
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.metrics import balanced_accuracy_score, make_scorer
import lightgbm as lgb
import xgboost as xgb

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
# DataLoader removed for simplicity - using manual batching for stability

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase4" / "experiment_4.1_normalization"

# Feature configuration
SIGNALS = ['eda', 'bvp', 'resp']  # Exclude SpO2 (lowercase to match file names)
FEATURE_COLS = ['pe', 'comp', 'fisher_shannon', 'fisher_info',
                'renyipe', 'renyicomp', 'tsallispe', 'tsalliscomp']

# Best (dimension, tau) from Stage 0
BEST_DIMENSION = 7
BEST_TAU = 2

# Class mapping (state values in CSV are: baseline, rest, low, high)
CLASS_MAP = {'baseline': 0, 'rest': 0, 'low': 1, 'high': 2}

# Cross-validation
N_CV_FOLDS = 5
RANDOM_STATE = 42

# Neural network settings
DEVICE_LOG = {}  # Track which device worked for each model


def test_mps_compatibility():
    """
    Test MPS (Metal Performance Shaders) compatibility and return status report.

    Returns
    -------
    dict
        MPS compatibility report with test results.
    """
    report = {
        'mps_available': False,
        'mps_test_passed': False,
        'mps_practical': False,
        'reason': '',
        'recommendation': 'cpu'
    }

    if not torch.backends.mps.is_available():
        report['reason'] = 'MPS backend not available on this system'
        return report

    report['mps_available'] = True

    try:
        # Quick MPS test
        device = torch.device('mps')
        test_model = nn.Sequential(
            nn.Linear(24, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        ).to(device)
        test_X = torch.randn(100, 24).to(device)
        test_y = torch.randint(0, 3, (100,)).to(device)

        optimizer = torch.optim.Adam(test_model.parameters())
        criterion = nn.CrossEntropyLoss()

        for _ in range(5):
            optimizer.zero_grad()
            out = test_model(test_X)
            loss = criterion(out, test_y)
            loss.backward()
            optimizer.step()

        del test_model, test_X, test_y, optimizer
        report['mps_test_passed'] = True
        report['reason'] = 'MPS quick test passed but may hang on larger training'
        report['mps_practical'] = False  # Known issue with variable batch training
        report['recommendation'] = 'cpu (MPS hangs on extended training)'

    except Exception as e:
        report['reason'] = f'MPS test failed: {str(e)[:100]}'
        report['recommendation'] = 'cpu'

    return report


def get_device():
    """
    Get PyTorch device - uses CPU for stability, logs MPS status.

    Returns
    -------
    torch.device
        CPU device (for stability).
    str
        Device status message for logging.
    """
    # Use CPU for stability - MPS has known issues with variable batch training
    return torch.device('cpu'), 'cpu (stable mode)'

# Global MPS report - set during initialization
MPS_REPORT = None


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data():
    """
    Load feature data from train and validation sets for EDA, BVP, RESP.

    Filters to best (dimension, tau) = (7, 2) from Stage 0.

    Returns
    -------
    pd.DataFrame
        Combined dataframe with all features and metadata.
    """
    dfs = []

    for split in ['train', 'validation']:
        for signal in SIGNALS:
            # File names have capitalized signal names
            signal_cap = signal.capitalize()
            filepath = DATA_DIR / f"results_{split}_{signal_cap}.csv"

            if filepath.exists():
                df = pd.read_csv(filepath)

                # Filter to best (d, tau) from Stage 0
                df = df[(df['dimension'] == BEST_DIMENSION) & (df['tau'] == BEST_TAU)].copy()

                # Add signal type identifier
                df['signal_type'] = signal
                df['split'] = split

                # Extract subject ID from file_name
                df['subject_id'] = df['file_name'].apply(
                    lambda x: int(x.split('/')[-1].replace('.csv', ''))
                )

                dfs.append(df)
            else:
                print(f"Warning: {filepath} not found")

    combined = pd.concat(dfs, ignore_index=True)

    return combined


def pivot_to_feature_matrix(df):
    """
    Pivot data from long format to wide feature matrix.

    Creates one row per sample with signal-prefixed feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with signal_type column.

    Returns
    -------
    pd.DataFrame
        Wide-format dataframe.
    np.ndarray
        Feature matrix (n_samples x n_features).
    np.ndarray
        Labels array.
    np.ndarray
        Subject IDs array.
    list
        Feature column names.
    """
    # Create unique sample identifier using segment ID (in 'signal' column) and subject
    df['sample_id'] = df['signal'] + '_' + df['subject_id'].astype(str)

    wide_dfs = []

    for signal in SIGNALS:
        signal_df = df[df['signal_type'] == signal].copy()

        # Select only feature columns and metadata
        cols_to_keep = ['sample_id', 'subject_id', 'state', 'split'] + FEATURE_COLS
        signal_df = signal_df[cols_to_keep].copy()

        # Rename feature columns with signal prefix
        rename_dict = {col: f'{signal}_{col}' for col in FEATURE_COLS}
        signal_df = signal_df.rename(columns=rename_dict)

        wide_dfs.append(signal_df)

    # Merge all signals on sample_id
    result_df = wide_dfs[0]
    for wdf in wide_dfs[1:]:
        # Only keep feature columns from subsequent merges
        feature_cols = [c for c in wdf.columns if c not in ['sample_id', 'subject_id', 'state', 'split']]
        merge_cols = ['sample_id'] + feature_cols
        result_df = result_df.merge(wdf[merge_cols], on='sample_id', how='inner')

    # Get feature columns
    feature_columns = [f"{s}_{f}" for s in SIGNALS for f in FEATURE_COLS]

    # Extract arrays
    X = result_df[feature_columns].values.astype(np.float64)
    y = result_df['state'].map(CLASS_MAP).values.astype(np.int64)
    subjects = result_df['subject_id'].values

    return result_df, X, y, subjects, feature_columns


# =============================================================================
# NORMALIZATION STRATEGIES
# =============================================================================

def normalize_per_subject_baseline(X, y, subjects, feature_columns):
    """
    Per-subject baseline normalization.

    For each subject, normalize features relative to their no-pain (baseline/rest) samples.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels (0 = no-pain, 1 = low, 2 = high).
    subjects : np.ndarray
        Subject IDs.
    feature_columns : list
        Feature column names.

    Returns
    -------
    np.ndarray
        Normalized feature matrix.
    """
    X_normalized = np.zeros_like(X, dtype=np.float64)

    for subj in np.unique(subjects):
        subj_mask = subjects == subj
        subj_indices = np.where(subj_mask)[0]

        # Get no-pain samples for this subject
        no_pain_mask = (subjects == subj) & (y == 0)
        no_pain_indices = np.where(no_pain_mask)[0]

        if len(no_pain_indices) > 0:
            baseline_mean = X[no_pain_indices].mean(axis=0)
            baseline_std = X[no_pain_indices].std(axis=0)
            baseline_std[baseline_std < 1e-8] = 1.0  # Avoid division by zero

            # Normalize all samples for this subject
            for idx in subj_indices:
                X_normalized[idx] = (X[idx] - baseline_mean) / baseline_std
        else:
            # Fallback: use subject's own mean/std
            subj_X = X[subj_indices]
            mean = subj_X.mean(axis=0)
            std = subj_X.std(axis=0)
            std[std < 1e-8] = 1.0
            for idx in subj_indices:
                X_normalized[idx] = (X[idx] - mean) / std

    return X_normalized


def normalize_global_zscore(X, y, subjects, feature_columns):
    """
    Global z-score normalization (Paper 1 style).

    Normalize each feature across ALL samples using global mean and std.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels (unused, kept for consistent API).
    subjects : np.ndarray
        Subject IDs (unused, kept for consistent API).
    feature_columns : list
        Feature column names (unused, kept for consistent API).

    Returns
    -------
    np.ndarray
        Normalized feature matrix.
    """
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized


def normalize_raw(X, y, subjects, feature_columns):
    """
    No normalization - use raw feature values.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels (unused).
    subjects : np.ndarray
        Subject IDs (unused).
    feature_columns : list
        Feature column names (unused).

    Returns
    -------
    np.ndarray
        Original feature matrix (unchanged).
    """
    return X.copy()


NORMALIZATION_METHODS = {
    'per_subject_baseline': normalize_per_subject_baseline,
    'global_zscore': normalize_global_zscore,
    'raw': normalize_raw,
}


# =============================================================================
# NEURAL NETWORK MODEL
# =============================================================================

class SimpleMLP(nn.Module):
    """
    Simple 2-layer MLP for 3-class classification.

    Architecture based on Phase 2 best hyperparameters, adjusted for 24 features.
    """

    def __init__(self, input_dim=24, layer1=256, layer2=32, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, layer1),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(layer1, layer2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(layer2, 3)  # 3 classes
        )

    def forward(self, x):
        return self.network(x)


class SimpleMLP_SKLearn:
    """
    Scikit-learn compatible wrapper for SimpleMLP.

    Enables use with cross_val_score and other sklearn utilities.
    """

    def __init__(self, input_dim=24, layer1=256, layer2=32, dropout=0.3,
                 lr=0.005, batch_size=16, epochs=30, device=None):  # 30 epochs for comparison
        self.input_dim = input_dim
        self.layer1 = layer1
        self.layer2 = layer2
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.model = None
        self.device_used = None

    def fit(self, X, y):
        """Train the model using simple batch training (no DataLoader for stability)."""
        import sys

        # Determine device
        if self.device is None:
            self.device, self.device_used = get_device()
        else:
            self.device_used = str(self.device)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # Initialize model
        self.model = SimpleMLP(
            input_dim=self.input_dim,
            layer1=self.layer1,
            layer2=self.layer2,
            dropout=self.dropout
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        n_samples = X_tensor.shape[0]
        n_batches = max(1, n_samples // self.batch_size)

        # Training loop - simple batch iteration without DataLoader
        self.model.train()
        for epoch in range(self.epochs):
            # Shuffle indices each epoch
            indices = torch.randperm(n_samples)

            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_X = X_tensor[batch_indices]
                batch_y = y_tensor[batch_indices]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'input_dim': self.input_dim,
            'layer1': self.layer1,
            'layer2': self.layer2,
            'dropout': self.dropout,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': self.device
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


def create_simple_mlp_with_fallback(input_dim=24):
    """
    Create SimpleMLP with CPU device (MPS disabled for stability).

    Parameters
    ----------
    input_dim : int
        Number of input features.

    Returns
    -------
    SimpleMLP_SKLearn
        Configured MLP model.
    str
        Device name for logging.
    """
    device = torch.device('cpu')
    device_name = 'cpu (stable mode)'

    model = SimpleMLP_SKLearn(
        input_dim=input_dim,
        layer1=256,
        layer2=32,
        dropout=0.3,
        lr=0.005,
        batch_size=16,
        epochs=30,
        device=device
    )

    return model, device_name


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_lightgbm():
    """
    Get LightGBM classifier with Phase 3 best hyperparameters.

    Returns
    -------
    lgb.LGBMClassifier
        Configured LightGBM model.
    """
    return lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=-1,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=0,
        random_state=RANDOM_STATE,
        verbose=-1,
        force_col_wise=True
    )


def get_stacked_ensemble():
    """
    Get Stacked Ensemble (RF + XGB + LGB) with 3-fold internal CV.

    Returns
    -------
    StackingClassifier
        Configured stacking ensemble.
    """
    # Use n_jobs=1 to avoid multiprocessing deadlocks in subprocess execution
    base_estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            n_jobs=1  # Avoid multiprocessing deadlock
        )),
        ('xgb', xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0,
            n_jobs=1  # Avoid multiprocessing deadlock
        )),
        ('lgb', lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=-1,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            verbose=-1,
            force_col_wise=True,
            n_jobs=1  # Avoid multiprocessing deadlock
        ))
    ]

    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=lgb.LGBMClassifier(
            n_estimators=50,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            verbose=-1,
            force_col_wise=True,
            n_jobs=1  # Avoid multiprocessing deadlock
        ),
        cv=3,  # Reduced from 5 to 3 per Decision 4
        n_jobs=1,  # Avoid multiprocessing deadlock
        passthrough=False
    )


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def run_normalization_experiment(X, y, subjects, feature_columns):
    """
    Run the full normalization comparison experiment.

    Parameters
    ----------
    X : np.ndarray
        Raw feature matrix.
    y : np.ndarray
        Labels.
    subjects : np.ndarray
        Subject IDs.
    feature_columns : list
        Feature column names.

    Returns
    -------
    pd.DataFrame
        Results dataframe with all comparisons.
    dict
        Summary of best configurations.
    """
    results = []
    n_features = X.shape[1]

    # Get neural network device info
    nn_device_name = None

    print("=" * 70)
    print("EXPERIMENT 4.1: NORMALIZATION COMPARISON")
    print("=" * 70)
    print(f"Features: {n_features}")
    print(f"Samples: {X.shape[0]}")
    print(f"Cross-validation: {N_CV_FOLDS}-fold")
    print()

    # Balanced accuracy scorer
    scorer = make_scorer(balanced_accuracy_score)

    for norm_name, norm_func in NORMALIZATION_METHODS.items():
        print(f"\n{'='*50}")
        print(f"Normalization: {norm_name}")
        print('='*50)

        # Apply normalization
        X_norm = norm_func(X, y, subjects, feature_columns)

        # Handle any NaN values from normalization
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Cross-validation setup
        cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        # 1. LightGBM
        print("\n  LightGBM...")
        lgbm_model = get_lightgbm()
        lgbm_scores = cross_val_score(lgbm_model, X_norm, y, cv=cv, scoring=scorer)
        lgbm_mean = lgbm_scores.mean() * 100
        lgbm_std = lgbm_scores.std() * 100
        print(f"    Balanced Accuracy: {lgbm_mean:.2f}% +/- {lgbm_std:.2f}%")

        results.append({
            'normalization': norm_name,
            'model': 'LightGBM',
            'model_type': 'tree',
            'balanced_accuracy_mean': lgbm_mean,
            'balanced_accuracy_std': lgbm_std,
            'fold_scores': lgbm_scores.tolist(),
            'device': 'cpu'
        })

        # 2. Simple MLP (with MPS fallback)
        print("\n  Simple MLP...")
        try:
            mlp_model, nn_device_name = create_simple_mlp_with_fallback(input_dim=n_features)
            print(f"    Device: {nn_device_name}")

            mlp_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_norm, y)):
                print(f"      Fold {fold_idx+1}/{N_CV_FOLDS}...", end=" ", flush=True)
                X_train, X_val = X_norm[train_idx], X_norm[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Create fresh model for each fold
                mlp_model, _ = create_simple_mlp_with_fallback(input_dim=n_features)
                mlp_model.fit(X_train, y_train)
                y_pred = mlp_model.predict(X_val)
                fold_score = balanced_accuracy_score(y_val, y_pred)
                mlp_scores.append(fold_score)
                print(f"{fold_score*100:.1f}%", flush=True)

            mlp_scores = np.array(mlp_scores)
            mlp_mean = mlp_scores.mean() * 100
            mlp_std = mlp_scores.std() * 100
            print(f"    Balanced Accuracy: {mlp_mean:.2f}% +/- {mlp_std:.2f}%")

            results.append({
                'normalization': norm_name,
                'model': 'Simple MLP',
                'model_type': 'neural',
                'balanced_accuracy_mean': mlp_mean,
                'balanced_accuracy_std': mlp_std,
                'fold_scores': mlp_scores.tolist(),
                'device': nn_device_name
            })

        except Exception as e:
            print(f"    ERROR: {e}")
            print("    Falling back to CPU...")

            # CPU fallback
            device = torch.device('cpu')
            nn_device_name = 'cpu (MPS failed)'

            mlp_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_norm, y)):
                print(f"      Fold {fold_idx+1}/{N_CV_FOLDS}...", end=" ", flush=True)
                X_train, X_val = X_norm[train_idx], X_norm[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                mlp_model = SimpleMLP_SKLearn(
                    input_dim=n_features, layer1=256, layer2=32,
                    dropout=0.3, lr=0.005, batch_size=16, epochs=30,  # Reduced for comparison
                    device=device
                )
                mlp_model.fit(X_train, y_train)
                y_pred = mlp_model.predict(X_val)
                fold_score = balanced_accuracy_score(y_val, y_pred)
                mlp_scores.append(fold_score)
                print(f"{fold_score*100:.1f}%", flush=True)

            mlp_scores = np.array(mlp_scores)
            mlp_mean = mlp_scores.mean() * 100
            mlp_std = mlp_scores.std() * 100
            print(f"    Balanced Accuracy (CPU): {mlp_mean:.2f}% +/- {mlp_std:.2f}%")

            results.append({
                'normalization': norm_name,
                'model': 'Simple MLP',
                'model_type': 'neural',
                'balanced_accuracy_mean': mlp_mean,
                'balanced_accuracy_std': mlp_std,
                'fold_scores': mlp_scores.tolist(),
                'device': nn_device_name
            })

        # 3. Stacked Ensemble
        print("\n  Stacked Ensemble (3-fold internal CV)...")
        stacked_model = get_stacked_ensemble()
        stacked_scores = cross_val_score(stacked_model, X_norm, y, cv=cv, scoring=scorer)
        stacked_mean = stacked_scores.mean() * 100
        stacked_std = stacked_scores.std() * 100
        print(f"    Balanced Accuracy: {stacked_mean:.2f}% +/- {stacked_std:.2f}%")

        results.append({
            'normalization': norm_name,
            'model': 'Stacked',
            'model_type': 'ensemble',
            'balanced_accuracy_mean': stacked_mean,
            'balanced_accuracy_std': stacked_std,
            'fold_scores': stacked_scores.tolist(),
            'device': 'cpu'
        })

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Determine best normalization
    summary = analyze_results(results_df, nn_device_name)

    return results_df, summary


def analyze_results(results_df, nn_device_name):
    """
    Analyze results to determine best normalization strategy.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from all experiments.
    nn_device_name : str
        Device used for neural networks.

    Returns
    -------
    dict
        Summary of findings and recommendations.
    """
    summary = {
        'timestamp': datetime.now().isoformat(),
        'neural_network_device': nn_device_name,
        'mps_compatibility_report': MPS_REPORT,
        'best_per_model': {},
        'best_overall': None,
        'recommendation': None
    }

    # Best normalization per model
    for model in ['LightGBM', 'Simple MLP', 'Stacked']:
        model_df = results_df[results_df['model'] == model]
        best_idx = model_df['balanced_accuracy_mean'].idxmax()
        best_row = results_df.loc[best_idx]
        summary['best_per_model'][model] = {
            'normalization': best_row['normalization'],
            'accuracy': best_row['balanced_accuracy_mean'],
            'std': best_row['balanced_accuracy_std']
        }

    # Check if all models agree
    best_norms = [v['normalization'] for v in summary['best_per_model'].values()]

    if len(set(best_norms)) == 1:
        # All agree
        summary['best_overall'] = best_norms[0]
        summary['recommendation'] = f"UNIVERSAL: Use {best_norms[0]} for all models"
    else:
        # Models disagree - find overall best
        avg_by_norm = results_df.groupby('normalization')['balanced_accuracy_mean'].mean()
        best_norm = avg_by_norm.idxmax()
        summary['best_overall'] = best_norm
        summary['recommendation'] = f"MODEL-SPECIFIC: Models prefer different normalizations. Suggest {best_norm} as default."
        summary['model_specific_note'] = {m: v['normalization'] for m, v in summary['best_per_model'].items()}

    return summary


def save_results(results_df, summary, feature_columns):
    """
    Save experiment results to files.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    summary : dict
        Summary of findings.
    feature_columns : list
        Feature column names used.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_df.to_csv(RESULTS_DIR / "normalization_comparison_results.csv", index=False)

    # Save summary
    with open(RESULTS_DIR / "normalization_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Save feature list for reference
    with open(RESULTS_DIR / "feature_columns.json", 'w') as f:
        json.dump(feature_columns, f, indent=2)

    # Generate text report
    generate_report(results_df, summary)


def generate_report(results_df, summary):
    """
    Generate human-readable report.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.
    summary : dict
        Summary of findings.
    """
    report_lines = [
        "=" * 70,
        "EXPERIMENT 4.1: NORMALIZATION COMPARISON REPORT",
        "=" * 70,
        "",
        f"Generated: {summary['timestamp']}",
        f"Neural Network Device: {summary['neural_network_device']}",
        "",
        "-" * 70,
        "MPS (Metal Performance Shaders) COMPATIBILITY",
        "-" * 70,
        ""
    ]

    mps = summary.get('mps_compatibility_report', {})
    if mps:
        report_lines.append(f"  MPS Available: {mps.get('mps_available', 'N/A')}")
        report_lines.append(f"  MPS Quick Test Passed: {mps.get('mps_test_passed', 'N/A')}")
        report_lines.append(f"  Practical for Training: {mps.get('mps_practical', 'N/A')}")
        report_lines.append(f"  Recommendation: {mps.get('recommendation', 'N/A')}")
        report_lines.append(f"  Notes: {mps.get('reason', 'N/A')}")
    else:
        report_lines.append("  MPS test not performed")

    report_lines.extend([
        "",
        "-" * 70,
        "RESULTS SUMMARY",
        "-" * 70,
        ""
    ])

    # Pivot table for display
    pivot = results_df.pivot(index='normalization', columns='model', values='balanced_accuracy_mean')
    report_lines.append("Balanced Accuracy (%) by Normalization and Model:")
    report_lines.append("")
    report_lines.append(pivot.to_string())
    report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("BEST NORMALIZATION PER MODEL")
    report_lines.append("-" * 70)
    report_lines.append("")

    for model, info in summary['best_per_model'].items():
        report_lines.append(f"  {model}: {info['normalization']} ({info['accuracy']:.2f}% +/- {info['std']:.2f}%)")

    report_lines.append("")
    report_lines.append("-" * 70)
    report_lines.append("RECOMMENDATION")
    report_lines.append("-" * 70)
    report_lines.append("")
    report_lines.append(f"  {summary['recommendation']}")
    report_lines.append("")

    if 'model_specific_note' in summary:
        report_lines.append("  Model-specific preferences:")
        for model, norm in summary['model_specific_note'].items():
            report_lines.append(f"    - {model}: {norm}")

    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)

    # Print to console
    print("\n" + report_text)

    # Save to file
    with open(RESULTS_DIR / "experiment_4.1_report.txt", 'w') as f:
        f.write(report_text)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function for Experiment 4.1."""
    global MPS_REPORT

    print("\n" + "=" * 70)
    print("PHASE 4 - EXPERIMENT 4.1: NORMALIZATION COMPARISON")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Test MPS compatibility first
    print("Testing MPS (Metal Performance Shaders) compatibility...")
    MPS_REPORT = test_mps_compatibility()
    print(f"  MPS Available: {MPS_REPORT['mps_available']}")
    print(f"  MPS Quick Test Passed: {MPS_REPORT['mps_test_passed']}")
    print(f"  Practical for Training: {MPS_REPORT['mps_practical']}")
    print(f"  Recommendation: {MPS_REPORT['recommendation']}")
    print(f"  Note: {MPS_REPORT['reason']}")
    print()

    # Load data
    print("Loading data...")
    df = load_all_data()
    print(f"  Loaded {len(df)} rows from {len(SIGNALS)} signals")

    # Pivot to feature matrix
    print("\nCreating feature matrix...")
    result_df, X, y, subjects, feature_columns = pivot_to_feature_matrix(df)
    print(f"  Shape: {X.shape}")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Subjects: {len(np.unique(subjects))}")
    print(f"  Class distribution: {np.bincount(y)}")

    # Run experiment
    results_df, summary = run_normalization_experiment(X, y, subjects, feature_columns)

    # Save results
    print("\nSaving results...")
    save_results(results_df, summary, feature_columns)

    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results_df, summary


if __name__ == "__main__":
    results_df, summary = main()
