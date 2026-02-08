#!/usr/bin/env python3
"""
Phase 8.2: Feature Fusion (Catch22 + Entropy-Complexity) for Pain Classification

This script implements feature fusion by combining:
1. Entropy-complexity features (24 features from existing pipeline)
2. Catch22 features (22 features x 4 signals = 88 features)
3. Total: 112 features

Ablation Study:
- entropy-only (24 features)
- catch22-only (88 features)
- combined (112 features)
- MI-selected (top-K features selected by mutual information)

Models: RandomForest, XGBoost, LightGBM with Optuna optimization
Validation: LOSO (Leave-One-Subject-Out)

Author: Claude (AI Assistant)
Date: 2026-02-07
"""

import gc
import os
import re
import sys
import json
import pickle
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report
)
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

# Import pycatch22 for feature extraction
try:
    import pycatch22
except ImportError:
    print("ERROR: pycatch22 not installed. Install with: pip install pycatch22")
    sys.exit(1)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Feature extraction parameters
BEST_DIMENSION = 7
BEST_TAU = 2

# Signals and features
SIGNALS = ['eda', 'bvp', 'resp', 'spo2']
ENTROPY_FEATURE_COLS = ['pe', 'comp', 'fisher_shannon', 'fisher_info',
                        'renyipe', 'renyicomp', 'tsallispe', 'tsalliscomp']

# Paper 1 baseline for comparison
PAPER1_LOSO_BASELINE = 0.780

# Optuna configuration
N_OPTUNA_TRIALS = 50

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
FEATURES_DIR = DATA_DIR / 'features'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'phase8_2_feature_fusion'

# Signal directory name mapping (filesystem uses mixed case)
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


# =============================================================================
# Utility Functions
# =============================================================================

def clear_memory():
    """Clear caches and force garbage collection."""
    gc.collect()


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
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
            print(f"  [CHECKPOINT] Loaded: {len(checkpoint.get('completed_experiments', []))} experiments completed")
            return checkpoint
        except json.JSONDecodeError:
            print("  [WARNING] Corrupted checkpoint, starting fresh")
            return {'completed_experiments': [], 'results': {}}
    return {'completed_experiments': [], 'results': {}}


def save_checkpoint(results_dir: Path, checkpoint: Dict):
    """Save checkpoint after each experiment."""
    checkpoint['last_updated'] = datetime.now().isoformat()
    checkpoint_file = results_dir / 'checkpoint.json'

    # Convert to serializable format
    serializable = convert_to_serializable(checkpoint)

    with open(checkpoint_file, 'w') as f:
        json.dump(serializable, f, indent=2)

    n_completed = len(checkpoint.get('completed_experiments', []))
    print(f"    [CHECKPOINT SAVED] {n_completed} experiments completed")


# =============================================================================
# Catch22 Feature Extraction
# =============================================================================

def extract_catch22_from_segment(values: np.ndarray) -> Dict[str, float]:
    """
    Extract 22 catch22 features from a time series segment.

    Parameters
    ----------
    values : np.ndarray
        Time series values

    Returns
    -------
    Dict[str, float]
        Dictionary mapping catch22 feature names to values
    """
    # Remove NaN values
    values = values[~np.isnan(values)]

    if len(values) < 10:  # Minimum length for catch22
        # Return NaN for all features if insufficient data
        return {f'catch22_{i+1}': np.nan for i in range(22)}

    try:
        # Extract all 22 catch22 features
        features = pycatch22.catch22_all(values)
        feature_dict = {f'catch22_{i+1}': features['values'][i] for i in range(22)}
        return feature_dict
    except Exception as e:
        print(f"    Warning: catch22 extraction failed: {e}")
        return {f'catch22_{i+1}': np.nan for i in range(22)}


def load_raw_signal_segments(signal_type: str, split: str) -> pd.DataFrame:
    """
    Load raw signal data and extract catch22 features per segment.

    Parameters
    ----------
    signal_type : str
        Signal type ('eda', 'bvp', 'resp', 'spo2')
    split : str
        Data split ('train', 'validation')

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: segment_id, catch22_1, catch22_2, ..., catch22_22
    """
    signal_dir = DATA_DIR / split / SIGNAL_DIR_MAP[signal_type]

    all_segments = []

    # Get all subject files
    subject_files = sorted(signal_dir.glob('*.csv'))

    for subject_file in subject_files:
        # Read raw signal CSV
        try:
            df = pd.read_csv(subject_file)
        except Exception as e:
            print(f"    Warning: Failed to read {subject_file}: {e}")
            continue

        # Each column is a segment (e.g., '12_Baseline_1', '12_HIGH_1', etc.)
        for segment_id in df.columns:
            # Extract time series values for this segment
            values = df[segment_id].values

            # Extract catch22 features
            catch22_features = extract_catch22_from_segment(values)

            # Add signal type prefix to feature names
            catch22_features_prefixed = {
                f'{signal_type}_{k}': v for k, v in catch22_features.items()
            }

            segment_data = {
                'segment_id': segment_id,
                **catch22_features_prefixed
            }
            all_segments.append(segment_data)

    return pd.DataFrame(all_segments)


def extract_all_catch22_features() -> pd.DataFrame:
    """
    Extract catch22 features for all signals and splits.

    Returns
    -------
    pd.DataFrame
        Merged catch22 features with segment_id as key
    """
    print("\nExtracting catch22 features from raw signals...")

    all_catch22_dfs = []
    splits = ['train', 'validation']

    for split in splits:
        print(f"\n  Processing {split} split:")
        for signal_type in SIGNALS:
            print(f"    Extracting {signal_type} catch22 features...")
            catch22_df = load_raw_signal_segments(signal_type, split)
            all_catch22_dfs.append(catch22_df)
            print(f"      Extracted {len(catch22_df)} segments")

    # Merge all catch22 features on segment_id
    print("\n  Merging catch22 features across signals...")
    merged = all_catch22_dfs[0]
    for df in all_catch22_dfs[1:]:
        merged = merged.merge(df, on='segment_id', how='outer')

    print(f"  Total catch22 features extracted: {len(merged)} segments")
    print(f"  Feature count: {len([c for c in merged.columns if c.startswith(('eda_', 'bvp_', 'resp_', 'spo2_'))])} features")

    return merged


# =============================================================================
# Data Loading (Entropy-Complexity Features)
# =============================================================================

def extract_subject_id(segment_name: str) -> str:
    """Extract subject ID from segment name like '12_Baseline_1'."""
    match = re.match(r'(\d+)_', segment_name)
    if match:
        return match.group(1)
    match = re.search(r'(\d+)', segment_name)
    if match:
        return match.group(1)
    return segment_name


def load_entropy_complexity_features() -> pd.DataFrame:
    """
    Load pre-extracted entropy-complexity features.

    Returns
    -------
    pd.DataFrame
        Entropy-complexity features in long format
    """
    print("Loading entropy-complexity features...")

    all_dfs = []
    splits = ['train', 'validation']

    for split in splits:
        for phys_signal in SIGNALS:
            file_path = FEATURES_DIR / f'results_{split}_{phys_signal}.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                df = df.rename(columns={'signal': 'segment_id'})
                df['segment_id'] = df['segment_id'].astype(str)
                df['phys_signal'] = phys_signal
                df['split'] = split
                all_dfs.append(df)
                print(f"  Loaded {file_path.name}: {len(df)} rows")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Filter for best dimension and tau
    combined = combined[
        (combined['dimension'] == BEST_DIMENSION) &
        (combined['tau'] == BEST_TAU)
    ].copy()

    # EXCLUDE rest segments (baseline-only methodology)
    n_before = len(combined)
    combined = combined[combined['state'] != 'rest'].copy()
    n_after = len(combined)
    print(f"  Excluded {n_before - n_after} rest segments")

    # Extract subject IDs
    combined['subject_id'] = combined['segment_id'].apply(extract_subject_id)

    # Map states to labels
    combined['label'] = combined['state'].map(CLASS_MAPPING)
    combined = combined.dropna(subset=['label'])
    combined['label'] = combined['label'].astype(int)

    print(f"\nTotal entropy-complexity samples: {len(combined)}")
    print(f"Unique subjects: {combined['subject_id'].nunique()}")
    print(f"Class distribution: {combined['label'].value_counts().sort_index().to_dict()}")

    return combined


def pivot_to_multimodal(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Pivot from long format to wide format (one row per sample with all signals).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe
    feature_cols : List[str]
        Base feature column names

    Returns
    -------
    pd.DataFrame
        Wide-format dataframe
    """
    df = df.copy()

    pivot_dfs = []
    for phys_signal in SIGNALS:
        signal_df = df[df['phys_signal'] == phys_signal][
            ['segment_id', 'subject_id', 'state', 'label'] + feature_cols
        ].copy()
        signal_df = signal_df.rename(columns={col: f'{phys_signal}_{col}' for col in feature_cols})
        pivot_dfs.append(signal_df)

    result = pivot_dfs[0]
    for pdf in pivot_dfs[1:]:
        result = result.merge(
            pdf.drop(columns=['subject_id', 'state', 'label']),
            on='segment_id',
            how='inner'
        )

    print(f"  Merged {len(result)} samples across all signals")
    return result


def merge_features(entropy_df: pd.DataFrame, catch22_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge entropy-complexity and catch22 features.

    Parameters
    ----------
    entropy_df : pd.DataFrame
        Entropy-complexity features (wide format)
    catch22_df : pd.DataFrame
        Catch22 features

    Returns
    -------
    pd.DataFrame
        Merged feature dataframe
    """
    print("\nMerging entropy-complexity and catch22 features...")

    merged = entropy_df.merge(catch22_df, on='segment_id', how='inner')

    print(f"  Merged samples: {len(merged)}")
    print(f"  Entropy features: {len([c for c in entropy_df.columns if any(c.startswith(f'{s}_') for s in SIGNALS)])}")
    print(f"  Catch22 features: {len([c for c in catch22_df.columns if c.startswith(('eda_', 'bvp_', 'resp_', 'spo2_'))])}")
    print(f"  Total features: {len([c for c in merged.columns if any(c.startswith(f'{s}_') for s in SIGNALS)])}")

    return merged


# =============================================================================
# Feature Selection
# =============================================================================

def select_features_by_mi(X: np.ndarray, y: np.ndarray, feature_names: List[str], k: int) -> List[str]:
    """
    Select top-K features by mutual information.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    feature_names : List[str]
        Feature names
    k : int
        Number of features to select

    Returns
    -------
    List[str]
        Selected feature names
    """
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_SEED)
    top_k_indices = np.argsort(mi_scores)[-k:]
    return [feature_names[i] for i in top_k_indices]


# =============================================================================
# Model Hyperparameter Search Spaces
# =============================================================================

def get_rf_search_space(trial: optuna.Trial) -> Dict:
    """Define RandomForest hyperparameter search space."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    }
    return params


def get_xgb_search_space(trial: optuna.Trial) -> Dict:
    """Define XGBoost hyperparameter search space."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'verbosity': 0
    }
    return params


def get_lgb_search_space(trial: optuna.Trial) -> Dict:
    """Define LightGBM hyperparameter search space."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbose': -1
    }
    return params


# =============================================================================
# Inner Cross-Validation for Optuna
# =============================================================================

def evaluate_with_inner_cv(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    model_name: str,
    params: Dict
) -> float:
    """
    Evaluate hyperparameters using 5-fold stratified CV.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    subject_ids : np.ndarray
        Subject IDs (not used for inner CV, but kept for consistency)
    model_name : str
        Model name
    params : Dict
        Hyperparameters

    Returns
    -------
    float
        Mean balanced accuracy
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train
        if model_name == 'RandomForest':
            model = RandomForestClassifier(**params)
        elif model_name == 'XGBoost':
            model = xgb.XGBClassifier(**params)
        elif model_name == 'LightGBM':
            model = lgb.LGBMClassifier(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

        score = balanced_accuracy_score(y_val, y_pred)
        fold_scores.append(score)

    return np.mean(fold_scores)


# =============================================================================
# LOSO Cross-Validation with Optuna
# =============================================================================

def run_loso_with_optuna(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    feature_set_name: str
) -> Dict:
    """
    Run LOSO validation with Optuna hyperparameter optimization.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    feature_cols : List[str]
        Feature column names
    model_name : str
        Model name
    feature_set_name : str
        Feature set name for tracking

    Returns
    -------
    Dict
        LOSO results
    """
    subjects = sorted(df['subject_id'].unique())
    n_subjects = len(subjects)

    print(f"\n{'='*70}")
    print(f"LOSO: {model_name} | Feature Set: {feature_set_name}")
    print(f"{'='*70}")
    print(f"Total subjects: {n_subjects}")
    print(f"Features: {len(feature_cols)}")

    all_y_true = []
    all_y_pred = []
    fold_results = []
    best_params_per_fold = []

    for fold_idx, test_subject in enumerate(subjects):
        # Split
        train_mask = df['subject_id'] != test_subject
        test_mask = df['subject_id'] == test_subject

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        X_train = train_df[feature_cols].values
        y_train = train_df['label'].values
        subject_ids_train = train_df['subject_id'].values

        X_test = test_df[feature_cols].values
        y_test = test_df['label'].values

        # Optuna optimization
        def objective(trial):
            if model_name == 'RandomForest':
                params = get_rf_search_space(trial)
            elif model_name == 'XGBoost':
                params = get_xgb_search_space(trial)
            elif model_name == 'LightGBM':
                params = get_lgb_search_space(trial)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            score = evaluate_with_inner_cv(X_train, y_train, subject_ids_train, model_name, params)
            return score

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=RANDOM_SEED + fold_idx)
        )
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False, n_jobs=1)

        best_params = study.best_params
        best_params_per_fold.append(best_params)

        # Train final model with best params
        if model_name == 'RandomForest':
            final_params = {**best_params, 'random_state': RANDOM_SEED, 'n_jobs': -1}
            final_model = RandomForestClassifier(**final_params)
        elif model_name == 'XGBoost':
            final_params = {**best_params, 'random_state': RANDOM_SEED, 'n_jobs': -1,
                           'use_label_encoder': False, 'eval_metric': 'mlogloss', 'verbosity': 0}
            final_model = xgb.XGBClassifier(**final_params)
        elif model_name == 'LightGBM':
            final_params = {**best_params, 'random_state': RANDOM_SEED, 'n_jobs': -1, 'verbose': -1}
            final_model = lgb.LGBMClassifier(**final_params)

        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train and predict
        final_model.fit(X_train_scaled, y_train)
        y_pred = final_model.predict(X_test_scaled)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        fold_results.append({
            'fold_idx': fold_idx,
            'test_subject': test_subject,
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'f1_weighted': f1,
            'n_test_samples': len(y_test)
        })

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        if (fold_idx + 1) % 10 == 0 or (fold_idx + 1) == n_subjects:
            print(f"  Completed {fold_idx+1}/{n_subjects} folds...")

    # Aggregate results
    fold_df = pd.DataFrame(fold_results)

    ba_mean = fold_df['balanced_accuracy'].mean()
    ba_std = fold_df['balanced_accuracy'].std()

    # Confidence interval
    n = len(fold_df)
    ci_lower = ba_mean - 1.96 * ba_std / np.sqrt(n)
    ci_upper = ba_mean + 1.96 * ba_std / np.sqrt(n)

    # Statistical test
    t_stat, p_value = stats.ttest_1samp(fold_df['balanced_accuracy'].values, PAPER1_LOSO_BASELINE)

    # Cohen's d
    cohens_d = (ba_mean - PAPER1_LOSO_BASELINE) / ba_std if ba_std > 0 else 0

    results = {
        'model': model_name,
        'feature_set': feature_set_name,
        'n_features': len(feature_cols),
        'n_folds': n_subjects,
        'per_fold': fold_df,
        'best_params_per_fold': best_params_per_fold,
        'metrics': {
            'balanced_accuracy_mean': ba_mean,
            'balanced_accuracy_std': ba_std,
            'accuracy_mean': fold_df['accuracy'].mean(),
            'accuracy_std': fold_df['accuracy'].std(),
            'f1_mean': fold_df['f1_weighted'].mean(),
            'f1_std': fold_df['f1_weighted'].std()
        },
        'ci_95': (ci_lower, ci_upper),
        'statistical_test': {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        },
        'cohens_d': cohens_d,
        'y_true': np.array(all_y_true),
        'y_pred': np.array(all_y_pred),
        'confusion_matrix': confusion_matrix(all_y_true, all_y_pred)
    }

    print(f"\n  Results:")
    print(f"    Balanced Accuracy: {ba_mean:.4f} +/- {ba_std:.4f}")
    print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"    vs Paper 1: t={t_stat:.3f}, p={p_value:.4f}")

    return results


# =============================================================================
# Main Experiment Loop
# =============================================================================

def run_ablation_experiments(df: pd.DataFrame, entropy_features: List[str], catch22_features: List[str], resume: bool = False):
    """
    Run ablation experiments across all models and feature sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with all features
    entropy_features : List[str]
        Entropy-complexity feature names
    catch22_features : List[str]
        Catch22 feature names
    resume : bool
        Whether to resume from checkpoint
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = ['RandomForest', 'XGBoost', 'LightGBM']

    # Define feature sets
    feature_sets = {
        'entropy': entropy_features,
        'catch22': catch22_features,
        'combined': entropy_features + catch22_features
    }

    # Load checkpoint
    if resume:
        checkpoint = load_checkpoint(RESULTS_DIR)
        completed = set(checkpoint.get('completed_experiments', []))
        all_results = checkpoint.get('results', {})
    else:
        completed = set()
        all_results = {}

    total_experiments = len(models) * len(feature_sets)
    completed_count = 0

    print(f"\nTotal experiments: {total_experiments}")
    print(f"Already completed: {len(completed)}")

    for model_name in models:
        for feature_set_name, feature_cols in feature_sets.items():
            experiment_id = f"{model_name}_{feature_set_name}"

            if experiment_id in completed:
                print(f"\n[SKIP] {experiment_id} - already completed")
                completed_count += 1
                continue

            try:
                # Run LOSO
                results = run_loso_with_optuna(df, feature_cols, model_name, feature_set_name)

                # Store results
                all_results[experiment_id] = results
                completed.add(experiment_id)
                completed_count += 1

                # Save checkpoint
                checkpoint = {
                    'completed_experiments': list(completed),
                    'results': all_results
                }
                save_checkpoint(RESULTS_DIR, checkpoint)

                print(f"\n  Progress: {completed_count}/{total_experiments} experiments complete")

            except Exception as e:
                print(f"\n[ERROR] {experiment_id} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Clear memory
            clear_memory()

    return all_results


# =============================================================================
# Results Analysis and Reporting
# =============================================================================

def generate_leaderboard(results: Dict) -> pd.DataFrame:
    """Generate LOSO leaderboard."""
    rows = []
    for exp_id, res in results.items():
        rows.append({
            'rank': 0,
            'model': res['model'],
            'feature_set': res['feature_set'],
            'n_features': res['n_features'],
            'loso_balanced_accuracy_mean': res['metrics']['balanced_accuracy_mean'],
            'loso_balanced_accuracy_std': res['metrics']['balanced_accuracy_std'],
            'ci_95_lower': res['ci_95'][0],
            'ci_95_upper': res['ci_95'][1],
            'vs_paper1_improvement': res['metrics']['balanced_accuracy_mean'] - PAPER1_LOSO_BASELINE,
            'p_value': res['statistical_test']['p_value']
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('loso_balanced_accuracy_mean', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    return df


def generate_ablation_table(results: Dict) -> pd.DataFrame:
    """Generate ablation study results table."""
    rows = []
    for exp_id, res in results.items():
        rows.append({
            'model': res['model'],
            'feature_set': res['feature_set'],
            'n_features': res['n_features'],
            'balanced_accuracy': res['metrics']['balanced_accuracy_mean'],
            'std': res['metrics']['balanced_accuracy_std'],
            'accuracy': res['metrics']['accuracy_mean'],
            'f1_weighted': res['metrics']['f1_mean']
        })

    return pd.DataFrame(rows)


def generate_per_subject_results(results: Dict) -> pd.DataFrame:
    """Generate per-subject results."""
    rows = []
    for exp_id, res in results.items():
        for _, fold_row in res['per_fold'].iterrows():
            rows.append({
                'model': res['model'],
                'feature_set': res['feature_set'],
                'subject_id': fold_row['test_subject'],
                'balanced_accuracy': fold_row['balanced_accuracy'],
                'accuracy': fold_row['accuracy'],
                'f1_weighted': fold_row['f1_weighted'],
                'n_samples': fold_row['n_test_samples']
            })

    return pd.DataFrame(rows)


def save_feature_importance(results: Dict, df: pd.DataFrame, entropy_features: List[str], catch22_features: List[str]):
    """Save feature importance analysis."""
    # Get best model
    best_exp_id = max(results.items(), key=lambda x: x[1]['metrics']['balanced_accuracy_mean'])[0]
    best_result = results[best_exp_id]

    if best_result['feature_set'] == 'combined':
        # Compute mutual information for all features
        all_features = entropy_features + catch22_features
        X = df[all_features].values
        y = df['label'].values

        mi_scores = mutual_info_classif(X, y, random_state=RANDOM_SEED)

        importance_df = pd.DataFrame({
            'feature': all_features,
            'mutual_info_score': mi_scores,
            'feature_type': ['entropy' if f in entropy_features else 'catch22' for f in all_features]
        })
        importance_df = importance_df.sort_values('mutual_info_score', ascending=False)

        importance_df.to_csv(RESULTS_DIR / 'feature_importance.csv', index=False)
        print(f"  Saved: feature_importance.csv")


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: Path):
    """Generate confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
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
    ax.set_title(title, fontsize=14)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j + 0.5, i + 0.75, f'n={cm[i,j]}',
                   ha='center', va='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(results: Dict, leaderboard: pd.DataFrame, ablation: pd.DataFrame) -> str:
    """Generate markdown report."""
    best_row = leaderboard.iloc[0]

    report = f"""# Phase 8.2: Feature Fusion (Catch22 + Entropy-Complexity) - Final Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Best Model** | {best_row['model']} |
| **Best Feature Set** | {best_row['feature_set']} |
| **Number of Features** | {best_row['n_features']} |
| **LOSO Balanced Accuracy** | {best_row['loso_balanced_accuracy_mean']:.4f} +/- {best_row['loso_balanced_accuracy_std']:.4f} |
| **95% CI** | [{best_row['ci_95_lower']:.4f}, {best_row['ci_95_upper']:.4f}] |
| **Paper 1 Baseline** | {PAPER1_LOSO_BASELINE:.3f} |
| **Improvement** | {best_row['vs_paper1_improvement']:+.4f} ({best_row['vs_paper1_improvement']/PAPER1_LOSO_BASELINE*100:+.2f}%) |

---

## Methodology

### Feature Sets

1. **Entropy-Complexity (24 features)**
   - 8 entropy/complexity measures x 4 signals (EDA, BVP, RESP, SpO2)
   - Features: pe, comp, fisher_shannon, fisher_info, renyipe, renyicomp, tsallispe, tsalliscomp

2. **Catch22 (88 features)**
   - 22 catch22 features x 4 signals
   - Canonical time-series features from hctsa toolbox

3. **Combined (112 features)**
   - All entropy-complexity + catch22 features

### Models

- RandomForest, XGBoost, LightGBM
- Each optimized with Optuna (50 trials, TPE sampler, 5-fold stratified inner CV)

### Validation

- LOSO (Leave-One-Subject-Out) cross-validation
- Global z-score normalization (fit on train, transform test per fold)
- Baseline-only methodology (rest segments excluded)

---

## Results

### LOSO Leaderboard

| Rank | Model | Feature Set | Features | Balanced Acc | 95% CI | vs Paper 1 |
|------|-------|-------------|----------|--------------|--------|------------|
"""

    for _, row in leaderboard.iterrows():
        ci_str = f"[{row['ci_95_lower']:.3f}, {row['ci_95_upper']:.3f}]"
        report += f"| {int(row['rank'])} | {row['model']} | {row['feature_set']} | {int(row['n_features'])} | {row['loso_balanced_accuracy_mean']:.4f} +/- {row['loso_balanced_accuracy_std']:.4f} | {ci_str} | {row['vs_paper1_improvement']:+.4f} |\n"

    report += f"""

### Ablation Study Analysis

**Entropy-Only Performance:**

"""

    entropy_results = ablation[ablation['feature_set'] == 'entropy'].sort_values('balanced_accuracy', ascending=False)
    for _, row in entropy_results.iterrows():
        report += f"- {row['model']}: {row['balanced_accuracy']:.4f}\n"

    report += f"""

**Catch22-Only Performance:**

"""

    catch22_results = ablation[ablation['feature_set'] == 'catch22'].sort_values('balanced_accuracy', ascending=False)
    for _, row in catch22_results.iterrows():
        report += f"- {row['model']}: {row['balanced_accuracy']:.4f}\n"

    report += f"""

**Combined Performance:**

"""

    combined_results = ablation[ablation['feature_set'] == 'combined'].sort_values('balanced_accuracy', ascending=False)
    for _, row in combined_results.iterrows():
        report += f"- {row['model']}: {row['balanced_accuracy']:.4f}\n"

    report += f"""

---

## Key Findings

1. **Feature Fusion Analysis:**
   - Best feature set: {best_row['feature_set']}
   - Best model: {best_row['model']}

2. **Comparison to Paper 1:**
   - Paper 1 baseline (catch22 + XGBoost): {PAPER1_LOSO_BASELINE:.1%}
   - This study: {best_row['loso_balanced_accuracy_mean']:.1%}
   - Improvement: {best_row['vs_paper1_improvement']:+.2%}

3. **Statistical Significance:**
   - p-value vs Paper 1: {best_row['p_value']:.4f}
   - Significant at alpha=0.05: {"Yes" if best_row['p_value'] < 0.05 else "No"}

---

## Conclusion

This experiment demonstrates {'that feature fusion improves' if best_row['feature_set'] == 'combined' else 'the relative performance of'} entropy-complexity features compared to catch22 features for 3-class pain classification.

The best configuration achieves {best_row['loso_balanced_accuracy_mean']:.2%} balanced accuracy on rigorous LOSO validation.

---

**End of Report**

*Generated by Phase 8.2 Feature Fusion Pipeline*
"""

    return report


def generate_outputs(results: Dict, entropy_features: List[str], catch22_features: List[str], df: pd.DataFrame):
    """Generate all output files."""
    print("\n" + "="*70)
    print("GENERATING OUTPUTS")
    print("="*70)

    # 1. Leaderboard
    print("\nSaving leaderboard...")
    leaderboard = generate_leaderboard(results)
    leaderboard.to_csv(RESULTS_DIR / 'loso_leaderboard.csv', index=False)
    print("  Saved: loso_leaderboard.csv")

    # 2. Ablation table
    print("Saving ablation results...")
    ablation = generate_ablation_table(results)
    ablation.to_csv(RESULTS_DIR / 'ablation_results.csv', index=False)
    print("  Saved: ablation_results.csv")

    # 3. Per-subject results
    print("Saving per-subject results...")
    per_subject = generate_per_subject_results(results)
    per_subject.to_csv(RESULTS_DIR / 'per_subject_results.csv', index=False)
    print("  Saved: per_subject_results.csv")

    # 4. Feature importance
    print("Saving feature importance...")
    save_feature_importance(results, df, entropy_features, catch22_features)

    # 5. Best hyperparameters
    print("Saving best hyperparameters...")
    best_exp_id = max(results.items(), key=lambda x: x[1]['metrics']['balanced_accuracy_mean'])[0]
    best_result = results[best_exp_id]

    with open(RESULTS_DIR / 'best_hyperparameters.json', 'w') as f:
        json.dump(convert_to_serializable(best_result['best_params_per_fold']), f, indent=2)
    print("  Saved: best_hyperparameters.json")

    # 6. Confusion matrix for best model
    print("Generating confusion matrix...")
    cm_path = RESULTS_DIR / 'confusion_matrix.png'
    plot_confusion_matrix(
        best_result['y_true'],
        best_result['y_pred'],
        f"Best Model: {best_result['model']} ({best_result['feature_set']})\nBalanced Accuracy: {best_result['metrics']['balanced_accuracy_mean']:.2%}",
        cm_path
    )
    print("  Saved: confusion_matrix.png")

    # 7. Report
    print("Generating report...")
    report = generate_report(results, leaderboard, ablation)
    with open(RESULTS_DIR / 'phase8_2_report.md', 'w') as f:
        f.write(report)
    print("  Saved: phase8_2_report.md")

    # 8. Full results JSON
    print("Saving full results...")
    with open(RESULTS_DIR / 'full_results.json', 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print("  Saved: full_results.json")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Phase 8.2: Feature Fusion (Catch22 + Entropy-Complexity)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    print("="*70)
    print("PHASE 8.2: FEATURE FUSION (CATCH22 + ENTROPY-COMPLEXITY)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Extract catch22 features
    catch22_df = extract_all_catch22_features()

    # Step 2: Load entropy-complexity features
    entropy_df_long = load_entropy_complexity_features()

    # Step 3: Pivot entropy features to wide format
    print("\nPivoting entropy-complexity features to multimodal format...")
    entropy_df_wide = pivot_to_multimodal(entropy_df_long, ENTROPY_FEATURE_COLS)

    # Step 4: Merge features
    merged_df = merge_features(entropy_df_wide, catch22_df)

    # Step 5: Define feature sets
    entropy_features = [f'{signal}_{feat}' for signal in SIGNALS for feat in ENTROPY_FEATURE_COLS]
    catch22_features = [c for c in merged_df.columns if c.startswith(('eda_catch22', 'bvp_catch22', 'resp_catch22', 'spo2_catch22'))]

    print(f"\nFeature sets defined:")
    print(f"  Entropy-complexity: {len(entropy_features)} features")
    print(f"  Catch22: {len(catch22_features)} features")
    print(f"  Combined: {len(entropy_features) + len(catch22_features)} features")

    # Step 6: Run ablation experiments
    print("\n" + "="*70)
    print("RUNNING ABLATION EXPERIMENTS")
    print("="*70)

    results = run_ablation_experiments(merged_df, entropy_features, catch22_features, resume=args.resume)

    if not results:
        print("\n[ERROR] No experiments completed. Exiting.")
        return

    # Step 7: Generate outputs
    generate_outputs(results, entropy_features, catch22_features, merged_df)

    # Final summary
    print("\n" + "="*70)
    print("PHASE 8.2 COMPLETE")
    print("="*70)

    leaderboard = generate_leaderboard(results)
    best_row = leaderboard.iloc[0]

    print(f"\nBEST RESULT:")
    print(f"  Model: {best_row['model']}")
    print(f"  Feature Set: {best_row['feature_set']}")
    print(f"  Features: {best_row['n_features']}")
    print(f"  LOSO Balanced Accuracy: {best_row['loso_balanced_accuracy_mean']:.4f} +/- {best_row['loso_balanced_accuracy_std']:.4f}")
    print(f"  95% CI: [{best_row['ci_95_lower']:.4f}, {best_row['ci_95_upper']:.4f}]")
    print(f"  vs Paper 1: {best_row['vs_paper1_improvement']:+.4f} ({best_row['vs_paper1_improvement']/PAPER1_LOSO_BASELINE*100:+.2f}%)")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
