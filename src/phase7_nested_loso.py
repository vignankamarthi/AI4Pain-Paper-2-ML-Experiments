#!/usr/bin/env python3
"""
Phase 7: Nested Optuna-LOSO to Beat Paper 1's 78.0%

This experiment uses rigorous nested cross-validation where hyperparameters
are optimized per LOSO fold using inner LOSO validation.

Target: Beat Paper 1's LOSO baseline of 78.0% (XGBoost with default params)

Methodology:
- Outer CV: LOSO (53 folds, one per subject)
- Inner CV: LOSO (52 folds per Optuna trial)
- Optuna trials: 50 per outer fold
- Model: RandomForest
- Normalization: Global z-score (fit on train, transform test per fold)

Author: Claude (AI Assistant)
Date: 2026-01-18
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
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report
)
import optuna
from optuna.samplers import TPESampler

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
FEATURE_COLS = ['pe', 'comp', 'fisher_shannon', 'fisher_info',
                'renyipe', 'renyicomp', 'tsallispe', 'tsalliscomp']

# Paper 1 baseline for comparison
PAPER1_LOSO_BASELINE = 0.780  # XGBoost with default params

# Optuna configuration
N_OPTUNA_TRIALS = 50

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'features'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'phase7_nested_loso'

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
            print(f"  [CHECKPOINT] Loaded: {len(checkpoint.get('completed_folds', []))} folds completed")
            return checkpoint
        except json.JSONDecodeError:
            print("  [WARNING] Corrupted checkpoint, starting fresh")
            return {'completed_folds': [], 'fold_results': {}, 'start_time': None}
    return {'completed_folds': [], 'fold_results': {}, 'start_time': None}


def save_checkpoint(results_dir: Path, checkpoint: Dict):
    """Save checkpoint after each fold."""
    checkpoint['last_updated'] = datetime.now().isoformat()
    checkpoint_file = results_dir / 'checkpoint.json'

    # Convert to serializable format
    serializable = convert_to_serializable(checkpoint)

    with open(checkpoint_file, 'w') as f:
        json.dump(serializable, f, indent=2)

    n_completed = len(checkpoint.get('completed_folds', []))
    print(f"    [CHECKPOINT SAVED] {n_completed} folds completed")


def save_fold_result(results_dir: Path, subject_id: str, fold_data: Dict):
    """Save individual fold result to disk."""
    fold_dir = results_dir / 'fold_results'
    fold_dir.mkdir(exist_ok=True)

    fold_file = fold_dir / f'fold_{subject_id}.json'
    with open(fold_file, 'w') as f:
        json.dump(convert_to_serializable(fold_data), f, indent=2)


# =============================================================================
# Data Loading
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


def load_all_data() -> pd.DataFrame:
    """
    Load and combine all feature data from train and validation sets.
    Test set excluded (has 'unknown' states without labels).
    """
    print("Loading all feature data...")

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

    print(f"\nTotal samples after filtering: {len(combined)}")
    print(f"Unique subjects: {combined['subject_id'].nunique()}")
    print(f"Class distribution: {combined['label'].value_counts().sort_index().to_dict()}")

    return combined


def pivot_to_multimodal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot from long format to wide format (one row per sample with all signals).
    """
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

    print(f"  Merged {len(result)} samples across all signals")
    return result


# =============================================================================
# RandomForest Hyperparameter Search Space
# =============================================================================

def get_rf_search_space(trial: optuna.Trial) -> Dict:
    """Define RandomForest hyperparameter search space for Optuna."""
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


# =============================================================================
# Inner LOSO (for Optuna evaluation)
# =============================================================================

def evaluate_with_inner_loso(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    params: Dict
) -> float:
    """
    Evaluate hyperparameters using inner LOSO on the training pool.
    Returns mean balanced accuracy across all inner folds.
    """
    unique_subjects = np.unique(subject_ids)
    fold_scores = []

    for test_subject in unique_subjects:
        # Split
        train_mask = subject_ids != test_subject
        test_mask = subject_ids == test_subject

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Global z-score normalization (fit on train, transform both)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train and predict
        model = RandomForestClassifier(**params)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Score
        score = balanced_accuracy_score(y_test, y_pred)
        fold_scores.append(score)

    return np.mean(fold_scores)


# =============================================================================
# Nested Optuna-LOSO Main Loop
# =============================================================================

def run_outer_fold(
    fold_idx: int,
    test_subject: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    all_subjects: List[str]
) -> Dict:
    """
    Run a single outer LOSO fold with Optuna optimization.

    Parameters
    ----------
    fold_idx : int
        Index of the outer fold (0-based)
    test_subject : str
        Subject ID to hold out for testing
    df : pd.DataFrame
        Full dataset
    feature_cols : List[str]
        Feature column names
    all_subjects : List[str]
        List of all subject IDs

    Returns
    -------
    Dict
        Fold results including predictions and best hyperparameters
    """
    n_subjects = len(all_subjects)
    fold_start = datetime.now()

    print(f"\n{'='*60}")
    print(f"OUTER FOLD {fold_idx + 1}/{n_subjects}: Test Subject = {test_subject}")
    print(f"{'='*60}")

    # Split data
    train_mask = df['subject_id'] != test_subject
    test_mask = df['subject_id'] == test_subject

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    # Extract features and labels
    X_train_pool = train_df[feature_cols].values
    y_train_pool = train_df['label'].values
    subject_ids_train = train_df['subject_id'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values

    print(f"  Train pool: {len(X_train_pool)} samples from {len(np.unique(subject_ids_train))} subjects")
    print(f"  Test set: {len(X_test)} samples from subject {test_subject}")

    # =========================================================================
    # Optuna Optimization with Inner LOSO
    # =========================================================================

    print(f"\n  Running Optuna ({N_OPTUNA_TRIALS} trials, inner LOSO)...")
    optuna_start = datetime.now()

    def objective(trial):
        params = get_rf_search_space(trial)
        score = evaluate_with_inner_loso(X_train_pool, y_train_pool, subject_ids_train, params)
        return score

    # Create study with TPE sampler
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_SEED)
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=False,
        n_jobs=1  # Sequential for reproducibility
    )

    optuna_duration = (datetime.now() - optuna_start).total_seconds()
    best_params = study.best_params
    best_inner_score = study.best_value

    print(f"  Optuna completed in {format_duration(optuna_duration)}")
    print(f"  Best inner LOSO score: {best_inner_score:.4f}")
    print(f"  Best params: n_est={best_params['n_estimators']}, depth={best_params['max_depth']}")

    # =========================================================================
    # Train Final Model and Evaluate on Test Subject
    # =========================================================================

    print(f"\n  Training final model on all {len(np.unique(subject_ids_train))} training subjects...")

    # Prepare final params
    final_params = {
        **best_params,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    }

    # Global z-score normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pool)
    X_test_scaled = scaler.transform(X_test)

    # Train final model
    final_model = RandomForestClassifier(**final_params)
    final_model.fit(X_train_scaled, y_train_pool)

    # Predict
    y_pred = final_model.predict(X_test_scaled)
    y_pred_proba = final_model.predict_proba(X_test_scaled)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    fold_duration = (datetime.now() - fold_start).total_seconds()

    print(f"\n  FOLD RESULTS:")
    print(f"    Balanced Accuracy: {balanced_acc:.4f}")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    F1 (weighted): {f1:.4f}")
    print(f"    Fold duration: {format_duration(fold_duration)}")

    # Compile results
    fold_result = {
        'fold_idx': fold_idx,
        'test_subject': test_subject,
        'n_train_samples': len(X_train_pool),
        'n_train_subjects': len(np.unique(subject_ids_train)),
        'n_test_samples': len(X_test),
        'best_params': best_params,
        'best_inner_score': best_inner_score,
        'optuna_n_trials': len(study.trials),
        'optuna_duration_seconds': optuna_duration,
        'metrics': {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_weighted': f1
        },
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'fold_duration_seconds': fold_duration
    }

    return fold_result


def run_nested_loso(
    df: pd.DataFrame,
    feature_cols: List[str],
    resume: bool = False
) -> Dict:
    """
    Run the full nested Optuna-LOSO experiment.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    feature_cols : List[str]
        Feature column names
    resume : bool
        Whether to resume from checkpoint

    Returns
    -------
    Dict
        Complete experiment results
    """
    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'fold_results').mkdir(exist_ok=True)

    # Get all subjects
    all_subjects = sorted(df['subject_id'].unique())
    n_subjects = len(all_subjects)

    print(f"\nNested Optuna-LOSO Configuration:")
    print(f"  Total subjects: {n_subjects}")
    print(f"  Outer folds: {n_subjects} (LOSO)")
    print(f"  Optuna trials per fold: {N_OPTUNA_TRIALS}")
    print(f"  Inner CV per trial: {n_subjects - 1} folds (LOSO)")
    print(f"  Total model fits: {n_subjects} x {N_OPTUNA_TRIALS} x {n_subjects - 1} = {n_subjects * N_OPTUNA_TRIALS * (n_subjects - 1):,}")

    # Load checkpoint if resuming
    if resume:
        checkpoint = load_checkpoint(RESULTS_DIR)
        completed_folds = set(checkpoint.get('completed_folds', []))
        fold_results = checkpoint.get('fold_results', {})
        start_time = checkpoint.get('start_time')
        if start_time:
            print(f"  Experiment started: {start_time}")
    else:
        completed_folds = set()
        fold_results = {}
        start_time = None

    if not start_time:
        start_time = datetime.now().isoformat()

    # Main outer loop
    experiment_start = datetime.now()

    for fold_idx, test_subject in enumerate(all_subjects):
        # Skip if already completed
        if test_subject in completed_folds:
            print(f"\n[SKIP] Fold {fold_idx + 1}/{n_subjects} (subject {test_subject}) - already completed")
            continue

        try:
            # Run fold
            fold_result = run_outer_fold(
                fold_idx=fold_idx,
                test_subject=test_subject,
                df=df,
                feature_cols=feature_cols,
                all_subjects=all_subjects
            )

            # Save fold result
            fold_results[test_subject] = fold_result
            completed_folds.add(test_subject)
            save_fold_result(RESULTS_DIR, test_subject, fold_result)

            # Save checkpoint
            checkpoint = {
                'start_time': start_time,
                'completed_folds': list(completed_folds),
                'fold_results': fold_results,
                'n_total_folds': n_subjects
            }
            save_checkpoint(RESULTS_DIR, checkpoint)

            # Progress estimate
            elapsed = (datetime.now() - experiment_start).total_seconds()
            avg_fold_time = elapsed / len(completed_folds)
            remaining_folds = n_subjects - len(completed_folds)
            eta_seconds = avg_fold_time * remaining_folds

            print(f"\n  PROGRESS: {len(completed_folds)}/{n_subjects} folds complete")
            print(f"  ETA: {format_duration(eta_seconds)} remaining")

        except Exception as e:
            print(f"\n[ERROR] Fold {fold_idx + 1} (subject {test_subject}) failed: {e}")
            import traceback
            traceback.print_exc()

            # Save error to checkpoint
            checkpoint = {
                'start_time': start_time,
                'completed_folds': list(completed_folds),
                'fold_results': fold_results,
                'n_total_folds': n_subjects,
                'last_error': {
                    'fold_idx': fold_idx,
                    'subject': test_subject,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }
            save_checkpoint(RESULTS_DIR, checkpoint)

            # Continue to next fold
            continue

        # Clear memory
        clear_memory()

    # Compile final results
    total_duration = (datetime.now() - experiment_start).total_seconds()

    results = {
        'model': 'RandomForest',
        'method': 'Nested Optuna-LOSO',
        'n_subjects': n_subjects,
        'n_optuna_trials': N_OPTUNA_TRIALS,
        'fold_results': fold_results,
        'start_time': start_time,
        'end_time': datetime.now().isoformat(),
        'total_duration_seconds': total_duration
    }

    return results


# =============================================================================
# Results Aggregation and Reporting
# =============================================================================

def aggregate_results(results: Dict) -> Dict:
    """Aggregate fold results into summary statistics."""
    fold_results = results['fold_results']

    # Extract per-fold metrics
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

    # 95% confidence interval
    n = len(balanced_accs)
    ci_lower = ba_mean - 1.96 * ba_std / np.sqrt(n)
    ci_upper = ba_mean + 1.96 * ba_std / np.sqrt(n)

    # Statistical test against Paper 1 baseline
    t_stat, p_value = stats.ttest_1samp(balanced_accs, PAPER1_LOSO_BASELINE)

    # Cohen's d effect size
    cohens_d = (ba_mean - PAPER1_LOSO_BASELINE) / ba_std if ba_std > 0 else 0

    # Overall confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)

    summary = {
        'n_folds': n,
        'balanced_accuracy': {
            'mean': ba_mean,
            'std': ba_std,
            'median': ba_median,
            'min': np.min(balanced_accs),
            'max': np.max(balanced_accs),
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper
        },
        'accuracy': {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies)
        },
        'f1_weighted': {
            'mean': np.mean(f1_scores),
            'std': np.std(f1_scores)
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


def generate_report(results: Dict, summary: Dict) -> str:
    """Generate markdown report."""
    ba = summary['balanced_accuracy']
    vs_p1 = summary['vs_paper1']

    # Determine status
    if vs_p1['beats_baseline']:
        status = "SUCCESS - BEAT PAPER 1 BASELINE"
    else:
        status = "BELOW PAPER 1 BASELINE"

    report = f"""# Phase 7: Nested Optuna-LOSO - Final Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Model** | RandomForest |
| **Method** | Nested Optuna-LOSO |
| **LOSO Balanced Accuracy** | {ba['mean']:.4f} +/- {ba['std']:.4f} |
| **95% CI** | [{ba['ci_95_lower']:.4f}, {ba['ci_95_upper']:.4f}] |
| **Paper 1 Baseline** | {vs_p1['paper1_baseline']:.4f} |
| **Improvement** | {vs_p1['improvement']:+.4f} ({vs_p1['improvement_pct']:+.2f}%) |
| **Statistical Significance** | p = {vs_p1['p_value']:.4f} ({'Yes' if vs_p1['significant'] else 'No'}) |
| **Status** | **{status}** |

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Total Subjects | {results['n_subjects']} |
| Outer CV | LOSO ({results['n_subjects']} folds) |
| Inner CV | LOSO ({results['n_subjects'] - 1} folds per trial) |
| Optuna Trials per Fold | {results['n_optuna_trials']} |
| Total Model Fits | {results['n_subjects'] * results['n_optuna_trials'] * (results['n_subjects'] - 1):,} |

---

## Results

### Overall Performance

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Balanced Accuracy | {ba['mean']:.4f} | {ba['std']:.4f} | {ba['min']:.4f} | {ba['max']:.4f} |
| Accuracy | {summary['accuracy']['mean']:.4f} | {summary['accuracy']['std']:.4f} | - | - |
| F1 (weighted) | {summary['f1_weighted']['mean']:.4f} | {summary['f1_weighted']['std']:.4f} | - | - |

### Comparison to Paper 1

| Metric | This Study | Paper 1 | Difference |
|--------|------------|---------|------------|
| LOSO Balanced Acc | {ba['mean']:.4f} | {vs_p1['paper1_baseline']:.4f} | {vs_p1['improvement']:+.4f} |
| Method | Nested Optuna-LOSO | LOSO (default params) | - |
| Model | RandomForest (optimized) | XGBoost (default) | - |

**Statistical Test (one-sample t-test vs 78.0%):**
- t-statistic: {vs_p1['t_statistic']:.4f}
- p-value: {vs_p1['p_value']:.6f}
- Cohen's d: {vs_p1['cohens_d']:.4f}
- Significant at alpha=0.05: {'Yes' if vs_p1['significant'] else 'No'}

---

## Confusion Matrix

```
              Predicted
              no_pain  low_pain  high_pain
Actual
no_pain         {summary['confusion_matrix'][0][0]:4d}      {summary['confusion_matrix'][0][1]:4d}       {summary['confusion_matrix'][0][2]:4d}
low_pain        {summary['confusion_matrix'][1][0]:4d}      {summary['confusion_matrix'][1][1]:4d}       {summary['confusion_matrix'][1][2]:4d}
high_pain       {summary['confusion_matrix'][2][0]:4d}      {summary['confusion_matrix'][2][1]:4d}       {summary['confusion_matrix'][2][2]:4d}
```

---

## Per-Subject Results

| Subject | Balanced Acc | vs Baseline |
|---------|--------------|-------------|
"""

    # Sort by balanced accuracy
    sorted_folds = sorted(
        results['fold_results'].items(),
        key=lambda x: x[1]['metrics']['balanced_accuracy'],
        reverse=True
    )

    for subject_id, fold_data in sorted_folds:
        ba_fold = fold_data['metrics']['balanced_accuracy']
        diff = ba_fold - PAPER1_LOSO_BASELINE
        report += f"| {subject_id} | {ba_fold:.4f} | {diff:+.4f} |\n"

    report += f"""

---

## Timing

| Metric | Value |
|--------|-------|
| Start Time | {results['start_time']} |
| End Time | {results['end_time']} |
| Total Duration | {format_duration(results['total_duration_seconds'])} |
| Avg Fold Duration | {format_duration(results['total_duration_seconds'] / results['n_subjects'])} |

---

## Conclusion

"""

    if vs_p1['beats_baseline']:
        report += f"""**SUCCESS!** The nested Optuna-LOSO approach achieved {ba['mean']:.2%} balanced accuracy,
which is {vs_p1['improvement']:+.2%} percentage points above Paper 1's baseline of {vs_p1['paper1_baseline']:.2%}.

This improvement is {'statistically significant' if vs_p1['significant'] else 'not statistically significant'} (p = {vs_p1['p_value']:.4f}).
"""
    else:
        report += f"""The nested Optuna-LOSO approach achieved {ba['mean']:.2%} balanced accuracy,
which is {abs(vs_p1['improvement']):.2%} percentage points below Paper 1's baseline of {vs_p1['paper1_baseline']:.2%}.

Despite rigorous hyperparameter optimization, the entropy-complexity features did not surpass
catch22 features for 3-class pain classification on this dataset.
"""

    report += """

---

**End of Report**

*Generated by Phase 7 Nested Optuna-LOSO Pipeline*
"""

    return report


def generate_outputs(results: Dict, summary: Dict):
    """Generate all output files."""
    print("\n" + "="*60)
    print("GENERATING OUTPUTS")
    print("="*60)

    # 1. Leaderboard CSV
    print("\nSaving leaderboard...")
    leaderboard = pd.DataFrame([{
        'rank': 1,
        'model': 'RandomForest',
        'method': 'Nested Optuna-LOSO',
        'loso_balanced_accuracy_mean': summary['balanced_accuracy']['mean'],
        'loso_balanced_accuracy_std': summary['balanced_accuracy']['std'],
        'ci_95_lower': summary['balanced_accuracy']['ci_95_lower'],
        'ci_95_upper': summary['balanced_accuracy']['ci_95_upper'],
        'vs_paper1': summary['vs_paper1']['improvement'],
        'p_value': summary['vs_paper1']['p_value']
    }])
    leaderboard.to_csv(RESULTS_DIR / 'loso_leaderboard.csv', index=False)
    print("  Saved: loso_leaderboard.csv")

    # 2. Per-subject results CSV
    print("Saving per-subject results...")
    per_subject_data = []
    for subject_id, fold_data in results['fold_results'].items():
        per_subject_data.append({
            'subject_id': subject_id,
            'balanced_accuracy': fold_data['metrics']['balanced_accuracy'],
            'accuracy': fold_data['metrics']['accuracy'],
            'f1_weighted': fold_data['metrics']['f1_weighted'],
            'n_samples': fold_data['n_test_samples'],
            'best_n_estimators': fold_data['best_params']['n_estimators'],
            'best_max_depth': fold_data['best_params']['max_depth'],
            'best_inner_score': fold_data['best_inner_score']
        })
    pd.DataFrame(per_subject_data).to_csv(RESULTS_DIR / 'per_subject_results.csv', index=False)
    print("  Saved: per_subject_results.csv")

    # 3. Best hyperparameters JSON
    print("Saving hyperparameters...")
    all_params = {
        subject_id: fold_data['best_params']
        for subject_id, fold_data in results['fold_results'].items()
    }
    with open(RESULTS_DIR / 'best_hyperparameters.json', 'w') as f:
        json.dump(convert_to_serializable(all_params), f, indent=2)
    print("  Saved: best_hyperparameters.json")

    # 4. Confusion matrix plot
    print("Generating confusion matrix plot...")
    cm = np.array(summary['confusion_matrix'])
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
    ax.set_title(f'Nested Optuna-LOSO Confusion Matrix\nBalanced Accuracy: {summary["balanced_accuracy"]["mean"]:.2%}', fontsize=14)

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
    report = generate_report(results, summary)
    with open(RESULTS_DIR / 'phase7_report.md', 'w') as f:
        f.write(report)
    print("  Saved: phase7_report.md")

    # 6. Full results JSON
    print("Saving full results...")
    with open(RESULTS_DIR / 'full_results.json', 'w') as f:
        json.dump(convert_to_serializable({
            **results,
            'summary': summary
        }), f, indent=2)
    print("  Saved: full_results.json")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Phase 7: Nested Optuna-LOSO')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    print("="*70)
    print("PHASE 7: NESTED OPTUNA-LOSO TO BEAT PAPER 1")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: Beat Paper 1 LOSO baseline of {PAPER1_LOSO_BASELINE:.1%}")

    # Load data
    raw_df = load_all_data()

    # Pivot to multimodal format
    print("\nPivoting to multimodal format...")
    df = pivot_to_multimodal(raw_df)
    print(f"Multimodal samples: {len(df)}")

    # Feature columns
    feature_cols = [f'{signal}_{feat}' for signal in SIGNALS for feat in FEATURE_COLS]
    print(f"Features: {len(feature_cols)}")

    # Run nested LOSO
    results = run_nested_loso(df, feature_cols, resume=args.resume)

    # Check if we have results
    if not results['fold_results']:
        print("\n[ERROR] No fold results. Experiment incomplete.")
        return

    # Aggregate and generate outputs
    print("\n" + "="*70)
    print("AGGREGATING RESULTS")
    print("="*70)

    summary = aggregate_results(results)
    generate_outputs(results, summary)

    # Final summary
    ba = summary['balanced_accuracy']
    vs_p1 = summary['vs_paper1']

    print("\n" + "="*70)
    print("PHASE 7 COMPLETE")
    print("="*70)

    print(f"\nFINAL RESULT:")
    print(f"  Model: RandomForest (Nested Optuna-LOSO)")
    print(f"  LOSO Balanced Accuracy: {ba['mean']:.4f} +/- {ba['std']:.4f}")
    print(f"  95% CI: [{ba['ci_95_lower']:.4f}, {ba['ci_95_upper']:.4f}]")
    print(f"  Paper 1 Baseline: {vs_p1['paper1_baseline']:.4f}")
    print(f"  Improvement: {vs_p1['improvement']:+.4f} ({vs_p1['improvement_pct']:+.2f}%)")
    print(f"  p-value: {vs_p1['p_value']:.6f}")

    if vs_p1['beats_baseline']:
        print(f"\n  *** SUCCESS: BEAT PAPER 1 BASELINE! ***")
    else:
        print(f"\n  Result below Paper 1 baseline")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
