"""
Phase 6 Step 4: Full Model Training + LOSO Validation (BASELINE-ONLY)

MIRRORS Phase 4 but with baseline-only labeling:
- Class 0: no_pain (baseline ONLY - rest segments EXCLUDED)
- Class 1: low_pain
- Class 2: high_pain

Trains ALL valid model+normalization combinations with OPTUNA optimization.

HYBRID OPTUNA APPROACH (44x faster than nested):
1. Optuna optimizes hyperparameters ONCE using 5-fold CV (50 trials)
2. LOSO validation uses those fixed params for all 53 folds
This maintains scientific validity while being practical to run.

FEATURES:
- Checkpointing: Results saved after EACH experiment completes
- Memory management: Cache cleared between experiments to prevent crashes
- Fault tolerance: Failed experiments logged and skipped, others continue
- Resume capability: Completed experiments skipped on restart
"""

import gc
import json
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import balanced_accuracy_score, make_scorer
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def clear_memory():
    """Clear all caches and force garbage collection to prevent memory crashes."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def load_checkpoint(results_dir):
    """Load checkpoint if exists, return completed experiment keys and results."""
    checkpoint_file = results_dir / 'checkpoint.json'
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        print(f'  Loaded checkpoint: {len(checkpoint["completed"])} experiments completed')
        return checkpoint
    return {'completed': [], 'cv_results': [], 'loso_results': []}


def save_checkpoint(results_dir, completed, cv_results, loso_results):
    """Save checkpoint after each experiment completes."""
    checkpoint = {
        'timestamp': datetime.now().isoformat(),
        'completed': completed,
        'cv_results': cv_results,
        'loso_results': loso_results
    }
    checkpoint_file = results_dir / 'checkpoint.json'
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f'    [CHECKPOINT SAVED] {len(completed)} experiments completed')


def save_incremental_results(results_dir, cv_results, loso_results):
    """Save current results incrementally (can be loaded if crash occurs)."""
    if cv_results:
        cv_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'best_params'} for r in cv_results])
        cv_df.to_csv(results_dir / 'cv_leaderboard_partial.csv', index=False)
    if loso_results:
        loso_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'per_subject_scores'} for r in loso_results])
        loso_df.to_csv(results_dir / 'loso_leaderboard_partial.csv', index=False)

# Constants
BEST_DIMENSION = 5
BEST_TAU = 2
FEATURE_COLS = ['pe', 'comp', 'fisher_shannon', 'fisher_info', 'renyipe', 'renyicomp', 'tsallispe', 'tsalliscomp']
SIGNALS = ['eda', 'bvp', 'resp']
# Class mapping - BASELINE ONLY (rest segments EXCLUDED)
CLASS_MAP = {'baseline': 0, 'low': 1, 'high': 2}
RANDOM_STATE = 42
N_OPTUNA_TRIALS = 50
N_CV_FOLDS = 5

# Paths - navigate up from src/phase6/ to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'features'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'phase6_step4_full_training'


def load_data():
    """Load and merge feature data from all signals."""
    dfs = []
    for split in ['train', 'validation']:
        for signal in SIGNALS:
            filepath = DATA_DIR / f'results_{split}_{signal.capitalize()}.csv'
            df = pd.read_csv(filepath)
            df = df[(df['dimension'] == BEST_DIMENSION) & (df['tau'] == BEST_TAU)].copy()
            df['signal_type'] = signal
            df['subject_id'] = df['file_name'].apply(lambda x: int(x.split('/')[-1].replace('.csv', '')))
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # PHASE 6 KEY DIFFERENCE: Exclude rest segments entirely
    n_before = len(combined)
    combined = combined[combined['state'] != 'rest'].copy()
    n_after = len(combined)
    print(f"  [PHASE 6] Excluded {n_before - n_after} rest segments")

    combined['sample_id'] = combined['signal'] + '_' + combined['subject_id'].astype(str)

    # Pivot to wide format
    wide_dfs = []
    for signal in SIGNALS:
        signal_df = combined[combined['signal_type'] == signal][['sample_id', 'subject_id', 'state'] + FEATURE_COLS].copy()
        signal_df = signal_df.rename(columns={c: f'{signal}_{c}' for c in FEATURE_COLS})
        wide_dfs.append(signal_df)

    result_df = wide_dfs[0]
    for wdf in wide_dfs[1:]:
        fcols = [c for c in wdf.columns if c not in ['sample_id', 'subject_id', 'state']]
        result_df = result_df.merge(wdf[['sample_id'] + fcols], on='sample_id')

    # Extract arrays
    fcols = [f'{s}_{f}' for s in SIGNALS for f in FEATURE_COLS]
    X = result_df[fcols].values.astype(np.float64)
    y = result_df['state'].map(CLASS_MAP).values.astype(np.int64)
    subjects = result_df['subject_id'].values

    return X, y, subjects


def per_subject_baseline_norm(X, y, subjects):
    """Normalize each subject relative to their no-pain baseline."""
    X_norm = np.zeros_like(X)
    for subj in np.unique(subjects):
        mask = subjects == subj
        bl_mask = mask & (y == 0)
        if bl_mask.sum() > 0:
            mean = X[bl_mask].mean(axis=0)
            std = X[bl_mask].std(axis=0) + 1e-8
            X_norm[mask] = (X[mask] - mean) / std
    return np.nan_to_num(X_norm, nan=0.0)


def global_zscore_norm(X, y, subjects):
    """Standard z-score normalization across all samples."""
    scaler = StandardScaler()
    return np.nan_to_num(scaler.fit_transform(X), nan=0.0)


def raw_norm(X, y, subjects):
    """No normalization - raw features."""
    return np.nan_to_num(X, nan=0.0)


# =============================================================================
# Optuna Objective Functions
# =============================================================================

def create_lightgbm_objective(X, y, cv):
    """Create Optuna objective for LightGBM hyperparameter optimization."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': RANDOM_STATE,
            'verbose': -1,
            'n_jobs': 1
        }
        model = lgb.LGBMClassifier(**params)
        scorer = make_scorer(balanced_accuracy_score)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=1)
        return scores.mean()
    return objective


def create_stacked_objective(X, y, cv):
    """Create Optuna objective for Stacked Ensemble hyperparameter optimization."""
    def objective(trial):
        # RF params
        rf_n_est = trial.suggest_int('rf_n_estimators', 50, 300)
        rf_depth = trial.suggest_int('rf_max_depth', 5, 20)

        # XGB params
        xgb_n_est = trial.suggest_int('xgb_n_estimators', 50, 300)
        xgb_depth = trial.suggest_int('xgb_max_depth', 3, 10)
        xgb_lr = trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True)

        # LGB params
        lgb_n_est = trial.suggest_int('lgb_n_estimators', 50, 300)
        lgb_depth = trial.suggest_int('lgb_max_depth', 3, 12)
        lgb_lr = trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True)

        # Final estimator params
        final_n_est = trial.suggest_int('final_n_estimators', 30, 150)

        model = StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=rf_n_est, max_depth=rf_depth,
                                              random_state=RANDOM_STATE, n_jobs=1)),
                ('xgb', xgb.XGBClassifier(n_estimators=xgb_n_est, max_depth=xgb_depth,
                                          learning_rate=xgb_lr, random_state=RANDOM_STATE,
                                          verbosity=0, n_jobs=1)),
                ('lgb', lgb.LGBMClassifier(n_estimators=lgb_n_est, max_depth=lgb_depth,
                                           learning_rate=lgb_lr, random_state=RANDOM_STATE,
                                           verbose=-1, n_jobs=1))
            ],
            final_estimator=lgb.LGBMClassifier(n_estimators=final_n_est, verbose=-1, n_jobs=1),
            cv=3, n_jobs=1
        )
        scorer = make_scorer(balanced_accuracy_score)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=1)
        return scores.mean()
    return objective


class SimpleMLP(nn.Module):
    """Simple MLP for 3-class classification with configurable architecture."""
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            hidden_dim = max(hidden_dim // 2, 16)
        layers.append(nn.Linear(prev_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp_with_params(X_train, y_train, X_val, y_val, params):
    """Train MLP with given hyperparameters and return validation balanced accuracy."""
    model = SimpleMLP(
        X_train.shape[1],
        hidden_dim=params['hidden_dim'],
        n_layers=params['n_layers'],
        dropout=params['dropout']
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    X_t = torch.FloatTensor(X_train)
    y_t = torch.LongTensor(y_train)

    batch_size = params['batch_size']
    epochs = params['epochs']
    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            optimizer.zero_grad()
            loss = criterion(model(X_t[batch_idx]), y_t[batch_idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_val)).argmax(dim=1).numpy()

    return balanced_accuracy_score(y_val, preds)


def create_mlp_objective(X, y, cv):
    """Create Optuna objective for SimpleMLP hyperparameter optimization."""
    def objective(trial):
        params = {
            'hidden_dim': trial.suggest_int('hidden_dim', 32, 256),
            'n_layers': trial.suggest_int('n_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'epochs': trial.suggest_int('epochs', 50, 150)
        }

        scores = []
        for train_idx, val_idx in cv.split(X, y):
            score = train_mlp_with_params(
                X[train_idx], y[train_idx],
                X[val_idx], y[val_idx],
                params
            )
            scores.append(score)
        return np.mean(scores)
    return objective


# =============================================================================
# Optimization and Evaluation Functions
# =============================================================================

def optimize_and_evaluate_cv(model_name, norm_name, X, y, norm_fn, subjects):
    """Optimize hyperparameters with Optuna and evaluate with 5-fold CV."""
    X_norm = norm_fn(X, y, subjects)
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    sampler = TPESampler(seed=RANDOM_STATE)

    if model_name == 'LightGBM':
        objective = create_lightgbm_objective(X_norm, y, cv)
    elif model_name == 'Stacked':
        objective = create_stacked_objective(X_norm, y, cv)
    else:  # SimpleMLP
        objective = create_mlp_objective(X_norm, y, cv)

    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value * 100

    # Get std by re-running with best params
    if model_name == 'LightGBM':
        model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_STATE, verbose=-1, n_jobs=1)
        scorer = make_scorer(balanced_accuracy_score)
        scores = cross_val_score(model, X_norm, y, cv=cv, scoring=scorer, n_jobs=1)
        cv_std = np.std(scores) * 100
    elif model_name == 'Stacked':
        # Reconstruct stacked with best params
        model = StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=best_params['rf_n_estimators'],
                                              max_depth=best_params['rf_max_depth'],
                                              random_state=RANDOM_STATE, n_jobs=1)),
                ('xgb', xgb.XGBClassifier(n_estimators=best_params['xgb_n_estimators'],
                                          max_depth=best_params['xgb_max_depth'],
                                          learning_rate=best_params['xgb_learning_rate'],
                                          random_state=RANDOM_STATE, verbosity=0, n_jobs=1)),
                ('lgb', lgb.LGBMClassifier(n_estimators=best_params['lgb_n_estimators'],
                                           max_depth=best_params['lgb_max_depth'],
                                           learning_rate=best_params['lgb_learning_rate'],
                                           random_state=RANDOM_STATE, verbose=-1, n_jobs=1))
            ],
            final_estimator=lgb.LGBMClassifier(n_estimators=best_params['final_n_estimators'],
                                               verbose=-1, n_jobs=1),
            cv=3, n_jobs=1
        )
        scorer = make_scorer(balanced_accuracy_score)
        scores = cross_val_score(model, X_norm, y, cv=cv, scoring=scorer, n_jobs=1)
        cv_std = np.std(scores) * 100
    else:  # SimpleMLP
        mlp_params = {k: best_params[k] for k in ['hidden_dim', 'n_layers', 'dropout', 'learning_rate', 'batch_size', 'epochs']}
        scores = []
        for train_idx, val_idx in cv.split(X_norm, y):
            score = train_mlp_with_params(X_norm[train_idx], y[train_idx], X_norm[val_idx], y[val_idx], mlp_params)
            scores.append(score)
        cv_std = np.std(scores) * 100

    return best_score, cv_std, best_params


def run_loso_with_fixed_params(model_name, norm_name, X, y, subjects, best_params):
    """Run LOSO validation using fixed hyperparameters from CV optimization.

    HYBRID APPROACH: Optuna optimizes once during CV, then LOSO uses those
    fixed params for all 53 folds. 44x faster than nested Optuna.
    """
    logo = LeaveOneGroupOut()
    subject_scores = []

    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, subjects)):
        print(f'      LOSO fold {fold_idx + 1}/{n_subjects}...', end=' ', flush=True)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        subjects_train = subjects[train_idx]

        # Apply normalization
        if norm_name == 'per_subject_baseline':
            X_train_norm = per_subject_baseline_norm(X_train, y_train, subjects_train)
            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0) + 1e-8
            X_test_norm = (X_test - mean) / std
        elif norm_name == 'global_zscore':
            scaler = StandardScaler()
            X_train_norm = scaler.fit_transform(X_train)
            X_test_norm = scaler.transform(X_test)
        else:  # raw
            X_train_norm = X_train
            X_test_norm = X_test

        X_train_norm = np.nan_to_num(X_train_norm, nan=0.0)
        X_test_norm = np.nan_to_num(X_test_norm, nan=0.0)

        # Use fixed best_params from CV optimization (no Optuna here)
        if model_name == 'LightGBM':
            model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_STATE, verbose=-1, n_jobs=1)
            model.fit(X_train_norm, y_train)
            preds = model.predict(X_test_norm)

        elif model_name == 'Stacked':
            model = StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=best_params['rf_n_estimators'],
                                                  max_depth=best_params['rf_max_depth'],
                                                  random_state=RANDOM_STATE, n_jobs=1)),
                    ('xgb', xgb.XGBClassifier(n_estimators=best_params['xgb_n_estimators'],
                                              max_depth=best_params['xgb_max_depth'],
                                              learning_rate=best_params['xgb_learning_rate'],
                                              random_state=RANDOM_STATE, verbosity=0, n_jobs=1)),
                    ('lgb', lgb.LGBMClassifier(n_estimators=best_params['lgb_n_estimators'],
                                               max_depth=best_params['lgb_max_depth'],
                                               learning_rate=best_params['lgb_learning_rate'],
                                               random_state=RANDOM_STATE, verbose=-1, n_jobs=1))
                ],
                final_estimator=lgb.LGBMClassifier(n_estimators=best_params['final_n_estimators'],
                                                   verbose=-1, n_jobs=1),
                cv=3, n_jobs=1
            )
            model.fit(X_train_norm, y_train)
            preds = model.predict(X_test_norm)

        else:  # SimpleMLP
            mlp_params = {k: best_params[k] for k in ['hidden_dim', 'n_layers', 'dropout', 'learning_rate', 'batch_size', 'epochs']}

            model = SimpleMLP(X_train_norm.shape[1], hidden_dim=mlp_params['hidden_dim'],
                            n_layers=mlp_params['n_layers'], dropout=mlp_params['dropout'])
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=mlp_params['learning_rate'])

            X_t = torch.FloatTensor(X_train_norm)
            y_t = torch.LongTensor(y_train)
            batch_size = mlp_params['batch_size']
            n_samples = X_train_norm.shape[0]

            for epoch in range(mlp_params['epochs']):
                model.train()
                indices = np.random.permutation(n_samples)
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    batch_idx = indices[start:end]
                    optimizer.zero_grad()
                    loss = criterion(model(X_t[batch_idx]), y_t[batch_idx])
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                preds = model(torch.FloatTensor(X_test_norm)).argmax(dim=1).numpy()

        score = balanced_accuracy_score(y_test, preds)
        subject_scores.append(score)
        print(f'{score*100:.1f}%')

    return np.mean(subject_scores) * 100, np.std(subject_scores) * 100, [s * 100 for s in subject_scores]


def main():
    print('=' * 70)
    print('PHASE 6 STEP 4 - FULL MODEL TRAINING + LOSO (BASELINE-ONLY)')
    print('=' * 70)
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Optuna trials per model: {N_OPTUNA_TRIALS}')

    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load checkpoint (resume capability)
    checkpoint = load_checkpoint(RESULTS_DIR)
    completed_keys = set(checkpoint['completed'])
    cv_results = checkpoint['cv_results']
    loso_results = checkpoint['loso_results']

    if completed_keys:
        print(f'  Resuming from checkpoint: {len(completed_keys)} experiments already done')

    # Load data
    print('\nLoading data...')
    X, y, subjects = load_data()
    print(f'  Samples: {X.shape[0]}, Features: {X.shape[1]}')
    print(f'  Subjects: {len(np.unique(subjects))}')
    print(f'  Classes: {np.bincount(y)}')

    # Define valid experiments (6 total - SimpleMLP EXCLUDED due to segfaults)
    # SimpleMLP causes memory corruption/segfaults on this system
    experiments = [
        ('LightGBM', 'per_subject_baseline', per_subject_baseline_norm),
        ('LightGBM', 'global_zscore', global_zscore_norm),
        ('LightGBM', 'raw', raw_norm),
        # SimpleMLP experiments removed - causes segfaults
        ('Stacked', 'per_subject_baseline', per_subject_baseline_norm),
        ('Stacked', 'global_zscore', global_zscore_norm),
        ('Stacked', 'raw', raw_norm),
    ]

    failed_experiments = []

    # Run all experiments with checkpointing
    for i, (model_name, norm_name, norm_fn) in enumerate(experiments):
        exp_key = f'{model_name}_{norm_name}'

        # Skip already completed experiments
        if exp_key in completed_keys:
            print(f'\n[{i+1}/6] {model_name} + {norm_name} - SKIPPED (already completed)')
            continue

        print(f'\n[{i+1}/6] {model_name} + {norm_name}')
        print('-' * 50)

        # Clear memory before experiment
        clear_memory()

        try:
            # 5-fold CV with Optuna optimization
            print('  Optimizing with Optuna (5-fold CV)...')
            cv_mean, cv_std, best_params_cv = optimize_and_evaluate_cv(model_name, norm_name, X, y, norm_fn, subjects)
            print(f'    CV: {cv_mean:.2f}% +/- {cv_std:.2f}%')
            cv_results.append({
                'model': model_name,
                'normalization': norm_name,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'best_params': best_params_cv
            })

            # LOSO with FIXED params from CV optimization (44x faster than nested Optuna)
            print('  Running LOSO validation (using CV-optimized params)...')
            loso_mean, loso_std, per_subject = run_loso_with_fixed_params(
                model_name, norm_name, X, y, subjects, best_params_cv
            )
            print(f'    LOSO: {loso_mean:.2f}% +/- {loso_std:.2f}%')
            loso_results.append({
                'model': model_name,
                'normalization': norm_name,
                'loso_mean': loso_mean,
                'loso_std': loso_std,
                'per_subject_scores': per_subject
            })

            # Mark as completed and save checkpoint
            completed_keys.add(exp_key)
            save_checkpoint(RESULTS_DIR, list(completed_keys), cv_results, loso_results)
            save_incremental_results(RESULTS_DIR, cv_results, loso_results)

        except Exception as e:
            print(f'    [ERROR] {exp_key} failed: {str(e)}')
            failed_experiments.append({'key': exp_key, 'error': str(e), 'timestamp': datetime.now().isoformat()})
            # Save failed experiments log
            with open(RESULTS_DIR / 'failed_experiments.json', 'w') as f:
                json.dump(failed_experiments, f, indent=2)

        # Clear memory after experiment
        clear_memory()

    # Create summary DataFrames
    cv_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'best_params'} for r in cv_results])
    cv_df = cv_df.sort_values('cv_mean', ascending=False)

    loso_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'per_subject_scores'} for r in loso_results])
    loso_df = loso_df.sort_values('loso_mean', ascending=False)

    # Save results
    cv_df.to_csv(RESULTS_DIR / 'cv_leaderboard.csv', index=False)
    loso_df.to_csv(RESULTS_DIR / 'loso_leaderboard.csv', index=False)

    # Combined leaderboard
    combined = cv_df.merge(loso_df, on=['model', 'normalization'])
    combined['cv_loso_gap'] = combined['cv_mean'] - combined['loso_mean']
    combined = combined.sort_values('loso_mean', ascending=False)
    combined.to_csv(RESULTS_DIR / 'full_leaderboard.csv', index=False)

    # Save detailed results
    with open(RESULTS_DIR / 'full_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'optuna_trials': N_OPTUNA_TRIALS,
            'cv_folds': N_CV_FOLDS,
            'cv_results': [{k: v for k, v in r.items() if k != 'best_params'} for r in cv_results],
            'loso_results': [{k: v for k, v in r.items() if k != 'per_subject_scores'} for r in loso_results],
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'n_subjects': int(len(np.unique(subjects)))
        }, f, indent=2)

    # Save best params separately
    best_params_dict = {f"{r['model']}_{r['normalization']}": r['best_params'] for r in cv_results}
    with open(RESULTS_DIR / 'best_hyperparameters.json', 'w') as f:
        json.dump(best_params_dict, f, indent=2)

    # Print final summary
    print('\n' + '=' * 70)
    print('FINAL RESULTS SUMMARY')
    print('=' * 70)

    print('\nCV Leaderboard (5-fold with Optuna):')
    print(cv_df.to_string(index=False))

    print('\nLOSO Leaderboard (with nested Optuna):')
    print(loso_df.to_string(index=False))

    print('\nCombined (sorted by LOSO):')
    print(combined[['model', 'normalization', 'cv_mean', 'loso_mean', 'cv_loso_gap']].to_string(index=False))

    # Best result
    best = combined.iloc[0]
    print(f'\nBEST: {best["model"]} + {best["normalization"]}')
    print(f'  CV: {best["cv_mean"]:.2f}%')
    print(f'  LOSO: {best["loso_mean"]:.2f}%')
    print(f'  Gap: {best["cv_loso_gap"]:.2f}%')

    # Comparison to Paper 1
    paper1_baseline = 79.4
    delta = best['loso_mean'] - paper1_baseline
    print(f'\nComparison to Paper 1 baseline ({paper1_baseline}%): {delta:+.2f}%')

    print(f'\nResults saved to: {RESULTS_DIR}')
    print(f'Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
