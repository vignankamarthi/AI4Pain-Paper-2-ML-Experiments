"""
Phase 4 Experiment 4.3+4.4: Full Model Training + LOSO Validation

Trains ALL valid model+normalization combinations:
- LightGBM x {per_subject_baseline, global_zscore, raw}
- Simple MLP x {per_subject_baseline, global_zscore}  (NO raw - invalid)
- Stacked x {per_subject_baseline, global_zscore, raw}

Total: 8 valid experiments

Then LOSO validates all trained models.
"""

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

warnings.filterwarnings('ignore')

# Constants
BEST_DIMENSION = 5
BEST_TAU = 2
FEATURE_COLS = ['pe', 'comp', 'fisher_shannon', 'fisher_info', 'renyipe', 'renyicomp', 'tsallispe', 'tsalliscomp']
SIGNALS = ['eda', 'bvp', 'resp']
CLASS_MAP = {'baseline': 0, 'rest': 0, 'low': 1, 'high': 2}
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'features'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'phase4' / 'experiment_4.3_full_training'


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


class SimpleMLP(nn.Module):
    """Simple 2-layer MLP for 3-class classification."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, epochs=50):
    """Train MLP and return validation balanced accuracy."""
    model = SimpleMLP(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_t = torch.FloatTensor(X_train)
    y_t = torch.LongTensor(y_train)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_val)).argmax(dim=1).numpy()

    return balanced_accuracy_score(y_val, preds)


def get_lightgbm():
    """Get LightGBM classifier with good defaults."""
    return lgb.LGBMClassifier(
        n_estimators=150, max_depth=8, learning_rate=0.05,
        num_leaves=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=1
    )


def get_stacked():
    """Get Stacked Ensemble (RF + XGB + LGB)."""
    return StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=1)),
            ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0, n_jobs=1)),
            ('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=8, random_state=RANDOM_STATE, verbose=-1, n_jobs=1))
        ],
        final_estimator=lgb.LGBMClassifier(n_estimators=50, verbose=-1, n_jobs=1),
        cv=3, n_jobs=1
    )


def run_cv_evaluation(model_name, norm_name, X, y, norm_fn, subjects):
    """Run 5-fold CV evaluation."""
    X_norm = norm_fn(X, y, subjects)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scorer = make_scorer(balanced_accuracy_score)

    if model_name == 'SimpleMLP':
        scores = []
        for train_idx, val_idx in cv.split(X_norm, y):
            score = train_mlp(X_norm[train_idx], y[train_idx], X_norm[val_idx], y[val_idx], epochs=50)
            scores.append(score)
        return np.mean(scores) * 100, np.std(scores) * 100
    else:
        model = get_lightgbm() if model_name == 'LightGBM' else get_stacked()
        scores = cross_val_score(model, X_norm, y, cv=cv, scoring=scorer, n_jobs=1)
        return np.mean(scores) * 100, np.std(scores) * 100


def run_loso_evaluation(model_name, norm_name, X, y, norm_fn, subjects):
    """Run Leave-One-Subject-Out validation."""
    logo = LeaveOneGroupOut()
    subject_scores = []

    for train_idx, test_idx in logo.split(X, y, subjects):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        subjects_train = subjects[train_idx]

        # Apply normalization (fit on train, transform test using train stats)
        if norm_name == 'per_subject_baseline':
            # For LOSO, we normalize train subjects using their own baselines
            X_train_norm = per_subject_baseline_norm(X_train, y_train, subjects_train)
            # For test subject, use global mean/std from train (no baseline available)
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

        if model_name == 'SimpleMLP':
            score = train_mlp(X_train_norm, y_train, X_test_norm, y_test, epochs=50)
        else:
            model = get_lightgbm() if model_name == 'LightGBM' else get_stacked()
            model.fit(X_train_norm, y_train)
            preds = model.predict(X_test_norm)
            score = balanced_accuracy_score(y_test, preds)

        subject_scores.append(score)

    return np.mean(subject_scores) * 100, np.std(subject_scores) * 100, [s * 100 for s in subject_scores]


def main():
    print('=' * 70)
    print('PHASE 4 - FULL MODEL TRAINING + LOSO VALIDATION')
    print('=' * 70)
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print('\nLoading data...')
    X, y, subjects = load_data()
    print(f'  Samples: {X.shape[0]}, Features: {X.shape[1]}')
    print(f'  Subjects: {len(np.unique(subjects))}')
    print(f'  Classes: {np.bincount(y)}')

    # Define valid experiments (8 total - excluding SimpleMLP + raw)
    experiments = [
        ('LightGBM', 'per_subject_baseline', per_subject_baseline_norm),
        ('LightGBM', 'global_zscore', global_zscore_norm),
        ('LightGBM', 'raw', raw_norm),
        ('SimpleMLP', 'per_subject_baseline', per_subject_baseline_norm),
        ('SimpleMLP', 'global_zscore', global_zscore_norm),
        # SimpleMLP + raw is INVALID (excluded)
        ('Stacked', 'per_subject_baseline', per_subject_baseline_norm),
        ('Stacked', 'global_zscore', global_zscore_norm),
        ('Stacked', 'raw', raw_norm),
    ]

    cv_results = []
    loso_results = []

    # Run all experiments
    for i, (model_name, norm_name, norm_fn) in enumerate(experiments):
        print(f'\n[{i+1}/8] {model_name} + {norm_name}')
        print('-' * 50)

        # 5-fold CV
        print('  Running 5-fold CV...')
        cv_mean, cv_std = run_cv_evaluation(model_name, norm_name, X, y, norm_fn, subjects)
        print(f'    CV: {cv_mean:.2f}% +/- {cv_std:.2f}%')
        cv_results.append({
            'model': model_name,
            'normalization': norm_name,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        })

        # LOSO
        print('  Running LOSO validation...')
        loso_mean, loso_std, per_subject = run_loso_evaluation(model_name, norm_name, X, y, norm_fn, subjects)
        print(f'    LOSO: {loso_mean:.2f}% +/- {loso_std:.2f}%')
        loso_results.append({
            'model': model_name,
            'normalization': norm_name,
            'loso_mean': loso_mean,
            'loso_std': loso_std,
            'per_subject_scores': per_subject
        })

    # Create summary DataFrames
    cv_df = pd.DataFrame(cv_results).sort_values('cv_mean', ascending=False)
    loso_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'per_subject_scores'} for r in loso_results])
    loso_df = loso_df.sort_values('loso_mean', ascending=False)

    # Save results
    cv_df.to_csv(RESULTS_DIR / 'cv_leaderboard.csv', index=False)
    loso_df.to_csv(RESULTS_DIR / 'loso_leaderboard.csv', index=False)

    # Combined leaderboard
    combined = cv_df.merge(loso_df, on=['model', 'normalization'])
    combined['cv_loso_gap'] = combined['cv_mean'] - combined['loso_mean']
    combined = combined.sort_values('loso_mean', ascending=False)
    combined.to_csv(RESULTS_DIR / 'combined_leaderboard.csv', index=False)

    # Save detailed results
    with open(RESULTS_DIR / 'full_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'cv_results': cv_results,
            'loso_results': [{k: v for k, v in r.items()} for r in loso_results],
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'n_subjects': int(len(np.unique(subjects)))
        }, f, indent=2)

    # Print final summary
    print('\n' + '=' * 70)
    print('FINAL RESULTS SUMMARY')
    print('=' * 70)

    print('\nCV Leaderboard (5-fold):')
    print(cv_df.to_string(index=False))

    print('\nLOSO Leaderboard:')
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
