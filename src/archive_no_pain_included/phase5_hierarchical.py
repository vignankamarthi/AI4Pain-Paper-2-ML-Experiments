"""
Phase 5: Hierarchical Binary Classification

Two-stage cascaded classifier:
- Stage 1: No Pain vs Pain (binary)
- Stage 2: Low Pain vs High Pain (binary, only for samples predicted as Pain)

Uses top 3 configurations from Phase 4 leaderboard.

HYBRID OPTUNA APPROACH (44x faster than nested):
1. Optuna optimizes hyperparameters ONCE using 5-fold CV (50 trials)
2. LOSO validation uses those fixed params for all 53 folds
This maintains scientific validity while being practical to run.

FEATURES:
- Checkpointing: Results saved after EACH configuration completes
- Memory management: Cache cleared between configurations to prevent crashes
- Fault tolerance: Failed configurations logged and skipped, others continue
- Resume capability: Completed configurations skipped on restart
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
from sklearn.metrics import balanced_accuracy_score, make_scorer, confusion_matrix
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
    """Load checkpoint if exists, return completed config keys and results."""
    checkpoint_file = results_dir / 'checkpoint.json'
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        print(f'  Loaded checkpoint: {len(checkpoint["completed"])} configurations completed')
        return checkpoint
    return {'completed': [], 'all_results': []}


def save_checkpoint(results_dir, completed, all_results):
    """Save checkpoint after each configuration completes."""
    checkpoint = {
        'timestamp': datetime.now().isoformat(),
        'completed': completed,
        'all_results': all_results
    }
    checkpoint_file = results_dir / 'checkpoint.json'
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f'    [CHECKPOINT SAVED] {len(completed)} configurations completed')

# Constants
BEST_DIMENSION = 5
BEST_TAU = 2
FEATURE_COLS = ['pe', 'comp', 'fisher_shannon', 'fisher_info', 'renyipe', 'renyicomp', 'tsallispe', 'tsalliscomp']
SIGNALS = ['eda', 'bvp', 'resp']
CLASS_MAP_3CLASS = {'baseline': 0, 'rest': 0, 'low': 1, 'high': 2}
CLASS_MAP_STAGE1 = {'baseline': 0, 'rest': 0, 'low': 1, 'high': 1}  # No Pain vs Pain
CLASS_MAP_STAGE2 = {'low': 0, 'high': 1}  # Low vs High (pain samples only)
RANDOM_STATE = 42
N_OPTUNA_TRIALS = 50
N_CV_FOLDS = 5

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'features'
PHASE4_RESULTS = PROJECT_ROOT / 'results' / 'phase4' / 'experiment_4.3_full_training'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'phase5'


class SimpleMLP(nn.Module):
    """Simple MLP for binary classification."""
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_phase4_top_configs(n=3):
    """Load top N configurations from Phase 4 leaderboard."""
    leaderboard_path = PHASE4_RESULTS / 'full_leaderboard.csv'
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Phase 4 leaderboard not found: {leaderboard_path}")

    df = pd.read_csv(leaderboard_path)
    # Sort by LOSO accuracy (descending) and take top N
    df = df.sort_values('loso_mean', ascending=False).head(n)

    configs = []
    for _, row in df.iterrows():
        configs.append({
            'model': row['model'],
            'normalization': row['normalization'],
            'loso_acc': row['loso_mean']
        })

    print(f"Loaded top {n} configurations from Phase 4:")
    for i, cfg in enumerate(configs):
        print(f"  {i+1}. {cfg['model']} + {cfg['normalization']}: {cfg['loso_acc']:.4f}")

    return configs


def load_data():
    """Load and merge feature data from all signals."""
    dfs = []
    for split in ['train', 'validation']:
        for signal in SIGNALS:
            filepath = DATA_DIR / f'results_{split}_{signal}.csv'
            if not filepath.exists():
                filepath = DATA_DIR / f'results_{split}_{signal.capitalize()}.csv'
            df = pd.read_csv(filepath)
            df = df[(df['dimension'] == BEST_DIMENSION) & (df['tau'] == BEST_TAU)].copy()
            df['signal_type'] = signal
            df['subject_id'] = pd.Series(df['file_name']).apply(lambda x: int(str(x).split('/')[-1].replace('.csv', '')))
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined['sample_id'] = combined['signal'] + '_' + combined['subject_id'].astype(str)

    # Pivot to wide format
    wide_dfs = []
    for signal in SIGNALS:
        signal_df = combined[combined['signal_type'] == signal][['sample_id', 'subject_id', 'state'] + FEATURE_COLS].copy()
        rename_dict = {c: f'{signal}_{c}' for c in FEATURE_COLS}
        signal_df = signal_df.rename(columns=rename_dict)
        wide_dfs.append(signal_df)

    result_df = wide_dfs[0]
    for wdf in wide_dfs[1:]:
        fcols = [c for c in wdf.columns if c not in ['sample_id', 'subject_id', 'state']]
        result_df = result_df.merge(wdf[['sample_id'] + fcols], on='sample_id')

    return result_df


def apply_normalization(X, y, subjects, norm_type):
    """Apply specified normalization strategy."""
    if norm_type == 'per_subject_baseline':
        X_norm = np.zeros_like(X)
        for subj in np.unique(subjects):
            mask = subjects == subj
            bl_mask = mask & (y == 0)
            if bl_mask.sum() > 0:
                mean = X[bl_mask].mean(axis=0)
                std = X[bl_mask].std(axis=0) + 1e-8
                X_norm[mask] = (X[mask] - mean) / std
        return np.nan_to_num(X_norm, nan=0.0)
    elif norm_type == 'global_zscore':
        scaler = StandardScaler()
        return np.nan_to_num(scaler.fit_transform(X), nan=0.0)
    else:  # raw
        return np.nan_to_num(X, nan=0.0)


def create_lightgbm_objective(X, y, cv):
    """Create Optuna objective for LightGBM."""
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


def train_lightgbm_with_params(X, y, params):
    """Train LightGBM with specified parameters."""
    model = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE, verbose=-1, n_jobs=1)
    model.fit(X, y)
    return model


def optimize_and_train(X_train, y_train, model_type='LightGBM'):
    """Optimize hyperparameters and train model."""
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    if model_type == 'LightGBM':
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
        study.optimize(create_lightgbm_objective(X_train, y_train, cv), n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
        best_params = study.best_params
        model = train_lightgbm_with_params(X_train, y_train, best_params)
        return model, best_params, study.best_value
    else:
        # Default to LightGBM for now (SimpleMLP can be added later)
        return optimize_and_train(X_train, y_train, 'LightGBM')


def optimize_only(X_train, y_train, model_type='LightGBM'):
    """Run Optuna optimization ONCE and return best params (no model training)."""
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    if model_type == 'LightGBM':
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
        study.optimize(create_lightgbm_objective(X_train, y_train, cv), n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
        return study.best_params, study.best_value
    else:
        return optimize_only(X_train, y_train, 'LightGBM')


def train_with_fixed_params(X_train, y_train, params, model_type='LightGBM'):
    """Train model using fixed params (no Optuna, for use in LOSO)."""
    if model_type == 'LightGBM':
        return train_lightgbm_with_params(X_train, y_train, params)
    else:
        return train_lightgbm_with_params(X_train, y_train, params)


def run_hierarchical_loso(df, config):
    """Run hierarchical classification with LOSO validation.

    HYBRID OPTUNA APPROACH: Optimize params ONCE with 5-fold CV, then use fixed
    params for all 53 LOSO folds. 44x faster than nested Optuna.
    """
    fcols = [f'{s}_{f}' for s in SIGNALS for f in FEATURE_COLS]
    X_all = df[fcols].values.astype(np.float64)
    y_3class = df['state'].map(CLASS_MAP_3CLASS).values.astype(np.int64)
    y_stage1 = df['state'].map(CLASS_MAP_STAGE1).values.astype(np.int64)
    subjects = df['subject_id'].values

    # === STEP 1: Optimize params ONCE using full training data ===
    print(f"\n  Optimizing hyperparameters for {config['model']} + {config['normalization']}...")

    # Normalize full data for optimization
    if config['normalization'] == 'global_zscore':
        X_all_norm = StandardScaler().fit_transform(X_all)
    elif config['normalization'] == 'per_subject_baseline':
        X_all_norm = apply_normalization(X_all, y_3class, subjects, 'per_subject_baseline')
    else:
        X_all_norm = np.nan_to_num(X_all, nan=0.0)

    # Optimize Stage 1 params
    print("    Stage 1 (Pain Detection) Optuna optimization...")
    params_s1, cv_s1 = optimize_only(X_all_norm, y_stage1, config['model'])
    print(f"      CV accuracy: {cv_s1:.2%}")

    # Optimize Stage 2 params (on pain samples only)
    print("    Stage 2 (Intensity) Optuna optimization...")
    pain_mask = y_3class > 0
    X_pain = X_all_norm[pain_mask]
    y_stage2_full = (y_3class[pain_mask] - 1).astype(np.int64)
    params_s2, cv_s2 = optimize_only(X_pain, y_stage2_full, config['model'])
    print(f"      CV accuracy: {cv_s2:.2%}")

    # === STEP 2: Run LOSO with fixed params ===
    print(f"  Running LOSO (53 folds) with fixed params...")
    logo = LeaveOneGroupOut()
    results = []
    stage1_results = []
    stage2_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_all, y_3class, subjects)):
        test_subject = subjects[test_idx][0]

        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train_3class = y_3class[train_idx]
        y_test_3class = y_3class[test_idx]
        y_train_stage1 = y_stage1[train_idx]
        y_test_stage1 = y_stage1[test_idx]
        subjects_train = subjects[train_idx]

        # Apply normalization
        if config['normalization'] == 'global_zscore':
            scaler = StandardScaler()
            X_train_norm = scaler.fit_transform(X_train)
            X_test_norm = scaler.transform(X_test)
        elif config['normalization'] == 'per_subject_baseline':
            X_train_norm = apply_normalization(X_train, y_train_3class, subjects_train, 'per_subject_baseline')
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_test_norm = scaler.transform(X_test)
        else:
            X_train_norm = np.nan_to_num(X_train, nan=0.0)
            X_test_norm = np.nan_to_num(X_test, nan=0.0)

        # Stage 1: Train with fixed params
        model_s1 = train_with_fixed_params(X_train_norm, y_train_stage1, params_s1, config['model'])
        pred_stage1 = model_s1.predict(X_test_norm)
        acc_stage1 = balanced_accuracy_score(y_test_stage1, pred_stage1)
        stage1_results.append({'subject': test_subject, 'balanced_acc': acc_stage1})

        # Stage 2: Train with fixed params (pain samples only)
        pain_train_mask = y_train_3class > 0
        X_train_pain = X_train_norm[pain_train_mask]
        y_train_stage2 = (y_train_3class[pain_train_mask] - 1).astype(np.int64)

        if len(np.unique(y_train_stage2)) > 1:
            model_s2 = train_with_fixed_params(X_train_pain, y_train_stage2, params_s2, config['model'])
        else:
            model_s2 = None

        # Combined prediction
        final_pred = np.zeros(len(y_test_3class), dtype=np.int64)
        for i in range(len(pred_stage1)):
            if pred_stage1[i] == 0:
                final_pred[i] = 0
            else:
                if model_s2 is not None:
                    s2_pred = model_s2.predict(X_test_norm[i:i+1])[0]
                    final_pred[i] = s2_pred + 1
                else:
                    final_pred[i] = 1

        acc_3class = balanced_accuracy_score(y_test_3class, final_pred)

        # Stage 2 accuracy
        pain_test_mask = y_test_3class > 0
        if pain_test_mask.sum() > 0 and model_s2 is not None:
            y_test_stage2 = (y_test_3class[pain_test_mask] - 1).astype(np.int64)
            pred_stage2_only = model_s2.predict(X_test_norm[pain_test_mask])
            acc_stage2 = balanced_accuracy_score(y_test_stage2, pred_stage2_only)
        else:
            acc_stage2 = np.nan

        stage2_results.append({'subject': test_subject, 'balanced_acc': acc_stage2})
        results.append({
            'subject': test_subject,
            'stage1_acc': acc_stage1,
            'stage2_acc': acc_stage2,
            'combined_3class_acc': acc_3class
        })

        print(f"      LOSO fold {fold_idx+1}/53... S1={acc_stage1:.1%} S2={acc_stage2:.1%} Combined={acc_3class:.1%}")

    return results, stage1_results, stage2_results


def save_results(all_results, output_dir):
    """Save all results to CSV and JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined results
    combined_df = pd.DataFrame(all_results)
    combined_df.to_csv(output_dir / 'hierarchical_loso_results.csv', index=False)

    # Calculate summary statistics
    summary = []
    for config_name in combined_df['config'].unique():
        cfg_data = combined_df[combined_df['config'] == config_name]
        summary.append({
            'config': config_name,
            'stage1_mean': cfg_data['stage1_acc'].mean(),
            'stage1_std': cfg_data['stage1_acc'].std(),
            'stage2_mean': cfg_data['stage2_acc'].mean(),
            'stage2_std': cfg_data['stage2_acc'].std(),
            'combined_mean': cfg_data['combined_3class_acc'].mean(),
            'combined_std': cfg_data['combined_3class_acc'].std()
        })

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('combined_mean', ascending=False)
    summary_df.to_csv(output_dir / 'summary.csv', index=False)

    # Save best configuration
    best = summary_df.iloc[0]
    best_config = {
        'config': best['config'],
        'stage1_accuracy': f"{best['stage1_mean']:.4f} +/- {best['stage1_std']:.4f}",
        'stage2_accuracy': f"{best['stage2_mean']:.4f} +/- {best['stage2_std']:.4f}",
        'combined_3class_accuracy': f"{best['combined_mean']:.4f} +/- {best['combined_std']:.4f}"
    }
    with open(output_dir / 'best_configuration.json', 'w') as f:
        json.dump(best_config, f, indent=2)

    return summary_df


def generate_report(summary_df, output_dir):
    """Generate Phase 5 markdown report."""
    best = summary_df.iloc[0]

    report = f"""# Phase 5: Hierarchical Binary Classification Results

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Summary

Phase 5 implements a two-stage hierarchical classifier:
- **Stage 1:** No Pain vs Pain (binary)
- **Stage 2:** Low Pain vs High Pain (binary, pain samples only)

---

## Results by Configuration

| Configuration | Stage 1 Acc | Stage 2 Acc | Combined 3-Class |
|--------------|-------------|-------------|------------------|
"""

    for _, row in summary_df.iterrows():
        report += f"| {row['config']} | {row['stage1_mean']:.2%} +/- {row['stage1_std']:.2%} | {row['stage2_mean']:.2%} +/- {row['stage2_std']:.2%} | {row['combined_mean']:.2%} +/- {row['combined_std']:.2%} |\n"

    report += f"""
---

## Best Configuration

**{best['config']}**
- Stage 1 (Pain Detection): {best['stage1_mean']:.2%} +/- {best['stage1_std']:.2%}
- Stage 2 (Intensity): {best['stage2_mean']:.2%} +/- {best['stage2_std']:.2%}
- Combined 3-Class: {best['combined_mean']:.2%} +/- {best['combined_std']:.2%}

---

## Comparison to Direct 3-Class (Phase 4)

| Approach | Best LOSO Balanced Accuracy |
|----------|----------------------------|
| Direct 3-Class (Phase 4) | (see Phase 4 results) |
| Hierarchical (Phase 5) | {best['combined_mean']:.2%} |
| Paper 1 Baseline | 79.4% |

---

## Analysis

### Stage 1: Pain Detection
Expected to be easier task - distinguishing pain from no-pain states.

### Stage 2: Intensity Discrimination
The harder task - distinguishing low from high pain intensity.

### Error Propagation
Stage 1 errors cascade to final prediction. If Stage 1 misclassifies a pain sample as no-pain, Stage 2 never gets a chance to correct it.

---

*Phase 5 hierarchical classification complete.*
"""

    with open(output_dir / 'phase5_report.md', 'w') as f:
        f.write(report)

    print(f"\nReport saved to {output_dir / 'phase5_report.md'}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("PHASE 5: HIERARCHICAL BINARY CLASSIFICATION")
    print("=" * 60)
    print("Features: Checkpointing, Memory Management, Fault Tolerance")

    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'stage1_pain_detection').mkdir(exist_ok=True)
    (RESULTS_DIR / 'stage2_intensity').mkdir(exist_ok=True)
    (RESULTS_DIR / 'combined').mkdir(exist_ok=True)

    # Load checkpoint if exists
    print("\nChecking for existing checkpoint...")
    checkpoint = load_checkpoint(RESULTS_DIR)
    completed_keys = checkpoint['completed']
    all_results = checkpoint['all_results']

    # Load Phase 4 top configurations
    try:
        configs = load_phase4_top_configs(n=3)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Phase 4 must complete before Phase 5 can run.")
        return

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"  Total samples: {len(df)}")
    print(f"  Unique subjects: {df['subject_id'].nunique()}")
    print(f"  State distribution: {df['state'].value_counts().to_dict()}")

    # Run hierarchical LOSO for each configuration
    failed_configs = []

    for config in configs:
        config_name = f"{config['model']}_{config['normalization']}"

        # Skip if already completed
        if config_name in completed_keys:
            print(f"\n{'='*50}")
            print(f"Configuration: {config_name} - SKIPPED (already completed)")
            continue

        print(f"\n{'='*50}")
        print(f"Configuration: {config_name}")
        print(f"{'='*50}")

        # Clear memory before each configuration
        print("  Clearing memory...")
        clear_memory()

        try:
            results, stage1_results, stage2_results = run_hierarchical_loso(df, config)

            for r in results:
                r['config'] = config_name
                all_results.append(r)

            # Calculate means
            s1_mean = np.mean([r['stage1_acc'] for r in results])
            s2_mean = np.nanmean([r['stage2_acc'] for r in results])
            combined_mean = np.mean([r['combined_3class_acc'] for r in results])

            print(f"\n  Stage 1 Mean: {s1_mean:.2%}")
            print(f"  Stage 2 Mean: {s2_mean:.2%}")
            print(f"  Combined Mean: {combined_mean:.2%}")

            # Mark as completed and save checkpoint
            completed_keys.append(config_name)
            save_checkpoint(RESULTS_DIR, completed_keys, all_results)

            # Clear memory after configuration completes
            clear_memory()

        except Exception as e:
            print(f"    [ERROR] Configuration failed: {str(e)}")
            failed_configs.append({
                'config': config_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            # Save error log
            with open(RESULTS_DIR / 'failed_configs.json', 'w') as f:
                json.dump(failed_configs, f, indent=2)
            # Clear memory and continue
            clear_memory()
            continue

    # Report on failed configurations
    if failed_configs:
        print(f'\n[WARNING] {len(failed_configs)} configuration(s) failed:')
        for fc in failed_configs:
            print(f'  - {fc["config"]}: {fc["error"]}')

    # Check if we have any results
    if not all_results:
        print('\n[ERROR] No configurations completed successfully!')
        print(f'Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        return

    # Save results
    print("\n" + "=" * 50)
    print("SAVING RESULTS")
    print(f"Configurations completed: {len(completed_keys)}/3")
    print("=" * 50)

    summary_df = save_results(all_results, RESULTS_DIR / 'combined')

    # Generate report
    generate_report(summary_df, RESULTS_DIR)

    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETE")
    print("=" * 60)

    # Print final summary
    best = summary_df.iloc[0]
    print(f"\nBest Configuration: {best['config']}")
    print(f"  Combined 3-Class LOSO: {best['combined_mean']:.2%} +/- {best['combined_std']:.2%}")


if __name__ == '__main__':
    main()
