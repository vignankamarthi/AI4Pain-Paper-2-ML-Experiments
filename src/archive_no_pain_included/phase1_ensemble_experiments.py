"""
Phase 1: 3-Class Ensemble Exploration with Per-Subject Baseline Normalization

This script trains and evaluates ensemble ML models for 3-class pain classification:
- Class 0: no_pain (baseline + rest states)
- Class 1: low_pain
- Class 2: high_pain

Key approach:
- Per-subject baseline normalization using no_pain samples as reference
- All 4 signals (EDA, BVP, RESP, SpO2) with d=7, tau=2 (best from Stage 0)
- 8 entropy/complexity features per signal = 32 total features
- Optuna hyperparameter optimization with 5-fold CV
- Models: Random Forest, XGBoost, LightGBM, Stacked Ensembles

Author: Claude (AI4Pain Paper 2)
Date: 2026-01-14
"""

import os
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configuration
RANDOM_SEED = 42
BEST_DIMENSION = 7  # From Stage 0
BEST_TAU = 2        # From Stage 0
SIGNALS = ['eda', 'bvp', 'resp', 'spo2']
FEATURE_COLS = ['pe', 'comp', 'fisher_shannon', 'fisher_info',
                'renyipe', 'renyicomp', 'tsallispe', 'tsalliscomp']
N_OPTUNA_TRIALS = 50
CV_FOLDS = 5

# Class mapping
CLASS_MAP = {
    'baseline': 0,  # no_pain
    'rest': 0,      # no_pain
    'low': 1,       # low_pain
    'high': 2       # high_pain
}
CLASS_NAMES = ['no_pain', 'low_pain', 'high_pain']


class Phase1Experiment:
    """
    Main experiment class for Phase 1 ensemble exploration.

    Attributes
    ----------
    data_dir : Path
        Path to feature data directory
    results_dir : Path
        Path to output results directory
    """

    def __init__(self, base_dir: str = None):
        """
        Initialize the experiment.

        Parameters
        ----------
        base_dir : str, optional
            Base directory of the project
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        else:
            base_dir = Path(base_dir)

        self.data_dir = base_dir / 'data' / 'features'
        self.results_dir = base_dir / 'results' / 'phase1_ensembles'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.results_dir / 'confusion_matrices').mkdir(exist_ok=True)
        (self.results_dir / 'feature_importance_plots').mkdir(exist_ok=True)
        (self.results_dir / 'models').mkdir(exist_ok=True)

        self.models = {}
        self.results = {}
        self.best_params = {}

    def load_data(self) -> pd.DataFrame:
        """
        Load and combine feature data from all signals.

        Returns
        -------
        pd.DataFrame
            Combined feature dataframe with all signals
        """
        print("Loading feature data...")

        all_data = []

        for split in ['train', 'validation']:
            for signal in SIGNALS:
                filepath = self.data_dir / f'results_{split}_{signal}.csv'
                df = pd.read_csv(filepath)

                # Filter to best (d, tau) from Stage 0
                df = df[(df['dimension'] == BEST_DIMENSION) & (df['tau'] == BEST_TAU)].copy()

                # Add signal identifier
                df['signal_type'] = signal
                df['split'] = split

                # Extract subject ID
                df['subject_id'] = df['file_name'].apply(
                    lambda x: int(x.split('/')[-1].replace('.csv', ''))
                )

                all_data.append(df)

        combined = pd.concat(all_data, ignore_index=True)

        print(f"  Loaded {len(combined)} rows from {len(SIGNALS)} signals")
        print(f"  Unique subjects: {combined['subject_id'].nunique()}")
        print(f"  States: {combined['state'].unique()}")

        return combined

    def pivot_to_wide_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot data from long to wide format (one row per sample).

        Creates feature columns like: eda_pe, eda_comp, bvp_pe, bvp_comp, etc.

        Parameters
        ----------
        df : pd.DataFrame
            Long format dataframe

        Returns
        -------
        pd.DataFrame
            Wide format dataframe with signal-prefixed features
        """
        print("Pivoting to wide format...")

        # Create unique sample identifier
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
        result = wide_dfs[0]
        for wdf in wide_dfs[1:]:
            # Only keep feature columns from subsequent merges
            feature_cols = [c for c in wdf.columns if c not in ['sample_id', 'subject_id', 'state', 'split']]
            merge_cols = ['sample_id'] + feature_cols
            result = result.merge(wdf[merge_cols], on='sample_id', how='inner')

        print(f"  Wide format shape: {result.shape}")

        return result

    def apply_baseline_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply per-subject baseline normalization.

        For each subject, normalize all features using mean/std from their
        no_pain samples (baseline + rest states) as reference.

        Parameters
        ----------
        df : pd.DataFrame
            Wide format dataframe

        Returns
        -------
        pd.DataFrame
            Normalized dataframe
        """
        print("Applying per-subject baseline normalization...")

        # Get all feature columns
        feature_cols = [c for c in df.columns if any(
            c.startswith(s + '_') for s in SIGNALS
        )]

        df_norm = df.copy()

        # Identify no_pain states for normalization
        no_pain_states = ['baseline', 'rest']

        subjects = df['subject_id'].unique()

        for subj in subjects:
            mask = df['subject_id'] == subj
            no_pain_mask = mask & df['state'].isin(no_pain_states)

            n_nopain = no_pain_mask.sum()

            if n_nopain >= 2:
                for col in feature_cols:
                    ref_mean = df.loc[no_pain_mask, col].mean()
                    ref_std = df.loc[no_pain_mask, col].std()

                    if ref_std > 1e-8:
                        df_norm.loc[mask, col] = (
                            df.loc[mask, col] - ref_mean
                        ) / ref_std
                    else:
                        df_norm.loc[mask, col] = df.loc[mask, col] - ref_mean
            else:
                # Fallback to global normalization for this subject
                for col in feature_cols:
                    col_mean = df[col].mean()
                    col_std = df[col].std()
                    if col_std > 1e-8:
                        df_norm.loc[mask, col] = (
                            df.loc[mask, col] - col_mean
                        ) / col_std

        print(f"  Normalized {len(feature_cols)} features for {len(subjects)} subjects")

        return df_norm

    def prepare_ml_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix and labels for ML.

        Parameters
        ----------
        df : pd.DataFrame
            Normalized wide format dataframe

        Returns
        -------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Label array (0: no_pain, 1: low_pain, 2: high_pain)
        feature_names : List[str]
            Names of features
        """
        print("Preparing ML data...")

        # Map states to classes
        df['label'] = df['state'].map(CLASS_MAP)

        # Get feature columns
        feature_cols = [c for c in df.columns if any(
            c.startswith(s + '_') for s in SIGNALS
        )]

        X = df[feature_cols].values
        y = df['label'].values

        print(f"  X shape: {X.shape}")
        print(f"  Class distribution:")
        for i, name in enumerate(CLASS_NAMES):
            count = (y == i).sum()
            pct = count / len(y) * 100
            print(f"    {name}: {count} ({pct:.1f}%)")

        return X, y, feature_cols

    def optimize_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Optimize Random Forest hyperparameters using Optuna.

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels

        Returns
        -------
        Dict
            Best hyperparameters
        """
        print("\nOptimizing Random Forest...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 500]),
                'max_depth': trial.suggest_categorical('max_depth', [10, 20, 30, 40, None]),
                'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10]),
                'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 4]),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                'random_state': RANDOM_SEED,
                'n_jobs': -1
            }

            model = RandomForestClassifier(**params)
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='balanced_accuracy')

            return scores.mean()

        sampler = TPESampler(seed=RANDOM_SEED)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

        print(f"  Best CV balanced accuracy: {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")

        return study.best_params

    def optimize_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Optimize XGBoost hyperparameters using Optuna.

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels

        Returns
        -------
        Dict
            Best hyperparameters
        """
        print("\nOptimizing XGBoost...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 500]),
                'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7, 9]),
                'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2]),
                'subsample': trial.suggest_categorical('subsample', [0.6, 0.8, 1.0]),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6, 0.8, 1.0]),
                'gamma': trial.suggest_categorical('gamma', [0, 0.1, 0.2, 0.5]),
                'reg_alpha': trial.suggest_categorical('reg_alpha', [0, 0.1, 0.5, 1.0]),
                'reg_lambda': trial.suggest_categorical('reg_lambda', [0, 0.1, 0.5, 1.0]),
                'random_state': RANDOM_SEED,
                'n_jobs': -1,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss'
            }

            model = xgb.XGBClassifier(**params)
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='balanced_accuracy')

            return scores.mean()

        sampler = TPESampler(seed=RANDOM_SEED)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

        print(f"  Best CV balanced accuracy: {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")

        return study.best_params

    def optimize_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Optimize LightGBM hyperparameters using Optuna.

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels

        Returns
        -------
        Dict
            Best hyperparameters
        """
        print("\nOptimizing LightGBM...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 500]),
                'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7, 9, -1]),
                'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2]),
                'num_leaves': trial.suggest_categorical('num_leaves', [15, 31, 63, 127]),
                'subsample': trial.suggest_categorical('subsample', [0.6, 0.8, 1.0]),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6, 0.8, 1.0]),
                'reg_alpha': trial.suggest_categorical('reg_alpha', [0, 0.1, 0.5, 1.0]),
                'reg_lambda': trial.suggest_categorical('reg_lambda', [0, 0.1, 0.5, 1.0]),
                'random_state': RANDOM_SEED,
                'n_jobs': -1,
                'verbose': -1
            }

            model = lgb.LGBMClassifier(**params)
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='balanced_accuracy')

            return scores.mean()

        sampler = TPESampler(seed=RANDOM_SEED)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

        print(f"  Best CV balanced accuracy: {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")

        return study.best_params

    def train_and_evaluate(
        self,
        model,
        model_name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        """
        Train model and evaluate on test set.

        Parameters
        ----------
        model : estimator
            Sklearn-compatible model
        model_name : str
            Name of the model
        X_train, X_test : np.ndarray
            Training and test features
        y_train, y_test : np.ndarray
            Training and test labels
        feature_names : List[str]
            Feature names for importance analysis

        Returns
        -------
        Dict
            Evaluation metrics
        """
        print(f"\nTraining and evaluating {model_name}...")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrics
        train_acc = accuracy_score(y_train, y_pred)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_bal_acc = balanced_accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')

        # Per-class accuracy
        cm = confusion_matrix(y_test, y_pred_test)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)

        results = {
            'model_name': model_name,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'balanced_accuracy': test_bal_acc,
            'f1_weighted': test_f1,
            'acc_no_pain': per_class_acc[0],
            'acc_low_pain': per_class_acc[1],
            'acc_high_pain': per_class_acc[2],
            'confusion_matrix': cm
        }

        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Balanced Accuracy: {test_bal_acc:.4f}")
        print(f"  F1 Weighted: {test_f1:.4f}")
        print(f"  Per-class: no_pain={per_class_acc[0]:.4f}, low={per_class_acc[1]:.4f}, high={per_class_acc[2]:.4f}")

        # Save model
        model_path = self.results_dir / 'models' / f'{model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Plot confusion matrix
        self.plot_confusion_matrix(cm, model_name)

        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            self.plot_feature_importance(model, feature_names, model_name)
            results['feature_importances'] = dict(zip(feature_names, model.feature_importances_))

        return results

    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str):
        """
        Plot and save confusion matrix.

        Parameters
        ----------
        cm : np.ndarray
            Confusion matrix
        model_name : str
            Model name for title and filename
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES
        )
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()

        filename = f'{model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")}_confusion_matrix.png'
        plt.savefig(self.results_dir / 'confusion_matrices' / filename, dpi=150)
        plt.close()

    def plot_feature_importance(
        self,
        model,
        feature_names: List[str],
        model_name: str,
        top_n: int = 20
    ):
        """
        Plot and save feature importance.

        Parameters
        ----------
        model : estimator
            Trained model with feature_importances_
        feature_names : List[str]
            Feature names
        model_name : str
            Model name for title and filename
        top_n : int
            Number of top features to show
        """
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 8))
        plt.barh(
            range(top_n),
            importances[indices][::-1],
            color='steelblue'
        )
        plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name} - Top {top_n} Features')
        plt.tight_layout()

        filename = f'{model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")}_feature_importance.png'
        plt.savefig(self.results_dir / 'feature_importance_plots' / filename, dpi=150)
        plt.close()

    def build_stacking_ensemble(
        self,
        base_models: List[Tuple[str, object]],
        ensemble_name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        """
        Build and evaluate a stacking ensemble.

        Parameters
        ----------
        base_models : List[Tuple[str, object]]
            List of (name, model) tuples for base estimators
        ensemble_name : str
            Name for the ensemble
        X_train, X_test : np.ndarray
            Training and test features
        y_train, y_test : np.ndarray
            Training and test labels
        feature_names : List[str]
            Feature names

        Returns
        -------
        Dict
            Evaluation metrics
        """
        print(f"\nBuilding {ensemble_name}...")

        meta_learner = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=CV_FOLDS,
            n_jobs=-1
        )

        return self.train_and_evaluate(
            stacking_clf,
            ensemble_name,
            X_train, X_test,
            y_train, y_test,
            feature_names
        )

    def generate_leaderboard(self, results: List[Dict]) -> pd.DataFrame:
        """
        Generate and save model leaderboard.

        Parameters
        ----------
        results : List[Dict]
            List of result dictionaries from each model

        Returns
        -------
        pd.DataFrame
            Leaderboard dataframe
        """
        print("\nGenerating leaderboard...")

        leaderboard_data = []
        for r in results:
            leaderboard_data.append({
                'model': r['model_name'],
                'accuracy': r['test_accuracy'],
                'balanced_accuracy': r['balanced_accuracy'],
                'f1_weighted': r['f1_weighted'],
                'acc_no_pain': r['acc_no_pain'],
                'acc_low_pain': r['acc_low_pain'],
                'acc_high_pain': r['acc_high_pain']
            })

        df = pd.DataFrame(leaderboard_data)
        df = df.sort_values('balanced_accuracy', ascending=False).reset_index(drop=True)
        df.insert(0, 'rank', range(1, len(df) + 1))

        # Save
        df.to_csv(self.results_dir / 'leaderboard.csv', index=False)

        print("\n" + "=" * 80)
        print("LEADERBOARD")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

        return df

    def generate_report(
        self,
        leaderboard: pd.DataFrame,
        results: List[Dict],
        best_params: Dict
    ) -> str:
        """
        Generate Phase 1 markdown report.

        Parameters
        ----------
        leaderboard : pd.DataFrame
            Model leaderboard
        results : List[Dict]
            All model results
        best_params : Dict
            Best hyperparameters for each model

        Returns
        -------
        str
            Report content
        """
        print("\nGenerating report...")

        best_model = leaderboard.iloc[0]
        paper1_baseline = 79.4
        improvement = best_model['balanced_accuracy'] * 100 - paper1_baseline

        report = f"""# Phase 1: Ensemble Exploration Results

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Executive Summary

**Best Model:** {best_model['model']}
**Balanced Accuracy:** {best_model['balanced_accuracy']*100:.2f}%
**Improvement over Paper 1 Baseline:** {improvement:+.2f}% (from {paper1_baseline}%)

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Embedding Dimension (d) | {BEST_DIMENSION} |
| Time Delay (tau) | {BEST_TAU} |
| Signals | {', '.join(SIGNALS).upper()} |
| Features per Signal | {len(FEATURE_COLS)} |
| Total Features | {len(FEATURE_COLS) * len(SIGNALS)} |
| Normalization | Per-subject baseline (no-pain reference) |
| Train/Test Split | 80/20 stratified |
| Optuna Trials | {N_OPTUNA_TRIALS} per model |
| CV Folds | {CV_FOLDS} |

---

## Leaderboard

| Rank | Model | Balanced Acc | Accuracy | F1 | No Pain | Low Pain | High Pain |
|------|-------|--------------|----------|-----|---------|----------|-----------|
"""
        for _, row in leaderboard.iterrows():
            report += f"| {int(row['rank'])} | {row['model']} | {row['balanced_accuracy']*100:.2f}% | {row['accuracy']*100:.2f}% | {row['f1_weighted']:.4f} | {row['acc_no_pain']*100:.1f}% | {row['acc_low_pain']*100:.1f}% | {row['acc_high_pain']*100:.1f}% |\n"

        report += f"""
---

## Comparison to Paper 1 Baseline

| Metric | Paper 1 (catch22) | Paper 2 (Entropy) | Difference |
|--------|-------------------|-------------------|------------|
| Balanced Accuracy | {paper1_baseline:.1f}% | {best_model['balanced_accuracy']*100:.2f}% | {improvement:+.2f}% |

---

## Best Hyperparameters

"""
        for model_name, params in best_params.items():
            report += f"### {model_name}\n```json\n{json.dumps(params, indent=2)}\n```\n\n"

        report += f"""---

## Per-Class Analysis

The best model ({best_model['model']}) achieves:
- **No Pain (baseline + rest):** {best_model['acc_no_pain']*100:.1f}% accuracy
- **Low Pain:** {best_model['acc_low_pain']*100:.1f}% accuracy
- **High Pain:** {best_model['acc_high_pain']*100:.1f}% accuracy

---

## Decision Recommendation

"""
        if best_model['balanced_accuracy'] >= 0.85:
            report += f"""**PROCEED TO PHASE 3 (LOSO VALIDATION)**

The best model achieves {best_model['balanced_accuracy']*100:.2f}% balanced accuracy, meeting the 85% threshold.
Phase 2 (Neural Net Exploration) is NOT needed.

Recommended models for LOSO validation:
"""
            top_models = leaderboard.head(5)
            for _, row in top_models.iterrows():
                report += f"- {row['model']}: {row['balanced_accuracy']*100:.2f}%\n"
        elif best_model['balanced_accuracy'] >= 0.82:
            report += f"""**CONSIDER PHASE 2 (NEURAL NET EXPLORATION)**

The best model achieves {best_model['balanced_accuracy']*100:.2f}% balanced accuracy, below the 85% threshold but above 82%.
Neural nets may provide improvement through non-linear feature interactions.
"""
        else:
            report += f"""**TRIGGER PHASE 2 (NEURAL NET EXPLORATION)**

The best model achieves {best_model['balanced_accuracy']*100:.2f}% balanced accuracy, below 82%.
Neural net exploration is recommended to improve performance.
"""

        report += f"""
---

## Output Files

```
results/phase1_ensembles/
├── leaderboard.csv
├── hyperparameters.json
├── phase1_report.md
├── confusion_matrices/
│   └── [model]_confusion_matrix.png
├── feature_importance_plots/
│   └── [model]_feature_importance.png
└── models/
    └── [model].pkl
```

---

## Next Steps

1. Review leaderboard and confusion matrices
2. Select top 2-5 models for LOSO validation
3. Proceed to Phase 3 for rigorous cross-validation
"""

        # Save report
        with open(self.results_dir / 'phase1_report.md', 'w') as f:
            f.write(report)

        # Save hyperparameters
        with open(self.results_dir / 'hyperparameters.json', 'w') as f:
            json.dump(best_params, f, indent=2)

        return report

    def run(self):
        """
        Execute the full Phase 1 experiment pipeline.
        """
        print("=" * 80)
        print("PHASE 1: ENSEMBLE EXPLORATION")
        print("=" * 80)

        # Load and prepare data
        df = self.load_data()
        df_wide = self.pivot_to_wide_format(df)
        df_norm = self.apply_baseline_normalization(df_wide)
        X, y, feature_names = self.prepare_ml_data(df_norm)

        # Train/test split
        print("\nSplitting data (80/20 stratified)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=RANDOM_SEED
        )
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")

        # Standardize features
        print("\nStandardizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save scaler
        with open(self.results_dir / 'models' / 'scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        all_results = []

        # Optimize and train Random Forest
        rf_params = self.optimize_random_forest(X_train_scaled, y_train)
        rf_params['random_state'] = RANDOM_SEED
        rf_params['n_jobs'] = -1
        rf_model = RandomForestClassifier(**rf_params)
        rf_results = self.train_and_evaluate(
            rf_model, 'Random Forest',
            X_train_scaled, X_test_scaled,
            y_train, y_test,
            feature_names
        )
        all_results.append(rf_results)
        self.best_params['Random Forest'] = rf_params

        # Optimize and train XGBoost
        xgb_params = self.optimize_xgboost(X_train_scaled, y_train)
        xgb_params['random_state'] = RANDOM_SEED
        xgb_params['n_jobs'] = -1
        xgb_params['use_label_encoder'] = False
        xgb_params['eval_metric'] = 'mlogloss'
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_results = self.train_and_evaluate(
            xgb_model, 'XGBoost',
            X_train_scaled, X_test_scaled,
            y_train, y_test,
            feature_names
        )
        all_results.append(xgb_results)
        self.best_params['XGBoost'] = xgb_params

        # Optimize and train LightGBM
        lgb_params = self.optimize_lightgbm(X_train_scaled, y_train)
        lgb_params['random_state'] = RANDOM_SEED
        lgb_params['n_jobs'] = -1
        lgb_params['verbose'] = -1
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_results = self.train_and_evaluate(
            lgb_model, 'LightGBM',
            X_train_scaled, X_test_scaled,
            y_train, y_test,
            feature_names
        )
        all_results.append(lgb_results)
        self.best_params['LightGBM'] = lgb_params

        # Build stacking ensembles
        # Need to retrain base models for stacking
        rf_for_stack = RandomForestClassifier(**rf_params)
        xgb_for_stack = xgb.XGBClassifier(**xgb_params)
        lgb_for_stack = lgb.LGBMClassifier(**lgb_params)

        # Stacking Ensemble 1: RF + XGBoost
        stack1_results = self.build_stacking_ensemble(
            [('rf', rf_for_stack), ('xgb', xgb_for_stack)],
            'Stacked (RF+XGB)',
            X_train_scaled, X_test_scaled,
            y_train, y_test,
            feature_names
        )
        all_results.append(stack1_results)

        # Stacking Ensemble 2: RF + XGBoost + LightGBM
        rf_for_stack2 = RandomForestClassifier(**rf_params)
        xgb_for_stack2 = xgb.XGBClassifier(**xgb_params)
        lgb_for_stack2 = lgb.LGBMClassifier(**lgb_params)

        stack2_results = self.build_stacking_ensemble(
            [('rf', rf_for_stack2), ('xgb', xgb_for_stack2), ('lgb', lgb_for_stack2)],
            'Stacked (RF+XGB+LGB)',
            X_train_scaled, X_test_scaled,
            y_train, y_test,
            feature_names
        )
        all_results.append(stack2_results)

        # Generate outputs
        leaderboard = self.generate_leaderboard(all_results)
        report = self.generate_report(leaderboard, all_results, self.best_params)

        print("\n" + "=" * 80)
        print("PHASE 1 COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {self.results_dir}")

        return leaderboard, all_results


if __name__ == '__main__':
    experiment = Phase1Experiment()
    leaderboard, results = experiment.run()
