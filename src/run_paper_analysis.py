"""
Execute all paper analysis computations and generate figures.

This script:
1. Runs parameter sweep (D x tau x signal) for Silhouette and Dunn indices
2. Generates C-H plane figures for each signal
3. Generates Fisher-Shannon plane figures for each signal
4. Generates parameter sensitivity heatmaps
5. Generates experiment pipeline diagram
6. Recomputes statistical comparisons with CORRECT baselines
7. Saves all numerical results as CSV for LaTeX source tracing

GATE 9: Paper 1 baselines - 78.0% LOSO, 79.4% 80/20
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from paper_analysis import (
    compute_silhouette,
    compute_dunn_index,
    compute_statistical_comparison,
    filter_baseline_only,
    generate_parameter_sweep_table,
    load_features,
    make_binary_labels,
    remove_outliers_iqr,
)
from paper_figures import (
    plot_ch_plane_grid,
    plot_fisher_shannon_grid,
    plot_confusion_matrix,
    plot_experiment_pipeline,
    plot_parameter_sensitivity,
)

# Configuration
SIGNALS = ["eda", "bvp", "resp", "spo2"]
D_VALUES = [3, 4, 5, 6, 7]
TAU_VALUES = [1, 2, 3]

PAPER1_LOSO_BASELINE = 0.780   # GATE 9: correct LOSO baseline
PAPER1_8020_BASELINE = 0.794   # GATE 9: correct 80/20 baseline

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_parameter_sweep():
    """Run Silhouette + Dunn sweep across all parameter combinations."""
    print("=" * 60)
    print("PARAMETER SWEEP: D x tau x signal")
    print("=" * 60)

    sweep_df = generate_parameter_sweep_table(
        signals=SIGNALS,
        d_values=D_VALUES,
        tau_values=TAU_VALUES,
        splits=["train", "validation"],
    )

    # Save raw results
    sweep_path = RESULTS_DIR / "parameter_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"Saved parameter sweep to {sweep_path}")
    print(f"Total combinations: {len(sweep_df)}")

    # Print best per signal
    for signal in SIGNALS:
        sig_data = sweep_df[sweep_df["signal"] == signal]
        if len(sig_data) == 0:
            continue
        best_sil = sig_data.loc[sig_data["silhouette"].idxmax()]
        print(f"\n{signal.upper()} best Silhouette: {best_sil['silhouette']:.4f} "
              f"(d={int(best_sil['d'])}, tau={int(best_sil['tau'])})")

    return sweep_df


def generate_figures(sweep_df):
    """Generate all paper figures."""
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    # C-H plane grids for each signal
    for signal in SIGNALS:
        path = RESULTS_DIR / f"ch_plane_{signal}.pdf"
        print(f"  C-H plane: {signal.upper()} -> {path.name}")
        plot_ch_plane_grid(signal, D_VALUES, TAU_VALUES, str(path))

    # Fisher-Shannon plane grids for each signal
    for signal in SIGNALS:
        path = RESULTS_DIR / f"fisher_shannon_{signal}.pdf"
        print(f"  Fisher-Shannon: {signal.upper()} -> {path.name}")
        plot_fisher_shannon_grid(signal, D_VALUES, TAU_VALUES, str(path))

    # Parameter sensitivity heatmaps
    path = RESULTS_DIR / "parameter_sensitivity.pdf"
    print(f"  Parameter sensitivity -> {path.name}")
    plot_parameter_sensitivity(sweep_df, str(path))

    # Experiment pipeline diagram
    path = RESULTS_DIR / "experiment_pipeline.pdf"
    print(f"  Pipeline diagram -> {path.name}")
    plot_experiment_pipeline(str(path))


def recompute_statistics():
    """Recompute statistical comparisons with CORRECT baselines.

    GATE 9: Uses 78.0% for LOSO comparison (NOT 79.4%)
    """
    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISONS (CORRECTED BASELINES)")
    print(f"  LOSO baseline: {PAPER1_LOSO_BASELINE:.1%}")
    print(f"  80/20 baseline: {PAPER1_8020_BASELINE:.1%}")
    print("=" * 60)

    # Load per-subject LOSO results
    loso_path = PROJECT_ROOT / "results" / "phase3_loso" / "per_subject_results.csv"
    per_subject = pd.read_csv(loso_path)

    results = []
    for model in per_subject["model"].unique():
        model_data = per_subject[per_subject["model"] == model]
        scores = model_data["balanced_accuracy"].values

        # Compare against CORRECT LOSO baseline (78.0%)
        comparison = compute_statistical_comparison(scores, PAPER1_LOSO_BASELINE)

        results.append({
            "model": model,
            "loso_balanced_acc": comparison["mean"],
            "loso_std": comparison["std"],
            "paper1_loso_baseline": PAPER1_LOSO_BASELINE,
            "improvement_pct": (comparison["mean"] - PAPER1_LOSO_BASELINE) / PAPER1_LOSO_BASELINE * 100,
            "t_statistic": comparison["t_statistic"],
            "p_value": comparison["p_value"],
            "cohens_d": comparison["cohens_d"],
            "ci_lower": comparison["ci_lower"],
            "ci_upper": comparison["ci_upper"],
            "n_subjects": comparison["n"],
            "significant_p05": comparison["p_value"] < 0.05,
        })

        print(f"\n  {model}:")
        print(f"    Mean BA: {comparison['mean']:.4f} +/- {comparison['std']:.4f}")
        print(f"    vs Paper 1 LOSO ({PAPER1_LOSO_BASELINE:.1%}): "
              f"delta = {comparison['mean'] - PAPER1_LOSO_BASELINE:+.4f}")
        print(f"    t = {comparison['t_statistic']:.4f}, p = {comparison['p_value']:.4f}")
        print(f"    Cohen's d = {comparison['cohens_d']:.4f}")
        print(f"    95% CI: [{comparison['ci_lower']:.4f}, {comparison['ci_upper']:.4f}]")
        print(f"    Significant (p<0.05): {comparison['p_value'] < 0.05}")

    stats_df = pd.DataFrame(results)
    stats_path = RESULTS_DIR / "corrected_statistical_comparison.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSaved corrected statistics to {stats_path}")

    return stats_df


def compute_binary_metrics():
    """Compute binary classification metrics for the C-H plane."""
    print("\n" + "=" * 60)
    print("BINARY CLASSIFICATION METRICS (C-H PLANE)")
    print("=" * 60)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    results = []
    for signal in SIGNALS:
        for d in D_VALUES:
            for tau in TAU_VALUES:
                try:
                    # Combine train+validation (matches Stage 0)
                    dfs = []
                    for split in ["train", "validation"]:
                        try:
                            dfs.append(load_features(split, signal, d=d, tau=tau))
                        except FileNotFoundError:
                            continue
                    if not dfs:
                        continue
                    df = pd.concat(dfs, ignore_index=True)
                    df = filter_baseline_only(df)

                    # True binary labels
                    labels = make_binary_labels(df)

                    if len(np.unique(labels)) < 2:
                        continue

                    # With outlier removal
                    df_clean = remove_outliers_iqr(df, columns=["pe", "comp"])
                    lab_clean = make_binary_labels(df_clean)

                    if len(np.unique(lab_clean)) < 2:
                        continue

                    # Normalize (matches Stage 0)
                    feat_raw = df_clean[["pe", "comp"]].values
                    scaler = StandardScaler()
                    feat_norm = scaler.fit_transform(feat_raw)

                    sil_norm = compute_silhouette(feat_norm, lab_clean)
                    dunn_norm = compute_dunn_index(feat_norm, lab_clean)

                    # Linear accuracy
                    clf = LogisticRegression(random_state=42, max_iter=1000)
                    clf.fit(feat_norm, lab_clean)
                    acc = accuracy_score(lab_clean, clf.predict(feat_norm))

                    results.append({
                        "signal": signal,
                        "d": d,
                        "tau": tau,
                        "n_samples": len(feat_raw),
                        "n_baseline": int(np.sum(lab_clean == 0)),
                        "n_pain": int(np.sum(lab_clean == 1)),
                        "silhouette": sil_norm,
                        "dunn": dunn_norm,
                        "linear_accuracy": acc,
                    })
                except Exception:
                    continue

    binary_df = pd.DataFrame(results)
    binary_path = RESULTS_DIR / "binary_metrics.csv"
    binary_df.to_csv(binary_path, index=False)
    print(f"Saved binary metrics to {binary_path}")

    # Print headline result
    if len(binary_df) > 0:
        best = binary_df.loc[binary_df["silhouette"].idxmax()]
        print(f"\nBest binary separation:")
        print(f"  Signal: {best['signal'].upper()}, d={int(best['d'])}, tau={int(best['tau'])}")
        print(f"  Silhouette: {best['silhouette']:.4f}")
        print(f"  Dunn: {best['dunn']:.4f}")
        print(f"  Linear accuracy: {best['linear_accuracy']:.4f}")
        print(f"  Samples: {int(best['n_baseline'])} baseline, {int(best['n_pain'])} pain")

        # Also print best linear accuracy
        best_acc = binary_df.loc[binary_df["linear_accuracy"].idxmax()]
        print(f"\nHighest linear accuracy:")
        print(f"  Signal: {best_acc['signal'].upper()}, d={int(best_acc['d'])}, tau={int(best_acc['tau'])}")
        print(f"  Linear accuracy: {best_acc['linear_accuracy']:.4f}")
        print(f"  Silhouette: {best_acc['silhouette']:.4f}")

    return binary_df


if __name__ == "__main__":
    sweep_df = run_parameter_sweep()
    generate_figures(sweep_df)
    stats_df = recompute_statistics()
    binary_df = compute_binary_metrics()

    print("\n" + "=" * 60)
    print("ALL ANALYSIS COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 60)
