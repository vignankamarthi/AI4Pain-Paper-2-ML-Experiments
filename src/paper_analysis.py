"""
Paper analysis module for AI4Pain Paper 2.

Provides statistical analysis, clustering quality metrics, and data
processing functions used to generate numbers and tables for the paper.
All functions are tested via tests/test_paper_analysis.py.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# Feature columns in the dataset
FEATURE_COLS = [
    "pe", "comp", "fisher_shannon", "fisher_info",
    "renyipe", "renyicomp", "tsallispe", "tsalliscomp",
]

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "features"


def compute_silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    """Compute Silhouette Score for given features and labels.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    labels : np.ndarray
        Cluster/class labels of shape (n_samples,).

    Returns
    -------
    float
        Silhouette score in [-1, 1].
    """
    return float(silhouette_score(features, labels))


def compute_dunn_index(features: np.ndarray, labels: np.ndarray) -> float:
    """Compute the Dunn Index for given features and labels.

    Dunn Index = min(inter-cluster distance) / max(intra-cluster diameter)

    Higher values indicate better-defined clusters.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    labels : np.ndarray
        Cluster/class labels of shape (n_samples,).

    Returns
    -------
    float
        Dunn index (positive value, higher is better).
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        raise ValueError("Need at least 2 clusters for Dunn index")

    # Compute intra-cluster diameters (max pairwise distance within each cluster)
    max_intra = 0.0
    cluster_points = {}
    for label in unique_labels:
        mask = labels == label
        points = features[mask]
        cluster_points[label] = points
        if len(points) > 1:
            intra_dists = cdist(points, points, metric="euclidean")
            diameter = intra_dists.max()
            max_intra = max(max_intra, diameter)

    if max_intra == 0.0:
        return float("inf")

    # Compute inter-cluster distances (min pairwise distance between clusters)
    min_inter = float("inf")
    for i, label_i in enumerate(unique_labels):
        for label_j in unique_labels[i + 1:]:
            inter_dists = cdist(
                cluster_points[label_i],
                cluster_points[label_j],
                metric="euclidean",
            )
            min_dist = inter_dists.min()
            min_inter = min(min_inter, min_dist)

    return float(min_inter / max_intra)


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str],
    factor: float = 1.5,
) -> pd.DataFrame:
    """Remove outliers using IQR method.

    A row is removed if ANY of the specified columns has a value
    outside [Q1 - factor*IQR, Q3 + factor*IQR].

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list of str
        Columns to check for outliers.
    factor : float
        IQR multiplier (default 1.5).

    Returns
    -------
    pd.DataFrame
        Dataframe with outlier rows removed.
    """
    mask = pd.Series(True, index=df.index)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask &= (df[col] >= lower) & (df[col] <= upper)
    return df[mask].reset_index(drop=True)


def load_features(
    split: str,
    signal: str,
    d: Optional[int] = None,
    tau: Optional[int] = None,
) -> pd.DataFrame:
    """Load feature CSV for a specific split and signal.

    Parameters
    ----------
    split : str
        One of 'train', 'validation', 'test'.
    signal : str
        One of 'eda', 'bvp', 'resp', 'spo2'.
    d : int, optional
        Embedding dimension to filter. If None, return all.
    tau : int, optional
        Embedding delay to filter. If None, return all.

    Returns
    -------
    pd.DataFrame
        Feature dataframe.
    """
    path = DATA_DIR / f"results_{split}_{signal}.csv"
    df = pd.read_csv(path)
    if d is not None:
        df = df[df["dimension"] == d]
    if tau is not None:
        df = df[df["tau"] == tau]
    return df.reset_index(drop=True)


def apply_global_zscore(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Apply global z-score normalization (StandardScaler) to feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature_cols : list of str
        Columns to normalize.

    Returns
    -------
    pd.DataFrame
        Copy of df with feature columns z-score normalized.
    """
    result = df.copy()
    scaler = StandardScaler()
    result[feature_cols] = scaler.fit_transform(result[feature_cols].values)
    return result


def filter_baseline_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to exclude rest segments (baseline-only methodology).

    Keeps: baseline, low, high
    Removes: rest

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe with 'state' column.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe without rest segments.
    """
    return df[df["state"] != "rest"].reset_index(drop=True)


def make_binary_labels(df: pd.DataFrame) -> np.ndarray:
    """Create proper binary labels: 0=baseline, 1=pain (low+high).

    The 'binaryclass' column in the CSV is actually a 4-class label
    (0=baseline, 1=low, 2=high, 3=rest). This function creates true
    binary labels from the 'state' column.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe with 'state' column.

    Returns
    -------
    np.ndarray
        Binary labels: 0 for baseline, 1 for pain (low or high).
    """
    return np.where(df["state"] == "baseline", 0, 1)


def compute_cohens_d(scores: np.ndarray, baseline: float) -> float:
    """Compute Cohen's d effect size (one-sample).

    Parameters
    ----------
    scores : np.ndarray
        Array of per-subject scores.
    baseline : float
        Known baseline value to compare against.

    Returns
    -------
    float
        Cohen's d = (mean(scores) - baseline) / std(scores, ddof=1).
    """
    return float((np.mean(scores) - baseline) / np.std(scores, ddof=1))


def compute_statistical_comparison(
    per_subject_scores: np.ndarray,
    baseline_value: float,
) -> Dict[str, float]:
    """Compute one-sample t-test and effect size against a known baseline.

    Parameters
    ----------
    per_subject_scores : np.ndarray
        Per-subject balanced accuracy scores.
    baseline_value : float
        Paper 1 baseline to compare against.

    Returns
    -------
    dict
        Keys: t_statistic, p_value, cohens_d, mean, std, n, ci_lower, ci_upper.
    """
    n = len(per_subject_scores)
    mean = float(np.mean(per_subject_scores))
    std = float(np.std(per_subject_scores, ddof=1))

    t_stat, p_value = stats.ttest_1samp(per_subject_scores, baseline_value)

    d = compute_cohens_d(per_subject_scores, baseline_value)

    # 95% CI for the mean
    se = std / np.sqrt(n)
    ci_lower = mean - 1.96 * se
    ci_upper = mean + 1.96 * se

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": d,
        "mean": mean,
        "std": std,
        "n": n,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def generate_parameter_sweep_table(
    signals: List[str],
    d_values: List[int],
    tau_values: List[int],
    splits: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute Silhouette and Dunn indices across all parameter combinations.

    Uses binary classification (baseline vs pain) on normalized C-H plane
    features (pe, comp) for each signal/d/tau combination.

    Parameters
    ----------
    signals : list of str
        Signal types to process.
    d_values : list of int
        Embedding dimensions.
    tau_values : list of int
        Embedding delays.
    splits : list of str, optional
        Data splits to combine (default ['train', 'validation']).

    Returns
    -------
    pd.DataFrame
        Table with columns: signal, d, tau, silhouette, dunn, n_samples.
    """
    if splits is None:
        splits = ["train", "validation"]
    rows = []
    for signal in signals:
        for d in d_values:
            for tau in tau_values:
                try:
                    # Load from all requested splits
                    dfs = []
                    for s in splits:
                        try:
                            dfs.append(load_features(s, signal, d=d, tau=tau))
                        except FileNotFoundError:
                            continue
                    if not dfs:
                        continue
                    df = pd.concat(dfs, ignore_index=True)
                    df = filter_baseline_only(df)

                    # True binary labels from state column
                    labels = make_binary_labels(df)

                    if len(np.unique(labels)) < 2:
                        continue

                    # Normalize features BEFORE computing metrics
                    # (matches Stage 0 methodology: NO outlier removal
                    #  before Silhouette/Dunn -- IQR can eliminate the
                    #  minority class when separation is strong)
                    features = df[["pe", "comp"]].values
                    scaler = StandardScaler()
                    features_norm = scaler.fit_transform(features)

                    sil = compute_silhouette(features_norm, labels)
                    dunn = compute_dunn_index(features_norm, labels)

                    rows.append({
                        "signal": signal,
                        "d": d,
                        "tau": tau,
                        "silhouette": sil,
                        "dunn": dunn,
                        "n_samples": len(features),
                    })
                except Exception:
                    continue

    return pd.DataFrame(rows)
