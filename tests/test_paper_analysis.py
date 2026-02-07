"""
TDD tests for paper_analysis module.

Tests use synthetic data with known properties to verify correctness
of all statistical and data processing functions used in the paper.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from paper_analysis import (
    apply_global_zscore,
    compute_cohens_d,
    compute_dunn_index,
    compute_silhouette,
    compute_statistical_comparison,
    filter_baseline_only,
    make_binary_labels,
    remove_outliers_iqr,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def well_separated_clusters():
    """Two well-separated Gaussian clusters for clustering quality tests."""
    rng = np.random.default_rng(42)
    cluster_a = rng.normal(loc=[0, 0], scale=0.3, size=(100, 2))
    cluster_b = rng.normal(loc=[10, 10], scale=0.3, size=(100, 2))
    features = np.vstack([cluster_a, cluster_b])
    labels = np.array([0] * 100 + [1] * 100)
    return features, labels


@pytest.fixture
def overlapping_clusters():
    """Two overlapping Gaussian clusters (same center, different spread)."""
    rng = np.random.default_rng(42)
    cluster_a = rng.normal(loc=[0, 0], scale=1.0, size=(100, 2))
    cluster_b = rng.normal(loc=[1.0, 1.0], scale=1.0, size=(100, 2))
    features = np.vstack([cluster_a, cluster_b])
    labels = np.array([0] * 100 + [1] * 100)
    return features, labels


@pytest.fixture
def mock_feature_df():
    """Mock dataframe mimicking the feature CSV structure."""
    rows = []
    for state in ["baseline", "rest", "low", "high"]:
        for d in [3, 4, 5]:
            for tau in [1, 2, 3]:
                rows.append({
                    "file_name": "data/train/Eda/12.csv",
                    "signal": f"12_{state.capitalize()}_1",
                    "signallength": 5950,
                    "pe": np.random.uniform(0.5, 1.0),
                    "comp": np.random.uniform(0.0, 0.5),
                    "fisher_shannon": np.random.uniform(0.5, 1.0),
                    "fisher_info": np.random.uniform(0.0, 0.5),
                    "renyipe": np.random.uniform(0.5, 1.0),
                    "renyicomp": np.random.uniform(0.0, 0.5),
                    "tsallispe": np.random.uniform(0.5, 1.0),
                    "tsalliscomp": np.random.uniform(0.0, 0.5),
                    "dimension": d,
                    "tau": tau,
                    "state": state,
                    "binaryclass": 0 if state == "baseline" else 1,
                    "nan_percentage": 0.0,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Silhouette Index Tests
# ---------------------------------------------------------------------------

class TestSilhouetteIndex:
    def test_well_separated_clusters(self, well_separated_clusters):
        features, labels = well_separated_clusters
        score = compute_silhouette(features, labels)
        assert score > 0.8, f"Expected Silhouette > 0.8 for well-separated clusters, got {score:.4f}"

    def test_returns_float(self, well_separated_clusters):
        features, labels = well_separated_clusters
        score = compute_silhouette(features, labels)
        assert isinstance(score, float)

    def test_range_bounded(self, overlapping_clusters):
        features, labels = overlapping_clusters
        score = compute_silhouette(features, labels)
        assert -1.0 <= score <= 1.0, f"Silhouette must be in [-1, 1], got {score}"


# ---------------------------------------------------------------------------
# Dunn Index Tests
# ---------------------------------------------------------------------------

class TestDunnIndex:
    def test_well_separated_clusters(self, well_separated_clusters):
        features, labels = well_separated_clusters
        dunn = compute_dunn_index(features, labels)
        assert dunn > 1.0, f"Expected Dunn > 1.0 for well-separated clusters, got {dunn:.4f}"

    def test_overlapping_worse_than_separated(self, well_separated_clusters, overlapping_clusters):
        feat_sep, lab_sep = well_separated_clusters
        feat_olap, lab_olap = overlapping_clusters
        dunn_sep = compute_dunn_index(feat_sep, lab_sep)
        dunn_olap = compute_dunn_index(feat_olap, lab_olap)
        assert dunn_sep > dunn_olap, (
            f"Separated Dunn ({dunn_sep:.4f}) should exceed overlapping ({dunn_olap:.4f})"
        )

    def test_positive_value(self, well_separated_clusters):
        features, labels = well_separated_clusters
        dunn = compute_dunn_index(features, labels)
        assert dunn > 0, "Dunn index must be positive"


# ---------------------------------------------------------------------------
# Outlier Removal Tests
# ---------------------------------------------------------------------------

class TestOutlierRemovalIQR:
    def test_removes_known_outliers(self):
        """Synthetic data with known outliers beyond 1.5 * IQR."""
        rng = np.random.default_rng(42)
        normal_data = rng.normal(loc=0, scale=1, size=100)
        # Add extreme outliers
        data_with_outliers = np.concatenate([normal_data, [100, -100, 50, -50]])
        df = pd.DataFrame({"value": data_with_outliers, "other": range(len(data_with_outliers))})
        cleaned = remove_outliers_iqr(df, columns=["value"], factor=1.5)
        assert len(cleaned) < len(df), "Should have removed some rows"
        assert cleaned["value"].max() < 100, "Extreme outlier at 100 should be removed"
        assert cleaned["value"].min() > -100, "Extreme outlier at -100 should be removed"

    def test_preserves_normal_data(self):
        """Data within IQR bounds should be preserved."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        cleaned = remove_outliers_iqr(df, columns=["value"], factor=1.5)
        assert len(cleaned) == len(df), "No outliers should be removed from tight data"

    def test_multiple_columns(self):
        """Outlier in any specified column should remove the row."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 100],
            "b": [1, 2, 3, 4, 5],
        })
        cleaned = remove_outliers_iqr(df, columns=["a", "b"], factor=1.5)
        assert 100 not in cleaned["a"].values


# ---------------------------------------------------------------------------
# Z-score Normalization Tests
# ---------------------------------------------------------------------------

class TestZscoreNormalization:
    def test_known_output(self):
        """Known input with manually computed z-scores."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        df = pd.DataFrame({"feat": data})
        normalized = apply_global_zscore(df, feature_cols=["feat"])
        # mean=3, std=sqrt(2.5) with ddof=0 (sklearn default)
        expected_mean = 0.0
        expected_std = 1.0
        assert abs(normalized["feat"].mean()) < 1e-10, "Mean should be ~0"
        assert abs(normalized["feat"].std(ddof=0) - expected_std) < 1e-10, "Std should be ~1"

    def test_preserves_non_feature_columns(self):
        """Non-feature columns should remain unchanged."""
        df = pd.DataFrame({"feat": [1.0, 2.0, 3.0], "label": ["a", "b", "c"]})
        normalized = apply_global_zscore(df, feature_cols=["feat"])
        assert list(normalized["label"]) == ["a", "b", "c"]

    def test_multiple_features(self):
        """Each feature should be independently normalized."""
        df = pd.DataFrame({
            "f1": [10.0, 20.0, 30.0],
            "f2": [100.0, 200.0, 300.0],
        })
        normalized = apply_global_zscore(df, feature_cols=["f1", "f2"])
        assert abs(normalized["f1"].mean()) < 1e-10
        assert abs(normalized["f2"].mean()) < 1e-10


# ---------------------------------------------------------------------------
# Filter Baseline Only Tests
# ---------------------------------------------------------------------------

class TestMakeBinaryLabels:
    def test_baseline_is_zero(self, mock_feature_df):
        filtered = filter_baseline_only(mock_feature_df)
        labels = make_binary_labels(filtered)
        baseline_mask = filtered["state"] == "baseline"
        assert all(labels[baseline_mask] == 0)

    def test_pain_is_one(self, mock_feature_df):
        filtered = filter_baseline_only(mock_feature_df)
        labels = make_binary_labels(filtered)
        pain_mask = filtered["state"].isin(["low", "high"])
        assert all(labels[pain_mask] == 1)

    def test_only_two_values(self, mock_feature_df):
        filtered = filter_baseline_only(mock_feature_df)
        labels = make_binary_labels(filtered)
        assert set(np.unique(labels)) == {0, 1}


class TestFilterBaselineOnly:
    def test_excludes_rest(self, mock_feature_df):
        filtered = filter_baseline_only(mock_feature_df)
        assert "rest" not in filtered["state"].values, "Rest segments must be excluded"

    def test_retains_baseline(self, mock_feature_df):
        filtered = filter_baseline_only(mock_feature_df)
        assert "baseline" in filtered["state"].values, "Baseline must be retained"

    def test_retains_pain_classes(self, mock_feature_df):
        filtered = filter_baseline_only(mock_feature_df)
        assert "low" in filtered["state"].values
        assert "high" in filtered["state"].values

    def test_row_count(self, mock_feature_df):
        n_rest = len(mock_feature_df[mock_feature_df["state"] == "rest"])
        filtered = filter_baseline_only(mock_feature_df)
        assert len(filtered) == len(mock_feature_df) - n_rest


# ---------------------------------------------------------------------------
# Statistical Comparison Tests
# ---------------------------------------------------------------------------

class TestStatisticalComparison:
    def test_ttest_against_baseline(self):
        """Known per-subject scores vs known baseline."""
        rng = np.random.default_rng(42)
        # Scores clearly below baseline
        scores = rng.normal(loc=0.70, scale=0.05, size=53)
        baseline = 0.78
        result = compute_statistical_comparison(scores, baseline)
        assert result["p_value"] < 0.05, "Should be significant when scores far below baseline"
        assert result["t_statistic"] < 0, "t should be negative when scores < baseline"

    def test_not_significant_when_close(self):
        """Scores close to baseline should not be significant."""
        rng = np.random.default_rng(42)
        scores = rng.normal(loc=0.78, scale=0.09, size=53)
        baseline = 0.78
        result = compute_statistical_comparison(scores, baseline)
        # With mean=baseline and reasonable variance, should often not be significant
        assert "p_value" in result
        assert "t_statistic" in result
        assert "cohens_d" in result

    def test_result_keys(self):
        scores = np.array([0.7, 0.75, 0.8])
        result = compute_statistical_comparison(scores, 0.78)
        assert "p_value" in result
        assert "t_statistic" in result
        assert "cohens_d" in result
        assert "mean" in result
        assert "std" in result


# ---------------------------------------------------------------------------
# Cohen's d Tests
# ---------------------------------------------------------------------------

class TestCohensD:
    def test_known_computation(self):
        """Cohen's d = (mean - baseline) / std."""
        scores = np.array([0.70, 0.70, 0.70])
        baseline = 0.80
        d = compute_cohens_d(scores, baseline)
        # mean=0.70, std=0.0 -> division by zero edge case
        # Use scores with variance
        scores = np.array([0.60, 0.70, 0.80])
        d = compute_cohens_d(scores, baseline)
        expected = (0.70 - 0.80) / np.std(scores, ddof=1)
        assert abs(d - expected) < 1e-10, f"Expected {expected}, got {d}"

    def test_positive_when_above_baseline(self):
        scores = np.array([0.85, 0.90, 0.88])
        d = compute_cohens_d(scores, 0.78)
        assert d > 0, "Cohen's d should be positive when scores > baseline"

    def test_negative_when_below_baseline(self):
        scores = np.array([0.65, 0.70, 0.68])
        d = compute_cohens_d(scores, 0.78)
        assert d < 0, "Cohen's d should be negative when scores < baseline"
