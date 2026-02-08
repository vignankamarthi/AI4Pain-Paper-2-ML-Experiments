"""
Comprehensive tests for Phase 8.2: Feature Fusion (Catch22 + Entropy-Complexity).

All tests use mock/synthetic data only -- no real dataset access.
No emojis anywhere in this file.
"""

import pytest
import json
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys

# ---------------------------------------------------------------------------
# Module import with dependency mocking
# ---------------------------------------------------------------------------
# pycatch22 may not be installed in test environments. The source module calls
# sys.exit(1) on import failure, so we must mock it before importing.
_pycatch22_mocked = False
try:
    import pycatch22 as _pc22_real
except ImportError:
    _pycatch22_mocked = True
    _mock_pycatch22 = MagicMock()
    sys.modules['pycatch22'] = _mock_pycatch22

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from phase8_2_feature_fusion import (
        extract_subject_id,
        extract_catch22_from_segment,
        SIGNAL_DIR_MAP,
        SIGNALS,
        ENTROPY_FEATURE_COLS,
        CLASS_MAPPING,
        CLASS_NAMES,
        RANDOM_SEED,
        PAPER1_LOSO_BASELINE,
        pivot_to_multimodal,
        merge_features,
        select_features_by_mi,
        evaluate_with_inner_cv,
        save_checkpoint,
        load_checkpoint,
        convert_to_serializable,
        format_duration,
        generate_leaderboard,
        generate_ablation_table,
    )
except (ImportError, SystemExit):
    pytest.skip("Required dependencies not available", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helper: build a synthetic results dict matching the structure expected by
# generate_leaderboard and generate_ablation_table.
# ---------------------------------------------------------------------------

def _make_synthetic_results():
    """Build a synthetic results dictionary for 3 experiments."""
    results = {}
    configs = [
        ("RF_entropy", "RandomForest", "entropy", 32, 0.82),
        ("XGB_catch22", "XGBoost", "catch22", 88, 0.79),
        ("LGB_combined", "LightGBM", "combined", 120, 0.85),
    ]
    for exp_id, model, fset, nf, ba_mean in configs:
        ba_std = 0.05
        ci_lower = ba_mean - 1.96 * ba_std / np.sqrt(53)
        ci_upper = ba_mean + 1.96 * ba_std / np.sqrt(53)
        results[exp_id] = {
            "model": model,
            "feature_set": fset,
            "n_features": nf,
            "metrics": {
                "balanced_accuracy_mean": ba_mean,
                "balanced_accuracy_std": ba_std,
                "accuracy_mean": ba_mean + 0.01,
                "accuracy_std": ba_std,
                "f1_mean": ba_mean - 0.01,
                "f1_std": ba_std,
            },
            "ci_95": (ci_lower, ci_upper),
            "statistical_test": {
                "t_statistic": 1.5,
                "p_value": 0.13,
                "significant": False,
            },
            "cohens_d": 0.2,
        }
    return results


# ===================================================================
# 1. extract_subject_id
# ===================================================================

class TestExtractSubjectId:
    """Tests for extract_subject_id parsing logic."""

    def test_standard_baseline_pattern(self):
        assert extract_subject_id("12_Baseline_1") == "12"

    def test_high_pain_pattern(self):
        assert extract_subject_id("54_HIGH_2") == "54"

    def test_low_pain_pattern(self):
        assert extract_subject_id("7_low_3") == "7"

    def test_rest_pattern(self):
        assert extract_subject_id("103_rest_1") == "103"

    def test_numeric_only(self):
        """A bare number should still be extracted."""
        assert extract_subject_id("42") == "42"

    def test_leading_digits_no_underscore(self):
        """Digits embedded in non-standard name -- regex fallback."""
        assert extract_subject_id("subj99end") == "99"

    def test_no_digits_returns_original(self):
        """If no digits at all, return the original string."""
        assert extract_subject_id("abc") == "abc"

    def test_multi_digit_prefix(self):
        assert extract_subject_id("1001_Baseline_5") == "1001"

    def test_empty_string(self):
        """Empty string has no digits -- returns itself."""
        assert extract_subject_id("") == ""


# ===================================================================
# 2. extract_catch22_from_segment
# ===================================================================

class TestExtractCatch22FromSegment:
    """Tests for catch22 feature extraction from a single segment."""

    def test_short_signal_returns_nan_dict(self):
        """Signals shorter than 10 samples should return 22 NaN values."""
        short_signal = np.array([1.0, 2.0, 3.0])
        result = extract_catch22_from_segment(short_signal)
        assert len(result) == 22
        assert all(k.startswith("catch22_") for k in result.keys())
        assert all(np.isnan(v) for v in result.values())

    def test_empty_signal_returns_nan_dict(self):
        result = extract_catch22_from_segment(np.array([]))
        assert len(result) == 22
        assert all(np.isnan(v) for v in result.values())

    def test_all_nan_input_returns_nan_dict(self):
        """An array of all NaNs becomes length-0 after filtering -> NaN dict."""
        result = extract_catch22_from_segment(np.full(50, np.nan))
        assert len(result) == 22
        assert all(np.isnan(v) for v in result.values())

    def test_partial_nan_input_long_enough(self):
        """After NaN removal, if >= 10 samples remain, should call pycatch22."""
        values = np.concatenate([np.sin(np.linspace(0, 4 * np.pi, 50)),
                                 np.full(5, np.nan)])
        mock_return = {
            "names": [f"feat_{i}" for i in range(22)],
            "values": [float(i) * 0.1 for i in range(22)],
        }
        with patch("phase8_2_feature_fusion.pycatch22") as mock_pc22:
            mock_pc22.catch22_all.return_value = mock_return
            result = extract_catch22_from_segment(values)

        assert len(result) == 22
        for i in range(22):
            assert result[f"catch22_{i+1}"] == pytest.approx(float(i) * 0.1)

    def test_valid_signal_returns_22_features(self):
        """A valid signal should produce exactly 22 named features."""
        values = np.sin(np.linspace(0, 6 * np.pi, 200))
        mock_return = {
            "names": [f"feat_{i}" for i in range(22)],
            "values": [1.0 + i for i in range(22)],
        }
        with patch("phase8_2_feature_fusion.pycatch22") as mock_pc22:
            mock_pc22.catch22_all.return_value = mock_return
            result = extract_catch22_from_segment(values)

        assert len(result) == 22
        assert set(result.keys()) == {f"catch22_{i+1}" for i in range(22)}
        # Values should match mock return
        for i in range(22):
            assert result[f"catch22_{i+1}"] == 1.0 + i

    def test_pycatch22_exception_returns_nan_dict(self):
        """If pycatch22 raises, the function should return NaN dict."""
        values = np.sin(np.linspace(0, 2 * np.pi, 100))
        with patch("phase8_2_feature_fusion.pycatch22") as mock_pc22:
            mock_pc22.catch22_all.side_effect = RuntimeError("computation failed")
            result = extract_catch22_from_segment(values)

        assert len(result) == 22
        assert all(np.isnan(v) for v in result.values())

    def test_key_names_are_sequential(self):
        """Keys must be catch22_1 through catch22_22."""
        result = extract_catch22_from_segment(np.array([1.0]))
        expected_keys = [f"catch22_{i}" for i in range(1, 23)]
        assert sorted(result.keys()) == sorted(expected_keys)


# ===================================================================
# 3. SIGNAL_DIR_MAP
# ===================================================================

class TestSignalDirMap:
    """Verify the filesystem directory name mapping for all signals."""

    def test_spo2_maps_to_SpO2(self):
        """Critical: spo2 must map to 'SpO2', not 'Spo2'."""
        assert SIGNAL_DIR_MAP["spo2"] == "SpO2"

    def test_eda_maps_to_Eda(self):
        assert SIGNAL_DIR_MAP["eda"] == "Eda"

    def test_bvp_maps_to_Bvp(self):
        assert SIGNAL_DIR_MAP["bvp"] == "Bvp"

    def test_resp_maps_to_Resp(self):
        assert SIGNAL_DIR_MAP["resp"] == "Resp"

    def test_all_four_signals_present(self):
        assert set(SIGNAL_DIR_MAP.keys()) == {"eda", "bvp", "resp", "spo2"}

    def test_signals_list_matches_dir_map_keys(self):
        assert set(SIGNALS) == set(SIGNAL_DIR_MAP.keys())


# ===================================================================
# 4. pivot_to_multimodal
# ===================================================================

class TestPivotToMultimodal:
    """Tests for long-to-wide pivoting of entropy features."""

    @staticmethod
    def _make_long_df(n_segments=5):
        """Create a synthetic long-format dataframe with 4 signals per segment."""
        rng = np.random.RandomState(42)
        rows = []
        for seg_idx in range(n_segments):
            seg_id = f"{seg_idx + 10}_Baseline_1"
            subj_id = str(seg_idx + 10)
            for signal in SIGNALS:
                row = {
                    "segment_id": seg_id,
                    "subject_id": subj_id,
                    "state": "baseline",
                    "label": 0,
                    "phys_signal": signal,
                }
                for feat in ENTROPY_FEATURE_COLS:
                    row[feat] = rng.rand()
                rows.append(row)
        return pd.DataFrame(rows)

    def test_output_shape_rows(self):
        """One row per segment after pivot."""
        df = self._make_long_df(n_segments=5)
        result = pivot_to_multimodal(df, ENTROPY_FEATURE_COLS)
        assert len(result) == 5

    def test_output_shape_columns(self):
        """Should have 4 signals x 8 features = 32 feature columns plus metadata."""
        df = self._make_long_df(n_segments=5)
        result = pivot_to_multimodal(df, ENTROPY_FEATURE_COLS)
        feature_cols = [c for c in result.columns
                        if any(c.startswith(f"{s}_") for s in SIGNALS)]
        assert len(feature_cols) == len(SIGNALS) * len(ENTROPY_FEATURE_COLS)

    def test_column_naming_convention(self):
        """Feature columns should follow the signal_feature pattern."""
        df = self._make_long_df(n_segments=3)
        result = pivot_to_multimodal(df, ENTROPY_FEATURE_COLS)
        for signal in SIGNALS:
            for feat in ENTROPY_FEATURE_COLS:
                col_name = f"{signal}_{feat}"
                assert col_name in result.columns, f"Missing column: {col_name}"

    def test_metadata_columns_preserved(self):
        """segment_id, subject_id, state, label must be in output."""
        df = self._make_long_df(n_segments=2)
        result = pivot_to_multimodal(df, ENTROPY_FEATURE_COLS)
        for col in ["segment_id", "subject_id", "state", "label"]:
            assert col in result.columns

    def test_inner_join_drops_incomplete_segments(self):
        """Segments missing data for one signal should be dropped (inner join)."""
        df = self._make_long_df(n_segments=3)
        # Remove all 'bvp' rows for segment 0
        seg0_id = df["segment_id"].unique()[0]
        mask = ~((df["segment_id"] == seg0_id) & (df["phys_signal"] == "bvp"))
        df_partial = df[mask].copy()
        result = pivot_to_multimodal(df_partial, ENTROPY_FEATURE_COLS)
        assert len(result) == 2  # only 2 complete segments remain


# ===================================================================
# 5. merge_features
# ===================================================================

class TestMergeFeatures:
    """Tests for merging entropy and catch22 feature dataframes."""

    def test_inner_join_on_segment_id(self):
        """Only segments present in both dataframes should survive."""
        entropy_df = pd.DataFrame({
            "segment_id": ["a", "b", "c"],
            "subject_id": ["1", "2", "3"],
            "eda_pe": [0.1, 0.2, 0.3],
        })
        catch22_df = pd.DataFrame({
            "segment_id": ["b", "c", "d"],
            "eda_catch22_1": [1.0, 2.0, 3.0],
        })
        result = merge_features(entropy_df, catch22_df)
        assert len(result) == 2
        assert set(result["segment_id"]) == {"b", "c"}

    def test_column_count_after_merge(self):
        """Merged df should contain columns from both inputs (minus duplicate segment_id)."""
        entropy_df = pd.DataFrame({
            "segment_id": ["a", "b"],
            "subject_id": ["1", "2"],
            "eda_pe": [0.1, 0.2],
            "bvp_pe": [0.3, 0.4],
        })
        catch22_df = pd.DataFrame({
            "segment_id": ["a", "b"],
            "eda_catch22_1": [1.0, 2.0],
            "bvp_catch22_1": [3.0, 4.0],
        })
        result = merge_features(entropy_df, catch22_df)
        # segment_id + subject_id + 2 entropy + 2 catch22 = 6
        assert len(result.columns) == 6

    def test_no_overlap_returns_empty(self):
        """Disjoint segment_ids -> empty merge result."""
        entropy_df = pd.DataFrame({"segment_id": ["a"], "x": [1]})
        catch22_df = pd.DataFrame({"segment_id": ["b"], "y": [2]})
        result = merge_features(entropy_df, catch22_df)
        assert len(result) == 0


# ===================================================================
# 6. select_features_by_mi
# ===================================================================

class TestSelectFeaturesByMI:
    """Tests for mutual-information-based feature selection."""

    def test_returns_correct_number_of_features(self):
        rng = np.random.RandomState(42)
        X = rng.rand(50, 10)
        y = np.array([0] * 25 + [1] * 25)
        feature_names = [f"feat_{i}" for i in range(10)]
        selected = select_features_by_mi(X, y, feature_names, k=5)
        assert len(selected) == 5

    def test_all_selected_are_valid_names(self):
        rng = np.random.RandomState(42)
        X = rng.rand(50, 10)
        y = np.array([0] * 25 + [1] * 25)
        feature_names = [f"feat_{i}" for i in range(10)]
        selected = select_features_by_mi(X, y, feature_names, k=3)
        for name in selected:
            assert name in feature_names

    def test_deterministic_with_same_data(self):
        """Calling twice with the same data should yield the same features."""
        rng = np.random.RandomState(42)
        X = rng.rand(60, 8)
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        feature_names = [f"f{i}" for i in range(8)]
        run1 = select_features_by_mi(X, y, feature_names, k=4)
        run2 = select_features_by_mi(X, y, feature_names, k=4)
        assert run1 == run2

    def test_k_equals_total_features(self):
        """Selecting k = total features should return all features."""
        rng = np.random.RandomState(0)
        X = rng.rand(30, 5)
        y = np.array([0] * 15 + [1] * 15)
        feature_names = [f"f{i}" for i in range(5)]
        selected = select_features_by_mi(X, y, feature_names, k=5)
        assert set(selected) == set(feature_names)

    def test_informative_feature_is_selected(self):
        """A perfectly informative feature should be in the top-1 selection."""
        rng = np.random.RandomState(42)
        n = 100
        # Feature 0 is a copy of the label; others are random noise
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        noise = rng.rand(n, 4)
        informative = y.reshape(-1, 1).astype(float)
        X = np.hstack([informative, noise])
        feature_names = ["informative", "noise1", "noise2", "noise3", "noise4"]
        selected = select_features_by_mi(X, y, feature_names, k=1)
        assert "informative" in selected


# ===================================================================
# 7. evaluate_with_inner_cv
# ===================================================================

class TestEvaluateWithInnerCV:
    """Tests for inner cross-validation scoring."""

    @staticmethod
    def _make_separable_data():
        """Create well-separated 3-class synthetic data.

        3 subjects, 5 samples each = 15 total, 4 features.
        """
        rng = np.random.RandomState(42)
        X = np.vstack([
            rng.randn(5, 4) + np.array([0, 0, 0, 0]),
            rng.randn(5, 4) + np.array([8, 8, 8, 8]),
            rng.randn(5, 4) + np.array([16, 16, 16, 16]),
        ])
        y = np.array([0] * 5 + [1] * 5 + [2] * 5)
        subject_ids = np.array(["s1"] * 5 + ["s2"] * 5 + ["s3"] * 5)
        return X, y, subject_ids

    def test_returns_float_score(self):
        X, y, sids = self._make_separable_data()
        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}
        score = evaluate_with_inner_cv(X, y, sids, "RandomForest", params)
        assert isinstance(score, (float, np.floating))

    def test_score_in_valid_range(self):
        X, y, sids = self._make_separable_data()
        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}
        score = evaluate_with_inner_cv(X, y, sids, "RandomForest", params)
        assert 0.0 <= score <= 1.0

    def test_separable_data_high_score(self):
        """Well-separated classes should yield a high balanced accuracy."""
        X, y, sids = self._make_separable_data()
        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}
        score = evaluate_with_inner_cv(X, y, sids, "RandomForest", params)
        assert score >= 0.8

    def test_unknown_model_raises(self):
        X, y, sids = self._make_separable_data()
        with pytest.raises(ValueError, match="Unknown model"):
            evaluate_with_inner_cv(X, y, sids, "FakeModel", {})


# ===================================================================
# 8. Checkpoint save / load cycle
# ===================================================================

class TestCheckpointing:
    """Tests for checkpoint persistence."""

    def test_save_then_load_round_trip(self, tmp_path):
        checkpoint = {
            "completed_experiments": ["RF_entropy", "XGB_catch22"],
            "results": {"RF_entropy": {"score": 0.85}},
        }
        save_checkpoint(tmp_path, checkpoint)
        loaded = load_checkpoint(tmp_path)
        assert set(loaded["completed_experiments"]) == {"RF_entropy", "XGB_catch22"}
        assert loaded["results"]["RF_entropy"]["score"] == 0.85

    def test_load_adds_last_updated(self, tmp_path):
        checkpoint = {"completed_experiments": [], "results": {}}
        save_checkpoint(tmp_path, checkpoint)
        loaded = load_checkpoint(tmp_path)
        # save_checkpoint adds 'last_updated' to the checkpoint before writing
        assert "last_updated" in loaded

    def test_load_nonexistent_returns_empty_state(self, tmp_path):
        loaded = load_checkpoint(tmp_path)
        assert loaded == {"completed_experiments": [], "results": {}}

    def test_corrupted_json_returns_empty_state(self, tmp_path):
        bad_file = tmp_path / "checkpoint.json"
        bad_file.write_text("{invalid json content!!!")
        loaded = load_checkpoint(tmp_path)
        assert loaded == {"completed_experiments": [], "results": {}}

    def test_checkpoint_file_is_valid_json(self, tmp_path):
        checkpoint = {
            "completed_experiments": ["exp1"],
            "results": {"exp1": {"val": np.float64(0.9)}},
        }
        save_checkpoint(tmp_path, checkpoint)
        raw = (tmp_path / "checkpoint.json").read_text()
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_numpy_types_serialized_correctly(self, tmp_path):
        """Checkpoint with numpy types should serialize to native Python types."""
        checkpoint = {
            "completed_experiments": [],
            "results": {
                "test": {
                    "int_val": np.int64(10),
                    "float_val": np.float64(3.14),
                    "array_val": np.array([1, 2, 3]),
                }
            },
        }
        save_checkpoint(tmp_path, checkpoint)
        loaded = load_checkpoint(tmp_path)
        assert loaded["results"]["test"]["int_val"] == 10
        assert loaded["results"]["test"]["float_val"] == pytest.approx(3.14)
        assert loaded["results"]["test"]["array_val"] == [1, 2, 3]


# ===================================================================
# 9. convert_to_serializable
# ===================================================================

class TestConvertToSerializable:
    """Tests for numpy-to-native type conversion."""

    def test_numpy_int(self):
        assert convert_to_serializable(np.int64(42)) == 42
        assert isinstance(convert_to_serializable(np.int64(42)), int)

    def test_numpy_float(self):
        result = convert_to_serializable(np.float64(3.14))
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_numpy_array(self):
        result = convert_to_serializable(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_numpy_bool(self):
        assert convert_to_serializable(np.bool_(True)) is True
        assert isinstance(convert_to_serializable(np.bool_(False)), bool)

    def test_nested_dict(self):
        data = {"a": np.int64(1), "b": {"c": np.float64(2.5)}}
        result = convert_to_serializable(data)
        assert result == {"a": 1, "b": {"c": 2.5}}
        assert isinstance(result["a"], int)
        assert isinstance(result["b"]["c"], float)

    def test_nested_list(self):
        data = [np.int64(1), [np.float64(2.0), np.array([3, 4])]]
        result = convert_to_serializable(data)
        assert result == [1, [2.0, [3, 4]]]

    def test_tuple_converted_to_list(self):
        data = (np.int64(1), np.float64(2.0))
        result = convert_to_serializable(data)
        assert result == [1, 2.0]
        assert isinstance(result, list)

    def test_plain_python_types_unchanged(self):
        assert convert_to_serializable(42) == 42
        assert convert_to_serializable("hello") == "hello"
        assert convert_to_serializable(None) is None

    def test_mixed_dict_list_nesting(self):
        data = {
            "scores": [np.float64(0.9), np.float64(0.8)],
            "meta": {"count": np.int32(5), "flag": np.bool_(True)},
        }
        result = convert_to_serializable(data)
        # Must be fully JSON-serializable now
        serialized = json.dumps(result)
        assert isinstance(serialized, str)


# ===================================================================
# 10. format_duration
# ===================================================================

class TestFormatDuration:
    """Tests for human-readable duration formatting."""

    def test_seconds_range(self):
        assert format_duration(30.0) == "30.0s"

    def test_zero_seconds(self):
        assert format_duration(0.0) == "0.0s"

    def test_just_under_one_minute(self):
        assert format_duration(59.9) == "59.9s"

    def test_exactly_one_minute(self):
        # 60 seconds -> minutes branch
        assert format_duration(60.0) == "1.0m"

    def test_minutes_range(self):
        assert format_duration(150.0) == "2.5m"

    def test_just_under_one_hour(self):
        assert format_duration(3599.0) == "60.0m"

    def test_exactly_one_hour(self):
        assert format_duration(3600.0) == "1.0h"

    def test_hours_range(self):
        assert format_duration(7200.0) == "2.0h"

    def test_fractional_seconds(self):
        assert format_duration(0.5) == "0.5s"

    def test_large_hours(self):
        assert format_duration(36000.0) == "10.0h"


# ===================================================================
# 11. LOSO split correctness
# ===================================================================

class TestLOSOSplitCorrectness:
    """Verify that LOSO splitting logic keeps the test subject out of training."""

    @staticmethod
    def _make_loso_df():
        """Create a small synthetic df with 4 subjects."""
        rng = np.random.RandomState(42)
        rows = []
        for subj in ["10", "20", "30", "40"]:
            for _ in range(5):
                rows.append({
                    "subject_id": subj,
                    "label": rng.choice([0, 1, 2]),
                    "f1": rng.rand(),
                    "f2": rng.rand(),
                })
        return pd.DataFrame(rows)

    def test_test_subject_not_in_training(self):
        df = self._make_loso_df()
        test_subject = "20"
        train_mask = df["subject_id"] != test_subject
        test_mask = df["subject_id"] == test_subject
        train_subjects = df[train_mask]["subject_id"].unique()
        assert test_subject not in train_subjects

    def test_all_other_subjects_in_training(self):
        df = self._make_loso_df()
        all_subjects = set(df["subject_id"].unique())
        test_subject = "30"
        train_mask = df["subject_id"] != test_subject
        train_subjects = set(df[train_mask]["subject_id"].unique())
        assert train_subjects == all_subjects - {test_subject}

    def test_no_data_leakage_across_split(self):
        df = self._make_loso_df()
        test_subject = "10"
        train_mask = df["subject_id"] != test_subject
        test_mask = df["subject_id"] == test_subject
        train_indices = set(df[train_mask].index)
        test_indices = set(df[test_mask].index)
        assert train_indices.isdisjoint(test_indices)

    def test_union_covers_full_dataset(self):
        df = self._make_loso_df()
        test_subject = "40"
        train_mask = df["subject_id"] != test_subject
        test_mask = df["subject_id"] == test_subject
        assert len(df[train_mask]) + len(df[test_mask]) == len(df)

    def test_each_subject_gets_a_turn_as_test(self):
        df = self._make_loso_df()
        subjects = sorted(df["subject_id"].unique())
        for subj in subjects:
            test_df = df[df["subject_id"] == subj]
            assert len(test_df) > 0, f"Subject {subj} has no test samples"


# ===================================================================
# 12. Feature set definition
# ===================================================================

class TestFeatureSetDefinition:
    """Verify feature set sizes and naming conventions."""

    def test_entropy_feature_cols_count(self):
        assert len(ENTROPY_FEATURE_COLS) == 8

    def test_signals_count(self):
        assert len(SIGNALS) == 4

    def test_entropy_features_total_32(self):
        """4 signals x 8 entropy features = 32 total."""
        entropy_features = [
            f"{signal}_{feat}" for signal in SIGNALS for feat in ENTROPY_FEATURE_COLS
        ]
        assert len(entropy_features) == 32

    def test_entropy_feature_names_contain_signal_prefix(self):
        entropy_features = [
            f"{signal}_{feat}" for signal in SIGNALS for feat in ENTROPY_FEATURE_COLS
        ]
        for feat in entropy_features:
            prefix = feat.split("_")[0]
            assert prefix in SIGNALS

    def test_catch22_features_correctly_prefixed(self):
        """Catch22 feature names should be signal_catch22_N."""
        for signal in SIGNALS:
            for i in range(1, 23):
                expected = f"{signal}_catch22_{i}"
                # Just verify the naming convention is constructible
                assert expected.startswith(signal)
                assert "catch22_" in expected

    def test_catch22_per_signal_is_22(self):
        """Each signal contributes 22 catch22 features."""
        per_signal = [f"catch22_{i+1}" for i in range(22)]
        assert len(per_signal) == 22

    def test_combined_feature_count(self):
        """32 entropy + 88 catch22 = 120 total."""
        n_entropy = len(SIGNALS) * len(ENTROPY_FEATURE_COLS)
        n_catch22 = len(SIGNALS) * 22
        assert n_entropy == 32
        assert n_catch22 == 88
        assert n_entropy + n_catch22 == 120

    def test_class_mapping_keys(self):
        assert set(CLASS_MAPPING.keys()) == {"baseline", "low", "high"}

    def test_class_mapping_values(self):
        assert CLASS_MAPPING["baseline"] == 0
        assert CLASS_MAPPING["low"] == 1
        assert CLASS_MAPPING["high"] == 2

    def test_class_names_list(self):
        assert CLASS_NAMES == ["no_pain", "low_pain", "high_pain"]

    def test_rest_excluded_from_class_mapping(self):
        """'rest' must NOT be in CLASS_MAPPING (baseline-only methodology)."""
        assert "rest" not in CLASS_MAPPING


# ===================================================================
# 13. generate_leaderboard
# ===================================================================

class TestGenerateLeaderboard:
    """Tests for the leaderboard generation function."""

    def test_correct_ranking_by_balanced_accuracy(self):
        results = _make_synthetic_results()
        lb = generate_leaderboard(results)
        # LGB_combined has 0.85, RF_entropy has 0.82, XGB_catch22 has 0.79
        assert lb.iloc[0]["model"] == "LightGBM"
        assert lb.iloc[1]["model"] == "RandomForest"
        assert lb.iloc[2]["model"] == "XGBoost"

    def test_rank_column_is_sequential(self):
        results = _make_synthetic_results()
        lb = generate_leaderboard(results)
        assert list(lb["rank"]) == [1, 2, 3]

    def test_leaderboard_has_expected_columns(self):
        results = _make_synthetic_results()
        lb = generate_leaderboard(results)
        expected_cols = {
            "rank", "model", "feature_set", "n_features",
            "loso_balanced_accuracy_mean", "loso_balanced_accuracy_std",
            "ci_95_lower", "ci_95_upper", "vs_paper1_improvement", "p_value",
        }
        assert expected_cols.issubset(set(lb.columns))

    def test_improvement_relative_to_paper1_baseline(self):
        results = _make_synthetic_results()
        lb = generate_leaderboard(results)
        best_row = lb.iloc[0]
        expected_improvement = best_row["loso_balanced_accuracy_mean"] - PAPER1_LOSO_BASELINE
        assert best_row["vs_paper1_improvement"] == pytest.approx(expected_improvement)

    def test_all_experiments_represented(self):
        results = _make_synthetic_results()
        lb = generate_leaderboard(results)
        assert len(lb) == len(results)

    def test_single_experiment(self):
        results = {
            "only_one": {
                "model": "RF",
                "feature_set": "entropy",
                "n_features": 32,
                "metrics": {
                    "balanced_accuracy_mean": 0.75,
                    "balanced_accuracy_std": 0.05,
                },
                "ci_95": (0.70, 0.80),
                "statistical_test": {"p_value": 0.1},
            }
        }
        lb = generate_leaderboard(results)
        assert len(lb) == 1
        assert lb.iloc[0]["rank"] == 1


# ===================================================================
# 14. generate_ablation_table
# ===================================================================

class TestGenerateAblationTable:
    """Tests for ablation table generation."""

    def test_all_experiments_represented(self):
        results = _make_synthetic_results()
        table = generate_ablation_table(results)
        assert len(table) == len(results)

    def test_expected_columns(self):
        results = _make_synthetic_results()
        table = generate_ablation_table(results)
        expected = {
            "model", "feature_set", "n_features",
            "balanced_accuracy", "std", "accuracy", "f1_weighted",
        }
        assert expected == set(table.columns)

    def test_feature_sets_present(self):
        results = _make_synthetic_results()
        table = generate_ablation_table(results)
        assert set(table["feature_set"]) == {"entropy", "catch22", "combined"}

    def test_values_match_input(self):
        results = _make_synthetic_results()
        table = generate_ablation_table(results)
        # Find the LightGBM combined row
        lgb_row = table[
            (table["model"] == "LightGBM") & (table["feature_set"] == "combined")
        ].iloc[0]
        assert lgb_row["balanced_accuracy"] == pytest.approx(0.85)
        assert lgb_row["n_features"] == 120

    def test_empty_results(self):
        table = generate_ablation_table({})
        assert len(table) == 0

    def test_single_experiment_ablation(self):
        results = {
            "single": {
                "model": "RF",
                "feature_set": "entropy",
                "n_features": 32,
                "metrics": {
                    "balanced_accuracy_mean": 0.77,
                    "balanced_accuracy_std": 0.04,
                    "accuracy_mean": 0.80,
                    "f1_mean": 0.76,
                },
            }
        }
        table = generate_ablation_table(results)
        assert len(table) == 1
        assert table.iloc[0]["model"] == "RF"
        assert table.iloc[0]["accuracy"] == pytest.approx(0.80)


# ===================================================================
# Additional edge-case and integration-like tests
# ===================================================================

class TestModuleConstants:
    """Verify important module-level constants."""

    def test_random_seed_is_42(self):
        assert RANDOM_SEED == 42

    def test_paper1_baseline_is_0_780(self):
        assert PAPER1_LOSO_BASELINE == pytest.approx(0.780)

    def test_entropy_feature_col_names(self):
        expected = [
            "pe", "comp", "fisher_shannon", "fisher_info",
            "renyipe", "renyicomp", "tsallispe", "tsalliscomp",
        ]
        assert ENTROPY_FEATURE_COLS == expected

    def test_signals_order(self):
        assert SIGNALS == ["eda", "bvp", "resp", "spo2"]
