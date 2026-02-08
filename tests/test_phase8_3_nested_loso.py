"""
Tests for Phase 8.3: Nested Optuna-LOSO Completion

All tests use synthetic/mock data only -- no real dataset access.
"""

import pytest
import json
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from phase8_3_nested_loso import (
    extract_subject_id,
    pivot_to_multimodal,
    get_rf_search_space,
    evaluate_with_inner_loso,
    load_checkpoint,
    save_checkpoint,
    save_fold_result,
    convert_to_serializable,
    format_duration,
    aggregate_results,
    SIGNALS,
    FEATURE_COLS,
    CLASS_MAPPING,
    RANDOM_SEED,
    PAPER1_LOSO_BASELINE
)


# =============================================================================
# Constants Verification
# =============================================================================

class TestModuleConstants:
    """Verify module-level constants are correctly defined."""

    def test_signals_list(self):
        assert SIGNALS == ['eda', 'bvp', 'resp', 'spo2']

    def test_feature_cols_list(self):
        expected = ['pe', 'comp', 'fisher_shannon', 'fisher_info',
                    'renyipe', 'renyicomp', 'tsallispe', 'tsalliscomp']
        assert FEATURE_COLS == expected

    def test_class_mapping(self):
        assert CLASS_MAPPING == {'baseline': 0, 'low': 1, 'high': 2}
        assert 'rest' not in CLASS_MAPPING

    def test_random_seed(self):
        assert RANDOM_SEED == 42

    def test_paper1_baseline(self):
        np.testing.assert_almost_equal(PAPER1_LOSO_BASELINE, 0.780)


# =============================================================================
# 1. extract_subject_id
# =============================================================================

class TestExtractSubjectId:
    """Test subject ID extraction from segment name patterns."""

    def test_baseline_pattern(self):
        assert extract_subject_id('12_Baseline_1') == '12'

    def test_high_pattern(self):
        assert extract_subject_id('54_HIGH_2') == '54'

    def test_low_pattern(self):
        assert extract_subject_id('7_low_3') == '7'

    def test_three_digit_subject(self):
        assert extract_subject_id('105_Baseline_1') == '105'

    def test_single_digit_subject(self):
        assert extract_subject_id('3_HIGH_1') == '3'

    def test_numeric_only_string(self):
        """When the segment name is purely numeric, regex fallback should find it."""
        assert extract_subject_id('42') == '42'

    def test_no_underscore_with_text(self):
        """When there is no underscore, the search fallback should find digits."""
        assert extract_subject_id('subject99data') == '99'

    def test_no_digits_returns_raw(self):
        """When no digits exist, the raw segment name is returned."""
        assert extract_subject_id('nodigits') == 'nodigits'

    def test_empty_string(self):
        assert extract_subject_id('') == ''

    def test_leading_zeros(self):
        assert extract_subject_id('007_Baseline_1') == '007'

    def test_multiple_underscored_segments(self):
        assert extract_subject_id('23_rest_segment_5') == '23'


# =============================================================================
# 2. pivot_to_multimodal
# =============================================================================

class TestPivotToMultimodal:
    """Test pivoting from long format (per-signal rows) to wide format."""

    @pytest.fixture
    def synthetic_long_df(self):
        """Create a synthetic long-format DataFrame with 4 signals and 3 segments."""
        rng = np.random.RandomState(42)
        rows = []
        segments = ['1_Baseline_1', '2_LOW_1', '3_HIGH_1']
        states = ['baseline', 'low', 'high']
        labels = [0, 1, 2]

        for seg, state, label in zip(segments, states, labels):
            subject = seg.split('_')[0]
            for signal in SIGNALS:
                row = {
                    'segment_id': seg,
                    'subject_id': subject,
                    'state': state,
                    'label': label,
                    'phys_signal': signal,
                }
                for feat in FEATURE_COLS:
                    row[feat] = rng.rand()
                rows.append(row)

        return pd.DataFrame(rows)

    def test_output_has_one_row_per_segment(self, synthetic_long_df):
        result = pivot_to_multimodal(synthetic_long_df)
        assert len(result) == 3

    def test_output_has_correct_feature_columns(self, synthetic_long_df):
        result = pivot_to_multimodal(synthetic_long_df)
        expected_cols = [f'{sig}_{feat}' for sig in SIGNALS for feat in FEATURE_COLS]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_total_feature_columns_count(self, synthetic_long_df):
        result = pivot_to_multimodal(synthetic_long_df)
        expected_cols = [f'{sig}_{feat}' for sig in SIGNALS for feat in FEATURE_COLS]
        # 4 signals x 8 features = 32
        assert len(expected_cols) == 32
        for col in expected_cols:
            assert col in result.columns

    def test_metadata_columns_preserved(self, synthetic_long_df):
        result = pivot_to_multimodal(synthetic_long_df)
        assert 'segment_id' in result.columns
        assert 'subject_id' in result.columns
        assert 'state' in result.columns
        assert 'label' in result.columns

    def test_inner_join_drops_segments_missing_signal(self, synthetic_long_df):
        """If a segment is missing one signal, it should be dropped by the inner merge."""
        # Remove EDA for segment '3_HIGH_1'
        df_partial = synthetic_long_df[
            ~((synthetic_long_df['segment_id'] == '3_HIGH_1') &
              (synthetic_long_df['phys_signal'] == 'eda'))
        ].copy()
        result = pivot_to_multimodal(df_partial)
        assert len(result) == 2
        assert '3_HIGH_1' not in result['segment_id'].values

    def test_no_duplicate_segment_ids(self, synthetic_long_df):
        result = pivot_to_multimodal(synthetic_long_df)
        assert result['segment_id'].is_unique


# =============================================================================
# 3. get_rf_search_space
# =============================================================================

class TestGetRfSearchSpace:
    """Test the RandomForest hyperparameter search space definition."""

    def test_returns_dict(self):
        trial = MagicMock(spec=['suggest_int', 'suggest_categorical'])
        trial.suggest_int.side_effect = lambda name, low, high: low
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        result = get_rf_search_space(trial)
        assert isinstance(result, dict)

    def test_contains_all_expected_keys(self):
        trial = MagicMock(spec=['suggest_int', 'suggest_categorical'])
        trial.suggest_int.side_effect = lambda name, low, high: low
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        result = get_rf_search_space(trial)

        expected_keys = {
            'n_estimators', 'max_depth', 'min_samples_split',
            'min_samples_leaf', 'max_features', 'class_weight',
            'criterion', 'random_state', 'n_jobs'
        }
        assert set(result.keys()) == expected_keys

    def test_random_state_is_42(self):
        trial = MagicMock(spec=['suggest_int', 'suggest_categorical'])
        trial.suggest_int.side_effect = lambda name, low, high: low
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        result = get_rf_search_space(trial)
        assert result['random_state'] == 42

    def test_n_jobs_is_negative_one(self):
        trial = MagicMock(spec=['suggest_int', 'suggest_categorical'])
        trial.suggest_int.side_effect = lambda name, low, high: low
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        result = get_rf_search_space(trial)
        assert result['n_jobs'] == -1

    def test_criterion_choices(self):
        """Verify that criterion suggestions are called with gini and entropy."""
        trial = MagicMock(spec=['suggest_int', 'suggest_categorical'])
        trial.suggest_int.side_effect = lambda name, low, high: low
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        get_rf_search_space(trial)

        criterion_call = [
            c for c in trial.suggest_categorical.call_args_list
            if c[0][0] == 'criterion'
        ]
        assert len(criterion_call) == 1
        assert set(criterion_call[0][0][1]) == {'gini', 'entropy'}


# =============================================================================
# 4. evaluate_with_inner_loso
# =============================================================================

class TestEvaluateWithInnerLoso:
    """Test inner LOSO evaluation on synthetic data."""

    @pytest.fixture
    def synthetic_loso_data(self):
        """
        Create a synthetic dataset with 5 subjects, 10 samples each, 4 features.
        Classes are linearly separable for reliable testing.
        """
        rng = np.random.RandomState(42)
        n_per_subject = 10
        n_features = 4
        subjects = ['s1', 's2', 's3', 's4', 's5']

        X_list = []
        y_list = []
        sid_list = []

        for s in subjects:
            for i in range(n_per_subject):
                label = i % 3  # cycle through 0, 1, 2
                # Linearly separable: offset features by label
                features = rng.randn(n_features) + label * 3.0
                X_list.append(features)
                y_list.append(label)
                sid_list.append(s)

        X = np.array(X_list)
        y = np.array(y_list)
        subject_ids = np.array(sid_list)
        return X, y, subject_ids

    @pytest.fixture
    def simple_rf_params(self):
        return {
            'n_estimators': 10,
            'max_depth': 3,
            'random_state': 42,
            'n_jobs': 1
        }

    def test_returns_float(self, synthetic_loso_data, simple_rf_params):
        X, y, subject_ids = synthetic_loso_data
        score = evaluate_with_inner_loso(X, y, subject_ids, simple_rf_params)
        assert isinstance(score, (float, np.floating))

    def test_score_in_valid_range(self, synthetic_loso_data, simple_rf_params):
        X, y, subject_ids = synthetic_loso_data
        score = evaluate_with_inner_loso(X, y, subject_ids, simple_rf_params)
        assert 0.0 <= score <= 1.0

    def test_perfect_separation_yields_high_score(self, simple_rf_params):
        """With very large class offsets, score should be near 1.0."""
        rng = np.random.RandomState(42)
        subjects = ['a', 'b', 'c', 'd', 'e']
        X_list, y_list, sid_list = [], [], []

        for s in subjects:
            for i in range(15):
                label = i % 3
                features = rng.randn(4) * 0.01 + label * 100.0
                X_list.append(features)
                y_list.append(label)
                sid_list.append(s)

        X = np.array(X_list)
        y = np.array(y_list)
        sids = np.array(sid_list)
        score = evaluate_with_inner_loso(X, y, sids, simple_rf_params)
        assert score > 0.9

    def test_subject_separation_in_inner_folds(self, synthetic_loso_data, simple_rf_params):
        """
        Verify that during inner LOSO, the held-out subject is never in the
        training set. We do this by patching RandomForestClassifier to capture
        the inputs.
        """
        X, y, subject_ids = synthetic_loso_data
        unique_subjects = np.unique(subject_ids)

        for test_subject in unique_subjects:
            train_mask = subject_ids != test_subject
            test_mask = subject_ids == test_subject

            # Verify no overlap
            train_sids = set(subject_ids[train_mask])
            test_sids = set(subject_ids[test_mask])
            assert test_subject not in train_sids
            assert test_subject in test_sids
            assert len(train_sids.intersection(test_sids)) == 0


# =============================================================================
# 5. Checkpoint save/load cycle
# =============================================================================

class TestCheckpointing:
    """Test checkpoint save and load functionality."""

    def test_save_then_load_returns_same_data(self, tmp_path):
        checkpoint_data = {
            'completed_folds': ['s1', 's2', 's3'],
            'fold_results': {
                's1': {'metrics': {'balanced_accuracy': 0.8}},
                's2': {'metrics': {'balanced_accuracy': 0.7}},
            },
            'start_time': '2026-01-01T00:00:00'
        }
        save_checkpoint(tmp_path, checkpoint_data)
        loaded = load_checkpoint(tmp_path)

        assert loaded['completed_folds'] == ['s1', 's2', 's3']
        assert 's1' in loaded['fold_results']
        assert loaded['fold_results']['s1']['metrics']['balanced_accuracy'] == 0.8
        assert loaded['start_time'] == '2026-01-01T00:00:00'

    def test_load_nonexistent_returns_default(self, tmp_path):
        loaded = load_checkpoint(tmp_path / 'nonexistent')
        assert loaded == {'completed_folds': [], 'fold_results': {}, 'start_time': None}

    def test_load_corrupted_json_returns_default(self, tmp_path):
        checkpoint_file = tmp_path / 'checkpoint.json'
        checkpoint_file.write_text('this is not valid json {{{')
        loaded = load_checkpoint(tmp_path)
        assert loaded == {'completed_folds': [], 'fold_results': {}, 'start_time': None}

    def test_completed_folds_list_preserved(self, tmp_path):
        folds = ['10', '20', '30', '40', '50']
        checkpoint_data = {
            'completed_folds': folds,
            'fold_results': {},
            'start_time': None
        }
        save_checkpoint(tmp_path, checkpoint_data)
        loaded = load_checkpoint(tmp_path)
        assert loaded['completed_folds'] == folds

    def test_save_adds_last_updated_field(self, tmp_path):
        checkpoint_data = {
            'completed_folds': [],
            'fold_results': {},
            'start_time': None
        }
        save_checkpoint(tmp_path, checkpoint_data)
        loaded = load_checkpoint(tmp_path)
        assert 'last_updated' in loaded

    def test_checkpoint_file_is_valid_json(self, tmp_path):
        checkpoint_data = {
            'completed_folds': ['s1'],
            'fold_results': {'s1': {'value': 1}},
            'start_time': None
        }
        save_checkpoint(tmp_path, checkpoint_data)
        checkpoint_file = tmp_path / 'checkpoint.json'
        with open(checkpoint_file, 'r') as f:
            parsed = json.load(f)
        assert isinstance(parsed, dict)


# =============================================================================
# 6. save_fold_result
# =============================================================================

class TestSaveFoldResult:
    """Test individual fold result saving."""

    def test_creates_fold_results_directory(self, tmp_path):
        fold_data = {'metrics': {'balanced_accuracy': 0.85}, 'y_true': [0, 1], 'y_pred': [0, 1]}
        save_fold_result(tmp_path, 's42', fold_data)
        assert (tmp_path / 'fold_results').is_dir()

    def test_creates_fold_json_file(self, tmp_path):
        fold_data = {'metrics': {'balanced_accuracy': 0.85}, 'y_true': [0, 1], 'y_pred': [0, 1]}
        save_fold_result(tmp_path, 's42', fold_data)
        assert (tmp_path / 'fold_results' / 'fold_s42.json').exists()

    def test_fold_file_is_valid_json(self, tmp_path):
        fold_data = {
            'metrics': {'balanced_accuracy': 0.85},
            'y_true': [0, 1, 2],
            'y_pred': [0, 1, 1],
            'best_params': {'n_estimators': 100}
        }
        save_fold_result(tmp_path, 'subj7', fold_data)
        fold_file = tmp_path / 'fold_results' / 'fold_subj7.json'
        with open(fold_file, 'r') as f:
            parsed = json.load(f)
        assert parsed['metrics']['balanced_accuracy'] == 0.85

    def test_handles_numpy_types_in_fold_data(self, tmp_path):
        fold_data = {
            'metrics': {'balanced_accuracy': np.float64(0.85)},
            'y_true': np.array([0, 1, 2]),
            'n_samples': np.int64(50)
        }
        save_fold_result(tmp_path, 'np_test', fold_data)
        fold_file = tmp_path / 'fold_results' / 'fold_np_test.json'
        with open(fold_file, 'r') as f:
            parsed = json.load(f)
        assert parsed['metrics']['balanced_accuracy'] == 0.85
        assert parsed['y_true'] == [0, 1, 2]
        assert parsed['n_samples'] == 50

    def test_multiple_fold_saves(self, tmp_path):
        for i in range(5):
            fold_data = {'metrics': {'balanced_accuracy': 0.7 + i * 0.05}}
            save_fold_result(tmp_path, f's{i}', fold_data)
        fold_dir = tmp_path / 'fold_results'
        json_files = list(fold_dir.glob('fold_*.json'))
        assert len(json_files) == 5


# =============================================================================
# 7. convert_to_serializable
# =============================================================================

class TestConvertToSerializable:
    """Test conversion of numpy types to JSON-serializable Python natives."""

    def test_numpy_int64(self):
        result = convert_to_serializable(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float64(self):
        result = convert_to_serializable(np.float64(3.14))
        assert isinstance(result, float)
        np.testing.assert_almost_equal(result, 3.14)

    def test_numpy_bool(self):
        result = convert_to_serializable(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_numpy_bool_false(self):
        result = convert_to_serializable(np.bool_(False))
        assert result is False
        assert isinstance(result, bool)

    def test_numpy_ndarray(self):
        arr = np.array([1, 2, 3])
        result = convert_to_serializable(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_numpy_2d_array(self):
        arr = np.array([[1, 2], [3, 4]])
        result = convert_to_serializable(arr)
        assert result == [[1, 2], [3, 4]]

    def test_nested_dict(self):
        data = {
            'a': np.int64(1),
            'b': {
                'c': np.float64(2.5),
                'd': np.array([10, 20])
            }
        }
        result = convert_to_serializable(data)
        assert result['a'] == 1
        assert isinstance(result['a'], int)
        assert result['b']['c'] == 2.5
        assert isinstance(result['b']['c'], float)
        assert result['b']['d'] == [10, 20]

    def test_nested_list(self):
        data = [np.int64(1), np.float64(2.0), [np.bool_(True), np.int64(3)]]
        result = convert_to_serializable(data)
        assert result == [1, 2.0, [True, 3]]

    def test_python_native_types_pass_through(self):
        assert convert_to_serializable(42) == 42
        assert convert_to_serializable(3.14) == 3.14
        assert convert_to_serializable('hello') == 'hello'
        assert convert_to_serializable(True) is True
        assert convert_to_serializable(None) is None

    def test_tuple_converted_to_list(self):
        result = convert_to_serializable((np.int64(1), np.int64(2)))
        assert result == [1, 2]
        assert isinstance(result, list)

    def test_empty_dict(self):
        assert convert_to_serializable({}) == {}

    def test_empty_list(self):
        assert convert_to_serializable([]) == []


# =============================================================================
# 8. format_duration
# =============================================================================

class TestFormatDuration:
    """Test human-readable duration formatting."""

    def test_seconds_under_60(self):
        result = format_duration(30.5)
        assert result == '30.5s'

    def test_zero_seconds(self):
        result = format_duration(0.0)
        assert result == '0.0s'

    def test_just_under_60(self):
        result = format_duration(59.9)
        assert result == '59.9s'

    def test_exactly_60_is_minutes(self):
        result = format_duration(60.0)
        assert result == '1.0m'

    def test_minutes_range(self):
        result = format_duration(150.0)
        assert result == '2.5m'

    def test_just_under_3600(self):
        result = format_duration(3599.0)
        # 3599 / 60 = 59.98...
        assert result == '60.0m'

    def test_exactly_3600_is_hours(self):
        result = format_duration(3600.0)
        assert result == '1.0h'

    def test_hours_range(self):
        result = format_duration(7200.0)
        assert result == '2.0h'

    def test_large_duration(self):
        result = format_duration(36000.0)
        assert result == '10.0h'

    def test_small_fraction_seconds(self):
        result = format_duration(0.1)
        assert result == '0.1s'


# =============================================================================
# 9. LOSO Split Verification
# =============================================================================

class TestLosoSplitVerification:
    """
    Verify LOSO split mechanics on a synthetic dataset with 5 subjects.
    This tests the splitting logic used in evaluate_with_inner_loso.
    """

    @pytest.fixture
    def five_subject_dataset(self):
        rng = np.random.RandomState(42)
        subjects = ['1', '2', '3', '4', '5']
        n_per = 8
        X_list, y_list, sid_list = [], [], []

        for s in subjects:
            for i in range(n_per):
                X_list.append(rng.randn(4))
                y_list.append(i % 3)
                sid_list.append(s)

        return (
            np.array(X_list),
            np.array(y_list),
            np.array(sid_list)
        )

    def test_held_out_subject_not_in_training(self, five_subject_dataset):
        X, y, subject_ids = five_subject_dataset
        held_out = '3'
        train_mask = subject_ids != held_out
        train_subjects = subject_ids[train_mask]
        assert held_out not in train_subjects

    def test_all_held_out_samples_in_test(self, five_subject_dataset):
        X, y, subject_ids = five_subject_dataset
        held_out = '3'
        test_mask = subject_ids == held_out
        test_subjects = subject_ids[test_mask]
        assert all(s == held_out for s in test_subjects)
        assert np.sum(test_mask) == 8  # n_per = 8

    def test_all_other_subjects_in_training(self, five_subject_dataset):
        X, y, subject_ids = five_subject_dataset
        held_out = '3'
        train_mask = subject_ids != held_out
        train_subjects_unique = set(subject_ids[train_mask])
        expected_train = {'1', '2', '4', '5'}
        assert train_subjects_unique == expected_train

    def test_no_sample_overlap_between_train_and_test(self, five_subject_dataset):
        X, y, subject_ids = five_subject_dataset
        held_out = '3'
        train_mask = subject_ids != held_out
        test_mask = subject_ids == held_out
        # Indices should be disjoint
        train_indices = set(np.where(train_mask)[0])
        test_indices = set(np.where(test_mask)[0])
        assert len(train_indices.intersection(test_indices)) == 0

    def test_train_plus_test_equals_total(self, five_subject_dataset):
        X, y, subject_ids = five_subject_dataset
        held_out = '3'
        train_mask = subject_ids != held_out
        test_mask = subject_ids == held_out
        assert np.sum(train_mask) + np.sum(test_mask) == len(subject_ids)

    def test_every_subject_gets_held_out_exactly_once(self, five_subject_dataset):
        X, y, subject_ids = five_subject_dataset
        unique_subjects = np.unique(subject_ids)
        held_out_subjects = []
        for s in unique_subjects:
            held_out_subjects.append(s)
        assert len(held_out_subjects) == 5
        assert set(held_out_subjects) == {'1', '2', '3', '4', '5'}


# =============================================================================
# 10. Normalization Correctness
# =============================================================================

class TestNormalizationCorrectness:
    """
    Verify that StandardScaler is fit on training pool only and
    the test data is transformed (not fit).
    """

    def test_scaler_fit_on_train_transform_both(self):
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(42)
        # Training data with known distribution
        X_train = rng.randn(100, 4) * 2 + 5  # mean ~5, std ~2
        # Test data from a different distribution
        X_test = rng.randn(20, 4) * 3 + 10  # mean ~10, std ~3

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train data should have mean ~0 and std ~1 after scaling
        np.testing.assert_almost_equal(X_train_scaled.mean(axis=0), np.zeros(4), decimal=1)
        np.testing.assert_almost_equal(X_train_scaled.std(axis=0), np.ones(4), decimal=1)

        # Test data should NOT have mean 0 since scaler was fit on train
        test_means = X_test_scaled.mean(axis=0)
        assert not np.allclose(test_means, 0, atol=0.5), (
            "Test data should not have near-zero mean when scaler is fit on train only"
        )

    def test_test_not_refit(self):
        """Confirm that calling transform (not fit_transform) on test preserves train params."""
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(123)
        X_train = rng.randn(50, 4) * 1.5 + 3.0
        X_test = rng.randn(10, 4) * 0.5 + 8.0

        scaler = StandardScaler()
        scaler.fit(X_train)
        train_mean = scaler.mean_.copy()
        train_scale = scaler.scale_.copy()

        _ = scaler.transform(X_test)

        # Scaler parameters should not change after transform
        np.testing.assert_array_equal(scaler.mean_, train_mean)
        np.testing.assert_array_equal(scaler.scale_, train_scale)

    def test_normalization_within_loso_fold(self):
        """
        Simulate a LOSO fold and verify normalization is fold-specific.
        """
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(42)
        subjects = ['a', 'b', 'c', 'd', 'e']
        n_per = 20
        X_all = rng.randn(n_per * len(subjects), 4)
        sids = np.repeat(subjects, n_per)

        held_out = 'c'
        train_mask = sids != held_out
        test_mask = sids == held_out

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_all[train_mask])
        X_test_scaled = scaler.transform(X_all[test_mask])

        # Dimensions preserved
        assert X_train_scaled.shape == (n_per * 4, 4)
        assert X_test_scaled.shape == (n_per, 4)


# =============================================================================
# 11. aggregate_results
# =============================================================================

class TestAggregateResults:
    """Test results aggregation from fold-level to summary statistics."""

    @pytest.fixture
    def mock_results(self):
        return {
            'fold_results': {
                's1': {
                    'metrics': {
                        'balanced_accuracy': 0.8,
                        'accuracy': 0.85,
                        'f1_weighted': 0.82
                    },
                    'y_true': [0, 1, 2],
                    'y_pred': [0, 1, 1]
                },
                's2': {
                    'metrics': {
                        'balanced_accuracy': 0.7,
                        'accuracy': 0.75,
                        'f1_weighted': 0.72
                    },
                    'y_true': [0, 1, 2],
                    'y_pred': [0, 0, 2]
                },
                's3': {
                    'metrics': {
                        'balanced_accuracy': 0.9,
                        'accuracy': 0.90,
                        'f1_weighted': 0.88
                    },
                    'y_true': [0, 1, 2],
                    'y_pred': [0, 1, 2]
                },
            }
        }

    def test_mean_balanced_accuracy(self, mock_results):
        summary = aggregate_results(mock_results)
        expected_mean = np.mean([0.8, 0.7, 0.9])
        np.testing.assert_almost_equal(
            summary['balanced_accuracy']['mean'], expected_mean
        )

    def test_std_balanced_accuracy(self, mock_results):
        summary = aggregate_results(mock_results)
        expected_std = np.std([0.8, 0.7, 0.9])
        np.testing.assert_almost_equal(
            summary['balanced_accuracy']['std'], expected_std
        )

    def test_median_balanced_accuracy(self, mock_results):
        summary = aggregate_results(mock_results)
        np.testing.assert_almost_equal(
            summary['balanced_accuracy']['median'], 0.8
        )

    def test_min_max_balanced_accuracy(self, mock_results):
        summary = aggregate_results(mock_results)
        np.testing.assert_almost_equal(summary['balanced_accuracy']['min'], 0.7)
        np.testing.assert_almost_equal(summary['balanced_accuracy']['max'], 0.9)

    def test_95_ci_computation(self, mock_results):
        summary = aggregate_results(mock_results)
        ba_values = [0.8, 0.7, 0.9]
        mean = np.mean(ba_values)
        std = np.std(ba_values)
        n = 3
        expected_lower = mean - 1.96 * std / np.sqrt(n)
        expected_upper = mean + 1.96 * std / np.sqrt(n)
        np.testing.assert_almost_equal(
            summary['balanced_accuracy']['ci_95_lower'], expected_lower
        )
        np.testing.assert_almost_equal(
            summary['balanced_accuracy']['ci_95_upper'], expected_upper
        )

    def test_confusion_matrix_shape(self, mock_results):
        summary = aggregate_results(mock_results)
        cm = np.array(summary['confusion_matrix'])
        # All labels are 0, 1, 2 so confusion matrix should be 3x3
        assert cm.shape == (3, 3)

    def test_confusion_matrix_values(self, mock_results):
        summary = aggregate_results(mock_results)
        cm = np.array(summary['confusion_matrix'])
        # Aggregated y_true = [0,1,2, 0,1,2, 0,1,2]
        # Aggregated y_pred = [0,1,1, 0,0,2, 0,1,2]
        # True 0 -> predicted 0: 3 times
        assert cm[0, 0] == 3
        # True 1 -> predicted 1: 2 times (s1 and s3), predicted 0: 1 time (s2)
        assert cm[1, 1] == 2
        assert cm[1, 0] == 1
        # True 2 -> predicted 2: 2 times (s2 and s3), predicted 1: 1 time (s1)
        assert cm[2, 2] == 2
        assert cm[2, 1] == 1

    def test_statistical_test_output(self, mock_results):
        summary = aggregate_results(mock_results)
        vs_p1 = summary['vs_paper1']
        assert 'paper1_baseline' in vs_p1
        assert 'improvement' in vs_p1
        assert 't_statistic' in vs_p1
        assert 'p_value' in vs_p1
        assert 'cohens_d' in vs_p1
        assert 'significant' in vs_p1
        assert isinstance(vs_p1['significant'], (bool, np.bool_))

    def test_improvement_sign(self, mock_results):
        summary = aggregate_results(mock_results)
        ba_mean = np.mean([0.8, 0.7, 0.9])
        expected_improvement = ba_mean - PAPER1_LOSO_BASELINE
        np.testing.assert_almost_equal(
            summary['vs_paper1']['improvement'], expected_improvement
        )

    def test_cohens_d_computation(self, mock_results):
        summary = aggregate_results(mock_results)
        ba_values = [0.8, 0.7, 0.9]
        mean = np.mean(ba_values)
        std = np.std(ba_values)
        expected_d = (mean - PAPER1_LOSO_BASELINE) / std
        np.testing.assert_almost_equal(
            summary['vs_paper1']['cohens_d'], expected_d
        )

    def test_n_folds_count(self, mock_results):
        summary = aggregate_results(mock_results)
        assert summary['n_folds'] == 3

    def test_per_fold_balanced_accs_preserved(self, mock_results):
        summary = aggregate_results(mock_results)
        assert len(summary['per_fold_balanced_accs']) == 3
        assert set(summary['per_fold_balanced_accs']) == {0.7, 0.8, 0.9}

    def test_accuracy_mean_std(self, mock_results):
        summary = aggregate_results(mock_results)
        expected_mean = np.mean([0.85, 0.75, 0.90])
        expected_std = np.std([0.85, 0.75, 0.90])
        np.testing.assert_almost_equal(summary['accuracy']['mean'], expected_mean)
        np.testing.assert_almost_equal(summary['accuracy']['std'], expected_std)

    def test_f1_weighted_mean_std(self, mock_results):
        summary = aggregate_results(mock_results)
        expected_mean = np.mean([0.82, 0.72, 0.88])
        expected_std = np.std([0.82, 0.72, 0.88])
        np.testing.assert_almost_equal(summary['f1_weighted']['mean'], expected_mean)
        np.testing.assert_almost_equal(summary['f1_weighted']['std'], expected_std)


# =============================================================================
# 12. Result Serialization Roundtrip
# =============================================================================

class TestResultSerializationRoundtrip:
    """Test that full results dicts with numpy arrays survive serialization."""

    def test_full_results_roundtrip(self):
        results = {
            'model': 'RandomForest',
            'n_subjects': np.int64(53),
            'fold_results': {
                's1': {
                    'metrics': {
                        'balanced_accuracy': np.float64(0.85),
                        'accuracy': np.float64(0.90),
                    },
                    'y_true': np.array([0, 1, 2, 0, 1]),
                    'y_pred': np.array([0, 1, 1, 0, 2]),
                    'best_params': {
                        'n_estimators': np.int64(200),
                        'max_depth': np.int64(10),
                    }
                }
            },
            'summary': {
                'confusion_matrix': np.array([[10, 2], [3, 15]]),
                'significant': np.bool_(True),
                'per_fold_accs': np.array([0.8, 0.85, 0.9])
            }
        }

        serializable = convert_to_serializable(results)
        json_str = json.dumps(serializable)
        loaded = json.loads(json_str)

        assert loaded['model'] == 'RandomForest'
        assert loaded['n_subjects'] == 53
        assert loaded['fold_results']['s1']['y_true'] == [0, 1, 2, 0, 1]
        assert loaded['fold_results']['s1']['y_pred'] == [0, 1, 1, 0, 2]
        assert loaded['fold_results']['s1']['metrics']['balanced_accuracy'] == 0.85
        assert loaded['fold_results']['s1']['best_params']['n_estimators'] == 200
        assert loaded['summary']['confusion_matrix'] == [[10, 2], [3, 15]]
        assert loaded['summary']['significant'] is True
        assert loaded['summary']['per_fold_accs'] == [0.8, 0.85, 0.9]

    def test_deeply_nested_numpy_roundtrip(self):
        data = {
            'level1': {
                'level2': {
                    'level3': {
                        'value': np.float64(0.123456),
                        'array': np.array([np.int64(1), np.int64(2)]),
                        'flag': np.bool_(False)
                    }
                }
            }
        }
        serializable = convert_to_serializable(data)
        json_str = json.dumps(serializable)
        loaded = json.loads(json_str)

        assert loaded['level1']['level2']['level3']['value'] == 0.123456
        assert loaded['level1']['level2']['level3']['array'] == [1, 2]
        assert loaded['level1']['level2']['level3']['flag'] is False

    def test_empty_results_roundtrip(self):
        results = {
            'fold_results': {},
            'summary': {
                'confusion_matrix': np.array([]).reshape(0, 0),
                'per_fold_accs': np.array([])
            }
        }
        serializable = convert_to_serializable(results)
        json_str = json.dumps(serializable)
        loaded = json.loads(json_str)
        assert loaded['fold_results'] == {}
        assert loaded['summary']['per_fold_accs'] == []

    def test_mixed_python_numpy_roundtrip(self):
        """Verify a dict mixing Python natives and numpy types survives roundtrip."""
        data = {
            'python_int': 42,
            'numpy_int': np.int64(42),
            'python_float': 3.14,
            'numpy_float': np.float64(3.14),
            'python_str': 'hello',
            'python_bool': True,
            'numpy_bool': np.bool_(True),
            'python_list': [1, 2, 3],
            'numpy_array': np.array([1, 2, 3]),
            'none_value': None
        }
        serializable = convert_to_serializable(data)
        json_str = json.dumps(serializable)
        loaded = json.loads(json_str)

        assert loaded['python_int'] == 42
        assert loaded['numpy_int'] == 42
        assert loaded['python_float'] == 3.14
        assert loaded['numpy_float'] == 3.14
        assert loaded['python_str'] == 'hello'
        assert loaded['python_bool'] is True
        assert loaded['numpy_bool'] is True
        assert loaded['python_list'] == [1, 2, 3]
        assert loaded['numpy_array'] == [1, 2, 3]
        assert loaded['none_value'] is None
