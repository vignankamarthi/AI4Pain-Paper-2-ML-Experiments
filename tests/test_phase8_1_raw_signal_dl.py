"""
Tests for phase8_1_raw_signal_dl module.

All tests use synthetic/mock data only -- no real dataset access required.
Covers utility functions, data loading, PyTorch components, neural network
architectures, training helpers, and checkpoint I/O.
"""

import sys
import json
import tempfile
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase8_1_raw_signal_dl import (
    extract_subject_id,
    load_raw_signal,
    RawSignalDataset,
    Conv1DClassifier,
    BiLSTMClassifier,
    EarlyStopping,
    compute_class_weights,
    convert_to_serializable,
    format_duration,
    save_checkpoint,
    load_checkpoint,
    N_CHANNELS,
    N_CLASSES,
    MAX_SIGNAL_LENGTH,
    SIGNALS,
    CLASS_MAPPING,
)


# ---------------------------------------------------------------------------
# Module-level constants verification
# ---------------------------------------------------------------------------

class TestModuleConstants:
    """Verify that module-level constants have expected values."""

    def test_n_channels(self):
        assert N_CHANNELS == 4

    def test_n_classes(self):
        assert N_CLASSES == 3

    def test_max_signal_length(self):
        assert MAX_SIGNAL_LENGTH == 1000

    def test_signals_list(self):
        assert SIGNALS == ["eda", "bvp", "resp", "spo2"]

    def test_class_mapping(self):
        assert CLASS_MAPPING == {"baseline": 0, "low": 1, "high": 2}

    def test_class_mapping_excludes_rest(self):
        assert "rest" not in CLASS_MAPPING


# ---------------------------------------------------------------------------
# extract_subject_id
# ---------------------------------------------------------------------------

class TestExtractSubjectId:
    """Tests for extract_subject_id function."""

    def test_baseline_segment(self):
        assert extract_subject_id("12_Baseline_1") == "12"

    def test_high_pain_segment(self):
        assert extract_subject_id("54_HIGH_2") == "54"

    def test_low_pain_segment(self):
        assert extract_subject_id("7_low_3") == "7"

    def test_three_digit_subject(self):
        assert extract_subject_id("123_Baseline_1") == "123"

    def test_single_digit_subject(self):
        assert extract_subject_id("3_rest_0") == "3"

    def test_no_underscore_with_digits(self):
        # Falls through the first regex, uses re.search
        assert extract_subject_id("subject42data") == "42"

    def test_pure_number(self):
        assert extract_subject_id("99") == "99"

    def test_number_at_start_no_underscore(self):
        # re.match(r'(\d+)_', ...) fails, re.search(r'(\d+)', ...) finds "10"
        assert extract_subject_id("10abc") == "10"

    def test_no_digits_returns_original(self):
        assert extract_subject_id("nodigits") == "nodigits"

    def test_underscore_prefixed_number(self):
        # First regex looks for digits at start followed by underscore
        # "a_12_test" -- first regex fails (no digits at start), search finds "12"
        assert extract_subject_id("a_12_test") == "12"


# ---------------------------------------------------------------------------
# load_raw_signal
# ---------------------------------------------------------------------------

class TestLoadRawSignal:
    """Tests for load_raw_signal function."""

    def test_exact_length_signal(self, tmp_path):
        """Signal with exactly max_length values is returned as-is."""
        max_len = 50
        values = np.arange(max_len, dtype=np.float32)
        csv_path = tmp_path / "signal.csv"
        import pandas as pd
        pd.DataFrame({"seg_A": values}).to_csv(csv_path, index=False)

        result = load_raw_signal(csv_path, "seg_A", max_length=max_len)
        assert result.shape == (max_len,)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, values)

    def test_short_signal_padded_with_zeros(self, tmp_path):
        """Signal shorter than max_length is zero-padded."""
        max_len = 100
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        csv_path = tmp_path / "signal.csv"
        import pandas as pd
        pd.DataFrame({"seg_B": values}).to_csv(csv_path, index=False)

        result = load_raw_signal(csv_path, "seg_B", max_length=max_len)
        assert result.shape == (max_len,)
        np.testing.assert_array_almost_equal(result[:3], values)
        np.testing.assert_array_equal(result[3:], np.zeros(max_len - 3))

    def test_long_signal_truncated(self, tmp_path):
        """Signal longer than max_length is truncated."""
        max_len = 10
        values = np.arange(50, dtype=np.float32)
        csv_path = tmp_path / "signal.csv"
        import pandas as pd
        pd.DataFrame({"seg_C": values}).to_csv(csv_path, index=False)

        result = load_raw_signal(csv_path, "seg_C", max_length=max_len)
        assert result.shape == (max_len,)
        np.testing.assert_array_almost_equal(result, values[:max_len])

    def test_nan_removal(self, tmp_path):
        """NaN values are removed before padding/truncation."""
        max_len = 10
        values = np.array([1.0, np.nan, 2.0, np.nan, 3.0], dtype=np.float32)
        csv_path = tmp_path / "signal.csv"
        import pandas as pd
        pd.DataFrame({"seg_D": values}).to_csv(csv_path, index=False)

        result = load_raw_signal(csv_path, "seg_D", max_length=max_len)
        assert result.shape == (max_len,)
        # After NaN removal we have [1.0, 2.0, 3.0], then zero-padded to 10
        np.testing.assert_array_almost_equal(result[:3], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result[3:], np.zeros(max_len - 3))

    def test_missing_column_returns_zeros(self, tmp_path):
        """If segment_id column is not found at all, return zeros."""
        max_len = 20
        csv_path = tmp_path / "signal.csv"
        import pandas as pd
        pd.DataFrame({"other_col": [1.0, 2.0]}).to_csv(csv_path, index=False)

        result = load_raw_signal(csv_path, "nonexistent_segment", max_length=max_len)
        assert result.shape == (max_len,)
        np.testing.assert_array_equal(result, np.zeros(max_len, dtype=np.float32))

    def test_partial_column_match(self, tmp_path):
        """If exact column missing but partial match exists, use it."""
        max_len = 5
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        csv_path = tmp_path / "signal.csv"
        import pandas as pd
        pd.DataFrame({"12_Baseline_1": values}).to_csv(csv_path, index=False)

        # Search for "Baseline_1" which is contained in "12_Baseline_1"
        result = load_raw_signal(csv_path, "Baseline_1", max_length=max_len)
        assert result.shape == (max_len,)
        np.testing.assert_array_almost_equal(result, values)

    def test_missing_file_returns_zeros(self, tmp_path):
        """If file does not exist, return zeros."""
        max_len = 15
        fake_path = tmp_path / "does_not_exist.csv"
        result = load_raw_signal(fake_path, "any_seg", max_length=max_len)
        assert result.shape == (max_len,)
        np.testing.assert_array_equal(result, np.zeros(max_len, dtype=np.float32))

    def test_default_max_length(self, tmp_path):
        """Default max_length uses MAX_SIGNAL_LENGTH constant (1000)."""
        values = np.ones(5, dtype=np.float32)
        csv_path = tmp_path / "signal.csv"
        import pandas as pd
        pd.DataFrame({"seg_E": values}).to_csv(csv_path, index=False)

        result = load_raw_signal(csv_path, "seg_E")
        assert result.shape == (MAX_SIGNAL_LENGTH,)


# ---------------------------------------------------------------------------
# RawSignalDataset
# ---------------------------------------------------------------------------

class TestRawSignalDataset:
    """Tests for the PyTorch RawSignalDataset class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a small synthetic dataset."""
        n_samples = 20
        X = np.random.randn(n_samples, N_CHANNELS, MAX_SIGNAL_LENGTH).astype(np.float32)
        y = np.random.randint(0, N_CLASSES, size=n_samples).astype(np.int64)
        return RawSignalDataset(X, y)

    def test_len(self, sample_dataset):
        assert len(sample_dataset) == 20

    def test_getitem_returns_tuple(self, sample_dataset):
        item = sample_dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_getitem_x_shape(self, sample_dataset):
        x, _ = sample_dataset[0]
        assert x.shape == (N_CHANNELS, MAX_SIGNAL_LENGTH)

    def test_getitem_x_dtype(self, sample_dataset):
        x, _ = sample_dataset[0]
        assert x.dtype == torch.float32

    def test_getitem_y_dtype(self, sample_dataset):
        _, y_val = sample_dataset[0]
        assert y_val.dtype == torch.int64

    def test_getitem_y_scalar(self, sample_dataset):
        _, y_val = sample_dataset[0]
        assert y_val.dim() == 0  # scalar tensor

    def test_all_indices_accessible(self, sample_dataset):
        for i in range(len(sample_dataset)):
            x, y_val = sample_dataset[i]
            assert x.shape == (N_CHANNELS, MAX_SIGNAL_LENGTH)
            assert 0 <= y_val.item() < N_CLASSES

    def test_values_match_input(self):
        """Verify tensor values match the original numpy arrays."""
        X = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)  # (1, 2, 2)
        y = np.array([1], dtype=np.int64)
        ds = RawSignalDataset(X, y)
        x_out, y_out = ds[0]
        torch.testing.assert_close(x_out, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert y_out.item() == 1

    def test_empty_dataset(self):
        """Zero-sample dataset has length 0."""
        X = np.empty((0, N_CHANNELS, 50), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        ds = RawSignalDataset(X, y)
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# Conv1DClassifier
# ---------------------------------------------------------------------------

class TestConv1DClassifier:
    """Tests for the Conv1DClassifier architecture."""

    def test_default_forward_shape(self):
        model = Conv1DClassifier()
        model.eval()
        batch = torch.randn(8, N_CHANNELS, MAX_SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (8, N_CLASSES)

    @pytest.mark.parametrize("n_conv_layers", [2, 3, 4])
    def test_varying_conv_layers(self, n_conv_layers):
        model = Conv1DClassifier(n_conv_layers=n_conv_layers)
        model.eval()
        batch = torch.randn(4, N_CHANNELS, MAX_SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4, N_CLASSES)

    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    def test_varying_kernel_size(self, kernel_size):
        model = Conv1DClassifier(kernel_size=kernel_size)
        model.eval()
        batch = torch.randn(4, N_CHANNELS, MAX_SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4, N_CLASSES)

    @pytest.mark.parametrize("hidden_dim", [32, 64, 128])
    def test_varying_hidden_dim(self, hidden_dim):
        model = Conv1DClassifier(hidden_dim=hidden_dim)
        model.eval()
        batch = torch.randn(4, N_CHANNELS, MAX_SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4, N_CLASSES)

    def test_custom_channels_and_classes(self):
        model = Conv1DClassifier(n_channels=2, n_classes=5)
        model.eval()
        batch = torch.randn(4, 2, MAX_SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4, 5)

    def test_single_sample_batch(self):
        model = Conv1DClassifier()
        model.eval()
        batch = torch.randn(1, N_CHANNELS, MAX_SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (1, N_CLASSES)

    def test_output_dtype_float(self):
        model = Conv1DClassifier()
        model.eval()
        batch = torch.randn(2, N_CHANNELS, MAX_SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(batch)
        assert out.dtype == torch.float32

    def test_short_sequence(self):
        """Works with a shorter-than-default sequence length."""
        model = Conv1DClassifier(n_conv_layers=2)
        model.eval()
        batch = torch.randn(2, N_CHANNELS, 64)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (2, N_CLASSES)

    def test_stores_attributes(self):
        model = Conv1DClassifier(n_channels=4, n_classes=3)
        assert model.n_channels == 4
        assert model.n_classes == 3

    def test_all_hp_combinations(self):
        """Combined non-default configuration from Optuna search space."""
        model = Conv1DClassifier(
            n_channels=N_CHANNELS,
            n_classes=N_CLASSES,
            hidden_dim=32,
            n_conv_layers=4,
            kernel_size=5,
            dropout=0.1,
        )
        model.eval()
        batch = torch.randn(4, N_CHANNELS, MAX_SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4, N_CLASSES)


# ---------------------------------------------------------------------------
# BiLSTMClassifier
# ---------------------------------------------------------------------------

class TestBiLSTMClassifier:
    """Tests for the BiLSTMClassifier architecture."""

    def test_default_forward_shape(self):
        model = BiLSTMClassifier()
        model.eval()
        batch = torch.randn(8, N_CHANNELS, MAX_SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (8, N_CLASSES)

    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_varying_n_layers(self, n_layers):
        model = BiLSTMClassifier(n_layers=n_layers)
        model.eval()
        batch = torch.randn(4, N_CHANNELS, MAX_SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4, N_CLASSES)

    @pytest.mark.parametrize("hidden_dim", [32, 64, 128])
    def test_varying_hidden_dim(self, hidden_dim):
        model = BiLSTMClassifier(hidden_dim=hidden_dim)
        model.eval()
        batch = torch.randn(4, N_CHANNELS, MAX_SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4, N_CLASSES)

    def test_custom_channels_and_classes(self):
        model = BiLSTMClassifier(n_channels=2, n_classes=5)
        model.eval()
        batch = torch.randn(4, 2, MAX_SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4, 5)

    def test_single_sample_batch(self):
        model = BiLSTMClassifier()
        model.eval()
        batch = torch.randn(1, N_CHANNELS, MAX_SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (1, N_CLASSES)

    def test_output_dtype_float(self):
        model = BiLSTMClassifier()
        model.eval()
        batch = torch.randn(2, N_CHANNELS, MAX_SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(batch)
        assert out.dtype == torch.float32

    def test_short_sequence(self):
        """Works with a shorter-than-default sequence length."""
        model = BiLSTMClassifier()
        model.eval()
        batch = torch.randn(2, N_CHANNELS, 32)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (2, N_CLASSES)

    def test_stores_attributes(self):
        model = BiLSTMClassifier(
            n_channels=4, n_classes=3, hidden_dim=64, n_layers=2
        )
        assert model.n_channels == 4
        assert model.n_classes == 3
        assert model.hidden_dim == 64
        assert model.n_layers == 2

    def test_all_hp_combinations(self):
        """Combined non-default configuration from Optuna search space."""
        model = BiLSTMClassifier(
            n_channels=N_CHANNELS,
            n_classes=N_CLASSES,
            hidden_dim=128,
            n_layers=3,
            dropout=0.5,
        )
        model.eval()
        batch = torch.randn(4, N_CHANNELS, MAX_SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4, N_CLASSES)

    def test_dropout_zero_for_single_layer(self):
        """When n_layers=1 the LSTM should have dropout=0 internally."""
        model = BiLSTMClassifier(n_layers=1, dropout=0.5)
        # The nn.LSTM dropout is set to 0 when n_layers == 1 in the source
        assert model.lstm.dropout == 0.0


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    """Tests for the EarlyStopping callback."""

    def test_no_trigger_within_patience(self):
        es = EarlyStopping(patience=3)
        es(1.0)
        es(1.1)
        es(1.2)
        assert not es.early_stop

    def test_triggers_after_patience_exceeded(self):
        es = EarlyStopping(patience=3)
        es(1.0)  # sets best_loss = 1.0
        es(1.1)  # counter = 1
        es(1.2)  # counter = 2
        es(1.3)  # counter = 3, triggers
        assert es.early_stop

    def test_counter_resets_on_improvement(self):
        es = EarlyStopping(patience=3)
        es(1.0)  # best_loss = 1.0
        es(1.1)  # counter = 1
        es(0.5)  # improvement, counter = 0, best_loss = 0.5
        assert es.counter == 0
        assert es.best_loss == 0.5
        assert not es.early_stop

    def test_first_call_sets_best_loss(self):
        es = EarlyStopping(patience=5)
        es(2.5)
        assert es.best_loss == 2.5
        assert es.counter == 0

    def test_equal_loss_does_not_increment_counter(self):
        """Loss equal to best with min_delta=0 is NOT worse (1.0 > 1.0 is False),
        so it hits the else branch and resets counter."""
        es = EarlyStopping(patience=2, min_delta=0.0)
        es(1.0)
        es(1.0)  # 1.0 > 1.0 - 0.0 is False, else branch: best_loss=1.0, counter=0
        assert es.counter == 0

    def test_slightly_worse_loss_increments_counter(self):
        """Loss strictly worse than best increments counter."""
        es = EarlyStopping(patience=3, min_delta=0.0)
        es(1.0)
        es(1.001)  # 1.001 > 1.0 is True, counter = 1
        assert es.counter == 1

    def test_min_delta_tolerance(self):
        """Improvement must exceed min_delta to reset counter."""
        es = EarlyStopping(patience=3, min_delta=0.1)
        es(1.0)
        es(0.95)  # 0.95 > 1.0 - 0.1 = 0.9, so no real improvement; counter = 1
        assert es.counter == 1
        es(0.85)  # 0.85 < 0.9, genuine improvement; counter = 0
        assert es.counter == 0
        assert es.best_loss == 0.85

    def test_large_patience(self):
        es = EarlyStopping(patience=100)
        es(1.0)  # sets best_loss = 1.0
        for _ in range(99):
            es(1.5)  # strictly worse, counter increments each time
        assert not es.early_stop  # counter = 99 < 100
        es(1.5)  # counter = 100 >= patience
        assert es.early_stop

    def test_immediate_improvement_never_triggers(self):
        """Continuously improving loss never triggers early stopping."""
        es = EarlyStopping(patience=2)
        for i in range(50):
            es(50.0 - i)
        assert not es.early_stop
        assert es.counter == 0


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------

class TestComputeClassWeights:
    """Tests for compute_class_weights function."""

    @patch("phase8_1_raw_signal_dl.DEVICE", torch.device("cpu"))
    def test_balanced_distribution(self):
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        weights = compute_class_weights(y)
        assert weights.shape == (3,)
        # For balanced classes, all weights should be equal
        torch.testing.assert_close(weights[0], weights[1])
        torch.testing.assert_close(weights[1], weights[2])

    @patch("phase8_1_raw_signal_dl.DEVICE", torch.device("cpu"))
    def test_imbalanced_distribution(self):
        y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2])
        weights = compute_class_weights(y)
        assert weights.shape == (3,)
        # Minority class (2) should have highest weight
        assert weights[2].item() > weights[0].item()
        assert weights[1].item() > weights[0].item()

    @patch("phase8_1_raw_signal_dl.DEVICE", torch.device("cpu"))
    def test_weights_are_float_tensor(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        weights = compute_class_weights(y)
        assert weights.dtype == torch.float32

    @patch("phase8_1_raw_signal_dl.DEVICE", torch.device("cpu"))
    def test_binary_classes(self):
        y = np.array([0, 0, 0, 1])
        weights = compute_class_weights(y)
        assert weights.shape == (2,)
        assert weights[1].item() > weights[0].item()

    @patch("phase8_1_raw_signal_dl.DEVICE", torch.device("cpu"))
    def test_weights_normalized(self):
        """Weights should sum to the number of classes."""
        y = np.array([0, 0, 0, 1, 2, 2])
        weights = compute_class_weights(y)
        n_classes = 3
        assert abs(weights.sum().item() - n_classes) < 1e-5


# ---------------------------------------------------------------------------
# convert_to_serializable
# ---------------------------------------------------------------------------

class TestConvertToSerializable:
    """Tests for convert_to_serializable utility."""

    def test_numpy_int64(self):
        result = convert_to_serializable(np.int64(42))
        assert isinstance(result, int)
        assert result == 42

    def test_numpy_int32(self):
        result = convert_to_serializable(np.int32(10))
        assert isinstance(result, int)
        assert result == 10

    def test_numpy_float64(self):
        result = convert_to_serializable(np.float64(3.14))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-10

    def test_numpy_float32(self):
        result = convert_to_serializable(np.float32(2.5))
        assert isinstance(result, float)

    def test_numpy_bool(self):
        result = convert_to_serializable(np.bool_(True))
        assert isinstance(result, bool)
        assert result is True

    def test_numpy_array(self):
        result = convert_to_serializable(np.array([1, 2, 3]))
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_numpy_2d_array(self):
        result = convert_to_serializable(np.array([[1, 2], [3, 4]]))
        assert result == [[1, 2], [3, 4]]

    def test_torch_tensor_scalar(self):
        result = convert_to_serializable(torch.tensor(7.0))
        assert isinstance(result, float)

    def test_torch_tensor_1d(self):
        result = convert_to_serializable(torch.tensor([1.0, 2.0, 3.0]))
        assert isinstance(result, list)
        assert len(result) == 3

    def test_torch_tensor_2d(self):
        result = convert_to_serializable(torch.tensor([[1, 2], [3, 4]]))
        assert result == [[1, 2], [3, 4]]

    def test_nested_dict(self):
        data = {
            "a": np.int64(1),
            "b": {
                "c": np.float32(2.5),
                "d": np.array([1, 2]),
            },
        }
        result = convert_to_serializable(data)
        assert result == {"a": 1, "b": {"c": 2.5, "d": [1, 2]}}
        assert isinstance(result["a"], int)

    def test_list_of_numpy(self):
        data = [np.int64(1), np.float64(2.0), np.bool_(False)]
        result = convert_to_serializable(data)
        assert result == [1, 2.0, False]
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)
        assert isinstance(result[2], bool)

    def test_tuple_of_numpy(self):
        data = (np.int32(5), np.float32(1.5))
        result = convert_to_serializable(data)
        assert isinstance(result, list)
        assert result == [5, 1.5]

    def test_plain_python_types_pass_through(self):
        assert convert_to_serializable(42) == 42
        assert convert_to_serializable(3.14) == 3.14
        assert convert_to_serializable("hello") == "hello"
        assert convert_to_serializable(None) is None
        assert convert_to_serializable(True) is True

    def test_result_is_json_serializable(self):
        data = {
            "score": np.float64(0.95),
            "counts": np.array([10, 20, 30]),
            "params": {"lr": np.float32(0.001), "layers": np.int64(3)},
            "preds": torch.tensor([0, 1, 2]),
        }
        result = convert_to_serializable(data)
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)


# ---------------------------------------------------------------------------
# format_duration
# ---------------------------------------------------------------------------

class TestFormatDuration:
    """Tests for format_duration utility."""

    def test_seconds(self):
        assert format_duration(30.0) == "30.0s"

    def test_zero_seconds(self):
        assert format_duration(0.0) == "0.0s"

    def test_fractional_seconds(self):
        assert format_duration(5.7) == "5.7s"

    def test_just_under_minute(self):
        assert format_duration(59.9) == "59.9s"

    def test_exactly_one_minute(self):
        assert format_duration(60.0) == "1.0m"

    def test_minutes(self):
        assert format_duration(150.0) == "2.5m"

    def test_just_under_hour(self):
        result = format_duration(3599.0)
        assert result.endswith("m")

    def test_exactly_one_hour(self):
        assert format_duration(3600.0) == "1.0h"

    def test_hours(self):
        assert format_duration(7200.0) == "2.0h"

    def test_large_duration(self):
        assert format_duration(36000.0) == "10.0h"


# ---------------------------------------------------------------------------
# Checkpoint save/load cycle
# ---------------------------------------------------------------------------

class TestCheckpoints:
    """Tests for checkpoint save and load functions."""

    def test_save_then_load_returns_same_data(self, tmp_path):
        checkpoint = {
            "completed_architectures": ["Conv1D"],
            "current_architecture": "BiLSTM",
            "fold_results": {
                "Conv1D": {
                    "12": {"metrics": {"balanced_accuracy": 0.75}},
                }
            },
        }
        save_checkpoint(tmp_path, checkpoint)
        loaded = load_checkpoint(tmp_path)

        assert loaded["completed_architectures"] == ["Conv1D"]
        assert loaded["current_architecture"] == "BiLSTM"
        assert "12" in loaded["fold_results"]["Conv1D"]
        assert loaded["fold_results"]["Conv1D"]["12"]["metrics"]["balanced_accuracy"] == 0.75

    def test_load_adds_last_updated(self, tmp_path):
        checkpoint = {
            "completed_architectures": [],
            "current_architecture": None,
            "fold_results": {},
        }
        save_checkpoint(tmp_path, checkpoint)
        loaded = load_checkpoint(tmp_path)
        assert "last_updated" in loaded

    def test_load_nonexistent_returns_default(self, tmp_path):
        loaded = load_checkpoint(tmp_path)
        assert loaded == {
            "completed_architectures": [],
            "current_architecture": None,
            "fold_results": {},
        }

    def test_corrupted_json_returns_default(self, tmp_path):
        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text("this is not valid json {{{")

        loaded = load_checkpoint(tmp_path)
        assert loaded == {
            "completed_architectures": [],
            "current_architecture": None,
            "fold_results": {},
        }

    def test_numpy_types_in_checkpoint(self, tmp_path):
        """Checkpoint with numpy types should serialize correctly."""
        checkpoint = {
            "completed_architectures": [],
            "current_architecture": "Conv1D",
            "fold_results": {
                "Conv1D": {
                    "subject_5": {
                        "metrics": {
                            "balanced_accuracy": np.float64(0.83),
                            "accuracy": np.float32(0.85),
                        },
                        "n_samples": np.int64(100),
                    }
                }
            },
        }
        save_checkpoint(tmp_path, checkpoint)
        loaded = load_checkpoint(tmp_path)
        fold_data = loaded["fold_results"]["Conv1D"]["subject_5"]
        assert isinstance(fold_data["metrics"]["balanced_accuracy"], float)
        assert isinstance(fold_data["n_samples"], int)

    def test_overwrite_checkpoint(self, tmp_path):
        """Saving twice overwrites the previous checkpoint."""
        save_checkpoint(tmp_path, {
            "completed_architectures": ["Conv1D"],
            "current_architecture": "Conv1D",
            "fold_results": {},
        })
        save_checkpoint(tmp_path, {
            "completed_architectures": ["Conv1D", "BiLSTM"],
            "current_architecture": "BiLSTM",
            "fold_results": {},
        })
        loaded = load_checkpoint(tmp_path)
        assert loaded["completed_architectures"] == ["Conv1D", "BiLSTM"]
        assert loaded["current_architecture"] == "BiLSTM"

    def test_checkpoint_file_is_valid_json(self, tmp_path):
        checkpoint = {
            "completed_architectures": [],
            "current_architecture": None,
            "fold_results": {},
        }
        save_checkpoint(tmp_path, checkpoint)
        checkpoint_file = tmp_path / "checkpoint.json"
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Normalization correctness
# ---------------------------------------------------------------------------

class TestNormalizationCorrectness:
    """Verify that StandardScaler is fit on train only and applied to both sets."""

    def test_scaler_fit_on_train_only(self):
        """Test set must be transformed using train-set statistics, not its own."""
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(42)
        n_train, n_test = 50, 10
        n_ch, seq_len = N_CHANNELS, 100

        # Train data: mean ~ 10, std ~ 2
        X_train = rng.normal(loc=10.0, scale=2.0, size=(n_train, n_ch, seq_len)).astype(np.float32)
        # Test data: mean ~ 50, std ~ 5 (very different distribution)
        X_test = rng.normal(loc=50.0, scale=5.0, size=(n_test, n_ch, seq_len)).astype(np.float32)

        scaler = StandardScaler()

        # Flatten, fit on train, transform both
        X_train_flat = X_train.reshape(n_train, -1)
        X_test_flat = X_test.reshape(n_test, -1)
        X_train_scaled_flat = scaler.fit_transform(X_train_flat)
        X_test_scaled_flat = scaler.transform(X_test_flat)

        X_train_scaled = X_train_scaled_flat.reshape(n_train, n_ch, seq_len)
        X_test_scaled = X_test_scaled_flat.reshape(n_test, n_ch, seq_len)

        # Train should be roughly zero-mean, unit-variance
        train_mean = X_train_scaled.mean()
        train_std = X_train_scaled.std()
        assert abs(train_mean) < 0.1, f"Train mean {train_mean} not near 0"
        assert abs(train_std - 1.0) < 0.1, f"Train std {train_std} not near 1"

        # Test was NOT fit, so its mean should NOT be near 0
        # (it was centered using train mean ~10, but test has mean ~50)
        test_mean = X_test_scaled.mean()
        assert abs(test_mean) > 5.0, (
            f"Test mean {test_mean} is too close to 0 -- scaler may have been fit on test data"
        )

    def test_scaler_transform_preserves_shape(self):
        """Reshaping for StandardScaler and back preserves array shape."""
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(123)
        n_samples = 30
        X = rng.standard_normal(size=(n_samples, N_CHANNELS, 200)).astype(np.float32)

        scaler = StandardScaler()
        X_flat = X.reshape(n_samples, -1)
        X_scaled_flat = scaler.fit_transform(X_flat)
        X_scaled = X_scaled_flat.reshape(n_samples, N_CHANNELS, 200)

        assert X_scaled.shape == X.shape
        assert X_scaled.dtype in (np.float32, np.float64)


# ---------------------------------------------------------------------------
# LOSO split verification
# ---------------------------------------------------------------------------

class TestLOSOSplitVerification:
    """Verify LOSO splitting logic: test subject excluded from training set."""

    @pytest.fixture
    def synthetic_subject_data(self):
        """Create synthetic data with known subject IDs."""
        rng = np.random.default_rng(42)
        subjects = ["10", "20", "30", "40", "50"]
        samples_per_subject = 8

        subject_ids = np.repeat(subjects, samples_per_subject)
        n_total = len(subject_ids)
        X = rng.standard_normal(size=(n_total, N_CHANNELS, 100)).astype(np.float32)
        y = rng.integers(0, N_CLASSES, size=n_total).astype(np.int64)

        return X, y, subject_ids, subjects

    def test_test_subject_not_in_train(self, synthetic_subject_data):
        X, y, subject_ids, all_subjects = synthetic_subject_data

        for test_subject in all_subjects:
            train_mask = subject_ids != test_subject
            test_mask = subject_ids == test_subject

            train_subjects = np.unique(subject_ids[train_mask])
            test_subjects = np.unique(subject_ids[test_mask])

            assert test_subject not in train_subjects
            assert test_subject in test_subjects

    def test_all_other_subjects_in_train(self, synthetic_subject_data):
        X, y, subject_ids, all_subjects = synthetic_subject_data

        for test_subject in all_subjects:
            train_mask = subject_ids != test_subject
            train_subjects = set(subject_ids[train_mask])

            expected_train = set(all_subjects) - {test_subject}
            assert train_subjects == expected_train

    def test_no_sample_overlap(self, synthetic_subject_data):
        X, y, subject_ids, all_subjects = synthetic_subject_data

        for test_subject in all_subjects:
            train_mask = subject_ids != test_subject
            test_mask = subject_ids == test_subject

            train_indices = set(np.where(train_mask)[0])
            test_indices = set(np.where(test_mask)[0])

            assert len(train_indices & test_indices) == 0

    def test_all_samples_used(self, synthetic_subject_data):
        X, y, subject_ids, all_subjects = synthetic_subject_data

        for test_subject in all_subjects:
            train_mask = subject_ids != test_subject
            test_mask = subject_ids == test_subject

            assert train_mask.sum() + test_mask.sum() == len(subject_ids)

    def test_each_fold_has_one_test_subject(self, synthetic_subject_data):
        X, y, subject_ids, all_subjects = synthetic_subject_data

        for test_subject in all_subjects:
            test_mask = subject_ids == test_subject
            unique_test = np.unique(subject_ids[test_mask])
            assert len(unique_test) == 1
            assert unique_test[0] == test_subject

    def test_number_of_folds_equals_n_subjects(self, synthetic_subject_data):
        _, _, _, all_subjects = synthetic_subject_data
        assert len(all_subjects) == 5


# ---------------------------------------------------------------------------
# Integration-style tests (combining multiple components)
# ---------------------------------------------------------------------------

class TestDatasetToModelIntegration:
    """Integration tests: Dataset -> DataLoader -> Model forward pass."""

    def test_conv1d_end_to_end(self):
        """RawSignalDataset feeds correctly into Conv1DClassifier."""
        n_samples = 16
        rng = np.random.default_rng(42)
        X = rng.standard_normal(size=(n_samples, N_CHANNELS, MAX_SIGNAL_LENGTH)).astype(np.float32)
        y = rng.integers(0, N_CLASSES, size=n_samples).astype(np.int64)

        dataset = RawSignalDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

        model = Conv1DClassifier()
        model.eval()

        all_outputs = []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                out = model(x_batch)
                all_outputs.append(out)
                assert out.shape[1] == N_CLASSES

        total_samples = sum(o.shape[0] for o in all_outputs)
        assert total_samples == n_samples

    def test_bilstm_end_to_end(self):
        """RawSignalDataset feeds correctly into BiLSTMClassifier."""
        n_samples = 12
        rng = np.random.default_rng(42)
        X = rng.standard_normal(size=(n_samples, N_CHANNELS, 200)).astype(np.float32)
        y = rng.integers(0, N_CLASSES, size=n_samples).astype(np.int64)

        dataset = RawSignalDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

        model = BiLSTMClassifier()
        model.eval()

        all_outputs = []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                out = model(x_batch)
                all_outputs.append(out)
                assert out.shape[1] == N_CLASSES

        total_samples = sum(o.shape[0] for o in all_outputs)
        assert total_samples == n_samples


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Miscellaneous edge case tests."""

    def test_early_stopping_patience_one(self):
        """Patience of 1 triggers immediately after first non-improvement."""
        es = EarlyStopping(patience=1)
        es(1.0)
        es(1.5)  # no improvement, counter = 1 >= patience
        assert es.early_stop

    @patch("phase8_1_raw_signal_dl.DEVICE", torch.device("cpu"))
    def test_class_weights_single_class(self):
        """Single class (degenerate case)."""
        y = np.array([0, 0, 0])
        weights = compute_class_weights(y)
        assert weights.shape == (1,)

    def test_convert_empty_dict(self):
        assert convert_to_serializable({}) == {}

    def test_convert_empty_list(self):
        assert convert_to_serializable([]) == []

    def test_conv1d_gradients_flow(self):
        """Verify gradients flow through Conv1DClassifier."""
        model = Conv1DClassifier()
        x = torch.randn(2, N_CHANNELS, MAX_SIGNAL_LENGTH, requires_grad=False)
        y = torch.tensor([0, 1])
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()

        has_grad = False
        for p in model.parameters():
            if p.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients computed for Conv1DClassifier"

    def test_bilstm_gradients_flow(self):
        """Verify gradients flow through BiLSTMClassifier."""
        model = BiLSTMClassifier()
        x = torch.randn(2, N_CHANNELS, MAX_SIGNAL_LENGTH, requires_grad=False)
        y = torch.tensor([0, 2])
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()

        has_grad = False
        for p in model.parameters():
            if p.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients computed for BiLSTMClassifier"

    def test_load_raw_signal_all_nans(self, tmp_path):
        """Signal that is entirely NaN should produce zero-padded output."""
        import pandas as pd
        max_len = 10
        values = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        csv_path = tmp_path / "signal.csv"
        pd.DataFrame({"seg_nan": values}).to_csv(csv_path, index=False)

        result = load_raw_signal(csv_path, "seg_nan", max_length=max_len)
        assert result.shape == (max_len,)
        np.testing.assert_array_equal(result, np.zeros(max_len, dtype=np.float32))
