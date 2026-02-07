"""
TDD tests for paper_figures module.

Tests verify figure generation functions produce correct outputs.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from paper_figures import (
    prepare_ch_plane_data,
    plot_confusion_matrix,
)


# ---------------------------------------------------------------------------
# C-H Plane Data Shape Tests
# ---------------------------------------------------------------------------

class TestCHPlaneDataShape:
    def test_correct_filtering(self):
        """Verify correct filtering produces expected array shapes."""
        rows = []
        # 10 baseline, 10 low, 10 high, 5 rest
        for i in range(10):
            rows.append({"pe": 0.5 + i * 0.01, "comp": 0.3 + i * 0.01, "state": "baseline", "binaryclass": 0, "dimension": 5, "tau": 1})
        for i in range(10):
            rows.append({"pe": 0.8 + i * 0.01, "comp": 0.1 + i * 0.01, "state": "low", "binaryclass": 1, "dimension": 5, "tau": 1})
        for i in range(10):
            rows.append({"pe": 0.85 + i * 0.01, "comp": 0.05 + i * 0.01, "state": "high", "binaryclass": 1, "dimension": 5, "tau": 1})
        for i in range(5):
            rows.append({"pe": 0.6 + i * 0.01, "comp": 0.2 + i * 0.01, "state": "rest", "binaryclass": 0, "dimension": 5, "tau": 1})
        df = pd.DataFrame(rows)

        result = prepare_ch_plane_data(df)
        # rest should be excluded
        assert len(result) == 30, f"Expected 30 rows (no rest), got {len(result)}"
        assert "rest" not in result["state"].values

    def test_returns_dataframe(self):
        rows = [
            {"pe": 0.5, "comp": 0.3, "state": "baseline", "binaryclass": 0, "dimension": 5, "tau": 1},
            {"pe": 0.8, "comp": 0.1, "state": "low", "binaryclass": 1, "dimension": 5, "tau": 1},
        ]
        df = pd.DataFrame(rows)
        result = prepare_ch_plane_data(df)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Figure Output Tests
# ---------------------------------------------------------------------------

class TestFigureOutput:
    def test_confusion_matrix_creates_file(self, tmp_path):
        """Verify PDF file is created at expected path."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 2, 2, 2])
        output_path = tmp_path / "test_cm.pdf"
        plot_confusion_matrix(
            y_true, y_pred,
            labels=["No Pain", "Low", "High"],
            output_path=str(output_path),
        )
        assert output_path.exists(), f"Expected {output_path} to be created"
        assert output_path.stat().st_size > 0, "File should not be empty"


# ---------------------------------------------------------------------------
# Confusion Matrix Values Tests
# ---------------------------------------------------------------------------

class TestConfusionMatrixValues:
    def test_known_values(self):
        """Known y_true/y_pred should produce expected confusion matrix."""
        from sklearn.metrics import confusion_matrix as sklearn_cm
        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
        cm = sklearn_cm(y_true, y_pred)
        # Row 0 (true=0): pred 0=2, pred 1=1, pred 2=0
        assert cm[0, 0] == 2
        assert cm[0, 1] == 1
        assert cm[0, 2] == 0
        # Row 1 (true=1): pred 0=0, pred 1=2, pred 2=1
        assert cm[1, 0] == 0
        assert cm[1, 1] == 2
        assert cm[1, 2] == 1
        # Row 2 (true=2): pred 0=0, pred 1=0, pred 2=3
        assert cm[2, 0] == 0
        assert cm[2, 1] == 0
        assert cm[2, 2] == 3
