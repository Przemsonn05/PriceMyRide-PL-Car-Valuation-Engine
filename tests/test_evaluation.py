"""Tests for evaluation metrics."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import calculate_metrics


def test_metrics_on_perfect_prediction():
    y = np.array([10_000, 20_000, 30_000, 40_000, 50_000], dtype=float)
    m = calculate_metrics(y, y)
    assert m["R2"] == pytest.approx(1.0)
    assert m["RMSE"] == pytest.approx(0.0)
    assert m["MAE"] == pytest.approx(0.0)
    assert m["MAPE"] == pytest.approx(0.0)
    assert m["MdAPE"] == pytest.approx(0.0)


def test_metrics_return_mdape():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 330.0, 360.0])  # 10%, 5%, 10%, 10%
    m = calculate_metrics(y_true, y_pred)
    assert m["MdAPE"] == pytest.approx(0.10, abs=1e-6)


def test_mdape_robust_to_outlier():
    """MdAPE must not be dominated by a single bad prediction."""
    y_true = np.array([100.0] * 10 + [100.0])
    y_pred = np.array([102.0] * 10 + [1000.0])  # one huge error
    m = calculate_metrics(y_true, y_pred)
    # MAPE is inflated by the outlier, MdAPE stays near 0.02.
    assert m["MAPE"] > 0.5
    assert m["MdAPE"] < 0.1
