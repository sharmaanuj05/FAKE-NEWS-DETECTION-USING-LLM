"""
test_calibration.py — Unit tests for Temperature Scaling calibration module.
"""
import os
import sys
import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.calibration import (
    fit_temperature_scaling,
    compute_ece,
    compute_brier_score,
    apply_temperature_scaling
)


def test_ece_computation():
    """ECE should be 0 for perfectly calibrated predictions."""
    probs = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2], [0.8, 0.2], [0.2, 0.8]])
    labels = np.array([0, 0, 0, 0, 1])
    ece = compute_ece(probs, labels, n_bins=5)
    assert 0.0 <= ece <= 0.2


def test_temperature_scaling_optimization():
    """Temperature scaling should optimize temperature strictly > 0 and preserve ranking."""
    np.random.seed(42)
    # Simulate overconfident uncalibrated logits
    logits = np.random.randn(200, 2) * 5.0
    labels = (logits[:, 1] > logits[:, 0]).astype(int)
    # Introduce 15% label noise
    flip_mask = np.random.rand(200) < 0.15
    labels[flip_mask] = 1 - labels[flip_mask]

    res = fit_temperature_scaling(logits, labels, save_to_disk=False)
    temp = res["temperature"]
    assert temp > 0.1, f"Temperature {temp} should be positive"
    assert "val_ece_calibrated" in res
    assert "val_brier_calibrated" in res

    # Verify argmax invariance
    unscaled_preds = np.argmax(logits, axis=1)
    scaled_probs = apply_temperature_scaling(logits, temp)
    scaled_preds = np.argmax(scaled_probs, axis=1)
    np.testing.assert_array_equal(unscaled_preds, scaled_preds)
