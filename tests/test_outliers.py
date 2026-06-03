"""Tests for openfit.outliers: ROUT adaptive outlier detection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from openfit.outliers import rout_outliers


def _clean_hill4p_data(n=30, seed=0):
    """Clean Hill4P data with very low noise (no outliers)."""
    rng = np.random.default_rng(seed)
    x = np.logspace(-1, 1, n)  # narrower range to avoid extreme tail values
    y_true = 0.0 + (100.0 - 0.0) / (1.0 + (1.0 / x) ** 1.0)
    y = y_true * (1.0 + 0.002 * rng.standard_normal(n))  # 0.2% noise -- very clean
    return x, np.maximum(y, 0.01)


# ---------------------------------------------------------------------------
# Structural / type tests
# ---------------------------------------------------------------------------


def test_rout_mask_shape():
    """ROUTResult.outlier_mask has the same shape as the input data."""
    x, y = _clean_hill4p_data()
    result = rout_outliers(x, y, "hill4p", Q=0.01)
    assert result.outlier_mask.shape == (len(x),)


def test_rout_mask_dtype_bool():
    """outlier_mask is a boolean array."""
    x, y = _clean_hill4p_data()
    result = rout_outliers(x, y, "hill4p", Q=0.01)
    assert result.outlier_mask.dtype == bool


def test_rout_result_has_expected_attributes():
    """ROUTResult exposes .outlier_mask, .n_outliers, and .Q."""
    x, y = _clean_hill4p_data()
    result = rout_outliers(x, y, "hill4p", Q=0.01)
    assert hasattr(result, "outlier_mask")
    assert hasattr(result, "n_outliers")
    assert hasattr(result, "Q")
    assert result.Q == 0.01


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


def test_rout_no_outliers_clean_data():
    """ROUT detects 0 outliers on clean low-noise Hill4P data."""
    x, y = _clean_hill4p_data(n=40)
    result = rout_outliers(x, y, "hill4p", Q=0.01)
    assert result.n_outliers == 0, (
        f"Expected 0 outliers in clean data, got {result.n_outliers} "
        f"at indices {result.outlier_indices}"
    )


def test_rout_detects_single_outlier():
    """ROUT flags a massively spiked point as an outlier."""
    x, y = _clean_hill4p_data(n=40, seed=1)
    # Inject an extreme outlier: set a mid-curve point to 10x the true value
    y_outlier = y.copy()
    spike_idx = len(x) // 2
    y_outlier[spike_idx] = y[spike_idx] * 10.0
    result = rout_outliers(x, y_outlier, "hill4p", Q=0.01)
    assert result.n_outliers >= 1, "Expected at least 1 outlier to be detected"
    assert result.outlier_mask[spike_idx], "Spiked point was not flagged"


def test_rout_q_parameter_respected():
    """Higher Q should flag same or more outliers than lower Q."""
    x, y = _clean_hill4p_data(n=40, seed=2)
    # Inject a moderate outlier
    y_mod = y.copy()
    y_mod[15] = y[15] * 3.5
    result_strict = rout_outliers(x, y_mod, "hill4p", Q=0.001)
    result_lenient = rout_outliers(x, y_mod, "hill4p", Q=0.05)
    assert result_lenient.n_outliers >= result_strict.n_outliers, (
        f"Expected higher Q to flag >= outliers: strict={result_strict.n_outliers}, "
        f"lenient={result_lenient.n_outliers}"
    )
