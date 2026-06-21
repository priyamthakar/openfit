# tests/test_bands.py
"""Tests for confidence and prediction bands."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path whether or not the package is installed editably.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
import scipy.stats as stats

from openfit import Fit
from openfit.bands import (
    BandResult,
    _numerical_jacobian,
    confidence_band,
    prediction_band,
)
from openfit.models.polynomial import Poly1


def test_confidence_band_narrows_with_more_data() -> None:
    """Verify that confidence band narrows with more data."""
    rng = np.random.default_rng(42)
    x_small = rng.uniform(1, 10, 15)
    y_small = 2.0 * x_small + 3.0 + 0.1 * rng.normal(size=len(x_small))

    x_large = rng.uniform(1, 10, 150)
    y_large = 2.0 * x_large + 3.0 + 0.1 * rng.normal(size=len(x_large))

    res_small = Fit("poly1", x_small, y_small, weights="uniform").run()
    res_large = Fit("poly1", x_large, y_large, weights="uniform").run()

    x_pred = np.array([5.0])
    band_small = confidence_band(res_small, x_pred)
    band_large = confidence_band(res_large, x_pred)

    width_small = band_small.upper[0] - band_small.lower[0]
    width_large = band_large.upper[0] - band_large.lower[0]

    assert width_large < width_small


def test_prediction_band_wider_than_confidence_band(hill4p_data) -> None:
    """Verify that prediction band is always wider than the confidence band."""
    x, y = hill4p_data
    result = Fit("hill4p", x, y, weights="uniform").run()

    x_pred = np.linspace(result.x.min(), result.x.max(), 50)
    band_conf = confidence_band(result, x_pred)
    band_pred = prediction_band(result, x_pred)

    width_conf = band_conf.upper - band_conf.lower
    width_pred = band_pred.upper - band_pred.lower

    assert np.all(width_pred > width_conf)


def test_band_contains_true_values_noiseless() -> None:
    """Verify that the band contains the true values for noiseless data."""
    x = np.linspace(1, 10, 20)
    y_true = 2.5 * x + 1.2

    # Fit noiseless data
    result = Fit("poly1", x, y_true, weights="uniform").run()

    band_conf = confidence_band(result, x)
    band_pred = prediction_band(result, x)

    # Allow a tiny tolerance for floating point precision
    assert np.all(y_true >= band_conf.lower - 1e-9)
    assert np.all(y_true <= band_conf.upper + 1e-9)
    assert np.all(y_true >= band_pred.lower - 1e-9)
    assert np.all(y_true <= band_pred.upper + 1e-9)


def test_linear_model_matches_closed_form() -> None:
    """Verify that linear model bands match simple linear regression closed-form formulas."""
    rng = np.random.default_rng(24)
    x = np.linspace(1, 10, 20)
    y = 3.0 * x + 5.0 + rng.normal(scale=0.5, size=len(x))

    result = Fit("poly1", x, y, weights="uniform").run()

    x_pred = np.linspace(0, 11, 15)
    band_conf = confidence_band(result, x_pred)
    band_pred = prediction_band(result, x_pred)

    # Calculate analytical closed form
    n = len(x)
    x_mean = np.mean(x)
    x_dev = x - x_mean
    sum_x_dev2 = np.sum(x_dev**2)
    df = n - 2
    s2 = result.rss / df

    var_conf_analytical = s2 * (1.0 / n + (x_pred - x_mean) ** 2 / sum_x_dev2)
    var_pred_analytical = s2 + var_conf_analytical

    alpha = 0.05
    t_val = float(stats.t.ppf(1.0 - alpha / 2.0, df))

    width_conf_analytical = t_val * np.sqrt(var_conf_analytical)
    width_pred_analytical = t_val * np.sqrt(var_pred_analytical)

    width_conf_delta = band_conf.upper - band_conf.y_pred
    width_pred_delta = band_pred.upper - band_pred.y_pred

    np.testing.assert_allclose(width_conf_delta, width_conf_analytical, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(width_pred_delta, width_pred_analytical, rtol=1e-12, atol=1e-12)

    # Verify predictions and bounds
    y_pred_analytical = result.params["a0"] + result.params["a1"] * x_pred
    np.testing.assert_allclose(band_conf.y_pred, y_pred_analytical, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        band_conf.lower, y_pred_analytical - width_conf_analytical, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        band_conf.upper, y_pred_analytical + width_conf_analytical, rtol=1e-12, atol=1e-12
    )


def test_x_pred_none_auto_grid() -> None:
    """Verify that x_pred=None auto-grid generation works correctly for linspace/logspace."""
    # Case 1: x.min() > 0 and x.max() / x.min() > 100 -> logspace
    x_log = np.logspace(-1, 2, 20)  # min=0.1, max=100. Ratio = 1000 > 100
    y_log = 2.0 * x_log + 3.0
    result_log = Fit("poly1", x_log, y_log, weights="uniform").run()

    band_log = confidence_band(result_log, x_pred=None, n_points=150)
    assert len(band_log.x) == 150
    assert band_log.band_type == "confidence"
    ratios = band_log.x[1:] / band_log.x[:-1]
    assert np.allclose(ratios, ratios[0])

    # Case 2: x.min() <= 0 or x.max() / x.min() <= 100 -> linspace
    x_lin = np.linspace(1, 10, 20)  # min=1, max=10. Ratio = 10 <= 100
    y_lin = 2.0 * x_lin + 3.0
    result_lin = Fit("poly1", x_lin, y_lin, weights="uniform").run()

    band_lin = confidence_band(result_lin, x_pred=None, n_points=100)
    assert len(band_lin.x) == 100
    diffs = np.diff(band_lin.x)
    assert np.allclose(diffs, diffs[0])


def test_fixed_parameter_intercept_zero_variance_at_zero() -> None:
    """Verify that fixing the intercept parameter results in zero band width at x=0."""
    x = np.linspace(1, 10, 15)
    y = 5.0 + 2.5 * x + np.random.default_rng(1).normal(scale=0.1, size=len(x))

    result = Fit("poly1", x, y, weights="uniform", fixed={"a0": 5.0}).run()

    x_pred = np.array([0.0])
    band_conf = confidence_band(result, x_pred)

    # Width at x=0 should be 0 since intercept is fixed
    width = band_conf.upper[0] - band_conf.lower[0]
    assert np.allclose(width, 0.0, atol=1e-12)
    assert np.allclose(band_conf.y_pred[0], 5.0, atol=1e-12)


def test_singular_fit_covariance_nans() -> None:
    """Verify that when covariance is all NaNs, lower and upper bounds are all NaNs."""
    x = np.linspace(1, 10, 15)
    y = 2.0 * x + 3.0
    result = Fit("poly1", x, y, weights="uniform").run()

    # Force covariance to be all NaNs
    result.covariance = np.full_like(result.covariance, np.nan)

    band_conf = confidence_band(result, x)
    band_pred = prediction_band(result, x)

    assert np.isnan(band_conf.lower).all()
    assert np.isnan(band_conf.upper).all()
    assert np.isnan(band_pred.lower).all()
    assert np.isnan(band_pred.upper).all()


def test_invalid_confidence() -> None:
    """Verify that invalid confidence values raise ValueError."""
    x = np.linspace(1, 10, 15)
    y = 2.0 * x + 3.0
    result = Fit("poly1", x, y, weights="uniform").run()

    with pytest.raises(ValueError, match="Confidence level must be between 0.0 and 1.0"):
        confidence_band(result, confidence=0.0)
    with pytest.raises(ValueError, match="Confidence level must be between 0.0 and 1.0"):
        confidence_band(result, confidence=1.0)
    with pytest.raises(ValueError, match="Confidence level must be between 0.0 and 1.0"):
        prediction_band(result, confidence=-0.5)
    with pytest.raises(ValueError, match="Confidence level must be between 0.0 and 1.0"):
        prediction_band(result, confidence=1.5)


def test_numerical_jacobian_matches_analytical() -> None:
    """Verify that numerical jacobian matches analytical jacobian for poly1."""
    model = Poly1()
    x = np.linspace(1, 10, 5)
    params = {"a0": 1.2, "a1": 3.4}

    jac_anal = model.jacobian(x, **params)
    assert jac_anal is not None

    jac_num = _numerical_jacobian(model, x, params, model.param_names, h=1e-6)

    np.testing.assert_allclose(jac_num, jac_anal, rtol=1e-5, atol=1e-5)


def test_fallback_numerical_jacobian_on_monoexp() -> None:
    """Verify fallback to numerical jacobian when model doesn't define analytical jacobian."""
    # MonoExp (monoexp) does not define analytical jacobian (returns None)
    x = np.linspace(0, 5, 20)
    y = 1.0 + 9.0 * np.exp(-0.5 * x)
    result = Fit("monoexp", x, y, weights="uniform").run()

    # This should fall back to numerical jacobian automatically and succeed
    band_conf = confidence_band(result, x)
    assert not np.isnan(band_conf.lower).any()
    assert not np.isnan(band_conf.upper).any()
    assert band_conf.band_type == "confidence"


def test_band_result_structure() -> None:
    """Verify the fields and types in the BandResult dataclass."""
    x = np.array([1.0])
    y_pred = np.array([2.0])
    lower = np.array([1.5])
    upper = np.array([2.5])
    band = BandResult(
        x=x, y_pred=y_pred, lower=lower, upper=upper, confidence=0.95, band_type="test"
    )
    assert np.array_equal(band.x, x)
    assert np.array_equal(band.y_pred, y_pred)
    assert np.array_equal(band.lower, lower)
    assert np.array_equal(band.upper, upper)
    assert band.confidence == 0.95
    assert band.band_type == "test"
