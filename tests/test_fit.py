"""Tests for openfit.fit.Fit and FitResult."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from openfit import Fit, FitSpec
from openfit.spec import compute_data_hash

# ---------------------------------------------------------------------------
# Parameter recovery
# ---------------------------------------------------------------------------


def test_hill4p_fit_recovers_params(hill4p_data) -> None:
    """Hill4P fit on low-noise data recovers EC50 within 5% and Top within 2%."""
    x, y = hill4p_data
    result = Fit("hill4p", x, y, weights="uniform").run()
    assert abs(result.params["EC50"] - 1.0) / 1.0 < 0.05
    assert abs(result.params["Top"] - 100.0) / 100.0 < 0.02


def test_monoexp_fit_recovers_params(mono_exp_data) -> None:
    """MonoExp fit recovers A (Y0) within 5% of 10 and k within 5% of 0.5."""
    x, y = mono_exp_data
    result = Fit("monoexp", x, y, weights="uniform").run()
    assert abs(result.params["Y0"] - 10.0) / 10.0 < 0.05
    assert abs(result.params["k"] - 0.5) / 0.5 < 0.05


# ---------------------------------------------------------------------------
# Required weights argument
# ---------------------------------------------------------------------------


def test_weights_required_no_default() -> None:
    """Fit() without the weights keyword raises TypeError."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    with pytest.raises(TypeError):
        Fit("poly2", x, y)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# NaN / Inf validation
# ---------------------------------------------------------------------------


def test_nan_in_x_raises() -> None:
    """x with NaN raises ValueError on .run()."""
    x = np.array([1.0, float("nan"), 3.0, 4.0, 5.0])
    y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    fit = Fit("poly2", x, y, weights="uniform")
    with pytest.raises(ValueError, match="NaN"):
        fit.run()


def test_nan_in_y_raises() -> None:
    """y with NaN raises ValueError on .run()."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, float("nan"), 9.0, 16.0, 25.0])
    fit = Fit("poly2", x, y, weights="uniform")
    with pytest.raises(ValueError, match="NaN"):
        fit.run()


def test_inf_in_data_raises() -> None:
    """Inf in y raises ValueError on .run()."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, float("inf"), 9.0, 16.0, 25.0])
    fit = Fit("poly2", x, y, weights="uniform")
    with pytest.raises(ValueError):
        fit.run()


# ---------------------------------------------------------------------------
# Residuals, R^2, AICc
# ---------------------------------------------------------------------------


def test_fit_result_residuals_close_to_zero_on_noiseless() -> None:
    """On exact (noiseless) Hill4P data the residuals are < 1e-6."""
    x = np.logspace(-2, 2, 30)
    y = 0.0 + (100.0 - 0.0) / (1.0 + (1.0 / x) ** 1.0)
    result = Fit("hill4p", x, y, weights="uniform").run()
    assert np.all(np.abs(result.residuals) < 1e-4)


def test_r_squared_near_one_for_good_fit(hill4p_data) -> None:
    """R-squared > 0.99 for a well-fitted low-noise dataset."""
    x, y = hill4p_data
    result = Fit("hill4p", x, y, weights="uniform").run()
    assert result.r_squared > 0.99


def test_aicc_finite(hill4p_data) -> None:
    """AICc is a finite float (not NaN or inf for a reasonable fit)."""
    x, y = hill4p_data
    result = Fit("hill4p", x, y, weights="uniform").run()
    assert np.isfinite(result.aicc)


# ---------------------------------------------------------------------------
# FitSpec attachment
# ---------------------------------------------------------------------------


def test_spec_is_attached(hill4p_data) -> None:
    """result.spec is a FitSpec instance."""
    x, y = hill4p_data
    result = Fit("hill4p", x, y, weights="uniform").run()
    assert isinstance(result.spec, FitSpec)


def test_spec_data_hash_matches(hill4p_data) -> None:
    """spec.data_hash matches compute_data_hash(x, y)."""
    x, y = hill4p_data
    result = Fit("hill4p", x, y, weights="uniform").run()
    expected_hash = compute_data_hash(x, y)
    assert result.spec.data_hash == expected_hash


# ---------------------------------------------------------------------------
# Weighted fit
# ---------------------------------------------------------------------------


def test_fit_with_1_over_y2_weights(hill4p_data) -> None:
    """1/y2 weighting runs without error and produces R^2 > 0.95."""
    x, y = hill4p_data
    result = Fit("hill4p", x, y, weights="1/y2").run()
    assert result.r_squared > 0.95


# ---------------------------------------------------------------------------
# Summary string
# ---------------------------------------------------------------------------


def test_fit_summary_is_string(hill4p_data) -> None:
    """result.summary() returns a non-empty string."""
    x, y = hill4p_data
    result = Fit("hill4p", x, y, weights="uniform").run()
    s = result.summary()
    assert isinstance(s, str)
    assert len(s) > 0


# ---------------------------------------------------------------------------
# TRF method
# ---------------------------------------------------------------------------


def test_trf_method_works(hill4p_data) -> None:
    """Fit with method='trf' converges successfully."""
    x, y = hill4p_data
    result = Fit("hill4p", x, y, weights="uniform", method="trf").run()
    assert result.r_squared > 0.95


# ---------------------------------------------------------------------------
# Standard errors and confidence intervals
# ---------------------------------------------------------------------------


def test_se_are_positive(hill4p_data) -> None:
    """All standard errors are strictly positive for a well-determined fit."""
    x, y = hill4p_data
    result = Fit("hill4p", x, y, weights="uniform").run()
    for name, se_val in result.se.items():
        assert se_val > 0, f"SE for {name!r} is not positive: {se_val}"


def test_ci_lower_lt_upper(hill4p_data) -> None:
    """For every parameter, the lower CI bound is strictly less than the upper."""
    x, y = hill4p_data
    result = Fit("hill4p", x, y, weights="uniform").run()
    for name, (lo, hi) in result.ci.items():
        assert lo < hi, f"CI for {name!r} is inverted: [{lo}, {hi}]"


# ---------------------------------------------------------------------------
# Warnings for unsupported parameters with method='lm'
# ---------------------------------------------------------------------------


def test_warns_x_scale_with_lm() -> None:
    """UserWarning when x_scale is provided but method='lm' ignores it."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    with pytest.warns(UserWarning, match="not supported with method"):
        Fit("poly2", x, y, weights="uniform", method="lm", x_scale="jac").run()


def test_warns_diff_method_with_lm() -> None:
    """UserWarning when diff_method is provided but method='lm' ignores it."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    with pytest.warns(UserWarning, match="not supported with method"):
        Fit("poly2", x, y, weights="uniform", method="lm", diff_method="3-point").run()


def test_no_warn_x_scale_with_trf(hill4p_data) -> None:
    """No warning when x_scale is used with method='trf' (supported)."""
    import warnings

    x, y = hill4p_data
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Fit("hill4p", x, y, weights="uniform", method="trf", x_scale="jac").run()


def test_no_warn_diff_method_with_trf(hill4p_data) -> None:
    """No warning when diff_method is used with method='trf' (supported)."""
    import warnings

    x, y = hill4p_data
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Fit("hill4p", x, y, weights="uniform", method="trf", diff_method="3-point").run()


def test_zero_rss_information_criteria() -> None:
    """AIC/BIC/AICc are -inf for a perfect (zero-RSS) fit; no RuntimeWarning emitted.

    A noiseless linear fit has RSS == 0 by construction.  The information-theoretic
    criteria are mathematically -inf in this case (log(0) = -inf).  The engine should
    return float('-inf') for all three without emitting a RuntimeWarning.
    """
    import warnings

    # Perfect linear data: y = 2*x + 1 (no noise)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * x + 1.0  # exact

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = Fit("poly1", x, y, weights="uniform").run()

    assert result.rss == 0.0 or result.rss < 1e-28, f"Expected zero RSS, got {result.rss}"
    assert result.aic == float("-inf"), f"AIC should be -inf for zero-RSS fit, got {result.aic}"
    assert result.bic == float("-inf"), f"BIC should be -inf for zero-RSS fit, got {result.bic}"
    assert result.aicc == float("-inf"), f"AICc should be -inf for zero-RSS fit, got {result.aicc}"
