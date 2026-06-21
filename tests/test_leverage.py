# tests/test_leverage.py
"""Tests for leverage and influence diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from openfit import Fit
from openfit.leverage import leverage_diagnostics


def test_hat_values_sum_to_k_poly1() -> None:
    """Verify that hat values sum to the number of active parameters for poly1."""
    # Poly1 has 2 parameters: a0, a1
    rng = np.random.default_rng(42)
    x = np.linspace(1, 10, 15)
    y = 3.0 + 2.0 * x + 0.5 * rng.standard_normal(len(x))

    result = Fit("poly1", x, y, weights="uniform").run()
    diag = leverage_diagnostics(result)

    assert np.all(diag.hat_values >= 0.0)
    assert np.all(diag.hat_values <= 1.0)
    # Sum of leverage values should equal k = 2
    np.testing.assert_allclose(np.sum(diag.hat_values), 2.0, rtol=1e-5)


def test_hat_values_sum_to_k_hill4p(hill4p_data) -> None:
    """Verify hat values sum to active parameters for hill4p (with and without fixed params)."""
    x, y = hill4p_data

    # 1. No fixed parameters (k = 4)
    result_full = Fit("hill4p", x, y, weights="uniform").run()
    diag_full = leverage_diagnostics(result_full)
    np.testing.assert_allclose(np.sum(diag_full.hat_values), 4.0, rtol=1e-5)

    # 2. One parameter fixed: Bottom=0.0 (k = 3)
    result_fixed = Fit("hill4p", x, y, weights="uniform", fixed={"Bottom": 0.0}).run()
    diag_fixed = leverage_diagnostics(result_fixed)
    np.testing.assert_allclose(np.sum(diag_fixed.hat_values), 3.0, rtol=1e-5)


def test_cooks_distance_zero_for_perfect_fit() -> None:
    """Verify Cook's distance and DFFITS are close to 0 for a perfect fit."""
    x = np.linspace(1, 5, 5)
    # Perfect linear relation
    y = 2.0 * x + 1.0

    result = Fit("poly1", x, y, weights="uniform").run()
    diag = leverage_diagnostics(result)

    # Cook's distance and DFFITS should be very close to 0
    np.testing.assert_allclose(diag.cooks_distance, 0.0, atol=1e-10)
    np.testing.assert_allclose(diag.dffits, 0.0, atol=1e-10)


def test_influential_point_detection_outlier() -> None:
    """Verify that an extreme outlier with high leverage is detected as influential."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 20.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 100.0])  # outlier at x=20 (expected y=40)

    result = Fit("poly1", x, y, weights="uniform").run()
    diag = leverage_diagnostics(result)

    # The point at x=20 is far from other points and deviates heavily from the line.
    # It should have high leverage and high Cook's distance, and be flagged.
    assert diag.high_leverage_mask[-1]
    assert diag.influential_mask[-1]
    # Other points should not be flagged as influential
    assert not np.any(diag.influential_mask[:-1])


def test_linear_model_leverage_matches_analytical() -> None:
    """Verify that leverage values match the analytical formula for unweighted and weighted fits."""
    rng = np.random.default_rng(24)
    x = np.linspace(1, 10, 10)
    # Use positive values for y to be compatible with 1/y2 weights
    y = 1.5 * x + 2.0 + 0.1 * rng.standard_normal(len(x))

    # 1. Unweighted (uniform) fit
    result_uni = Fit("poly1", x, y, weights="uniform").run()
    diag_uni = leverage_diagnostics(result_uni)

    # Analytical leverage: H = X(X^T X)^-1 X^T
    # For poly1, columns of Jacobian are 1 and x
    J = np.column_stack([np.ones_like(x), x])
    w_uni = result_uni._weights
    unscaled_cov_uni = np.linalg.inv(J.T @ (w_uni[:, None] * J))
    h_analytic_uni = w_uni * np.sum((J @ unscaled_cov_uni) * J, axis=1)

    np.testing.assert_allclose(diag_uni.hat_values, h_analytic_uni, rtol=1e-7)

    # 2. Weighted (1/y2) fit
    result_wt = Fit("poly1", x, y, weights="1/y2").run()
    diag_wt = leverage_diagnostics(result_wt)

    w_wt = result_wt._weights
    unscaled_cov_wt = np.linalg.inv(J.T @ (w_wt[:, None] * J))
    h_analytic_wt = w_wt * np.sum((J @ unscaled_cov_wt) * J, axis=1)

    np.testing.assert_allclose(diag_wt.hat_values, h_analytic_wt, rtol=1e-7)


def test_singular_fit_returns_nans() -> None:
    """Verify that leverage diagnostics handles singular/invalid covariance correctly."""
    # Create fake FitResult with NaN covariance
    from openfit.results import FitResult
    from openfit.spec import build_spec

    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    spec = build_spec(model_id="poly1", param_values={"a": 1.0}, weights="uniform", x=x, y=y)
    # Just a mock result
    res = FitResult(
        params={"a": 1.0},
        se={"a": float("inf")},
        ci={"a": (float("nan"), float("nan"))},
        covariance=np.full((1, 1), np.nan),
        r_squared=1.0,
        aic=0.0,
        bic=0.0,
        aicc=0.0,
        rss=0.0,
        x=x,
        y=y,
        y_fitted=y,
        residuals=np.zeros(2),
        weighted_residuals=np.zeros(2),
        standardized_residuals=np.zeros(2),
        n_obs=2,
        n_params=1,
        model_id="poly1",
        weight_scheme="uniform",
        spec=spec,
        _model=None,  # type: ignore
        _weights=np.ones(2),
    )

    diag = leverage_diagnostics(res)
    assert np.all(np.isnan(diag.hat_values))
    assert np.all(np.isnan(diag.cooks_distance))
    assert np.all(np.isnan(diag.dffits))
    assert not np.any(diag.high_leverage_mask)
    assert not np.any(diag.influential_mask)
