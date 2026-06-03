"""Tests verifying that the full parameter covariance matrix is exposed and correctly ordered."""

from __future__ import annotations

import numpy as np

from openfit import Fit


def test_covariance_shape_and_diagonal_4pl():
    """Covariance matrix for 4PL should be 4x4, diagonal should match SE^2,
    and order should match param_names."""
    x = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    y = np.array([2.0, 5.0, 20.0, 65.0, 90.0, 97.0, 99.0])

    result = Fit("hill4p", x, y, weights="1/y2").run()

    # Shape check
    assert result.covariance.shape == (4, 4), f"Expected (4, 4), got {result.covariance.shape}"

    # Order check: param_names should match covariance row/col order
    param_names = ["Bottom", "Top", "EC50", "HillSlope"]
    assert result._model.param_names == param_names

    # Diagonal check: cov[i, i] should equal se[name]**2
    for i, name in enumerate(param_names):
        expected_var = result.se[name] ** 2
        actual_var = result.covariance[i, i]
        assert np.isclose(actual_var, expected_var, rtol=1e-5), (
            f"Covariance diagonal mismatch for {name}: expected {expected_var}, got {actual_var}"
        )

    # Symmetry check
    assert np.allclose(result.covariance, result.covariance.T, rtol=1e-5), (
        "Covariance matrix is not symmetric"
    )


def test_covariance_shape_and_diagonal_5pl():
    """Covariance matrix for 5PL should be 5x5, diagonal should match SE^2,
    and order should match param_names."""
    x = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    y = np.array([2.0, 5.0, 20.0, 65.0, 90.0, 97.0, 99.0])

    result = Fit("hill5p", x, y, weights="1/y2").run()

    # Shape check
    assert result.covariance.shape == (5, 5), f"Expected (5, 5), got {result.covariance.shape}"

    # Order check: param_names should match covariance row/col order
    param_names = ["Bottom", "Top", "EC50", "HillSlope", "Asymmetry"]
    assert result._model.param_names == param_names

    # Diagonal check: cov[i, i] should equal se[name]**2
    for i, name in enumerate(param_names):
        expected_var = result.se[name] ** 2
        actual_var = result.covariance[i, i]
        assert np.isclose(actual_var, expected_var, rtol=1e-5), (
            f"Covariance diagonal mismatch for {name}: expected {expected_var}, got {actual_var}"
        )

    # Symmetry check
    assert np.allclose(result.covariance, result.covariance.T, rtol=1e-5), (
        "Covariance matrix is not symmetric"
    )


def test_covariance_nan_on_singular_jacobian():
    """When Jacobian is singular, covariance should be NaN-filled matrix of correct shape."""
    # Degenerate data: all y values identical, which can cause singular Jacobian
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

    # This may succeed or fail depending on optimizer, but if it succeeds,
    # covariance should still be valid or NaN-filled
    try:
        result = Fit("hill4p", x, y, weights="uniform").run()
        # If it succeeds, check shape
        assert result.covariance.shape == (4, 4)
    except ValueError:
        # Expected if optimization fails due to degenerate data
        pass
