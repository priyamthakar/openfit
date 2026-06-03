"""Tests verifying that the full parameter covariance matrix is exposed and correctly ordered."""

from __future__ import annotations

import numpy as np

from openfit import Fit
from openfit.models.custom import CustomModel
from openfit.report.html import render_html_report
from openfit.report.markdown import render_markdown_report
from openfit.uncertainty import profile_likelihood_ci


def _singular_redundant_line_result():
    def redundant_line(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return (a + b) * x

    model = CustomModel(
        model_id="redundant_line",
        func=redundant_line,
        param_names=["a", "b"],
        initial_guess_func=lambda x, y: {"a": 1.0, "b": 1.0},
    )
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    return Fit(model, x, y, weights="uniform").run()


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
    result = _singular_redundant_line_result()

    assert result.covariance.shape == (2, 2)
    assert np.isnan(result.covariance).all()
    assert all(np.isinf(value) for value in result.se.values())


def test_singular_fit_summary_ascii_safe():
    """summary() should not crash and should render non-finite SE/CI as NaN/inf."""
    result = _singular_redundant_line_result()

    # Should not raise
    summary_str = result.summary()
    assert "inf" in summary_str.lower() or "nan" in summary_str.lower()
    assert "a" in summary_str
    assert "b" in summary_str

    # CI should be NaN for singular fits
    for name in ("a", "b"):
        assert np.isnan(result.ci[name][0])
        assert np.isnan(result.ci[name][1])


def test_profile_ci_rejects_singular_fit():
    """Profile CI should fail clearly when singular SE cannot define a search range."""
    result = _singular_redundant_line_result()

    import pytest

    with pytest.raises(ValueError, match="profile-likelihood CI requires finite SE"):
        profile_likelihood_ci(result)


def test_singular_fit_reports_render():
    """HTML and Markdown reports should render singular fits without crashing."""
    result = _singular_redundant_line_result()

    html = render_html_report(result)
    markdown = render_markdown_report(result)

    assert "redundant_line" in html
    assert "redundant_line" in markdown
    assert "nan" in html.lower()
    assert "nan" in markdown.lower()
