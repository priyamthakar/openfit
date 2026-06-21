"""Tests for analytic Jacobians vs finite-difference reference (scipy.optimize.approx_fprime)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy.optimize import approx_fprime

from openfit.models.binding import CompetitiveBinding, OneSiteBinding, TwoSiteBinding
from openfit.models.gaussian import Gaussian, Lorentzian
from openfit.models.growth import AsymmetricGompertz, Gompertz, Logistic4P
from openfit.models.sigmoidal import Hill4P

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_jacobian(model, x, params, rtol=1e-4, atol=1e-6) -> None:
    """Compare analytic Jacobian to finite-difference reference.

    Parameters
    ----------
    model : object
        Model instance with .jacobian(x, **params) and .equation(x, **params).
    x : np.ndarray
        Sample x values.
    params : dict
        Parameter values.
    rtol, atol : float
        Tolerances passed to np.testing.assert_allclose.
    """
    J_analytic = model.jacobian(x, **params)
    assert J_analytic is not None, f"{model.model_id} must have analytic Jacobian"

    param_names = model.param_names
    param_values = np.array([params[n] for n in param_names])

    def f_wrapper(pvals):
        p = dict(zip(param_names, pvals, strict=False))
        return model.equation(x, **p)

    J_fd = approx_fprime(param_values, f_wrapper, 1e-8)

    # Check column by column
    for j, name in enumerate(param_names):
        np.testing.assert_allclose(
            J_analytic[:, j],
            J_fd[:, j],
            rtol=rtol,
            atol=atol,
            err_msg=f"Column {j} ({name}) of {model.model_id} Jacobian does not match finite diff",
        )


# ---------------------------------------------------------------------------
# Gaussian Jacobian
# ---------------------------------------------------------------------------


def test_gaussian_jacobian_vs_finite_diff() -> None:
    """Gaussian Jacobian matches scipy approx_fprime for a typical x array."""
    model = Gaussian()
    x = np.linspace(-5, 5, 50)
    params = {"A": 3.0, "mu": 1.0, "sigma": 2.0}
    _check_jacobian(model, x, params)


def test_gaussian_jacobian_at_peak() -> None:
    """Gaussian Jacobian at x=mu matches finite diff (sigma small)."""
    model = Gaussian()
    x = np.linspace(0.5, 1.5, 30)
    params = {"A": 5.0, "mu": 1.0, "sigma": 0.5}
    _check_jacobian(model, x, params)


def test_gaussian_jacobian_wide_sigma() -> None:
    """Gaussian Jacobian with wide sigma matches finite diff."""
    model = Gaussian()
    x = np.linspace(-20, 20, 60)
    params = {"A": 10.0, "mu": -2.0, "sigma": 10.0}
    _check_jacobian(model, x, params)


def test_gaussian_jacobian_small_sigma() -> None:
    """Gaussian Jacobian with small sigma matches finite diff."""
    model = Gaussian()
    x = np.linspace(-0.1, 0.1, 40)
    params = {"A": 1.0, "mu": 0.0, "sigma": 0.01}
    _check_jacobian(model, x, params)


def test_gaussian_jacobian_negative_A() -> None:
    """Gaussian Jacobian with negative amplitude matches finite diff."""
    model = Gaussian()
    x = np.linspace(-3, 3, 50)
    params = {"A": -2.5, "mu": 0.5, "sigma": 1.2}
    _check_jacobian(model, x, params)


# ---------------------------------------------------------------------------
# Lorentzian Jacobian
# ---------------------------------------------------------------------------


def test_lorentzian_jacobian_vs_finite_diff() -> None:
    """Lorentzian Jacobian matches scipy approx_fprime for a typical x array."""
    model = Lorentzian()
    x = np.linspace(-5, 5, 50)
    params = {"A": 4.0, "x0": 0.5, "gamma": 1.5}
    _check_jacobian(model, x, params)


def test_lorentzian_jacobian_at_peak() -> None:
    """Lorentzian Jacobian at x=x0 matches finite diff."""
    model = Lorentzian()
    x = np.linspace(0.5, 1.5, 30)
    params = {"A": 7.0, "x0": 1.0, "gamma": 0.3}
    _check_jacobian(model, x, params)


def test_lorentzian_jacobian_wide_gamma() -> None:
    """Lorentzian Jacobian with wide gamma matches finite diff."""
    model = Lorentzian()
    x = np.linspace(-50, 50, 80)
    params = {"A": 10.0, "x0": -5.0, "gamma": 20.0}
    _check_jacobian(model, x, params)


def test_lorentzian_jacobian_small_gamma() -> None:
    """Lorentzian Jacobian with small gamma matches finite diff."""
    model = Lorentzian()
    x = np.linspace(-0.05, 0.05, 50)
    params = {"A": 1.0, "x0": 0.0, "gamma": 0.005}
    _check_jacobian(model, x, params)


def test_lorentzian_jacobian_negative_A() -> None:
    """Lorentzian Jacobian with negative amplitude matches finite diff."""
    model = Lorentzian()
    x = np.linspace(-3, 3, 50)
    params = {"A": -3.0, "x0": 0.0, "gamma": 1.0}
    _check_jacobian(model, x, params)


# ---------------------------------------------------------------------------
# Hill4P (sigmoidal) Jacobian -- already has analytic
# ---------------------------------------------------------------------------


def test_hill4p_jacobian_vs_finite_diff() -> None:
    """Hill4P Jacobian matches scipy approx_fprime."""
    model = Hill4P()
    x = np.logspace(-2, 2, 50)
    params = {"Bottom": 0.0, "Top": 100.0, "EC50": 1.0, "HillSlope": 1.5}
    _check_jacobian(model, x, params)


def test_hill4p_jacobian_steep_slope() -> None:
    """Hill4P Jacobian with steep HillSlope matches finite diff."""
    model = Hill4P()
    x = np.logspace(-0.5, 0.5, 50)
    params = {"Bottom": 10.0, "Top": 90.0, "EC50": 0.5, "HillSlope": 5.0}
    _check_jacobian(model, x, params, rtol=5e-4, atol=1e-4)


def test_hill4p_jacobian_shallow_slope() -> None:
    """Hill4P Jacobian with shallow HillSlope matches finite diff."""
    model = Hill4P()
    x = np.logspace(-3, 3, 60)
    params = {"Bottom": -5.0, "Top": 50.0, "EC50": 10.0, "HillSlope": 0.5}
    _check_jacobian(model, x, params)


# ---------------------------------------------------------------------------
# Logistic4P Jacobian
# ---------------------------------------------------------------------------


def test_logistic4p_jacobian_vs_finite_diff() -> None:
    """Logistic4P analytic Jacobian matches scipy approx_fprime."""
    model = Logistic4P()
    x = np.linspace(0, 20, 30)
    params = {"K": 100.0, "r": 0.5, "x_mid": 5.0}
    _check_jacobian(model, x, params)


def test_logistic4p_jacobian_steep_r() -> None:
    """Logistic4P Jacobian with steep growth rate matches finite diff."""
    model = Logistic4P()
    x = np.linspace(0, 10, 30)
    params = {"K": 50.0, "r": 3.0, "x_mid": 4.0}
    _check_jacobian(model, x, params, rtol=2e-4, atol=1e-5)


def test_logistic4p_jacobian_shallow_r() -> None:
    """Logistic4P Jacobian with shallow growth rate matches finite diff."""
    model = Logistic4P()
    x = np.linspace(0, 100, 40)
    params = {"K": 200.0, "r": 0.05, "x_mid": 50.0}
    _check_jacobian(model, x, params)


# ---------------------------------------------------------------------------
# Gompertz Jacobian
# ---------------------------------------------------------------------------


def test_gompertz_jacobian_vs_finite_diff() -> None:
    """Gompertz analytic Jacobian matches scipy approx_fprime."""
    model = Gompertz()
    x = np.linspace(0, 30, 30)
    params = {"K": 100.0, "r": 0.3, "t_inf": 10.0}
    _check_jacobian(model, x, params)


def test_gompertz_jacobian_steep_r() -> None:
    """Gompertz Jacobian with steep growth rate matches finite diff."""
    model = Gompertz()
    x = np.linspace(0, 10, 30)
    params = {"K": 50.0, "r": 2.0, "t_inf": 4.0}
    _check_jacobian(model, x, params)


def test_gompertz_jacobian_shallow_r() -> None:
    """Gompertz Jacobian with shallow growth rate matches finite diff."""
    model = Gompertz()
    x = np.linspace(0, 100, 40)
    params = {"K": 200.0, "r": 0.05, "t_inf": 50.0}
    _check_jacobian(model, x, params)


# ---------------------------------------------------------------------------
# Receptor-Ligand Binding Jacobians
# ---------------------------------------------------------------------------


def test_one_site_binding_jacobian() -> None:
    """OneSiteBinding analytic Jacobian matches scipy approx_fprime."""
    model = OneSiteBinding()
    x = np.logspace(-2, 2, 30)
    params = {"Bmax": 150.0, "Kd": 5.0}
    _check_jacobian(model, x, params)


def test_two_site_binding_jacobian() -> None:
    """TwoSiteBinding analytic Jacobian matches scipy approx_fprime."""
    model = TwoSiteBinding()
    x = np.logspace(-2, 2, 40)
    params = {"Bmax1": 100.0, "Kd1": 2.0, "Bmax2": 50.0, "Kd2": 20.0}
    _check_jacobian(model, x, params)


def test_competitive_binding_jacobian() -> None:
    """CompetitiveBinding analytic Jacobian matches scipy approx_fprime."""
    # Test with non-zero inhibitor concentration
    model = CompetitiveBinding(inhibitor_conc=5.0)
    x = np.logspace(-2, 2, 30)
    params = {"Bmax": 120.0, "Kd": 4.0, "Ki": 2.0}
    _check_jacobian(model, x, params)


# ---------------------------------------------------------------------------
# Asymmetric Gompertz Jacobian
# ---------------------------------------------------------------------------


def test_asymmetric_gompertz_jacobian() -> None:
    """AsymmetricGompertz analytic Jacobian matches scipy approx_fprime."""
    model = AsymmetricGompertz()
    # Test points both below and above t_inf (5.0)
    x = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    params = {"K": 100.0, "r_left": 0.5, "r_right": 0.2, "t_inf": 5.0}
    _check_jacobian(model, x, params)
