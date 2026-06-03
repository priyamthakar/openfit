"""Tests for openfit model equations, Jacobians, and the model registry."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from openfit.models.sigmoidal import Hill3P, Hill4P, Hill5P, Boltzmann
from openfit.models.exponential import MonoExp, BiExp, ExpGrowth, ExpPlateau, ExpDecay
from openfit.models.enzyme import MichaelisMenten
from openfit.models.growth import Logistic3P, Logistic4P, Gompertz
from openfit.models.gaussian import Gaussian, Lorentzian
from openfit.models.polynomial import Poly1, Poly2, Poly3
from openfit.models.custom import CustomModel
from openfit.models import get_model, list_models, register_model


# ---------------------------------------------------------------------------
# Sigmoidal
# ---------------------------------------------------------------------------


def test_hill4p_midpoint():
    """At x=EC50, Hill4P returns (Top+Bottom)/2."""
    model = Hill4P()
    bottom, top, ec50, slope = 0.0, 100.0, 5.0, 1.0
    x = np.array([ec50])
    y = model.equation(x, Bottom=bottom, Top=top, EC50=ec50, HillSlope=slope)
    expected = (top + bottom) / 2.0
    np.testing.assert_allclose(y[0], expected, rtol=1e-10)


def test_hill4p_jacobian_shape():
    """Hill4P Jacobian has shape (n_points, 4)."""
    model = Hill4P()
    x = np.linspace(0.1, 10.0, 15)
    J = model.jacobian(x, Bottom=0.0, Top=100.0, EC50=1.0, HillSlope=1.0)
    assert J is not None
    assert J.shape == (15, 4)


def test_hill3p_fixed_bottom_zero():
    """Hill3P at x=EC50 returns Top/2 (since Bottom=0 by definition)."""
    model = Hill3P()
    top, ec50, slope = 80.0, 2.0, 1.0
    x = np.array([ec50])
    y = model.equation(x, Top=top, EC50=ec50, HillSlope=slope)
    np.testing.assert_allclose(y[0], top / 2.0, rtol=1e-10)


def test_hill5p_asymmetry_1_matches_hill4p():
    """Hill5P with Asymmetry=1 gives the same output as Hill4P."""
    x = np.logspace(-1, 1, 10)
    params = dict(Bottom=0.0, Top=100.0, EC50=1.0, HillSlope=1.5)

    h4 = Hill4P()
    h5 = Hill5P()
    y4 = h4.equation(x, **params)
    y5 = h5.equation(x, **params, Asymmetry=1.0)
    np.testing.assert_allclose(y5, y4, rtol=1e-7)


def test_boltzmann_midpoint():
    """At x=V50, Boltzmann returns (Top+Bottom)/2."""
    model = Boltzmann()
    bottom, top, v50, slope = 0.0, 100.0, 0.0, 10.0
    x = np.array([v50])
    y = model.equation(x, Bottom=bottom, Top=top, V50=v50, Slope=slope)
    np.testing.assert_allclose(y[0], (top + bottom) / 2.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# Exponential
# ---------------------------------------------------------------------------


def test_monoexp_at_zero():
    """MonoExp.equation(0) == Y0 (the initial value)."""
    model = MonoExp()
    y0, plateau, k = 10.0, 2.0, 0.5
    x = np.array([0.0])
    y = model.equation(x, Y0=y0, Plateau=plateau, k=k)
    np.testing.assert_allclose(y[0], y0, rtol=1e-12)


def test_biexp_reduces_to_mono_when_span_fast_zero():
    """BiExp with Span_fast=0 equals a mono-exp (slow component only)."""
    model = BiExp()
    x = np.linspace(0, 5, 20)
    plateau, span_slow, k_slow = 2.0, 8.0, 0.3
    y_bi = model.equation(
        x, Plateau=plateau, Span_fast=0.0, k_fast=100.0,
        Span_slow=span_slow, k_slow=k_slow,
    )
    y_mono = plateau + span_slow * np.exp(-k_slow * x)
    np.testing.assert_allclose(y_bi, y_mono, rtol=1e-6)


def test_expgrowth_initial_value():
    """ExpGrowth at x=0 returns Y0."""
    model = ExpGrowth()
    y0, k = 3.0, 0.5
    x = np.array([0.0])
    y = model.equation(x, Y0=y0, k=k)
    np.testing.assert_allclose(y[0], y0, rtol=1e-12)


def test_safe_exp_no_overflow():
    """Large x values in ExpGrowth do not raise OverflowError."""
    model = ExpGrowth()
    x = np.array([1e10])
    # Should return a finite (clipped) value, not raise.
    y = model.equation(x, Y0=1.0, k=1.0)
    assert np.isfinite(y[0])


# ---------------------------------------------------------------------------
# Enzyme
# ---------------------------------------------------------------------------


def test_mm_at_km():
    """MichaelisMenten at x=Km returns Vmax/2."""
    model = MichaelisMenten()
    vmax, km = 50.0, 10.0
    x = np.array([km])
    y = model.equation(x, Vmax=vmax, Km=km)
    np.testing.assert_allclose(y[0], vmax / 2.0, rtol=1e-12)


def test_mm_jacobian_shape():
    """MichaelisMenten Jacobian has shape (n_points, 2)."""
    model = MichaelisMenten()
    x = np.linspace(1.0, 100.0, 20)
    J = model.jacobian(x, Vmax=50.0, Km=10.0)
    assert J is not None
    assert J.shape == (20, 2)


# ---------------------------------------------------------------------------
# Growth
# ---------------------------------------------------------------------------


def test_logistic3p_initial_value():
    """Logistic3P at x=0 returns N0 (initial population)."""
    model = Logistic3P()
    k, n0, r = 100.0, 10.0, 0.5
    x = np.array([0.0])
    y = model.equation(x, K=k, N0=n0, r=r)
    np.testing.assert_allclose(y[0], n0, rtol=1e-10)


def test_logistic4p_midpoint():
    """Logistic4P at x=x_mid returns K/2."""
    model = Logistic4P()
    k, r, x_mid = 100.0, 1.0, 5.0
    x = np.array([x_mid])
    y = model.equation(x, K=k, r=r, x_mid=x_mid)
    np.testing.assert_allclose(y[0], k / 2.0, rtol=1e-10)


def test_gompertz_at_zero():
    """Gompertz at x=0 returns K * exp(-exp(r * t_inf)) from the exact equation."""
    model = Gompertz()
    # Equation: y = K * exp(-exp(-r * (x - t_inf)))
    # At x=0: y = K * exp(-exp(-r * (0 - t_inf))) = K * exp(-exp(r * t_inf))
    k, r, t_inf = 100.0, 0.5, 5.0
    x = np.array([0.0])
    y = model.equation(x, K=k, r=r, t_inf=t_inf)
    expected = k * np.exp(-np.exp(r * t_inf))
    np.testing.assert_allclose(y[0], expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Gaussian / Lorentzian
# ---------------------------------------------------------------------------


def test_gaussian_peak_at_mean():
    """Gaussian at x=mu returns exactly A."""
    model = Gaussian()
    a, mu, sigma = 5.0, 3.0, 1.0
    x = np.array([mu])
    y = model.equation(x, A=a, mu=mu, sigma=sigma)
    np.testing.assert_allclose(y[0], a, rtol=1e-12)


def test_lorentzian_peak_at_center():
    """Lorentzian at x=x0 returns A."""
    model = Lorentzian()
    a, x0, gamma = 7.0, 2.0, 0.5
    x = np.array([x0])
    y = model.equation(x, A=a, x0=x0, gamma=gamma)
    np.testing.assert_allclose(y[0], a, rtol=1e-12)


# ---------------------------------------------------------------------------
# Polynomial
# ---------------------------------------------------------------------------


def test_poly1_is_linear():
    """Poly1(x, a0, a1) = a0 + a1*x exactly."""
    model = Poly1()
    x = np.array([0.0, 1.0, 2.0, 3.0])
    a0, a1 = 2.0, 3.0
    y = model.equation(x, a0=a0, a1=a1)
    np.testing.assert_allclose(y, a0 + a1 * x, rtol=1e-15)


def test_poly3_matches_numpy_polyval():
    """Poly3 matches np.polyval with reversed (descending-degree) coefficients."""
    model = Poly3()
    x = np.array([0.0, 1.0, 2.0, 3.0])
    a0, a1, a2, a3 = 1.0, 2.0, 3.0, 4.0
    y = model.equation(x, a0=a0, a1=a1, a2=a2, a3=a3)
    # np.polyval expects descending order: [a3, a2, a1, a0]
    expected = np.polyval([a3, a2, a1, a0], x)
    np.testing.assert_allclose(y, expected, rtol=1e-14)


def test_poly2_jacobian_correct():
    """Poly2 Jacobian columns are [1, x, x^2] in ascending degree order."""
    model = Poly2()
    x = np.array([1.0, 2.0, 3.0])
    J = model.jacobian(x, a0=1.0, a1=2.0, a2=3.0)
    assert J is not None
    np.testing.assert_allclose(J[:, 0], np.ones_like(x))
    np.testing.assert_allclose(J[:, 1], x)
    np.testing.assert_allclose(J[:, 2], x ** 2)


# ---------------------------------------------------------------------------
# Custom model
# ---------------------------------------------------------------------------


def test_custom_model_callable():
    """CustomModel wraps a callable and evaluates correctly."""
    def linear(x, a, b):
        return a + b * x

    model = CustomModel("linear_custom", linear)
    x = np.array([0.0, 1.0, 2.0])
    y = model.equation(x, a=1.0, b=2.0)
    np.testing.assert_allclose(y, [1.0, 3.0, 5.0])


def test_custom_model_infers_param_names():
    """CustomModel infers param_names from the function signature."""
    def power_law(x, A, n):
        return A * x ** n

    model = CustomModel("power_law_custom", power_law)
    assert model.param_names == ["A", "n"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_list_models_returns_all_expected():
    """list_models() returns at least 25 registered built-in models."""
    models = list_models()
    assert len(models) >= 25


def test_get_model_case_insensitive():
    """get_model is case-insensitive: HILL4P == hill4p."""
    m1 = get_model("hill4p")
    m2 = get_model("HILL4P")
    assert m1.model_id == m2.model_id


def test_register_custom_model():
    """A CustomModel can be registered and retrieved by name."""
    def quad(x, a, b, c):
        return a * x ** 2 + b * x + c

    model = CustomModel("test_quad_unique", quad)
    register_model(model)
    retrieved = get_model("test_quad_unique")
    assert retrieved.model_id == "test_quad_unique"
    assert retrieved.param_names == ["a", "b", "c"]
