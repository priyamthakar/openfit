"""Tests for binding models: OneSiteBinding, TwoSiteBinding, CompetitiveBinding."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from openfit.models.binding import CompetitiveBinding, OneSiteBinding, TwoSiteBinding

# ---------------------------------------------------------------------------
# OneSiteBinding
# ---------------------------------------------------------------------------


def test_one_site_model_id() -> None:
    """OneSiteBinding model_id is 'one_site_binding'."""
    model = OneSiteBinding()
    assert model.model_id == "one_site_binding"


def test_one_site_param_names() -> None:
    """OneSiteBinding param_names are ['Bmax', 'Kd']."""
    model = OneSiteBinding()
    assert model.param_names == ["Bmax", "Kd"]


def test_one_site_equation_shape() -> None:
    """OneSiteBinding.equation returns array with same length as x."""
    model = OneSiteBinding()
    x = np.logspace(-2, 2, 20)
    y = model.equation(x, Bmax=100.0, Kd=10.0)
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_one_site_midpoint() -> None:
    """At x=Kd, OneSiteBinding returns Bmax/2."""
    model = OneSiteBinding()
    bmax, kd = 100.0, 5.0
    x = np.array([kd])
    y = model.equation(x, Bmax=bmax, Kd=kd)
    np.testing.assert_allclose(y[0], bmax / 2.0, rtol=1e-12)


def test_one_site_initial_guess() -> None:
    """OneSiteBinding.initial_guess returns dict with Bmax and Kd."""
    model = OneSiteBinding()
    x = np.logspace(-1, 2, 15)
    # Simulate a one-site binding curve: Bmax=100, Kd=10
    y_true = 100.0 * x / (10.0 + x)
    y = y_true * (1.0 + 0.01 * np.random.default_rng(42).standard_normal(len(x)))
    guesses = model.initial_guess(x, y)
    assert "Bmax" in guesses
    assert "Kd" in guesses
    assert len(guesses) == 2
    assert guesses["Bmax"] > 0
    assert guesses["Kd"] > 0


def test_one_site_initial_guess_all_zero_y() -> None:
    """OneSiteBinding.initial_guess handles all-zero y gracefully (Bmax fallback=1.0)."""
    model = OneSiteBinding()
    x = np.array([1.0, 2.0, 3.0])
    y = np.zeros(3)
    guesses = model.initial_guess(x, y)
    assert guesses["Bmax"] == 1.0  # fallback when max(y)==0
    assert guesses["Kd"] > 0


def test_one_site_bounds() -> None:
    """OneSiteBinding.bounds returns (lower, upper) each of length 2."""
    model = OneSiteBinding()
    lo, hi = model.bounds()
    assert len(lo) == 2
    assert len(hi) == 2
    assert all(val >= 0 for val in lo)
    assert hi == [np.inf, np.inf]


def test_one_site_jacobian_returns_array() -> None:
    """OneSiteBinding.jacobian returns analytical Jacobian array."""
    model = OneSiteBinding()
    J = model.jacobian(np.array([1.0, 2.0]), Bmax=100.0, Kd=10.0)
    assert isinstance(J, np.ndarray)
    assert J.shape == (2, 2)


# ---------------------------------------------------------------------------
# TwoSiteBinding
# ---------------------------------------------------------------------------


def test_two_site_model_id() -> None:
    """TwoSiteBinding model_id is 'two_site_binding'."""
    model = TwoSiteBinding()
    assert model.model_id == "two_site_binding"


def test_two_site_param_names() -> None:
    """TwoSiteBinding param_names are ['Bmax1', 'Kd1', 'Bmax2', 'Kd2']."""
    model = TwoSiteBinding()
    assert model.param_names == ["Bmax1", "Kd1", "Bmax2", "Kd2"]


def test_two_site_equation_shape() -> None:
    """TwoSiteBinding.equation returns array with same length as x."""
    model = TwoSiteBinding()
    x = np.logspace(-2, 2, 30)
    y = model.equation(x, Bmax1=60.0, Kd1=1.0, Bmax2=40.0, Kd2=100.0)
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_two_site_is_sum_of_two_langmuir() -> None:
    """TwoSiteBinding equals sum of two one-site isotherms with given params."""
    model = TwoSiteBinding()
    one_site = OneSiteBinding()
    x = np.logspace(-2, 3, 50)
    bmax1, kd1 = 60.0, 2.0
    bmax2, kd2 = 40.0, 50.0
    y_two = model.equation(x, Bmax1=bmax1, Kd1=kd1, Bmax2=bmax2, Kd2=kd2)
    y_site1 = one_site.equation(x, Bmax=bmax1, Kd=kd1)
    y_site2 = one_site.equation(x, Bmax=bmax2, Kd=kd2)
    np.testing.assert_allclose(y_two, y_site1 + y_site2, rtol=1e-12)


def test_two_site_initial_guess() -> None:
    """TwoSiteBinding.initial_guess returns dict with all 4 params."""
    model = TwoSiteBinding()
    x = np.logspace(-1, 3, 20)
    # Simulated two-site data: Bmax1=60, Kd1=1, Bmax2=40, Kd2=100
    y_true = 60.0 * x / (1.0 + x) + 40.0 * x / (100.0 + x)
    y = y_true * (1.0 + 0.01 * np.random.default_rng(99).standard_normal(len(x)))
    guesses = model.initial_guess(x, y)
    assert "Bmax1" in guesses
    assert "Kd1" in guesses
    assert "Bmax2" in guesses
    assert "Kd2" in guesses
    assert len(guesses) == 4
    assert guesses["Bmax1"] > 0
    assert guesses["Bmax2"] > 0
    assert guesses["Kd1"] > 0
    assert guesses["Kd2"] > 0
    # Kd1 should be < Kd2 by construction
    assert guesses["Kd1"] < guesses["Kd2"]


def test_two_site_initial_guess_few_points() -> None:
    """TwoSiteBinding.initial_guess works with fewer than 4 points."""
    model = TwoSiteBinding()
    x = np.array([1.0, 10.0, 100.0])
    y = np.array([30.0, 60.0, 90.0])
    guesses = model.initial_guess(x, y)
    assert len(guesses) == 4
    assert guesses["Kd1"] < guesses["Kd2"]


def test_two_site_bounds() -> None:
    """TwoSiteBinding.bounds returns (lower, upper) each of length 4."""
    model = TwoSiteBinding()
    lo, hi = model.bounds()
    assert len(lo) == 4
    assert len(hi) == 4
    assert all(val >= 0 for val in lo)
    assert hi == [np.inf, np.inf, np.inf, np.inf]


def test_two_site_jacobian_returns_array() -> None:
    """TwoSiteBinding.jacobian returns analytical Jacobian array."""
    model = TwoSiteBinding()
    J = model.jacobian(np.array([1.0, 2.0]), Bmax1=60.0, Kd1=1.0, Bmax2=40.0, Kd2=100.0)
    assert isinstance(J, np.ndarray)
    assert J.shape == (2, 4)


# ---------------------------------------------------------------------------
# CompetitiveBinding
# ---------------------------------------------------------------------------


def test_competitive_model_id() -> None:
    """CompetitiveBinding model_id is 'competitive_binding'."""
    model = CompetitiveBinding()
    assert model.model_id == "competitive_binding"


def test_competitive_param_names() -> None:
    """CompetitiveBinding param_names are ['Bmax', 'Kd', 'Ki']."""
    model = CompetitiveBinding()
    assert model.param_names == ["Bmax", "Kd", "Ki"]


def test_competitive_equation_shape() -> None:
    """CompetitiveBinding.equation returns array with same length as x."""
    model = CompetitiveBinding(inhibitor_conc=10.0)
    x = np.logspace(-2, 2, 25)
    y = model.equation(x, Bmax=100.0, Kd=5.0, Ki=2.0)
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_competitive_with_I_zero_matches_one_site() -> None:
    """CompetitiveBinding with I=0 matches OneSiteBinding with same Bmax, Kd."""
    comp = CompetitiveBinding(inhibitor_conc=0.0)
    one_site = OneSiteBinding()
    x = np.logspace(-2, 2, 30)
    bmax, kd = 100.0, 10.0
    y_comp = comp.equation(x, Bmax=bmax, Kd=kd, Ki=1.0)  # Ki doesn't matter when I=0
    y_one = one_site.equation(x, Bmax=bmax, Kd=kd)
    np.testing.assert_allclose(y_comp, y_one, rtol=1e-12)


def test_competitive_with_I_positive_shifts_kd_apparent() -> None:
    """With I>0, CompetitiveBinding has higher apparent Kd (right-shifted curve)."""
    comp_zero = CompetitiveBinding(inhibitor_conc=0.0)
    comp_pos = CompetitiveBinding(inhibitor_conc=10.0)
    x = np.linspace(0.1, 100.0, 100)
    bmax, kd, ki = 100.0, 5.0, 1.0
    comp_zero.equation(x, Bmax=bmax, Kd=kd, Ki=ki)
    comp_pos.equation(x, Bmax=bmax, Kd=kd, Ki=ki)
    # At any fixed x > 0, the inhibited curve should be lower (higher apparent Kd)
    # Check at several points
    for xi in [1.0, 5.0, 20.0]:
        yi_zero = comp_zero.equation(np.array([xi]), Bmax=bmax, Kd=kd, Ki=ki)[0]
        yi_pos = comp_pos.equation(np.array([xi]), Bmax=bmax, Kd=kd, Ki=ki)[0]
        assert yi_pos < yi_zero, f"At x={xi}: y_pos={yi_pos} should be < y_zero={yi_zero}"

    # Also: with I>0, the apparent Kd = Kd * (1 + I/Ki).
    # At x = apparent Kd, y should be Bmax/2.
    apparent_kd = kd * (1.0 + 10.0 / ki)  # = 5 * (1 + 10) = 55
    y_at_app_kd = comp_pos.equation(np.array([apparent_kd]), Bmax=bmax, Kd=kd, Ki=ki)[0]
    np.testing.assert_allclose(y_at_app_kd, bmax / 2.0, rtol=1e-12)


def test_competitive_initial_guess() -> None:
    """CompetitiveBinding.initial_guess returns dict with Bmax, Kd, Ki."""
    model = CompetitiveBinding(inhibitor_conc=5.0)
    x = np.logspace(-1, 2, 15)
    y_true = 100.0 * x / (10.0 + x)
    y = y_true * (1.0 + 0.01 * np.random.default_rng(77).standard_normal(len(x)))
    guesses = model.initial_guess(x, y)
    assert "Bmax" in guesses
    assert "Kd" in guesses
    assert "Ki" in guesses
    assert len(guesses) == 3
    assert guesses["Bmax"] > 0
    assert guesses["Kd"] > 0
    assert guesses["Ki"] > 0
    # Ki should equal Kd as initial estimate (equal affinity assumption)
    assert guesses["Ki"] == guesses["Kd"]


def test_competitive_bounds() -> None:
    """CompetitiveBinding.bounds returns (lower, upper) each of length 3."""
    model = CompetitiveBinding()
    lo, hi = model.bounds()
    assert len(lo) == 3
    assert len(hi) == 3
    assert all(val >= 0 for val in lo)
    assert hi == [np.inf, np.inf, np.inf]


def test_competitive_jacobian_returns_array() -> None:
    """CompetitiveBinding.jacobian returns analytical Jacobian array."""
    model = CompetitiveBinding(inhibitor_conc=5.0)
    J = model.jacobian(np.array([1.0, 2.0]), Bmax=100.0, Kd=5.0, Ki=1.0)
    assert isinstance(J, np.ndarray)
    assert J.shape == (2, 3)
