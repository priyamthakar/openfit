"""Tests for openfit.uncertainty: asymptotic_ci, bootstrap_ci, profile_likelihood_ci."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from openfit import Fit
from openfit.uncertainty import asymptotic_ci, bootstrap_ci, profile_likelihood_ci


def _make_mm_result(n=30, seed=0):
    """Fit Michaelis-Menten to synthetic data and return the FitResult."""
    rng = np.random.default_rng(seed)
    x = np.linspace(1, 100, n)
    y_true = 50.0 * x / (10.0 + x)
    y = y_true * (1.0 + 0.01 * rng.standard_normal(n))
    y = np.maximum(y, 0.01)
    return Fit("michaelis_menten", x, y, weights="uniform").run()


# ---------------------------------------------------------------------------
# asymptotic_ci
# ---------------------------------------------------------------------------


def test_asymptotic_ci_contains_true_param():
    """For noiseless MM data, the 95% asymptotic CI contains the true Vmax and Km."""
    x = np.linspace(1, 100, 40)
    y = 50.0 * x / (10.0 + x)  # exact, no noise
    result = Fit("michaelis_menten", x, y, weights="uniform").run()

    ci = result.ci  # asymptotic CI from Fit
    lo_vmax, hi_vmax = ci["Vmax"]
    lo_km, hi_km = ci["Km"]
    assert lo_vmax <= 50.0 <= hi_vmax, f"True Vmax=50 not in CI [{lo_vmax:.3f}, {hi_vmax:.3f}]"
    assert lo_km <= 10.0 <= hi_km, f"True Km=10 not in CI [{lo_km:.3f}, {hi_km:.3f}]"


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_shape():
    """bootstrap_ci returns a dict keyed by param names, each value a (lo, hi) tuple."""
    result = _make_mm_result()
    ci = bootstrap_ci(result, n_bootstrap=200, random_seed=42)
    assert set(ci.keys()) == set(result.params.keys())
    for name, (lo, hi) in ci.items():
        assert isinstance(lo, float)
        assert isinstance(hi, float)


def test_bootstrap_ci_seed_reproducible():
    """Same random_seed produces identical bootstrap CI bounds."""
    result = _make_mm_result()
    ci1 = bootstrap_ci(result, n_bootstrap=200, random_seed=42)
    ci2 = bootstrap_ci(result, n_bootstrap=200, random_seed=42)
    for name in result.params:
        assert ci1[name] == ci2[name], f"CI for {name!r} not reproducible"


def test_bootstrap_ci_lower_lt_upper():
    """bootstrap_ci lower < upper for all params."""
    result = _make_mm_result()
    ci = bootstrap_ci(result, n_bootstrap=200, random_seed=99)
    for name, (lo, hi) in ci.items():
        assert lo < hi, f"bootstrap CI for {name!r} is inverted: [{lo}, {hi}]"


# ---------------------------------------------------------------------------
# profile_likelihood_ci
# ---------------------------------------------------------------------------


def test_profile_ci_returns_dict():
    """profile_likelihood_ci returns a ProfileCIResult with a .ci dict."""
    result = _make_mm_result()
    pci = profile_likelihood_ci(result, n_steps=10)
    # ci is a dict keyed by param names
    assert set(pci.ci.keys()) == set(result.params.keys())
    for name, (lo, hi) in pci.ci.items():
        assert isinstance(lo, float)
        assert isinstance(hi, float)


# ---------------------------------------------------------------------------
# CI width decreases with more data (Michaelis-Menten, 2 params)
# ---------------------------------------------------------------------------


def test_ci_width_decreases_with_more_data():
    """Asymptotic SE (proxy for CI width) is smaller for n=50 than for n=8 on MM."""
    # Use the same noise magnitude; SE must shrink as n grows from 8 -> 50.
    rng_small = np.random.default_rng(12)
    rng_large = np.random.default_rng(12)

    noise_frac = 0.05  # 5% noise -- large enough that SE is nontrivial

    # 8-point fit (still enough to fit 2 params, but SE is larger)
    x8 = np.linspace(1, 100, 8)
    y8 = 50.0 * x8 / (10.0 + x8) * (1.0 + noise_frac * rng_small.standard_normal(8))
    y8 = np.maximum(y8, 0.01)
    r8 = Fit("michaelis_menten", x8, y8, weights="uniform").run()

    # 50-point fit — same x range, much more data
    x50 = np.linspace(1, 100, 50)
    y50 = 50.0 * x50 / (10.0 + x50) * (1.0 + noise_frac * rng_large.standard_normal(50))
    y50 = np.maximum(y50, 0.01)
    r50 = Fit("michaelis_menten", x50, y50, weights="uniform").run()

    # Vmax SE is well-behaved: should clearly shrink
    assert r50.se["Vmax"] < r8.se["Vmax"], (
        f"SE for Vmax did not decrease: 8pt={r8.se['Vmax']:.4f}, "
        f"50pt={r50.se['Vmax']:.4f}"
    )
