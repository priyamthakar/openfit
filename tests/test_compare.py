"""Tests for openfit.compare: model comparison, AICc, BIC, F-test."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from openfit import Fit, compare_models


def _hill3p_result(x, y):
    return Fit("hill3p", x, y, weights="uniform").run()


def _hill4p_result(x, y):
    return Fit("hill4p", x, y, weights="uniform").run()


def _mm_result(x, y):
    return Fit("michaelis_menten", x, y, weights="uniform").run()


# Common test data: a Hill3P (3-param) should fit well when Bottom=0.
def _make_hill3p_data():
    rng = np.random.default_rng(5)
    x = np.logspace(-2, 2, 25)
    y_true = 100.0 / (1.0 + (1.0 / x) ** 1.5)
    y = y_true * (1.0 + 0.01 * rng.standard_normal(25))
    return np.maximum(y, 0.01), x


# ---------------------------------------------------------------------------
# compare_models returns ComparisonResult
# ---------------------------------------------------------------------------


def test_compare_models_returns_comparison_result() -> None:
    """compare_models([r1, r2]) returns a ComparisonResult without error."""
    y, x = _make_hill3p_data()
    r3 = _hill3p_result(x, y)
    r4 = _hill4p_result(x, y)
    cr = compare_models([r3, r4])
    assert cr is not None
    assert len(cr.model_ids) == 2


def test_aicc_lower_for_better_model() -> None:
    """For data from a linear model, Poly1 has strictly lower AICc than Poly3."""
    rng = np.random.default_rng(99)
    x = np.linspace(0, 10, 30)
    y = 2.0 + 3.0 * x + 0.1 * rng.standard_normal(30)
    r1 = Fit("poly1", x, y, weights="uniform").run()
    r3 = Fit("poly3", x, y, weights="uniform").run()
    cr = compare_models([r1, r3])
    # Simpler (poly1) should be preferred for truly linear data
    assert cr.best_model_by_aicc == "poly1", (
        f"Expected poly1 as best model, got {cr.best_model_by_aicc}; "
        f"AICc poly1={cr.aicc_values['poly1']:.2f}, poly3={cr.aicc_values['poly3']:.2f}"
    )


def test_best_model_has_lowest_aicc() -> None:
    """ComparisonResult.best_model_by_aicc has the minimum AICc value."""
    y, x = _make_hill3p_data()
    r3 = _hill3p_result(x, y)
    r4 = _hill4p_result(x, y)
    cr = compare_models([r3, r4])
    best = cr.best_model_by_aicc
    min_aicc = min(cr.aicc_values.values())
    assert cr.aicc_values[best] == min_aicc


# ---------------------------------------------------------------------------
# F-test for nested models
# ---------------------------------------------------------------------------


def test_f_test_nested_hill3_in_hill4() -> None:
    """Hill3P is nested in Hill4P; compare_models should run the F-test."""
    y, x = _make_hill3p_data()
    r3 = _hill3p_result(x, y)
    r4 = _hill4p_result(x, y)
    cr = compare_models([r3, r4])
    # Hill3P param names are a subset of Hill4P: F-test should fire
    assert cr.f_test is not None


def test_f_test_not_run_for_non_nested() -> None:
    """Non-nested models (Hill4P vs Michaelis-Menten): f_test must be None."""
    rng = np.random.default_rng(0)
    x = np.logspace(-1, 2, 20)
    y = 100.0 / (1.0 + (1.0 / x) ** 1.0) + 2.0 * rng.standard_normal(20)
    y = np.maximum(y, 0.1)
    r4 = Fit("hill4p", x, y, weights="uniform").run()
    rm = _mm_result(x, y)
    cr = compare_models([r4, rm])
    assert cr.f_test is None


# ---------------------------------------------------------------------------
# Akaike weights
# ---------------------------------------------------------------------------


def test_akaike_weights_sum_to_one() -> None:
    """Akaike weights across all models sum to approximately 1.0."""
    y, x = _make_hill3p_data()
    r3 = _hill3p_result(x, y)
    r4 = _hill4p_result(x, y)
    cr = compare_models([r3, r4])
    total = sum(cr.akaike_weights.values())
    assert abs(total - 1.0) < 1e-10
