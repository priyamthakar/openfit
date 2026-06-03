"""Published-reference F-test validation for compare.py.

Validates the extra sum-of-squares F-test implementation against the formula
from Motulsky & Christopoulos (2003) "Fitting Models to Biological Data Using
Linear and Nonlinear Regression", GraphPad Software, Chapter 12.

Reference:
    Motulsky, H., & Christopoulos, A. (2003). Fitting Models to Biological
    Data Using Linear and Nonlinear Regression: A Practical Guide to Curve
    Fitting. GraphPad Software Inc., San Diego CA.
    ISBN: 0-19-517182-9 (Oxford University Press edition, 2004)

    F-test formula (Chapter 12, Equation 12.1):
        F = ((RSS_simple - RSS_complex) / (df_simple - df_complex)) /
            (RSS_complex / df_complex)

    where:
        - RSS_simple: residual sum of squares for simpler (nested) model
        - RSS_complex: residual sum of squares for complex model
        - df_simple = n - p_simple (degrees of freedom for simpler model)
        - df_complex = n - p_complex (degrees of freedom for complex model)
        - df_numerator = df_simple - df_complex = p_complex - p_simple
        - df_denominator = df_complex = n - p_complex

Application:
    Comparing one-site vs two-site binding models, a canonical example in
    receptor pharmacology where the F-test determines if the data justify
    the additional parameters of a two-site model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy import stats

from openfit import Fit, compare_models
from openfit.models import register_model
from openfit.models.custom import CustomModel

# ---------------------------------------------------------------------------
# One-site and two-site binding models (nested)
# ---------------------------------------------------------------------------


def one_site_binding(x: np.ndarray, Bmax: float, Kd: float) -> np.ndarray:
    """One-site binding (hyperbolic saturation).

    y = Bmax * x / (Kd + x)

    Parameters
    ----------
    x : np.ndarray
        Ligand concentration.
    Bmax : float
        Maximum binding (>0).
    Kd : float
        Equilibrium dissociation constant (>0).
    """
    Kd_safe = Kd if Kd != 0.0 else 1e-300
    return Bmax * x / (Kd_safe + x)


def two_site_binding(x: np.ndarray, Bmax: float, Kd: float, Bmax2: float, Kd2: float) -> np.ndarray:
    """Two-site binding (sum of two hyperbolic components).

    y = Bmax * x / (Kd + x) + Bmax2 * x / (Kd2 + x)

    Parameters
    ----------
    x : np.ndarray
        Ligand concentration.
    Bmax : float
        Maximum binding at site 1 (>0).
    Kd : float
        Kd for site 1 (>0).
    Bmax2 : float
        Maximum binding at site 2 (>0).
    Kd2 : float
        Kd for site 2 (>0).
    """
    Kd_safe = Kd if Kd != 0.0 else 1e-300
    Kd2_safe = Kd2 if Kd2 != 0.0 else 1e-300
    return Bmax * x / (Kd_safe + x) + Bmax2 * x / (Kd2_safe + x)


def initial_guess_one_site(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Data-driven initial estimates for one-site binding."""
    Bmax = float(np.max(y)) * 1.1
    if Bmax == 0.0:
        Bmax = 1.0
    half_max = Bmax / 2.0
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    Kd = float(np.interp(half_max, y_sorted, x_sorted)) if len(x_sorted) >= 2 else float(x[0])
    if Kd <= 0 or not np.isfinite(Kd):
        Kd = float(np.median(x[x > 0])) if np.any(x > 0) else 1.0
    return {"Bmax": Bmax, "Kd": Kd}


def initial_guess_two_site(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Data-driven initial estimates for two-site binding."""
    # Use one-site estimates as starting point, then split
    one_site_est = initial_guess_one_site(x, y)
    Bmax_total = one_site_est["Bmax"]
    Kd_est = one_site_est["Kd"]

    # Split into two sites with different affinities
    return {
        "Bmax": Bmax_total * 0.6,  # 60% at site 1
        "Kd": Kd_est * 0.3,  # Higher affinity (lower Kd)
        "Bmax2": Bmax_total * 0.4,  # 40% at site 2
        "Kd2": Kd_est * 3.0,  # Lower affinity (higher Kd)
    }


# Create and register models
one_site_model = CustomModel(
    model_id="one_site_binding",
    func=one_site_binding,
    param_names=["Bmax", "Kd"],
    initial_guess_func=initial_guess_one_site,
    bounds_dict={"Bmax": (1e-300, np.inf), "Kd": (1e-300, np.inf)},
)

two_site_model = CustomModel(
    model_id="two_site_binding",
    func=two_site_binding,
    param_names=["Bmax", "Kd", "Bmax2", "Kd2"],
    initial_guess_func=initial_guess_two_site,
    bounds_dict={
        "Bmax": (1e-300, np.inf),
        "Kd": (1e-300, np.inf),
        "Bmax2": (1e-300, np.inf),
        "Kd2": (1e-300, np.inf),
    },
)

register_model(one_site_model)
register_model(two_site_model)


# ---------------------------------------------------------------------------
# Synthetic dataset: true two-site binding with noise
# ---------------------------------------------------------------------------


def make_two_site_test_data(seed: int = 2003) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic two-site binding data.

    True parameters (representative of published receptor binding studies):
        Site 1 (high affinity): Bmax1=80, Kd1=2.0 nM
        Site 2 (low affinity):  Bmax2=40, Kd2=50.0 nM

    Returns
    -------
    x : np.ndarray
        Ligand concentrations (log-spaced from 0.1 to 1000 nM).
    y : np.ndarray
        Observed binding values with 3% Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    x = np.logspace(-1, 3, 30)  # 0.1 to 1000 nM, 30 points

    # True two-site binding
    y_true = two_site_binding(x, Bmax=80.0, Kd=2.0, Bmax2=40.0, Kd2=50.0)

    # Add 3% Gaussian noise (realistic for binding assays)
    noise = 0.03 * y_true * rng.standard_normal(len(x))
    y = y_true + noise

    # Clamp to non-negative (binding cannot be negative)
    y = np.maximum(y, 0.1)

    return x, y


# ---------------------------------------------------------------------------
# Test: F-test formula validation against hand-computed values
# ---------------------------------------------------------------------------


def test_f_test_formula_matches_manual_calculation() -> None:
    """Validate F-test implementation against manual calculation per M&C 2003.

    This test:
    1. Fits one-site and two-site models to synthetic data
    2. Manually computes F-statistic using the textbook formula
    3. Verifies compare_models() returns matching F-statistic and p-value

    Reference: Motulsky & Christopoulos (2003), Chapter 12, Equation 12.1
    """
    x, y = make_two_site_test_data(seed=2003)

    # Fit both models
    result_one = Fit("one_site_binding", x, y, weights="uniform").run()
    result_two = Fit("two_site_binding", x, y, weights="uniform").run()

    # Extract RSS and degrees of freedom
    rss_simple = result_one.rss
    rss_complex = result_two.rss
    n_obs = result_one.n_obs
    p_simple = len(result_one.params)  # 2 parameters
    p_complex = len(result_two.params)  # 4 parameters

    # Manual F-test calculation (M&C 2003, Eq. 12.1)
    df_numerator = p_complex - p_simple  # 4 - 2 = 2
    df_denominator = n_obs - p_complex  # 30 - 4 = 26

    f_stat_manual = ((rss_simple - rss_complex) / df_numerator) / (rss_complex / df_denominator)

    p_value_manual = float(stats.f.sf(f_stat_manual, df_numerator, df_denominator))

    # Run compare_models
    comparison = compare_models([result_one, result_two])

    # Verify F-test was computed (models are nested)
    assert comparison.f_test is not None, (
        "F-test should be computed for nested models (one_site ⊂ two_site)"
    )

    # Verify degrees of freedom
    assert comparison.f_test.df_numerator == df_numerator, (
        f"df_numerator: expected {df_numerator}, got {comparison.f_test.df_numerator}"
    )
    assert comparison.f_test.df_denominator == df_denominator, (
        f"df_denominator: expected {df_denominator}, got {comparison.f_test.df_denominator}"
    )

    # Verify F-statistic (tolerance 1e-6 for numerical precision)
    assert np.isclose(comparison.f_test.f_statistic, f_stat_manual, rtol=1e-6), (
        f"F-statistic mismatch:\n"
        f"  Expected (manual): {f_stat_manual:.6f}\n"
        f"  Got (compare.py):  {comparison.f_test.f_statistic:.6f}\n"
        f"  RSS_simple:        {rss_simple:.6f}\n"
        f"  RSS_complex:       {rss_complex:.6f}\n"
        f"  df_num:            {df_numerator}\n"
        f"  df_den:            {df_denominator}"
    )

    # Verify p-value (tolerance 1e-6)
    assert np.isclose(comparison.f_test.p_value, p_value_manual, rtol=1e-6), (
        f"p-value mismatch:\n"
        f"  Expected (manual): {p_value_manual:.6e}\n"
        f"  Got (compare.py):  {comparison.f_test.p_value:.6e}\n"
        f"  F-statistic:       {f_stat_manual:.6f}"
    )

    # Verify RSS values are stored correctly
    assert np.isclose(comparison.f_test.rss_simpler, rss_simple, rtol=1e-10), (
        f"RSS_simpler mismatch: {comparison.f_test.rss_simpler} vs {rss_simple}"
    )

    assert np.isclose(comparison.f_test.rss_complex, rss_complex, rtol=1e-10), (
        f"RSS_complex mismatch: {comparison.f_test.rss_complex} vs {rss_complex}"
    )


def test_f_test_prefers_complex_model_for_true_two_site_data() -> None:
    """For true two-site data, F-test should prefer the complex model.

    When data are generated from a two-site binding model, the F-test should
    yield p < 0.05, indicating the additional parameters are justified.
    """
    x, y = make_two_site_test_data(seed=2003)

    result_one = Fit("one_site_binding", x, y, weights="uniform").run()
    result_two = Fit("two_site_binding", x, y, weights="uniform").run()

    comparison = compare_models([result_one, result_two])

    assert comparison.f_test is not None
    assert comparison.f_test.p_value < 0.05, (
        f"For true two-site data, expected p < 0.05 but got p = "
        f"{comparison.f_test.p_value:.4e}. The two-site model should be "
        f"statistically preferred."
    )
    assert comparison.f_test.preferred_model == "complex", (
        f"Expected 'complex' model to be preferred, got '{comparison.f_test.preferred_model}'"
    )


def test_f_test_does_not_prefer_complex_for_one_site_data() -> None:
    """For true one-site data, F-test should NOT prefer the complex model.

    When data are generated from a one-site binding model, the F-test should
    yield p > 0.05, indicating the simpler model is adequate.
    """
    # Generate true one-site data
    rng = np.random.default_rng(2024)
    x = np.logspace(-1, 3, 30)
    y_true = one_site_binding(x, Bmax=100.0, Kd=10.0)
    noise = 0.03 * y_true * rng.standard_normal(len(x))
    y = np.maximum(y_true + noise, 0.1)

    result_one = Fit("one_site_binding", x, y, weights="uniform").run()
    result_two = Fit("two_site_binding", x, y, weights="uniform").run()

    comparison = compare_models([result_one, result_two])

    assert comparison.f_test is not None
    assert comparison.f_test.p_value > 0.05, (
        f"For true one-site data, expected p > 0.05 but got p = "
        f"{comparison.f_test.p_value:.4e}. The simpler model should be "
        f"adequate."
    )
    assert comparison.f_test.preferred_model == "simpler", (
        f"Expected 'simpler' model to be preferred for one-site data, got "
        f"'{comparison.f_test.preferred_model}'"
    )


def test_f_test_reduces_rss() -> None:
    """Complex model must have RSS <= simpler model (nested models property).

    For properly nested models, adding parameters cannot increase RSS.
    """
    x, y = make_two_site_test_data(seed=2003)

    result_one = Fit("one_site_binding", x, y, weights="uniform").run()
    result_two = Fit("two_site_binding", x, y, weights="uniform").run()

    comparison = compare_models([result_one, result_two])

    assert comparison.f_test is not None
    assert comparison.f_test.rss_complex <= comparison.f_test.rss_simpler, (
        f"Complex model RSS ({comparison.f_test.rss_complex:.6f}) should be "
        f"<= simpler model RSS ({comparison.f_test.rss_simpler:.6f}) for "
        f"nested models"
    )


# ---------------------------------------------------------------------------
# Edge case: F-test with different noise levels
# ---------------------------------------------------------------------------


def test_f_test_with_low_noise() -> None:
    """With low noise, F-test should strongly prefer complex model.

    Low noise (1%) makes the two-site signature clearer, yielding a more
    significant F-test result.
    """
    rng = np.random.default_rng(999)
    x = np.logspace(-1, 3, 30)
    y_true = two_site_binding(x, Bmax=80.0, Kd=2.0, Bmax2=40.0, Kd2=50.0)
    noise = 0.01 * y_true * rng.standard_normal(len(x))  # 1% noise
    y = np.maximum(y_true + noise, 0.1)

    result_one = Fit("one_site_binding", x, y, weights="uniform").run()
    result_two = Fit("two_site_binding", x, y, weights="uniform").run()

    comparison = compare_models([result_one, result_two])

    assert comparison.f_test is not None
    assert comparison.f_test.p_value < 0.01, (
        f"With 1% noise, expected highly significant F-test (p < 0.01) but "
        f"got p = {comparison.f_test.p_value:.4e}"
    )


# ---------------------------------------------------------------------------
# Validation: F-test formula components
# ---------------------------------------------------------------------------


def test_f_test_components_consistency() -> None:
    """Verify internal consistency of F-test result components.

    The F-statistic, RSS values, and degrees of freedom must satisfy the
    defining equation:
        F = ((RSS_simple - RSS_complex) / df_num) / (RSS_complex / df_den)
    """
    x, y = make_two_site_test_data(seed=2003)

    result_one = Fit("one_site_binding", x, y, weights="uniform").run()
    result_two = Fit("two_site_binding", x, y, weights="uniform").run()

    comparison = compare_models([result_one, result_two])
    f_test = comparison.f_test

    assert f_test is not None

    # Recompute F from stored components
    f_recomputed = ((f_test.rss_simpler - f_test.rss_complex) / f_test.df_numerator) / (
        f_test.rss_complex / f_test.df_denominator
    )

    assert np.isclose(f_test.f_statistic, f_recomputed, rtol=1e-10), (
        f"F-statistic ({f_test.f_statistic:.6f}) does not match recomputed "
        f"value ({f_recomputed:.6f}) from stored components"
    )

    # Verify p-value from F-distribution
    p_recomputed = float(stats.f.sf(f_test.f_statistic, f_test.df_numerator, f_test.df_denominator))

    assert np.isclose(f_test.p_value, p_recomputed, rtol=1e-6), (
        f"p-value ({f_test.p_value:.6e}) does not match recomputed value "
        f"({p_recomputed:.6e}) from F-distribution"
    )
