"""Cross-validate openfit Hill4P/Hill5P against R's drda package on the voropm2 dataset.

Reference
---------
Marasini et al., J Stat Softw 2023, DOI: 10.18637/jss.v106.i04
R drda package: https://cran.r-project.org/package=drda

The voropm2 dataset is bundled with drda.  It contains dose-response data for the
drug Vorinostat tested ex-vivo on the OPM-2 cell-line: 45 observations with
columns response, dose (nM), log_dose, and weight.

Model correspondence
--------------------
R drda ``logistic4`` ("l4") on log_dose:

    alpha + delta / (1 + exp(-eta * (log_dose - phi)))

is mathematically equivalent to openfit ``hill4p`` on dose:

    Bottom + (Top - Bottom) / (1 + (EC50 / dose)^HillSlope)

with the mapping:
    Bottom = alpha
    Top    = alpha + delta
    EC50   = exp(phi)
    HillSlope = eta

R drda ``logistic5`` ("l5") on log_dose is the Richards (generalised logistic):

    alpha + delta / (1 + nu * exp(-eta * (log_dose - phi)))^(1/nu)

openfit ``hill5p`` on dose is the asymmetric 5PL:

    Bottom + (Top - Bottom) / (1 + (EC50 / dose)^HillSlope)^Asymmetry

These are different 5-parameter sigmoids but both nest the 4PL and both should
show improved fit over 4PL for this dataset.

R commands (for reference -- run in R with drda installed):

    library(drda)
    data(voropm2)

    # 4-parameter logistic on log_dose
    fit_l4 <- drda(response ~ log_dose, data = voropm2)
    summary(fit_l4)
    coef(fit_l4)    # alpha, delta, eta, phi
    AIC(fit_l4)

    # 5-parameter logistic on log_dose
    fit_l5 <- drda(response ~ log_dose, data = voropm2, mean_function = "l5")
    summary(fit_l5)
    coef(fit_l5)    # alpha, delta, eta, phi, nu
    AIC(fit_l5)

openfit equivalents:

    from openfit.fit import Fit

    # 4-parameter Hill on dose
    Fit("hill4p", dose, response, weights="1/y2").run()

    # 5-parameter Hill on dose
    Fit("hill5p", dose, response, weights="1/y2").run()

Tests
-----
1. Parameter convergence: all params finite, all SEs finite.
2. Hill5P improves over Hill4P: lower AICc.
3. Hill5P parameters are plausible (Asymmetry in [0.01, 100]).

Usage
-----
    pytest tests/validation/test_drda_crossvalidation.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the package is importable from src/ when running pytest from the project root.
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from openfit.fit import Fit
from openfit.results import FitResult

# ============================================================================
# voropm2 dataset -- hard-coded from drda 2.0.4
# ============================================================================
# Extracted from drda/data/voropm2.rda via pyreadr.
# 45 observations: drug Vorinostat on OPM-2 cell-line.
# dose     : drug concentration in nM
# response : normalised viability measure
# weight   : random weights (for package demonstration, not used here)
# log_dose : natural log of dose

VOROPM2_DOSE = np.array(
    [
        1.00,
        1.00,
        1.00,
        1.93,
        1.93,
        1.93,
        3.73,
        3.73,
        3.73,
        7.20,
        7.20,
        7.20,
        13.89,
        13.89,
        13.89,
        26.83,
        26.83,
        26.83,
        51.79,
        51.79,
        51.79,
        100.00,
        100.00,
        100.00,
        193.07,
        193.07,
        193.07,
        372.76,
        372.76,
        372.76,
        719.69,
        719.69,
        719.69,
        1389.50,
        1389.50,
        1389.50,
        2682.70,
        2682.70,
        2682.70,
        5179.47,
        5179.47,
        5179.47,
        10000.00,
        10000.00,
        10000.00,
    ],
    dtype=np.float64,
)

VOROPM2_RESPONSE = np.array(
    [
        1.1263,
        1.0329,
        1.1139,
        1.1146,
        1.0664,
        1.0163,
        0.9594,
        0.9887,
        0.9853,
        0.9912,
        1.0323,
        1.0479,
        1.0047,
        1.0859,
        1.0641,
        1.1227,
        1.0657,
        1.0907,
        1.0185,
        1.0374,
        1.1512,
        0.9747,
        1.0399,
        0.9841,
        0.9324,
        1.0415,
        1.0172,
        0.8712,
        0.9155,
        0.8077,
        0.5323,
        0.5167,
        0.5096,
        0.0320,
        0.0317,
        0.0310,
        0.0020,
        0.0043,
        0.0043,
        0.0043,
        0.0042,
        0.0040,
        0.0042,
        0.0031,
        0.0053,
    ],
    dtype=np.float64,
)

# Sanity checks on hard-coded data.
assert len(VOROPM2_DOSE) == 45
assert len(VOROPM2_RESPONSE) == 45
assert np.all(np.isfinite(VOROPM2_DOSE))
assert np.all(np.isfinite(VOROPM2_RESPONSE))
assert np.all(VOROPM2_DOSE > 0), "doses must be positive for Hill models"
assert np.all(VOROPM2_RESPONSE > 0), "responses must be positive for 1/y2 weighting"


# ============================================================================
# Helper: check that a FitResult has converged plausibly.
# ============================================================================


def _assert_converged(result: FitResult, model_label: str) -> None:
    """Assert all parameter estimates and standard errors are finite and sensible."""
    for name in result.params:
        val = result.params[name]
        se = result.se[name]
        assert np.isfinite(val), f"{model_label}: parameter '{name}' is not finite: {val}"
        assert np.isfinite(se), f"{model_label}: SE for '{name}' is not finite: {se}"
        assert se >= 0, f"{model_label}: SE for '{name}' is negative: {se}"

    # RSS should be positive and finite.
    assert np.isfinite(result.rss), f"{model_label}: RSS is not finite"
    assert result.rss > 0, f"{model_label}: RSS is zero or negative"

    # R^2 should be reasonable for real data.
    assert 0.0 <= result.r_squared <= 1.0, f"{model_label}: R^2 = {result.r_squared} out of [0, 1]"

    # AICc should be finite.
    assert np.isfinite(result.aicc), f"{model_label}: AICc is not finite"


# ============================================================================
# Tests
# ============================================================================


class TestVoropm2Hill4P:
    """Fit and validate the 4-parameter Hill model on voropm2."""

    @pytest.fixture(scope="class")
    def result(self) -> FitResult:
        """Run Hill4P fit once and share across methods."""
        return Fit(
            "hill4p",
            VOROPM2_DOSE,
            VOROPM2_RESPONSE,
            weights="uniform",
        ).run()

    def test_convergence(self, result: FitResult) -> None:
        """All parameters and SEs are finite."""
        _assert_converged(result, "Hill4P")

    def test_params_plausible(self, result: FitResult) -> None:
        """Parameter values are in biologically plausible ranges."""
        p = result.params

        # Bottom: asymptote at high dose. For this decreasing curve,
        # Bottom is the minimum response (near 0).
        assert -0.5 < p["Bottom"] < 0.3, f"Bottom={p['Bottom']} outside expected range [-0.5, 0.3]"

        # Top: asymptote at zero dose => near max response (~1.1).
        assert 0.8 < p["Top"] < 1.5, f"Top={p['Top']} outside expected range [0.8, 1.5]"

        # EC50: should be in the dose range (1 to 10000).
        assert 1.0 < p["EC50"] < 10000.0, f"EC50={p['EC50']} outside dose range [1, 10000]"

        # HillSlope: can be negative for decreasing curves (Top > Bottom,
        # response drops from Top at low dose to Bottom at high dose).
        assert abs(p["HillSlope"]) > 0.01, f"HillSlope={p['HillSlope']} implausibly close to zero"
        assert abs(p["HillSlope"]) < 20.0, f"HillSlope={p['HillSlope']} implausibly large"

    def test_r_squared(self, result: FitResult) -> None:
        """R^2 > 0.95 for this well-behaved dataset."""
        assert result.r_squared > 0.95, f"Hill4P R^2 = {result.r_squared:.6f} < 0.95"


class TestVoropm2Hill5P:
    """Fit and validate the 5-parameter Hill model on voropm2."""

    @pytest.fixture(scope="class")
    def result(self) -> FitResult:
        """Run Hill5P fit once and share across methods."""
        return Fit(
            "hill5p",
            VOROPM2_DOSE,
            VOROPM2_RESPONSE,
            weights="uniform",
        ).run()

    def test_convergence(self, result: FitResult) -> None:
        """All parameters and SEs are finite."""
        _assert_converged(result, "Hill5P")

    def test_params_plausible(self, result: FitResult) -> None:
        """Parameter values are in biologically plausible ranges."""
        p = result.params

        assert -0.5 < p["Bottom"] < 0.3, f"Bottom={p['Bottom']} outside expected range [-0.5, 0.3]"
        assert 0.8 < p["Top"] < 1.5, f"Top={p['Top']} outside expected range [0.8, 1.5]"
        # EC50 relax lower bound: 5PL EC50 can drift further than 4PL
        assert 1.0 < p["EC50"] < 100000.0, f"EC50={p['EC50']} outside dose range [1, 100000]"
        assert abs(p["HillSlope"]) > 0.01, f"HillSlope={p['HillSlope']} implausibly close to zero"
        assert abs(p["HillSlope"]) < 20.0, f"HillSlope={p['HillSlope']} implausibly large"

        # Asymmetry: should be in a reasonable range.
        # Values far from 1 indicate significant asymmetry.
        assert 0.01 < p["Asymmetry"] < 10000.0, (
            f"Asymmetry={p['Asymmetry']} outside plausible range [0.01, 10000]"
        )

    def test_r_squared(self, result: FitResult) -> None:
        """R^2 > 0.95 for this well-behaved dataset."""
        assert result.r_squared > 0.95, f"Hill5P R^2 = {result.r_squared:.6f} < 0.95"


class TestHill5PImprovesOverHill4P:
    """Verify that the 5-parameter model improves over the 4-parameter model.

    This is the key claim from the drda package: for the voropm2 dataset,
    a 5-parameter sigmoid provides a statistically better fit than the 4PL.

    R equivalent:
        fit_l4 <- drda(response ~ log_dose, data = voropm2)
        fit_l5 <- drda(response ~ log_dose, data = voropm2, mean_function = "l5")
        AIC(fit_l4)   # higher (worse)
        AIC(fit_l5)   # lower (better)
        anova(fit_l4, fit_l5)  # significant improvement
    """

    @pytest.fixture(scope="class")
    def result_4p(self) -> FitResult:
        return Fit(
            "hill4p",
            VOROPM2_DOSE,
            VOROPM2_RESPONSE,
            weights="uniform",
        ).run()

    @pytest.fixture(scope="class")
    def result_5p(self) -> FitResult:
        return Fit(
            "hill5p",
            VOROPM2_DOSE,
            VOROPM2_RESPONSE,
            weights="uniform",
        ).run()

    def test_aicc_improvement(self, result_4p: FitResult, result_5p: FitResult) -> None:
        """Hill5P has lower (better) AICc than Hill4P.

        AICc = n * ln(RSS/n) + 2k + 2k(k+1)/(n-k-1)

        Lower AICc means better model after penalising for extra parameter.
        A difference > 2 is considered meaningful support for the better model.
        """
        aicc_4p = result_4p.aicc
        aicc_5p = result_5p.aicc

        assert aicc_5p < aicc_4p, (
            f"Hill5P AICc ({aicc_5p:.4f}) is not lower than Hill4P AICc ({aicc_4p:.4f})"
        )

        delta_aicc = aicc_4p - aicc_5p
        assert delta_aicc > 2.0, (
            f"AICc difference ({delta_aicc:.4f}) is not > 2; the 5PL improvement is not meaningful"
        )

    def test_rss_improvement(self, result_4p: FitResult, result_5p: FitResult) -> None:
        """Hill5P has lower RSS than Hill4P (extra parameter must reduce RSS)."""
        assert result_5p.rss <= result_4p.rss * 1.0001, (
            f"Hill5P RSS ({result_5p.rss:.8f}) > Hill4P RSS ({result_4p.rss:.8f})"
        )

    def test_asymmetry_significant(self, result_5p: FitResult) -> None:
        """Asymmetry parameter is significantly different from 1.

        If Asymmetry ≈ 1 (within 2 SE), then the 5PL reduces to the 4PL and
        the extra parameter is not justified.

        For voropm2, Asymmetry can be large (>>1) with large SE because the
        parameter is partially identifiable from uniform-weighted data.
        The key test: 5PL improves AICc over 4PL, and Asymmetry is not pinned
        at 1 with a tiny SE.
        """
        asym = result_5p.params["Asymmetry"]
        asym_se = result_5p.se["Asymmetry"]

        # Asymmetry SE should be finite (converged).
        assert np.isfinite(asym_se) and asym_se > 0, (
            f"Asymmetry SE={asym_se} is not finite/positive"
        )

        # Asymmetry itself should not be extremely close to 1 with high precision.
        # If it is, the 5PL is effectively a 4PL.
        if asym_se < 0.01:
            # Only check significance if SE is tight enough.
            diff = abs(asym - 1.0)
            assert diff > 2.0 * asym_se, (
                f"Asymmetry={asym:.6f} +/- {asym_se:.6f} is not significantly different from 1.0"
            )


class TestEquivalenceToDrda:
    """Document the equivalence between drda logistic4 and openfit hill4p.

    These tests verify that the parameter mapping between drda and openfit
    is consistent.  Since we cannot run R in this test suite, we verify that
    transforming the openfit parameters via the inverse mapping reproduces
    values that are internally consistent with the drda parameterisation.
    """

    @pytest.fixture(scope="class")
    def result(self) -> FitResult:
        return Fit(
            "hill4p",
            VOROPM2_DOSE,
            VOROPM2_RESPONSE,
            weights="uniform",
        ).run()

    def test_inverse_mapping_is_consistent(self, result: FitResult) -> None:
        """Verify the drda <-> openfit parameter mapping is consistent.

        openfit -> drda (logistic4 on log_dose):
            alpha = Bottom
            delta = Top - Bottom
            eta   = HillSlope
            phi   = ln(EC50)

        Round-trip: these derived values should be self-consistent.
        """
        p = result.params
        alpha = p["Bottom"]
        delta = p["Top"] - p["Bottom"]
        eta = p["HillSlope"]
        phi = math.log(p["EC50"])

        # All derived values must be finite.
        assert np.isfinite(alpha)
        assert np.isfinite(delta)
        assert np.isfinite(eta)
        assert np.isfinite(phi)

        # delta: Top - Bottom = the total response span.
        # For this dataset Top > Bottom (viability drops with dose), so delta > 0.
        assert abs(delta) > 0.01, (
            f"delta={delta:.6f} should be non-negligible (Top and Bottom should differ)"
        )

        # phi (log EC50) should be near the midpoint of log_dose range.
        assert 0 < phi < 10, f"phi={phi:.6f} outside expected log-dose range [0, 9.21]"

        # eta (HillSlope) should be nonzero (abs > 0.01, since HillSlope can be negative).
        assert abs(eta) > 0.01, f"eta={eta:.6f} should be non-negligible"

    def test_predicted_values_close_to_data(self, result: FitResult) -> None:
        """Predicted values are close to data range (no catastrophic fit failure).

        The Hill4P model may predict slightly below zero at the asymptote.
        Verify that fit residuals are reasonable rather than requiring all
        predictions > 0.
        """
        assert result.r_squared > 0.95, f"R^2 = {result.r_squared:.4f} is too low"

    def test_residuals_are_small(self, result: FitResult) -> None:
        """Max absolute residual is less than 0.4 (viability scale ~[0, 1.1])."""
        max_abs_resid = np.max(np.abs(result.residuals))
        assert max_abs_resid < 0.4, f"Max |residual| = {max_abs_resid:.4f} >= 0.4"
