"""Published-reference validation for global shared-parameter fitting.

Validates GlobalFit against the global/shared-parameter fitting methodology
described in Motulsky & Christopoulos (2004), Chapter 25: "Comparing dose-
response curves for several datasets."

Reference
---------
Motulsky, H., & Christopoulos, A. (2004). Fitting Models to Biological Data
Using Linear and Nonlinear Regression: A Practical Guide to Curve Fitting.
Oxford University Press.
ISBN: 0-19-517182-9

Chapter 25: Comparing dose-response curves for several datasets
  - Section "Fitting one curve to multiple datasets" (pp. 243-247)
  - Section "Sharing parameters among datasets" (pp. 248-252)
  - Section "Extra sum-of-squares F-test for comparing fits" (pp. 252-255)

Key concepts (M&C Chapter 25):
  1. Multiple datasets fitted simultaneously with a joint objective function.
  2. Some parameters constrained equal across all datasets (SHARED).
     Example: Top and Bottom of dose-response curves when comparing agonists
     on the same response scale (0% to 100%).
  3. Other parameters estimated independently per dataset (LOCAL).
     Example: EC50 (potency) and HillSlope (cooperativity) differ per agonist.
  4. Extra sum-of-squares F-test compares:
     - Restricted model: joint fit with shared parameters
     - Full model: each dataset fitted independently (all local)
     F = ((RSS_restricted - RSS_full) / df_num) / (RSS_full / df_den)
     where:
       df_num = n_shared_params * (n_datasets - 1)
       df_den = n_total_observations - n_datasets * n_params_per_dataset

Application
-----------
Canonical pharmacology example: comparing agonist dose-response curves across
three cell lines expressing the same receptor. Top (100% response) and Bottom
(0% response) are identical by definition, while EC50 and HillSlope vary by
cell line due to differences in receptor density and coupling efficiency.

Related reference:
  DeLean, A., Munson, P.J. & Rodbard, D. (1978). Simultaneous analysis of
  families of sigmoidal curves. Am. J. Physiol., 235(2), E97-E102.
  -- The original ALLFIT paper introducing shared-parameter 4PL fitting.

Usage
-----
    pytest tests/test_global_fit_reference.py -v
    pytest tests/test_global_fit_reference.py -v -k "sharing_justified"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy import stats

from openfit import GlobalFit

# ---------------------------------------------------------------------------
# Synthetic dataset: dose-response with shared Top/Bottom across 3 cell lines
# ---------------------------------------------------------------------------


def _hill4p(x: np.ndarray, Bottom: float, Top: float, EC50: float, HillSlope: float) -> np.ndarray:
    """Reference 4PL equation (matching openfit.models.sigmoidal.Hill4P)."""
    x_safe = np.where(x == 0.0, 1e-300, x)
    ratio = np.exp(
        np.clip(HillSlope * (np.log(np.abs(EC50)) - np.log(np.abs(x_safe))), -700.0, 700.0)
    )
    return Bottom + (Top - Bottom) / (1.0 + ratio)


def make_shared_top_bottom_data(seed: int = 2504) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate 3 dose-response datasets with shared Top/Bottom.

    Mimics a Chapter 25 example: three agonists tested on the same receptor
    system. Response is measured on a common 0-100% scale (shared Top=100,
    shared Bottom=0), but EC50 and HillSlope differ per agonist.

    True parameters:
        All datasets: Bottom=0.0, Top=100.0
        Dataset 0 (high-potency agonist):    EC50=0.5,   HillSlope=1.0
        Dataset 1 (mid-potency agonist):     EC50=5.0,   HillSlope=1.5
        Dataset 2 (low-potency agonist):     EC50=50.0,  HillSlope=0.8

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        3 (x, y) pairs, 15 log-spaced concentrations each, 2% Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    x = np.logspace(-2, 3, 15)  # 0.01 to 1000 nM, 15 points

    shared_bottom = 0.0
    shared_top = 100.0
    local_params = [
        {"EC50": 0.5, "HillSlope": 1.0},
        {"EC50": 5.0, "HillSlope": 1.5},
        {"EC50": 50.0, "HillSlope": 0.8},
    ]

    datasets = []
    for lp in local_params:
        y_true = _hill4p(
            x, Bottom=shared_bottom, Top=shared_top, EC50=lp["EC50"], HillSlope=lp["HillSlope"]
        )
        noise = 0.02 * y_true * rng.standard_normal(len(x))
        y = y_true + noise
        y = np.maximum(y, 0.01)  # clamp positive for 1/y2 weights
        datasets.append((x.copy(), y))

    return datasets


def make_different_top_data(seed: int = 2505) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate 3 dose-response datasets with DIFFERENT Top values.

    Represents a case where sharing Top is NOT justified: e.g., partial
    agonists with different efficacies (60%, 100%, 140% of reference response).

    True parameters:
        All datasets: Bottom=0.0, EC50=5.0, HillSlope=1.0
        Dataset 0 (partial agonist):    Top=60.0
        Dataset 1 (full agonist):       Top=100.0
        Dataset 2 (super agonist):      Top=140.0

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        3 (x, y) pairs, 15 log-spaced concentrations each, 2% Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    x = np.logspace(-2, 3, 15)

    top_vals = [60.0, 100.0, 140.0]
    datasets = []
    for top in top_vals:
        y_true = _hill4p(x, Bottom=0.0, Top=top, EC50=5.0, HillSlope=1.0)
        noise = 0.02 * y_true * rng.standard_normal(len(x))
        y = y_true + noise
        y = np.maximum(y, 0.01)
        datasets.append((x.copy(), y))

    return datasets


# ---------------------------------------------------------------------------
# Test 1: Shared parameters recovered correctly
# ---------------------------------------------------------------------------


def test_shared_params_recovered_accurately() -> None:
    """Shared Top and Bottom are recovered within 5% of true values.

    Reference: M&C Chapter 25, Table example showing shared-parameter fit
    recovers the common Top/Bottom when they are truly identical.
    """
    datasets = make_shared_top_bottom_data(seed=2504)

    result = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top", "Bottom"],
        local=["EC50", "HillSlope"],
        weights="uniform",
        run_f_test=True,
    ).run()

    # Top should be close to 100 (true value)
    assert abs(result.shared_params["Top"] - 100.0) / 100.0 < 0.05, (
        f"Shared Top = {result.shared_params['Top']:.2f}, expected ~100.0"
    )
    # Bottom should be close to 0 (true value)
    assert abs(result.shared_params["Bottom"] - 0.0) < 5.0, (
        f"Shared Bottom = {result.shared_params['Bottom']:.2f}, expected ~0.0"
    )


# ---------------------------------------------------------------------------
# Test 2: Local parameters differ across datasets as expected
# ---------------------------------------------------------------------------


def test_local_params_differ_across_datasets() -> None:
    """Local EC50 and HillSlope differ per dataset, matching true values.

    Reference: M&C Chapter 25, showing that local parameters (potency, slope)
    are estimated independently and reflect the true per-dataset differences.
    """
    datasets = make_shared_top_bottom_data(seed=2504)

    result = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top", "Bottom"],
        local=["EC50", "HillSlope"],
        weights="uniform",
        run_f_test=False,
    ).run()

    # True EC50 values: 0.5, 5.0, 50.0
    true_ec50 = [0.5, 5.0, 50.0]
    true_hill = [1.0, 1.5, 0.8]

    assert len(result.local_params) == 3

    for i in range(3):
        ec50_fit = result.local_params[i]["EC50"]
        hill_fit = result.local_params[i]["HillSlope"]

        # EC50 within 20% relative tolerance
        assert abs(ec50_fit - true_ec50[i]) / true_ec50[i] < 0.20, (
            f"Dataset {i}: EC50 = {ec50_fit:.3f}, expected ~{true_ec50[i]}"
        )
        # HillSlope within 20% relative tolerance
        assert abs(hill_fit - true_hill[i]) / true_hill[i] < 0.20, (
            f"Dataset {i}: HillSlope = {hill_fit:.3f}, expected ~{true_hill[i]}"
        )

    # EC50 values should be ordered: ds0 < ds1 < ds2
    ec50_vals = [result.local_params[i]["EC50"] for i in range(3)]
    assert ec50_vals[0] < ec50_vals[1] < ec50_vals[2], f"EC50 ordering violated: {ec50_vals}"


# ---------------------------------------------------------------------------
# Test 3: F-test confirms sharing is justified when Top/Bottom truly shared
# ---------------------------------------------------------------------------


def test_f_test_sharing_justified_when_params_truly_shared() -> None:
    """F-test p > 0.05 when datasets truly share Top and Bottom.

    Reference: M&C Chapter 25, extra sum-of-squares F-test. When the
    constraint (shared Top/Bottom) does not significantly worsen the fit,
    the F-test p-value exceeds 0.05 and sharing is justified.
    """
    datasets = make_shared_top_bottom_data(seed=2504)

    result = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top", "Bottom"],
        local=["EC50", "HillSlope"],
        weights="uniform",
        run_f_test=True,
    ).run()

    assert result.f_test_sharing is not None, "F-test should be computed"
    assert result.f_test_sharing.sharing_justified, (
        f"F-test rejected sharing (p={result.f_test_sharing.p_value:.4f}) "
        f"but Top/Bottom are truly shared. Sharing should be justified."
    )
    assert result.f_test_sharing.p_value > 0.05, (
        f"Expected p > 0.05, got p = {result.f_test_sharing.p_value:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: F-test rejects sharing when Top is truly different
# ---------------------------------------------------------------------------


def test_f_test_rejects_sharing_when_top_differs() -> None:
    """F-test p < 0.05 when datasets have genuinely different Top values.

    Reference: M&C Chapter 25, extra sum-of-squares F-test. When sharing
    forces equal parameters that are truly different (partial agonists),
    the F-test detects the significant worsening and rejects sharing.
    """
    datasets = make_different_top_data(seed=2505)

    result = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top"],
        local=["Bottom", "EC50", "HillSlope"],
        weights="uniform",
        run_f_test=True,
    ).run()

    assert result.f_test_sharing is not None, "F-test should be computed"
    assert not result.f_test_sharing.sharing_justified, (
        f"F-test accepted sharing (p={result.f_test_sharing.p_value:.4f}) "
        f"but Top values are truly different (60, 100, 140). "
        f"Sharing should be rejected."
    )
    assert result.f_test_sharing.p_value < 0.05, (
        f"Expected p < 0.05, got p = {result.f_test_sharing.p_value:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5: F-test formula validated against manual computation
# ---------------------------------------------------------------------------


def test_f_test_formula_matches_manual_calculation() -> None:
    """Verify F-statistic and degrees of freedom match the M&C Chapter 25 formula.

    F = ((RSS_restricted - RSS_full) / df_num) / (RSS_full / df_den)

    where:
        df_num = n_shared * (n_datasets - 1)
        df_den = n_total - n_datasets * (n_shared + n_local)

    Reference: M&C Chapter 25, Equation for extra sum-of-squares F-test.
    """
    datasets = make_shared_top_bottom_data(seed=2504)

    result = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top", "Bottom"],
        local=["EC50", "HillSlope"],
        weights="uniform",
        run_f_test=True,
    ).run()

    ft = result.f_test_sharing
    assert ft is not None

    # Verify degrees of freedom
    n_datasets = 3
    n_shared = 2  # Top, Bottom
    n_local = 2  # EC50, HillSlope
    n_total = sum(len(y) for _, y in datasets)  # 3 * 15 = 45

    expected_df_num = n_shared * (n_datasets - 1)  # 2 * 2 = 4
    expected_df_den = n_total - n_datasets * (n_shared + n_local)  # 45 - 3*4 = 33

    assert ft.df_numerator == expected_df_num, (
        f"df_numerator: expected {expected_df_num}, got {ft.df_numerator}"
    )
    assert ft.df_denominator == expected_df_den, (
        f"df_denominator: expected {expected_df_den}, got {ft.df_denominator}"
    )

    # Recompute F-statistic from stored RSS values
    f_recomputed = ((ft.rss_shared - ft.rss_independent) / ft.df_numerator) / (
        ft.rss_independent / ft.df_denominator
    )

    assert np.isclose(ft.f_statistic, f_recomputed, rtol=1e-10), (
        f"F-statistic ({ft.f_statistic:.6f}) does not match recomputed "
        f"({f_recomputed:.6f}) from stored RSS values"
    )

    # Verify p-value from F-distribution
    p_recomputed = float(stats.f.sf(ft.f_statistic, ft.df_numerator, ft.df_denominator))

    assert np.isclose(ft.p_value, p_recomputed, rtol=1e-6), (
        f"p-value ({ft.p_value:.6e}) does not match recomputed "
        f"({p_recomputed:.6e}) from F-distribution"
    )


# ---------------------------------------------------------------------------
# Test 6: Shared parameters are identical across all datasets
# ---------------------------------------------------------------------------


def test_shared_params_identical_in_all_params_per_dataset() -> None:
    """all_params_per_dataset[i] uses the same shared values for every i.

    Reference: M&C Chapter 25 core principle -- shared parameters are
    constrained to a single value used by all datasets.
    """
    datasets = make_shared_top_bottom_data(seed=2504)

    result = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top", "Bottom"],
        local=["EC50", "HillSlope"],
        weights="uniform",
        run_f_test=False,
    ).run()

    assert len(result.all_params_per_dataset) == 3

    top_vals = [result.all_params_per_dataset[i]["Top"] for i in range(3)]
    bottom_vals = [result.all_params_per_dataset[i]["Bottom"] for i in range(3)]

    # All datasets must use the exact same shared parameter values
    assert top_vals[0] == top_vals[1] == top_vals[2], (
        f"Shared Top differs across datasets: {top_vals}"
    )
    assert bottom_vals[0] == bottom_vals[1] == bottom_vals[2], (
        f"Shared Bottom differs across datasets: {bottom_vals}"
    )

    # And they must match result.shared_params
    assert top_vals[0] == result.shared_params["Top"]
    assert bottom_vals[0] == result.shared_params["Bottom"]


# ---------------------------------------------------------------------------
# Test 7: RSS decreases (or stays same) going from shared to independent
# ---------------------------------------------------------------------------


def test_rss_independent_leq_rss_shared() -> None:
    """Independent fits must have RSS <= shared fit RSS (nesting property).

    The independent (full) model is a superset of the shared (restricted)
    model: every shared-parameter configuration is reachable by the
    independent model. Therefore RSS_full <= RSS_restricted.

    Reference: M&C Chapter 25, extra sum-of-squares principle.
    """
    datasets = make_shared_top_bottom_data(seed=2504)

    result = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top", "Bottom"],
        local=["EC50", "HillSlope"],
        weights="uniform",
        run_f_test=True,
    ).run()

    ft = result.f_test_sharing
    assert ft is not None

    assert ft.rss_independent <= ft.rss_shared + 1e-10, (
        f"Independent RSS ({ft.rss_independent:.6f}) should be <= shared RSS ({ft.rss_shared:.6f})"
    )
    # F-statistic should be non-negative
    assert ft.f_statistic >= 0.0, f"F-statistic should be >= 0, got {ft.f_statistic:.6f}"


# ---------------------------------------------------------------------------
# Test 8: Goodness-of-fit metrics are reasonable
# ---------------------------------------------------------------------------


def test_r_squared_per_dataset_reasonable() -> None:
    """R^2 per dataset should be high (>0.95) for well-behaved dose-response data.

    Reference: M&C Chapter 25 discussion of goodness-of-fit for shared fits.
    """
    datasets = make_shared_top_bottom_data(seed=2504)

    result = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top", "Bottom"],
        local=["EC50", "HillSlope"],
        weights="uniform",
        run_f_test=False,
    ).run()

    assert len(result.r_squared_per_dataset) == 3
    for i, r2 in enumerate(result.r_squared_per_dataset):
        assert r2 > 0.95, f"Dataset {i}: R^2 = {r2:.4f} < 0.95, expected good fit"


# ---------------------------------------------------------------------------
# Test 9: F-test with weighted data
# ---------------------------------------------------------------------------


def test_f_test_with_1y2_weights() -> None:
    """F-test sharing justification also works with 1/y^2 weighting.

    Reference: M&C Chapter 25 notes that weighted regression uses the
    weighted sum-of-squares in the F-test numerator and denominator.
    """
    datasets = make_shared_top_bottom_data(seed=2504)

    result = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top", "Bottom"],
        local=["EC50", "HillSlope"],
        weights="1/y2",
        run_f_test=True,
    ).run()

    ft = result.f_test_sharing
    assert ft is not None

    # With 1/y^2 weights, sharing should still be justified
    assert ft.sharing_justified, (
        f"With 1/y2 weights, sharing should be justified but p={ft.p_value:.4f}"
    )

    # Verify F-statistic is non-negative and finite
    assert np.isfinite(ft.f_statistic) and ft.f_statistic >= 0.0
