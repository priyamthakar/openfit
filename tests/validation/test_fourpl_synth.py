"""Synthetic 4PL and 5PL parameter recovery validation tests.

Verifies that openfit recovers known exact parameters from synthetic datasets
with controlled noise levels. Tolerances are noise-dependent:
- 0% noise: relative error < 1e-6 (or absolute < 1e-6 for zero-certified params)
- 1% noise: relative error < 2% (or absolute < 2 for zero-certified params)
- 5% noise: relative error < 10% (or absolute < 10 for zero-certified params)

Also validates:
- R^2 > 0.99 for all datasets
- Residuals are randomly distributed for noisy datasets (runs test p > 0.05)

Usage
-----
    pytest tests/validation/test_fourpl_synth.py -v
    pytest tests/validation/test_fourpl_synth.py -v -k "fourpl"
    pytest tests/validation/test_fourpl_synth.py -v -k "noise0"
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest
from scipy import stats

# Ensure the package is importable from src/ when running pytest from the project root.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2] / "src"))

from openfit.fit import Fit
from tests.validation.fourpl_certified_values import SYNTH_DATASETS

# ---------------------------------------------------------------------------
# Tolerances by noise level
# ---------------------------------------------------------------------------
# For 0% noise: machine precision
# For 1% noise: allow ~2x the noise level (statistical variation with n=50)
# For 5% noise: allow ~2x the noise level
_RELATIVE_TOLERANCE_MAP = {
    0.0: 1e-6,
    0.01: 0.02,  # 2% relative error
    0.05: 0.10,  # 10% relative error
}

# Absolute tolerance for zero-certified parameters (scaled by response range)
_ABSOLUTE_TOLERANCE_MAP = {
    0.0: 1e-6,
    0.01: 2.0,   # absolute error < 2 (2% of 100)
    0.05: 10.0,  # absolute error < 10 (10% of 100)
}

# R^2 threshold for all datasets
R2_MIN = 0.99

# Runs test p-value threshold
RUNS_TEST_P_MIN = 0.05

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _runs_test(residuals: np.ndarray) -> float:
    """Perform Wald-Wolfowitz runs test for randomness.

    Tests whether the sequence of positive/negative residuals is random.
    Returns p-value; p > 0.05 indicates no evidence of non-randomness.

    Parameters
    ----------
    residuals : np.ndarray
        Array of residuals (y_obs - y_pred).

    Returns
    -------
    float
        Two-tailed p-value from the runs test.
    """
    # Convert to binary: positive = 1, negative = 0
    signs = (residuals > 0).astype(int)
    n1 = np.sum(signs)
    n2 = np.sum(1 - signs)

    # Edge case: all same sign
    if n1 == 0 or n2 == 0:
        return 0.0

    # Count runs (transitions from 0->1 or 1->0)
    runs = 1 + np.sum(np.diff(signs) != 0)

    # Expected runs under H0 (randomness)
    n = n1 + n2
    expected = (2.0 * n1 * n2) / n + 1.0

    # Variance of runs
    variance = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n)) / (n**2 * (n - 1.0))
    if variance <= 0:
        return 1.0

    # Z-score
    z = (runs - expected) / np.sqrt(variance)

    # Two-tailed p-value
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return p_value


def _check_params(
    result: Any,
    certified: dict[str, float],
    noise_level: float,
) -> list[str]:
    """Check parameter recovery within noise-dependent tolerance.

    Parameters
    ----------
    result : FitResult
        Completed fit result.
    certified : dict[str, float]
        Certified parameter values.
    noise_level : float
        Noise level (0.0, 0.01, or 0.05).

    Returns
    -------
    list[str]
        List of failure messages (empty if all pass).
    """
    failures = []
    rel_tol = _RELATIVE_TOLERANCE_MAP[noise_level]
    abs_tol = _ABSOLUTE_TOLERANCE_MAP[noise_level]

    for pname, cert_val in certified.items():
        fitted_val = result.params[pname]
        if abs(cert_val) < 1e-10:
            # Use absolute error for zero-certified parameters
            abs_err = abs(fitted_val - cert_val)
            if abs_err > abs_tol:
                failures.append(
                    f"{pname}: fitted={fitted_val:.10e}, certified={cert_val:.10e}, "
                    f"abs_err={abs_err:.2e} (tol={abs_tol:.0e})"
                )
        else:
            rel_err = abs(fitted_val - cert_val) / abs(cert_val)
            if rel_err > rel_tol:
                failures.append(
                    f"{pname}: fitted={fitted_val:.10e}, certified={cert_val:.10e}, "
                    f"rel_err={rel_err:.2e} (tol={rel_tol:.0e})"
                )
    return failures


def _check_r_squared(result: Any, r2_min: float = R2_MIN) -> str | None:
    """Check R^2 exceeds minimum threshold.

    Parameters
    ----------
    result : FitResult
        Completed fit result.
    r2_min : float
        Minimum acceptable R^2 value.

    Returns
    -------
    str | None
        Failure message if R^2 < r2_min, else None.
    """
    if result.r_squared < r2_min:
        return f"R^2: {result.r_squared:.6f} < {r2_min:.6f}"
    return None


def _check_residuals_random(result: Any, p_min: float = RUNS_TEST_P_MIN) -> str | None:
    """Check residuals are randomly distributed (runs test).

    Parameters
    ----------
    result : FitResult
        Completed fit result.
    p_min : float
        Minimum acceptable p-value.

    Returns
    -------
    str | None
        Failure message if p < p_min, else None.
    """
    p_value = _runs_test(result.residuals)
    if p_value < p_min:
        return (
            f"Residuals not random: runs test p={p_value:.6f} < {p_min:.6f} "
            f"(evidence of systematic pattern)"
        )
    return None


# ---------------------------------------------------------------------------
# Parametrize: build test cases for all datasets
# ---------------------------------------------------------------------------

_DATASET_ITEMS = [(name, ds) for name, ds in SYNTH_DATASETS.items()]


def _pytest_id(item: tuple[str, dict]) -> str:
    """Generate pytest test ID."""
    name, ds = item
    noise = ds["noise_level"]
    return f"{name}[noise={noise:.0%}]"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dataset_name,dataset",
    _DATASET_ITEMS,
    ids=[_pytest_id(item) for item in _DATASET_ITEMS],
)
def test_parameter_recovery(dataset_name: str, dataset: dict) -> None:
    """Verify parameter recovery within noise-dependent tolerance.

    Tests that Fit() recovers the certified parameter values to within:
    - 1e-6 relative error for noise-free data
    - 2% relative error for 1% noise
    - 10% relative error for 5% noise
    """
    x = np.asarray(dataset["x"], dtype=np.float64)
    y = np.asarray(dataset["y"], dtype=np.float64)
    model_type = dataset["model_type"]
    noise_level = dataset["noise_level"]
    certified = dataset["certified_params"]

    # Run fit with default solver settings
    result = Fit(model_type, x, y, weights="uniform").run()

    # Check parameter recovery
    param_failures = _check_params(result, certified, noise_level)

    if param_failures:
        msg = (
            f"Dataset: {dataset_name} (noise={noise_level:.0%})\n"
            + "\n".join(f"  {f}" for f in param_failures)
        )
        pytest.fail(msg)


@pytest.mark.parametrize(
    "dataset_name,dataset",
    _DATASET_ITEMS,
    ids=[_pytest_id(item) for item in _DATASET_ITEMS],
)
def test_r_squared(dataset_name: str, dataset: dict) -> None:
    """Verify R^2 > 0.99 for all datasets."""
    x = np.asarray(dataset["x"], dtype=np.float64)
    y = np.asarray(dataset["y"], dtype=np.float64)
    model_type = dataset["model_type"]

    result = Fit(model_type, x, y, weights="uniform").run()

    r2_failure = _check_r_squared(result)
    if r2_failure:
        pytest.fail(f"Dataset: {dataset_name}\n  {r2_failure}")


@pytest.mark.parametrize(
    "dataset_name,dataset",
    [(name, ds) for name, ds in _DATASET_ITEMS if ds["noise_level"] > 0],
    ids=[_pytest_id(item) for item in _DATASET_ITEMS if item[1]["noise_level"] > 0],
)
def test_residuals_random(dataset_name: str, dataset: dict) -> None:
    """Verify residuals are randomly distributed for noisy datasets (runs test p > 0.05).

    Only tests datasets with noise > 0. For noise-free data, residuals are
    machine epsilon and not expected to be random.

    A failed runs test (p < 0.05) indicates systematic patterns in residuals,
    suggesting model misspecification or convergence to a local minimum.
    """
    x = np.asarray(dataset["x"], dtype=np.float64)
    y = np.asarray(dataset["y"], dtype=np.float64)
    model_type = dataset["model_type"]

    result = Fit(model_type, x, y, weights="uniform").run()

    residual_failure = _check_residuals_random(result)
    if residual_failure:
        pytest.fail(f"Dataset: {dataset_name}\n  {residual_failure}")
