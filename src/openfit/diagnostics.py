# src/openfit/diagnostics.py
"""Residual diagnostics for openfit FitResult objects.

FitResult fields used: result.residuals, result.x, result.y, result.y_fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from openfit.results import FitResult


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticsResult:
    """Container for all residual diagnostic results."""

    runs_test_pvalue: float
    runs_test_passed: bool  # True if p > 0.05 (residuals appear random)
    n_runs: int
    replicates_test_pvalue: float | None  # None if no replicates
    replicates_test_passed: bool | None
    normality_test_pvalue: float  # Shapiro-Wilk or D'Agostino-Pearson
    normality_test_passed: bool  # True if p > 0.05
    normality_test_name: str  # "Shapiro-Wilk" or "D'Agostino-Pearson"
    outlier_flags: np.ndarray  # bool array, True = potential outlier (3-sigma rule)
    summary: str  # human-readable ASCII summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def residual_analysis(result: FitResult) -> DiagnosticsResult:
    """Full residual diagnostic suite on a FitResult.

    Runs: runs test, normality test, outlier flagging (3-sigma rule).

    Parameters
    ----------
    result : FitResult
        A completed fit result.

    Returns
    -------
    DiagnosticsResult
    """
    raw_residuals: np.ndarray = np.asarray(result.residuals, dtype=float)
    x: np.ndarray = np.asarray(result.x, dtype=float)
    y: np.ndarray = np.asarray(result.y, dtype=float)
    y_fitted: np.ndarray = np.asarray(result.y_fitted, dtype=float)

    # Runs test
    runs_pvalue, n_runs = runs_test(raw_residuals)
    runs_passed = runs_pvalue > 0.05

    # Replicates (lack-of-fit) test
    rep_pvalue = replicates_test(x, y, y_fitted)
    rep_passed = (rep_pvalue > 0.05) if rep_pvalue is not None else None

    # Normality test
    norm_pvalue, norm_name = normality_test(raw_residuals)
    norm_passed = norm_pvalue > 0.05

    # Outlier flags: |standardized residual| > 3
    std_resid = _standardize_residuals(raw_residuals)
    outlier_flags = np.abs(std_resid) > 3.0

    summary = _build_summary(
        runs_pvalue=runs_pvalue,
        runs_passed=runs_passed,
        n_runs=n_runs,
        rep_pvalue=rep_pvalue,
        rep_passed=rep_passed,
        norm_pvalue=norm_pvalue,
        norm_passed=norm_passed,
        norm_name=norm_name,
        n_outliers=int(outlier_flags.sum()),
        n_obs=len(raw_residuals),
    )

    return DiagnosticsResult(
        runs_test_pvalue=runs_pvalue,
        runs_test_passed=runs_passed,
        n_runs=n_runs,
        replicates_test_pvalue=rep_pvalue,
        replicates_test_passed=rep_passed,
        normality_test_pvalue=norm_pvalue,
        normality_test_passed=norm_passed,
        normality_test_name=norm_name,
        outlier_flags=outlier_flags,
        summary=summary,
    )


def runs_test(residuals: np.ndarray) -> tuple[float, int]:
    """Wald-Wolfowitz runs test for randomness of residuals.

    Tests H0: residuals are randomly distributed around zero.

    Parameters
    ----------
    residuals : np.ndarray
        Raw residuals.

    Returns
    -------
    tuple[float, int]
        (p_value, n_runs)

    Notes
    -----
    Uses normal approximation with continuity correction:
    z = (R - mu_R +/- 0.5) / sigma_R where R = observed runs,
    mu_R and sigma_R derived from combinatorics of n_pos, n_neg.

    Returns p=1.0 if all residuals have the same sign (trivially non-random
    by any other test, but the runs statistic is degenerate there).
    """
    residuals = np.asarray(residuals, dtype=float)

    # Classify: positive residuals -> True, non-positive -> False (ignoring zeros)
    nonzero = residuals[residuals != 0.0]
    if len(nonzero) < 2:
        return 1.0, 0

    signs = nonzero > 0  # bool array
    n_pos = int(signs.sum())
    n_neg = int((~signs).sum())

    # Count runs
    n_runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            n_runs += 1

    # Guard degenerate case: all same sign
    if n_pos == 0 or n_neg == 0:
        return 1.0, n_runs

    n = n_pos + n_neg
    # Expected runs and variance (combinatorial formulas)
    mu_r = (2.0 * n_pos * n_neg / n) + 1.0
    sigma_r_sq = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n)) / (n * n * (n - 1.0))
    if sigma_r_sq <= 0.0:
        return 1.0, n_runs

    sigma_r = float(np.sqrt(sigma_r_sq))

    # Continuity correction
    z = (n_runs - mu_r + 0.5) / sigma_r if n_runs < mu_r else (n_runs - mu_r - 0.5) / sigma_r

    p_value = float(2.0 * stats.norm.sf(abs(z)))
    p_value = min(p_value, 1.0)  # clamp floating-point overshoot
    return p_value, n_runs


def replicates_test(
    x: np.ndarray,
    y: np.ndarray,
    y_fitted: np.ndarray,
) -> float | None:
    """Test whether replicated points show lack-of-fit.

    Returns None if no replicates exist (all x unique).
    Returns F-test p-value for lack-of-fit vs pure error.

    Parameters
    ----------
    x : np.ndarray
        x values (may contain duplicates = replicates).
    y : np.ndarray
        Observed y values.
    y_fitted : np.ndarray
        Fitted y values.

    Returns
    -------
    float | None
        p-value for lack-of-fit F-test, or None if no replicates.

    Notes
    -----
    Degrees of freedom:
        df_PE = n - m  (pure error: within-group variation, m = distinct x levels)
        df_LOF = m - 1  (lack-of-fit: between model means and group means)
    This is the conservative partition that does not subtract the number of
    model parameters from df_PE, because compare.py already handles model
    complexity via AICc/BIC/F-test.  It matches the convention in
    Draper & Smith (1998), "Applied Regression Analysis", 3rd ed., Section 2.3.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_fitted = np.asarray(y_fitted, dtype=float)

    unique_x = np.unique(x)
    n = len(y)
    m = len(unique_x)  # number of distinct x levels

    if m == n:
        return None  # no replicates

    # Pure-error SS: within-group variation around group means
    ss_pe = 0.0
    ss_lof = 0.0
    for xi in unique_x:
        mask = x == xi
        y_group = y[mask]
        yhat_group = y_fitted[mask]
        group_mean = float(np.mean(y_group))
        ss_pe += float(np.sum((y_group - group_mean) ** 2))
        # Lack-of-fit: distance between fitted value (same for all replicates of xi)
        # and the group mean
        ss_lof += float(np.sum((yhat_group - group_mean) ** 2))

    df_pe = n - m
    df_lof = m - 1

    if df_pe <= 0 or df_lof <= 0:
        return None

    ms_pe = ss_pe / df_pe
    if ms_pe <= 0.0:
        return 1.0  # perfect pure error -> no lack of fit detectable

    ms_lof = ss_lof / df_lof
    f_stat = ms_lof / ms_pe
    p_value = float(stats.f.sf(f_stat, df_lof, df_pe))
    return p_value


def normality_test(residuals: np.ndarray) -> tuple[float, str]:
    """Test normality of residuals.

    Uses Shapiro-Wilk for n <= 50, D'Agostino-Pearson for n > 50.

    Parameters
    ----------
    residuals : np.ndarray
        Raw residuals.

    Returns
    -------
    tuple[float, str]
        (p_value, test_name)
    """
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)

    if n < 3:
        # Cannot run any normality test; assume normal by convention
        return 1.0, "Shapiro-Wilk"

    if n <= 50:
        _stat, p_value = stats.shapiro(residuals)
        return float(p_value), "Shapiro-Wilk"
    else:
        _stat, p_value = stats.normaltest(residuals)  # D'Agostino-Pearson
        return float(p_value), "D'Agostino-Pearson"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _standardize_residuals(residuals: np.ndarray) -> np.ndarray:
    """Return residuals scaled by their standard deviation."""
    std = float(np.std(residuals, ddof=1))
    if std == 0.0:
        return np.zeros_like(residuals)
    return residuals / std


def _build_summary(
    *,
    runs_pvalue: float,
    runs_passed: bool,
    n_runs: int,
    rep_pvalue: float | None,
    rep_passed: bool | None,
    norm_pvalue: float,
    norm_passed: bool,
    norm_name: str,
    n_outliers: int,
    n_obs: int,
) -> str:
    """Build a plain ASCII diagnostic summary string."""
    sep = "-" * 60
    lines = [
        sep,
        "openfit -- Residual Diagnostics",
        sep,
        f"Observations : {n_obs}",
        "",
        "Runs Test (Wald-Wolfowitz)",
        f"  Runs observed : {n_runs}",
        f"  p-value       : {runs_pvalue:.4f}",
        f"  Result        : {'PASS (random)' if runs_passed else 'FAIL (systematic pattern)'}",
        "",
        f"Normality Test ({norm_name})",
        f"  p-value       : {norm_pvalue:.4f}",
        f"  Result        : {'PASS (normal)' if norm_passed else 'FAIL (non-normal)'}",
        "",
        "Replicates / Lack-of-Fit Test",
    ]

    if rep_pvalue is None:
        lines.append("  No replicates found -- test not applicable.")
    else:
        rep_label = "PASS (no lack of fit)" if rep_passed else "FAIL (significant lack of fit)"
        lines.append(f"  p-value       : {rep_pvalue:.4f}")
        lines.append(f"  Result        : {rep_label}")

    lines += [
        "",
        "Outlier Flags (|standardized residual| > 3)",
        f"  Flagged       : {n_outliers} of {n_obs}",
        sep,
        "NOTE: Tests are exploratory. Small samples have low power.",
        "      Verify results before regulatory or clinical use.",
        sep,
    ]

    return "\n".join(lines)
