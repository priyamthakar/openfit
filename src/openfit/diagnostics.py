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


__all__ = [
    "DiagnosticsResult",
    "DurbinWatsonResult",
    "LackOfFitResult",
    "durbin_watson",
    "lack_of_fit_test",
    "normality_test",
    "replicates_test",
    "residual_analysis",
    "runs_test",
]


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


@dataclass
class DurbinWatsonResult:
    """Result of the Durbin-Watson autocorrelation test.

    Parameters
    ----------
    statistic : float
        The Durbin-Watson statistic (between 0 and 4).
    interpretation : str
        Interpretation of the statistic: "positive autocorrelation",
        "negative autocorrelation", or "no autocorrelation".
    """

    statistic: float
    interpretation: str


@dataclass
class LackOfFitResult:
    """Result of the Lack-of-Fit F-test.

    Parameters
    ----------
    statistic : float or None
        The F statistic, or None if the test cannot be performed.
    p_value : float or None
        The p-value of the F-test, or None if the test cannot be performed.
    passed : bool or None
        True if p_value > 0.05 (no significant lack of fit), False if <= 0.05,
        or None if the test cannot be performed.
    df_lof : int
        Degrees of freedom for lack-of-fit.
    df_pe : int
        Degrees of freedom for pure error.
    """

    statistic: float | None
    p_value: float | None
    passed: bool | None
    df_lof: int
    df_pe: int


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


def durbin_watson(result: FitResult) -> DurbinWatsonResult:
    """Calculate the Durbin-Watson statistic for residuals.

    The Durbin-Watson statistic tests for the presence of autocorrelation
    at lag 1 in the residuals of a regression analysis.

    Parameters
    ----------
    result : FitResult
        A completed fit result.

    Returns
    -------
    DurbinWatsonResult
        A dataclass containing the statistic and its interpretation.
    """
    residuals = np.asarray(result.residuals, dtype=float)
    if len(residuals) < 2:
        return DurbinWatsonResult(statistic=2.0, interpretation="no autocorrelation")

    numerator = float(np.sum(np.diff(residuals) ** 2))
    denominator = float(np.sum(residuals**2))

    dw = 2.0 if denominator == 0.0 else numerator / denominator

    if dw < 1.5:
        interpretation = "positive autocorrelation"
    elif dw > 2.5:
        interpretation = "negative autocorrelation"
    else:
        interpretation = "no autocorrelation"

    return DurbinWatsonResult(statistic=dw, interpretation=interpretation)


def lack_of_fit_test(
    result: FitResult,
    x_groups: np.ndarray | None = None,
    tolerance: float = 1e-10,
) -> LackOfFitResult:
    """Perform a Lack-of-Fit F-test on the fit residuals.

    This test partitions the residual sum of squares (SS_res) into pure error (SS_PE)
    and lack of fit (SS_LOF) using replicate groups.

    Parameters
    ----------
    result : FitResult
        A completed fit result.
    x_groups : np.ndarray, optional
        Custom grouping labels/values for replicate groups. If None, groups are
        determined by unique independent variable values (result.x) within tolerance.
    tolerance : float, default 1e-10
        Tolerance for grouping unique values.

    Returns
    -------
    LackOfFitResult
        A dataclass containing the F-statistic, p-value, and whether the test passed.
    """
    x = np.asarray(result.x, dtype=float)
    y = np.asarray(result.y, dtype=float)
    y_fitted = np.asarray(result.y_fitted, dtype=float)

    if x_groups is not None:
        x_groups = np.asarray(x_groups)
        if len(x_groups) != len(x):
            raise ValueError("x_groups must have the same length as the fit data.")
        groups_source = x_groups
    else:
        groups_source = x

    groups = _group_by_tolerance(groups_source, tolerance)
    N = len(y)
    M = len(groups)
    K = result.n_params

    df_pe = N - M
    df_lof = M - K

    if df_pe <= 0 or df_lof <= 0:
        return LackOfFitResult(
            statistic=None,
            p_value=None,
            passed=None,
            df_lof=df_lof,
            df_pe=df_pe,
        )

    # SS_PE = sum over all groups g of \sum_{i \in g} (y_i - \bar{y}_g)^2
    ss_pe = 0.0
    for g_indices in groups:
        if len(g_indices) > 1:
            y_g = y[g_indices]
            mean_y_g = np.mean(y_g)
            ss_pe += np.sum((y_g - mean_y_g) ** 2)

    ss_res = np.sum((y - y_fitted) ** 2)
    ss_lof = max(0.0, ss_res - ss_pe)

    ms_lof = ss_lof / df_lof
    ms_pe = ss_pe / df_pe

    if ms_pe <= 0.0:
        return LackOfFitResult(
            statistic=None,
            p_value=None,
            passed=None,
            df_lof=df_lof,
            df_pe=df_pe,
        )

    f_stat = ms_lof / ms_pe
    p_value = float(stats.f.sf(f_stat, df_lof, df_pe))
    passed = p_value > 0.05

    return LackOfFitResult(
        statistic=f_stat,
        p_value=p_value,
        passed=passed,
        df_lof=df_lof,
        df_pe=df_pe,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _group_by_tolerance(values: np.ndarray, tolerance: float) -> list[np.ndarray]:
    """Group indices of values that are within tolerance of each other.

    Returns a list of 1D numpy arrays, each containing the indices belonging to a group.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return []

    # Sort values to group consecutive ones
    sort_idx = np.argsort(v)
    v_sorted = v[sort_idx]

    # Find boundaries where difference exceeds tolerance
    diffs = np.diff(v_sorted)
    # A new group starts at index 0 and whenever diffs > tolerance
    boundaries = np.concatenate(([0], np.where(diffs > tolerance)[0] + 1, [n]))

    groups = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        group_indices = sort_idx[start:end]
        groups.append(group_indices)
    return groups


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
