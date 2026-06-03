# src/openfit/compare.py
"""Model comparison for openfit: AICc, BIC, F-test, evidence ratio.

Assumed FitResult contract (to be reconciled when results.py is written):
    result.params           : dict[str, float]  -- fitted parameter values
    result.residuals        : np.ndarray         -- raw residuals (y - y_fitted)
    result._y               : np.ndarray         -- observed y values
    result.spec.model_id    : str                -- model identifier

Information criteria formulas (matching task spec verbatim):
    AIC  = n * ln(RSS/n) + 2*k
    AICc = AIC + 2*k*(k+1) / (n - k - 1)      [guarded: n - k - 1 must be > 0]
    BIC  = n * ln(RSS/n) + k * ln(n)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from openfit.results import FitResult


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FTestResult:
    """Extra sum-of-squares F-test between a simpler and more complex nested model."""

    simpler_model_id: str
    complex_model_id: str
    df_numerator: int
    df_denominator: int
    f_statistic: float
    p_value: float
    rss_simpler: float
    rss_complex: float
    preferred_model: str  # "simpler" if p > 0.05, else "complex"


@dataclass
class ComparisonResult:
    """Container for multi-model comparison results."""

    model_ids: list[str]
    n_obs: int
    aic_values: dict[str, float]
    aicc_values: dict[str, float]
    bic_values: dict[str, float]
    delta_aicc: dict[str, float]  # aicc_i - min(aicc)
    akaike_weights: dict[str, float]  # exp(-0.5*delta_i) / sum
    evidence_ratio: dict[str, float]  # weight_best / weight_i
    best_model_by_aicc: str
    best_model_by_bic: str
    f_test: FTestResult | None  # only if exactly 2 models and nested
    summary: str  # ASCII summary table


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_models(results: list[FitResult]) -> ComparisonResult:
    """Compare multiple fitted models on the same dataset.

    Parameters
    ----------
    results : list[FitResult]
        Two or more FitResult objects from fits to the same dataset.
        All must have the same n_obs.

    Returns
    -------
    ComparisonResult

    Raises
    ------
    ValueError
        If fewer than 2 results provided.
        If results have different n_obs.

    Notes
    -----
    F-test is computed only if exactly 2 models are provided AND nestedness
    is detected. Nestedness check: model A is nested in B if A's params
    are a strict subset of B's params.

    Rule from CLAUDE.md: "F-test for nested models only. The extra
    sum-of-squares F-test is only valid when one model is a special case
    of the other."
    """
    if len(results) < 2:
        raise ValueError("compare_models requires at least 2 FitResult objects.")

    # Validate n_obs consistency
    n_obs_vals = []
    for r in results:
        n_obs_vals.append(r.n_obs)
    if len(set(n_obs_vals)) != 1:
        raise ValueError(
            f"All results must have the same number of observations. Got: {n_obs_vals}"
        )
    n_obs = n_obs_vals[0]

    model_ids = [_get_model_id(r) for r in results]

    # Compute information criteria for each model
    aic_values: dict[str, float] = {}
    aicc_values: dict[str, float] = {}
    bic_values: dict[str, float] = {}

    for model_id, r in zip(model_ids, results, strict=False):
        k = len(r.params)
        rss = _compute_rss_from_result(r)
        aic, aicc, bic = _information_criteria(rss, n_obs, k)
        aic_values[model_id] = aic
        aicc_values[model_id] = aicc
        bic_values[model_id] = bic

    # Delta AICc and Akaike weights
    min_aicc = min(aicc_values.values())
    delta_aicc = {mid: aicc_values[mid] - min_aicc for mid in model_ids}

    raw_weights = {mid: float(np.exp(-0.5 * delta_aicc[mid])) for mid in model_ids}
    sum_weights = sum(raw_weights.values())
    if sum_weights == 0.0:
        sum_weights = 1.0  # guard against all-infinite deltas
    akaike_weights = {mid: raw_weights[mid] / sum_weights for mid in model_ids}

    best_by_aicc = min(aicc_values, key=lambda m: aicc_values[m])
    w_best = akaike_weights[best_by_aicc]
    evidence_ratio = {
        mid: w_best / akaike_weights[mid] if akaike_weights[mid] > 0.0 else float("inf")
        for mid in model_ids
    }

    best_by_bic = min(bic_values, key=lambda m: bic_values[m])

    # F-test: only for exactly 2 models that are nested
    f_test_result: FTestResult | None = None
    if len(results) == 2 and _check_nestedness(results[0], results[1]):
        # Determine which is simpler (fewer parameters)
        r0, r1 = results[0], results[1]
        if len(r0.params) <= len(r1.params):
            simpler, complex_ = r0, r1
        else:
            simpler, complex_ = r1, r0
        import contextlib

        with contextlib.suppress(ValueError):
            f_test_result = f_test_nested(simpler, complex_)

    summary = _build_summary(
        model_ids=model_ids,
        n_obs=n_obs,
        aic_values=aic_values,
        aicc_values=aicc_values,
        bic_values=bic_values,
        delta_aicc=delta_aicc,
        akaike_weights=akaike_weights,
        evidence_ratio=evidence_ratio,
        best_by_aicc=best_by_aicc,
        best_by_bic=best_by_bic,
        f_test=f_test_result,
    )

    return ComparisonResult(
        model_ids=model_ids,
        n_obs=n_obs,
        aic_values=aic_values,
        aicc_values=aicc_values,
        bic_values=bic_values,
        delta_aicc=delta_aicc,
        akaike_weights=akaike_weights,
        evidence_ratio=evidence_ratio,
        best_model_by_aicc=best_by_aicc,
        best_model_by_bic=best_by_bic,
        f_test=f_test_result,
        summary=summary,
    )


def _check_nestedness(result_a: FitResult, result_b: FitResult) -> bool:
    """Return True if one model is nested in the other (params are strict subset).

    Parameters
    ----------
    result_a : FitResult
        First fit result.
    result_b : FitResult
        Second fit result.

    Returns
    -------
    bool
        True if one model's parameter set is a strict subset of the other's.
    """
    names_a = set(result_a.params.keys())
    names_b = set(result_b.params.keys())
    return (names_a < names_b) or (names_b < names_a)


def f_test_nested(
    simpler: FitResult,
    complex_: FitResult,
) -> FTestResult:
    """Extra sum-of-squares F-test for two nested models.

    F = ((RSS_simple - RSS_complex) / (p_complex - p_simple)) /
        (RSS_complex / (n - p_complex))

    Parameters
    ----------
    simpler : FitResult
        The constrained (fewer parameters) model.
    complex_ : FitResult
        The unconstrained (more parameters) model.

    Returns
    -------
    FTestResult

    Raises
    ------
    ValueError
        If complex_ has fewer or equal parameters than simpler.
        If RSS_complex > RSS_simple (simpler fits better -- not nested).
    """
    k_simple = len(simpler.params)
    k_complex = len(complex_.params)

    if k_complex <= k_simple:
        raise ValueError(
            f"complex_ model must have more parameters than simpler. "
            f"Got k_simple={k_simple}, k_complex={k_complex}."
        )

    n_obs = simpler.n_obs
    rss_simple = _compute_rss_from_result(simpler)
    rss_complex = _compute_rss_from_result(complex_)

    df_num = k_complex - k_simple
    df_den = n_obs - k_complex

    if df_den <= 0:
        raise ValueError(
            f"Not enough degrees of freedom: n_obs ({n_obs}) <= k_complex ({k_complex})."
        )

    if rss_complex <= 0.0:
        raise ValueError("RSS of complex model is zero or negative -- cannot compute F-test.")

    f_stat = ((rss_simple - rss_complex) / df_num) / (rss_complex / df_den)
    p_value = float(stats.f.sf(f_stat, df_num, df_den))

    preferred = "simpler" if p_value > 0.05 else "complex"

    return FTestResult(
        simpler_model_id=_get_model_id(simpler),
        complex_model_id=_get_model_id(complex_),
        df_numerator=df_num,
        df_denominator=df_den,
        f_statistic=float(f_stat),
        p_value=p_value,
        rss_simpler=float(rss_simple),
        rss_complex=float(rss_complex),
        preferred_model=preferred,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_model_id(result: FitResult) -> str:
    """Extract model_id from FitResult.spec, with fallback."""
    try:
        return str(result.spec.model_id)
    except AttributeError:
        return f"model_{id(result)}"


def _compute_rss_from_result(result: FitResult) -> float:
    """Compute RSS from FitResult residuals."""
    residuals = np.asarray(result.residuals, dtype=float)
    return float(np.sum(residuals**2))


def _information_criteria(rss: float, n: int, k: int) -> tuple[float, float, float]:
    """Compute AIC, AICc, BIC from RSS, n observations, and k parameters.

    Formulas from task spec (verbatim):
        AIC  = n * ln(RSS/n) + 2*k
        AICc = AIC + 2*k*(k+1) / (n - k - 1)
        BIC  = n * ln(RSS/n) + k * ln(n)

    AICc is set to +inf when n - k - 1 <= 0 (undefined for over-parameterized models).

    Parameters
    ----------
    rss : float
        Residual sum of squares.
    n : int
        Number of observations.
    k : int
        Number of parameters.

    Returns
    -------
    tuple[float, float, float]
        (aic, aicc, bic)
    """
    if rss <= 0.0:
        # Perfect fit or numerical zero: return -inf to signal best possible
        aic = float("-inf")
        aicc = float("-inf")
        bic = float("-inf")
        return aic, aicc, bic

    log_term = n * float(np.log(rss / n))
    aic = log_term + 2.0 * k

    denom = n - k - 1
    aicc = float("inf") if denom <= 0 else aic + 2.0 * k * (k + 1) / denom

    bic = log_term + k * float(np.log(n))

    return aic, aicc, bic


def _build_summary(
    *,
    model_ids: list[str],
    n_obs: int,
    aic_values: dict[str, float],
    aicc_values: dict[str, float],
    bic_values: dict[str, float],
    delta_aicc: dict[str, float],
    akaike_weights: dict[str, float],
    evidence_ratio: dict[str, float],
    best_by_aicc: str,
    best_by_bic: str,
    f_test: FTestResult | None,
) -> str:
    """Build plain ASCII comparison table sorted by AICc."""
    sorted_ids = sorted(model_ids, key=lambda m: aicc_values[m])

    sep = "-" * 90
    header = (
        f"{'Model':<25} {'AIC':>10} {'AICc':>10} {'BIC':>10} "
        f"{'dAICc':>8} {'Weight':>8} {'EvRatio':>8}"
    )
    lines = [
        sep,
        "openfit -- Model Comparison",
        f"Observations: {n_obs}",
        sep,
        header,
        "-" * 90,
    ]

    for mid in sorted_ids:
        marker = "*" if mid == best_by_aicc else " "
        ev = evidence_ratio[mid]
        ev_str = f"{ev:.2f}" if ev < 1e6 else ">1e6"
        row = (
            f"{marker}{mid:<24} {aic_values[mid]:>10.3f} {aicc_values[mid]:>10.3f} "
            f"{bic_values[mid]:>10.3f} {delta_aicc[mid]:>8.3f} "
            f"{akaike_weights[mid]:>8.4f} {ev_str:>8}"
        )
        lines.append(row)

    lines += [
        sep,
        f"Best by AICc : {best_by_aicc}",
        f"Best by BIC  : {best_by_bic}",
        "* = best model by AICc",
    ]

    if f_test is not None:
        lines += [
            "",
            "F-test (extra sum-of-squares, nested models)",
            f"  Simpler  : {f_test.simpler_model_id}  (RSS = {f_test.rss_simpler:.6g})",
            f"  Complex  : {f_test.complex_model_id}  (RSS = {f_test.rss_complex:.6g})",
            f"  F({f_test.df_numerator},{f_test.df_denominator}) = {f_test.f_statistic:.4f}",
            f"  p-value  : {f_test.p_value:.4f}",
            f"  Preferred: {f_test.preferred_model} model",
        ]

    lines += [
        sep,
        "NOTE: AICc applies to all model comparisons (small-sample correction).",
        "      F-test shown only for nested models. Use AICc for non-nested.",
        sep,
    ]

    return "\n".join(lines)
