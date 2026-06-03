"""ROUT adaptive outlier detection for nonlinear regression.

Implements the algorithm from:
    Motulsky HJ & Brown RE (2006). Detecting outliers when fitting data
    with nonlinear regression: a new method based on robust nonlinear
    regression and the false discovery rate. BMC Bioinformatics, 7:123.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC1472692/

Statistical notes (paper-faithful implementation):

Robust fitting uses a Lorentzian merit function:
    sum( ln(1 + (residual_i / RSDR)^2) )
minimized with a modified Marquardt-Levenberg algorithm.  This differs from
Tukey biweight / IRLS; Tukey biweight is NOT used in the original method.

RSDR is defined as:
    RSDR = P68 * N / (N - K)
where P68 is the 68.27th percentile of |residuals| (with proportional
interpolation), N is the number of observations, and K is the number of
fitted parameters.  The N/(N-K) factor corrects small-sample downward bias.

Outlier p-values use the t-distribution with N-K degrees of freedom, NOT
the normal distribution.

FDR is applied via the sequential rule from equation (17) of the paper
(Simes/FWER-based step-up test), NOT the Benjamini-Hochberg adjusted-p
formulation.  The test scans only the 30% most extreme residuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.optimize
import scipy.stats

from openfit.models.base import BaseModel
from openfit.models import get_model


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ROUTResult:
    """Results from the ROUT adaptive outlier detection algorithm.

    Attributes
    ----------
    outlier_mask : np.ndarray
        Boolean array of shape (n,).  True indicates an outlier.
    n_outliers : int
        Number of points flagged as outliers.
    outlier_indices : np.ndarray
        Integer indices of outlier points in the original data.
    Q : float
        FDR threshold used (user-supplied, default 0.01).
    p_values : np.ndarray
        Two-tailed p-values for each point from the robust fit residuals.
        Uses t-distribution with N-K degrees of freedom.
        Points NOT tested in the 30 %% scan are assigned p=1.0.
    fdr_adjusted_p : np.ndarray
        Adjusted p-values after the sequential FDR procedure.
        Points not scanned are set to 1.0.
    robust_params : dict[str, float]
        Model parameters from the Lorentzian robust fit (all N points).
    clean_params : dict[str, float]
        Model parameters re-fitted by ordinary least squares after removing
        outliers.  Equals robust_params when re-fitting is not possible
        (too few clean points) or when no outliers are found.
    robust_rss : float
        Residual sum of squares from the robust fit over all N points.
    clean_rss : float
        Residual sum of squares from the ordinary re-fit over clean points
        only.  Equals robust_rss when re-fitting is not possible.
    model_id : str
        Identifier of the model used.
    summary : str
        ASCII summary of results.
    """

    outlier_mask: np.ndarray
    n_outliers: int
    outlier_indices: np.ndarray
    Q: float
    p_values: np.ndarray
    fdr_adjusted_p: np.ndarray
    robust_params: dict[str, float]
    clean_params: dict[str, float]
    robust_rss: float
    clean_rss: float
    model_id: str
    summary: str


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _rsdr(residuals: np.ndarray, n_params: int) -> float:
    """Compute the Robust Standard Deviation of Residuals.

    RSDR = P68 * N / (N - K)

    where P68 is the 68.27th percentile of |residuals| (with proportional
    interpolation) and N/(N-K) is the small-sample degrees-of-freedom
    correction from Motulsky & Brown (2006), Equation 1.

    Parameters
    ----------
    residuals : np.ndarray
        Array of residuals (y - y_hat).
    n_params : int
        Number of fitted model parameters (K).

    Returns
    -------
    float
        RSDR value.  Returns 0.0 if degenerate (all residuals zero).

    Notes
    -----
    If N - K <= 0 or the 68.27th percentile is zero, returns 0.0.
    Callers must guard against rsdr == 0 before using it as a divisor.
    """
    n = len(residuals)
    df = n - n_params
    if df <= 0:
        return 0.0
    abs_res = np.abs(residuals)
    # 68.27th percentile with proportional interpolation (np.percentile default)
    p68 = float(np.percentile(abs_res, 68.27))
    return p68 * n / df


def _lorentzian_objective(
    params_vec: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    model: BaseModel,
    rsdr: float,
) -> np.ndarray:
    """Residual array for Lorentzian robust fitting.

    The Lorentzian merit function is sum(ln(1 + (r/RSDR)^2)).  scipy
    least_squares minimizes 0.5 * sum(f_i^2), so we pass:
        f_i = sqrt(2 * ln(1 + (r_i / RSDR)^2))

    to make the two objectives equivalent in value and gradient direction.

    Parameters
    ----------
    params_vec : np.ndarray
        Parameter vector, ordered as model.param_names.
    x : np.ndarray
        Independent variable values.
    y : np.ndarray
        Observed response values.
    model : BaseModel
        Fitted model object.
    rsdr : float
        Current robust scale estimate (must be > 0).

    Returns
    -------
    np.ndarray
        Transformed residual vector of shape (n,).
    """
    param_dict = dict(zip(model.param_names, params_vec))
    y_hat = model.equation(x, **param_dict)
    raw_res = y - y_hat
    # Avoid overflow: clip the argument of ln
    arg = 1.0 + (raw_res / rsdr) ** 2
    return np.sqrt(2.0 * np.log(np.maximum(arg, 1e-300)))


def _ordinary_residuals(
    params_vec: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    model: BaseModel,
) -> np.ndarray:
    """Plain (unweighted) residuals for ordinary least squares.

    Parameters
    ----------
    params_vec : np.ndarray
        Parameter vector, ordered as model.param_names.
    x : np.ndarray
        Independent variable values.
    y : np.ndarray
        Observed response values.
    model : BaseModel
        Fitted model object.

    Returns
    -------
    np.ndarray
        Residual vector (y - y_hat) of shape (n,).
    """
    param_dict = dict(zip(model.param_names, params_vec))
    y_hat = model.equation(x, **param_dict)
    return y - y_hat


def _fit_ols(
    x: np.ndarray,
    y: np.ndarray,
    model: BaseModel,
    p0: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Ordinary least-squares fit via scipy.optimize.least_squares (TRF).

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    y : np.ndarray
        Observed response values.
    model : BaseModel
        Model to fit.
    p0 : np.ndarray
        Initial parameter vector.

    Returns
    -------
    tuple[np.ndarray, bool]
        (fitted_params, success) where success indicates convergence.
    """
    lower, upper = model.bounds()
    result = scipy.optimize.least_squares(
        _ordinary_residuals,
        p0,
        args=(x, y, model),
        method="trf",
        bounds=(lower, upper),
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        max_nfev=2000 * len(p0),
    )
    return result.x, result.success


def _robust_fit(
    x: np.ndarray,
    y: np.ndarray,
    model: BaseModel,
    p0: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, float]:
    """Lorentzian robust fit via iterative Marquardt-Levenberg optimization.

    The Lorentzian merit function is:
        sum( ln(1 + (residual_i / RSDR)^2) )

    Each outer iteration:
        1. Compute RSDR from current residuals.
        2. Minimize the Lorentzian merit using scipy least_squares.
        3. Recompute merit of PREVIOUS params with NEW RSDR (paper rule).
        4. Converge when relative parameter change < tol.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    y : np.ndarray
        Observed response values.
    model : BaseModel
        Model to fit.
    p0 : np.ndarray
        Initial parameter vector (from model.initial_guess).
    max_iter : int
        Maximum outer iterations.
    tol : float
        Relative convergence tolerance on parameter change.

    Returns
    -------
    tuple[np.ndarray, float]
        (robust_params_vec, final_rsdr)

    Notes
    -----
    If RSDR collapses to zero (degenerate data), switches to ordinary
    least squares immediately.
    """
    n_params = len(p0)
    lower, upper = model.bounds()
    params = p0.copy()

    # Bootstrap: get initial residuals from OLS start
    ols_params, _ = _fit_ols(x, y, model, p0)
    params = ols_params.copy()

    param_dict = dict(zip(model.param_names, params))
    y_hat = model.equation(x, **param_dict)
    residuals = y - y_hat
    current_rsdr = _rsdr(residuals, n_params)

    if current_rsdr <= 0.0:
        # Degenerate: return OLS solution
        return params, 0.0

    for _iteration in range(max_iter):
        prev_params = params.copy()

        # Minimize Lorentzian merit with current RSDR
        result = scipy.optimize.least_squares(
            _lorentzian_objective,
            params,
            args=(x, y, model, current_rsdr),
            method="trf",
            bounds=(lower, upper),
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
            max_nfev=2000 * n_params,
        )
        params = result.x

        # Update RSDR using new residuals
        param_dict = dict(zip(model.param_names, params))
        y_hat = model.equation(x, **param_dict)
        residuals = y - y_hat
        new_rsdr = _rsdr(residuals, n_params)

        if new_rsdr <= 0.0:
            # Perfect fit or all residuals zero -- accept and stop
            current_rsdr = new_rsdr
            break

        current_rsdr = new_rsdr

        # Convergence: relative change in parameter vector
        scale = np.abs(prev_params) + 1e-12
        rel_change = float(np.max(np.abs(params - prev_params) / scale))
        if rel_change < tol:
            break

    return params, current_rsdr


def _fdr_sequential_test(
    abs_residuals: np.ndarray,
    rsdr: float,
    n_params: int,
    Q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sequential FDR outlier test (Motulsky & Brown 2006, Equation 17-18).

    Tests only the most extreme 30 %% of residuals by magnitude.  Proceeds
    from i = int(0.70 * N) to N (sorted ascending by |residual|).  A point
    is declared an outlier if its p-value falls below alpha_i = Q*(N-i+1)/N;
    all points further from the curve than that threshold are also flagged.

    Parameters
    ----------
    abs_residuals : np.ndarray
        Absolute residuals from the robust fit, shape (n,).
    rsdr : float
        Robust standard deviation of residuals (must be > 0).
    n_params : int
        Number of fitted parameters (K), for t-distribution df.
    Q : float
        FDR threshold (0 < Q < 1).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (outlier_mask, p_values_full, fdr_adjusted_p_full)

        outlier_mask[i] = True if point i is flagged as an outlier.
        p_values_full[i] = two-tailed t-distribution p-value for each point
            (points outside the 30 %% scan window have p = 1.0).
        fdr_adjusted_p_full[i] = alpha_i threshold that triggered flagging
            for outliers; 1.0 for non-outliers and untested points.

    Notes
    -----
    Uses t-distribution with max(df, 1) degrees of freedom (df = N - K).
    If rsdr == 0, no outliers are detected.
    """
    n = len(abs_residuals)
    df = max(n - n_params, 1)

    p_values = np.ones(n, dtype=float)
    alpha_thresholds = np.ones(n, dtype=float)
    outlier_mask = np.zeros(n, dtype=bool)

    if rsdr <= 0.0:
        return outlier_mask, p_values, alpha_thresholds

    # Sort indices by |residual| ascending; scan only the top 30 %%
    sorted_idx = np.argsort(abs_residuals)
    scan_start = int(0.70 * n)  # index in the sorted array to begin scanning

    # Compute p-values for all scanned points first
    for si in range(scan_start, n):
        orig_idx = sorted_idx[si]
        t_ratio = abs_residuals[orig_idx] / rsdr
        p_val = float(2.0 * scipy.stats.t.sf(t_ratio, df=df))
        p_values[orig_idx] = p_val

    # Sequential decision (i is 1-based rank in the paper, si is 0-based here)
    # alpha_i = Q * (N - (i-1)) / N  where i goes from scan_start+1 to N
    outlier_start_si: int | None = None
    for si in range(scan_start, n):
        orig_idx = sorted_idx[si]
        # i in the paper is the 1-based position (n points sorted low to high)
        # si = 0-based index in sorted array; paper's i = si + 1
        i_paper = si + 1
        alpha_i = Q * (n - (i_paper - 1)) / n
        alpha_thresholds[orig_idx] = alpha_i
        if p_values[orig_idx] < alpha_i:
            outlier_start_si = si
            break

    # All points further from the curve (higher |residual|) are also outliers
    if outlier_start_si is not None:
        for si in range(outlier_start_si, n):
            outlier_mask[sorted_idx[si]] = True

    return outlier_mask, p_values, alpha_thresholds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rout_outliers(
    x: np.ndarray,
    y: np.ndarray,
    model: str | BaseModel,
    Q: float = 0.01,
    max_iter: int = 50,
    tol: float = 1e-6,
    tukey_c: float = 4.685,
) -> ROUTResult:
    """Detect outliers using the ROUT method (Motulsky & Brown, 2006).

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    y : np.ndarray
        Observed response values.
    model : str | BaseModel
        Model to fit (must support initial_guess, equation, bounds, and
        param_names).  A string is looked up in the openfit model registry.
    Q : float
        False discovery rate threshold.  Default 0.01 (1 %%).
        Typical values: 0.001, 0.01, 0.05.
    max_iter : int
        Maximum outer iterations for the robust Lorentzian fit.  Default 50.
    tol : float
        Relative convergence tolerance on parameter change.  Default 1e-6.
    tukey_c : float
        Accepted for API compatibility; not used.  The paper uses a Lorentzian
        merit function, not Tukey biweight.  This parameter is ignored.

    Returns
    -------
    ROUTResult

    Raises
    ------
    ValueError
        If x or y contain NaN or Inf values.
        If len(x) != len(y).
        If Q is not strictly in (0, 1).
        If the number of observations does not exceed the number of model
        parameters (not enough degrees of freedom).

    Notes
    -----
    Phase 1: Robust fit via iterative Lorentzian minimization.
        The Lorentzian merit function sum(ln(1+(r/RSDR)^2)) is minimized
        iteratively using scipy.optimize.least_squares (TRF method).
        RSDR is updated between iterations: RSDR = P68 * N / (N-K),
        where P68 is the 68.27th percentile of |residuals|.

    Phase 2: Outlier identification via sequential FDR (paper Eq. 17-18).
        Only the 30 %% most extreme residuals are tested.  A point is
        flagged when its two-tailed p-value (t-distribution, N-K df)
        falls below alpha_i = Q * (N - i + 1) / N.  All points further
        from the curve than the first flagged point are also outliers.

    Phase 3: Clean re-fit.
        After removing outliers, an ordinary least-squares fit is performed
        on the remaining points to obtain clean_params.  If re-fitting is
        not possible (too few clean points), clean_params = robust_params.

    Reference
    ---------
    Motulsky HJ, Brown RE. Detecting outliers when fitting data with
    nonlinear regression -- a new method based on robust nonlinear regression
    and the false discovery rate. BMC Bioinformatics. 2006;7:123.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC1472692/
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # --- Input validation ---
    if not np.isfinite(x).all():
        raise ValueError(
            "x contains NaN or Inf values. Clean the input data before calling rout_outliers."
        )
    if not np.isfinite(y).all():
        raise ValueError(
            "y contains NaN or Inf values. Clean the input data before calling rout_outliers."
        )
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape. Got x.shape={x.shape}, y.shape={y.shape}."
        )
    if not (0.0 < Q < 1.0):
        raise ValueError(
            f"Q must be strictly between 0 and 1. Got Q={Q}."
        )

    # --- Resolve model ---
    if isinstance(model, str):
        model_obj: BaseModel = get_model(model)
    else:
        model_obj = model

    n_obs = len(x)
    n_params = len(model_obj.param_names)

    if n_obs <= n_params:
        raise ValueError(
            f"Number of observations ({n_obs}) must exceed number of model parameters "
            f"({n_params}) to fit model '{model_obj.model_id}'."
        )

    model_id = model_obj.model_id

    # --- Phase 1: Robust Lorentzian fit ---
    p0_dict = model_obj.initial_guess(x, y)
    p0 = np.array([p0_dict[name] for name in model_obj.param_names], dtype=float)

    robust_params_vec, final_rsdr = _robust_fit(
        x, y, model_obj, p0, max_iter=max_iter, tol=tol
    )

    robust_param_dict = dict(zip(model_obj.param_names, robust_params_vec))
    y_hat_robust = model_obj.equation(x, **robust_param_dict)
    raw_residuals = y - y_hat_robust
    robust_rss = float(np.sum(raw_residuals ** 2))

    # Recompute RSDR from final residuals (ensure consistency)
    final_rsdr = _rsdr(raw_residuals, n_params)

    # --- Phase 2: FDR outlier detection ---
    abs_residuals = np.abs(raw_residuals)
    outlier_mask, p_values, fdr_adjusted_p = _fdr_sequential_test(
        abs_residuals, final_rsdr, n_params, Q
    )

    n_outliers = int(outlier_mask.sum())
    outlier_indices = np.where(outlier_mask)[0]

    # --- Phase 3: Clean re-fit ---
    clean_mask = ~outlier_mask
    n_clean = int(clean_mask.sum())
    can_refit = (n_clean > n_params) and (n_outliers > 0)

    if can_refit:
        x_clean = x[clean_mask]
        y_clean = y[clean_mask]
        p0_clean_dict = model_obj.initial_guess(x_clean, y_clean)
        p0_clean = np.array(
            [p0_clean_dict[name] for name in model_obj.param_names], dtype=float
        )
        clean_params_vec, _ = _fit_ols(x_clean, y_clean, model_obj, p0_clean)
        clean_param_dict = dict(zip(model_obj.param_names, clean_params_vec))
        y_hat_clean = model_obj.equation(x_clean, **clean_param_dict)
        clean_rss = float(np.sum((y_clean - y_hat_clean) ** 2))
    else:
        clean_param_dict = dict(robust_param_dict)
        clean_rss = robust_rss

    # --- Summary ---
    summary = _build_summary(
        model_id=model_id,
        n_obs=n_obs,
        n_params=n_params,
        Q=Q,
        n_outliers=n_outliers,
        outlier_indices=outlier_indices,
        final_rsdr=final_rsdr,
        robust_rss=robust_rss,
        clean_rss=clean_rss,
        robust_params=robust_param_dict,
        clean_params=clean_param_dict,
        can_refit=can_refit,
    )

    return ROUTResult(
        outlier_mask=outlier_mask,
        n_outliers=n_outliers,
        outlier_indices=outlier_indices,
        Q=Q,
        p_values=p_values,
        fdr_adjusted_p=fdr_adjusted_p,
        robust_params=robust_param_dict,
        clean_params=clean_param_dict,
        robust_rss=robust_rss,
        clean_rss=clean_rss,
        model_id=model_id,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Private summary builder
# ---------------------------------------------------------------------------


def _build_summary(
    *,
    model_id: str,
    n_obs: int,
    n_params: int,
    Q: float,
    n_outliers: int,
    outlier_indices: np.ndarray,
    final_rsdr: float,
    robust_rss: float,
    clean_rss: float,
    robust_params: dict[str, float],
    clean_params: dict[str, float],
    can_refit: bool,
) -> str:
    """Build a plain ASCII summary of ROUT results.

    Parameters
    ----------
    model_id : str
        Model identifier.
    n_obs : int
        Total number of observations.
    n_params : int
        Number of model parameters.
    Q : float
        FDR threshold used.
    n_outliers : int
        Number of outliers flagged.
    outlier_indices : np.ndarray
        Integer indices of outlier points.
    final_rsdr : float
        Final robust standard deviation of residuals.
    robust_rss : float
        RSS from the robust fit (all N points).
    clean_rss : float
        RSS from the clean re-fit (non-outlier points only).
    robust_params : dict[str, float]
        Parameters from robust fit.
    clean_params : dict[str, float]
        Parameters from clean re-fit (or copy of robust if not re-fitted).
    can_refit : bool
        Whether a clean re-fit was possible.

    Returns
    -------
    str
        Formatted ASCII summary.
    """
    sep = "-" * 65
    lines = [
        sep,
        "openfit -- ROUT Outlier Detection (Motulsky & Brown, 2006)",
        sep,
        f"Model          : {model_id}",
        f"Observations   : {n_obs}  (parameters: {n_params})",
        f"FDR threshold  : Q = {Q}",
        f"RSDR           : {final_rsdr:.6g}",
        "",
        f"Outliers found : {n_outliers}",
    ]

    if n_outliers > 0:
        idx_str = ", ".join(str(i) for i in outlier_indices)
        lines.append(f"Outlier indices: {idx_str}")
    else:
        lines.append("                 (none)")

    lines += [
        "",
        "Robust fit parameters (all points):",
    ]
    for name, val in robust_params.items():
        lines.append(f"  {name:<16} {val:.6g}")
    lines.append(f"  RSS (N={n_obs})       {robust_rss:.6g}")

    lines += [
        "",
    ]
    if n_outliers == 0:
        lines.append("Clean re-fit   : not needed (no outliers)")
    elif not can_refit:
        lines.append(
            "Clean re-fit   : skipped (too few clean points to re-fit model)"
        )
        lines.append("                 clean_params = robust_params")
    else:
        n_clean = n_obs - n_outliers
        lines.append(f"Clean re-fit parameters ({n_clean} points, outliers removed):")
        for name, val in clean_params.items():
            lines.append(f"  {name:<16} {val:.6g}")
        lines.append(f"  RSS (N={n_clean})       {clean_rss:.6g}")

    lines += [
        "",
        sep,
        "Reference: Motulsky & Brown (2006) BMC Bioinformatics 7:123",
        "NOTE: Results should be independently verified for regulatory use.",
        sep,
    ]

    return "\n".join(lines)
