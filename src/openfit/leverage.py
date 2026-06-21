# src/openfit/leverage.py
"""Leverage and influence diagnostics for openfit FitResult objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openfit.results import FitResult


@dataclass
class LeverageResult:
    """Container for leverage and influence diagnostic results.

    Parameters
    ----------
    hat_values : np.ndarray
        Hat values (leverages) h_ii for each observation. Shape (n_obs,).
    cooks_distance : np.ndarray
        Cook's distance values for each observation. Shape (n_obs,).
    dffits : np.ndarray
        DFFITS values for each observation. Shape (n_obs,).
    high_leverage_mask : np.ndarray
        Boolean mask indicating observations with high leverage (h_ii > 2k/n). Shape (n_obs,).
    influential_mask : np.ndarray
        Boolean mask indicating influential observations (Cook's D > 4/n). Shape (n_obs,).
    """

    hat_values: np.ndarray
    cooks_distance: np.ndarray
    dffits: np.ndarray
    high_leverage_mask: np.ndarray
    influential_mask: np.ndarray


def _central_diff_jacobian(model, x: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Compute the central difference numerical Jacobian of shape (N, P)."""
    n = len(x)
    p_names = model.param_names
    p = len(p_names)
    J = np.zeros((n, p))
    step = 1e-8
    for j, name in enumerate(p_names):
        val = params[name]
        h = step * max(abs(val), 1.0)

        params_pos = params.copy()
        params_pos[name] = val + h

        params_neg = params.copy()
        params_neg[name] = val - h

        y_pos = model.equation(x, **params_pos)
        y_neg = model.equation(x, **params_neg)

        J[:, j] = (y_pos - y_neg) / (2.0 * h)
    return J


def leverage_diagnostics(result: FitResult) -> LeverageResult:
    """Compute leverage and influence diagnostics for a FitResult.

    Parameters
    ----------
    result : FitResult
        A completed fit result.

    Returns
    -------
    LeverageResult
        The computed leverage and influence diagnostics.
    """
    n = result.n_obs
    k = result.n_params

    # Handle singular fits, missing covariance, or invalid parameters
    if np.isnan(result.covariance).any() or k <= 0 or n <= 0:
        nan_arr = np.full(n, np.nan)
        false_arr = np.zeros(n, dtype=bool)
        return LeverageResult(
            hat_values=nan_arr,
            cooks_distance=nan_arr,
            dffits=nan_arr,
            high_leverage_mask=false_arr,
            influential_mask=false_arr,
        )

    model = result._model
    params = result.params
    x = result.x
    w = result._weights

    try:
        J = model.jacobian(x, **params)
    except Exception:
        J = None

    J = _central_diff_jacobian(model, x, params) if J is None else np.asarray(J, dtype=float)

    # 2. Compute s^2_weighted
    rss_weighted = float(np.sum(result.weighted_residuals**2))
    df = max(n - k, 1)
    s2_weighted = rss_weighted / df
    s2_weighted_clamped = max(s2_weighted, 1e-14)

    # 3. Compute unscaled covariance
    unscaled_cov = result.covariance / s2_weighted_clamped

    # 4. Compute hat values (leverage)
    h = w * np.sum((J @ unscaled_cov) * J, axis=1)
    h = np.clip(h, 0.0, 1.0 - 1e-10)

    # 5. Cook's distance
    r_w = result.weighted_residuals
    cooks_distance = (r_w**2 / (k * s2_weighted_clamped)) * (h / (1.0 - h) ** 2)

    # 6. DFFITS
    dffits = (r_w / np.sqrt(s2_weighted_clamped)) * (np.sqrt(h) / (1.0 - h))

    # 7. High leverage and influential masks
    high_leverage_mask = h > (2.0 * k / n)
    influential_mask = cooks_distance > (4.0 / n)

    return LeverageResult(
        hat_values=h,
        cooks_distance=cooks_distance,
        dffits=dffits,
        high_leverage_mask=high_leverage_mask,
        influential_mask=influential_mask,
    )
