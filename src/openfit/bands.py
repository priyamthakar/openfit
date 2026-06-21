# src/openfit/bands.py
"""Prediction and confidence bands for openfit fit results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.stats as stats

if TYPE_CHECKING:
    from openfit.models.base import BaseModel
    from openfit.results import FitResult


@dataclass
class BandResult:
    """Container for prediction or confidence band results.

    Attributes
    ----------
    x : np.ndarray
        The independent variable values where the band was evaluated.
    y_pred : np.ndarray
        The predicted response values at `x`.
    lower : np.ndarray
        The lower bound of the band at `x`.
    upper : np.ndarray
        The upper bound of the band at `x`.
    confidence : float
        The confidence level of the band (e.g., 0.95).
    band_type : str
        The type of band, either "confidence" or "prediction".
    """

    x: np.ndarray
    y_pred: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    confidence: float
    band_type: str


def _numerical_jacobian(
    model: BaseModel,
    x: np.ndarray,
    params: dict[str, float],
    param_names: list[str],
    h: float = 1e-6,
) -> np.ndarray:
    """Compute the numerical Jacobian matrix using central differences.

    Parameters
    ----------
    model : BaseModel
        The model object.
    x : np.ndarray
        Independent variable values.
    params : dict[str, float]
        Parameter values keyed by parameter name.
    param_names : list[str]
        Ordered parameter names.
    h : float, optional
        Step size. Default 1e-6.

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (n_obs, n_params).
    """
    if h <= 0.0:
        raise ValueError("Step size h must be positive.")

    n = len(x)
    p = len(param_names)
    jac = np.empty((n, p), dtype=np.float64)

    for i, name in enumerate(param_names):
        val = params[name]
        params_plus = params.copy()
        params_minus = params.copy()

        params_plus[name] = val + h
        params_minus[name] = val - h

        y_plus = model.equation(x, **params_plus)
        y_minus = model.equation(x, **params_minus)

        jac[:, i] = (y_plus - y_minus) / (2.0 * h)

    return jac


def confidence_band(
    result: FitResult,
    x_pred: np.ndarray | None = None,
    confidence: float = 0.95,
    n_points: int = 200,
) -> BandResult:
    """Compute the confidence band for a fitted model.

    The confidence band represents the uncertainty in the estimated curve itself,
    accounting for the uncertainty in the parameter estimates.

    Parameters
    ----------
    result : FitResult
        The completed fit result.
    x_pred : np.ndarray | None, optional
        The independent variable values at which to evaluate the band.
        If None, a grid of `n_points` is automatically generated between
        `result.x.min()` and `result.x.max()`.
    confidence : float, optional
        The confidence level of the band. Default is 0.95.
    n_points : int, optional
        The number of points to generate if `x_pred` is None. Default is 200.

    Returns
    -------
    BandResult
        The confidence band result containing `x`, `y_pred`, `lower`, `upper`,
        `confidence`, and `band_type`.
    """
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("Confidence level must be between 0.0 and 1.0 (exclusive).")

    if x_pred is None:
        if len(result.x) == 0:
            raise ValueError("Cannot generate prediction grid from empty result data.")
        x_min = float(result.x.min())
        x_max = float(result.x.max())
        if x_min > 0.0 and (x_max / x_min) > 100.0:
            x_pred = np.logspace(np.log10(x_min), np.log10(x_max), num=n_points)
        else:
            x_pred = np.linspace(x_min, x_max, num=n_points)
    else:
        x_pred = np.asarray(x_pred, dtype=np.float64)

    y_pred = result._model.equation(x_pred, **result.params)

    if np.isnan(result.covariance).all():
        lower = np.full_like(y_pred, np.nan)
        upper = np.full_like(y_pred, np.nan)
        return BandResult(
            x=x_pred,
            y_pred=y_pred,
            lower=lower,
            upper=upper,
            confidence=confidence,
            band_type="confidence",
        )

    # Evaluate Jacobian at x_pred
    jac = result._model.jacobian(x_pred, **result.params)
    if jac is None:
        jac = _numerical_jacobian(result._model, x_pred, result.params, result._model.param_names)

    # Delta method variance: v = g^T \Sigma g at each x_pred
    # Form: v = diag(jac @ cov @ jac.T)
    # Optimized vectorized form:
    v = np.sum((jac @ result.covariance) * jac, axis=1)
    v = np.maximum(v, 0.0)

    df = max(result.n_obs - result.n_params, 1)
    alpha = 1.0 - confidence
    t_val = float(stats.t.ppf(1.0 - alpha / 2.0, df))

    width = t_val * np.sqrt(v)
    lower = y_pred - width
    upper = y_pred + width

    return BandResult(
        x=x_pred,
        y_pred=y_pred,
        lower=lower,
        upper=upper,
        confidence=confidence,
        band_type="confidence",
    )


def prediction_band(
    result: FitResult,
    x_pred: np.ndarray | None = None,
    confidence: float = 0.95,
    n_points: int = 200,
) -> BandResult:
    """Compute the prediction band for a fitted model.

    The prediction band represents the uncertainty in future individual observations,
    accounting for both the uncertainty in the parameters and the residual variance.

    Parameters
    ----------
    result : FitResult
        The completed fit result.
    x_pred : np.ndarray | None, optional
        The independent variable values at which to evaluate the band.
        If None, a grid of `n_points` is automatically generated between
        `result.x.min()` and `result.x.max()`.
    confidence : float, optional
        The confidence level of the band. Default is 0.95.
    n_points : int, optional
        The number of points to generate if `x_pred` is None. Default is 200.

    Returns
    -------
    BandResult
        The prediction band result containing `x`, `y_pred`, `lower`, `upper`,
        `confidence`, and `band_type`.
    """
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("Confidence level must be between 0.0 and 1.0 (exclusive).")

    if x_pred is None:
        if len(result.x) == 0:
            raise ValueError("Cannot generate prediction grid from empty result data.")
        x_min = float(result.x.min())
        x_max = float(result.x.max())
        if x_min > 0.0 and (x_max / x_min) > 100.0:
            x_pred = np.logspace(np.log10(x_min), np.log10(x_max), num=n_points)
        else:
            x_pred = np.linspace(x_min, x_max, num=n_points)
    else:
        x_pred = np.asarray(x_pred, dtype=np.float64)

    y_pred = result._model.equation(x_pred, **result.params)

    if np.isnan(result.covariance).all():
        lower = np.full_like(y_pred, np.nan)
        upper = np.full_like(y_pred, np.nan)
        return BandResult(
            x=x_pred,
            y_pred=y_pred,
            lower=lower,
            upper=upper,
            confidence=confidence,
            band_type="prediction",
        )

    # Evaluate Jacobian at x_pred
    jac = result._model.jacobian(x_pred, **result.params)
    if jac is None:
        jac = _numerical_jacobian(result._model, x_pred, result.params, result._model.param_names)

    # Delta method variance: v = g^T \Sigma g at each x_pred
    v = np.sum((jac @ result.covariance) * jac, axis=1)
    v = np.maximum(v, 0.0)

    df = max(result.n_obs - result.n_params, 1)
    s2 = result.rss / df
    v_pred = s2 + v

    alpha = 1.0 - confidence
    t_val = float(stats.t.ppf(1.0 - alpha / 2.0, df))

    width = t_val * np.sqrt(v_pred)
    lower = y_pred - width
    upper = y_pred + width

    return BandResult(
        x=x_pred,
        y_pred=y_pred,
        lower=lower,
        upper=upper,
        confidence=confidence,
        band_type="prediction",
    )
