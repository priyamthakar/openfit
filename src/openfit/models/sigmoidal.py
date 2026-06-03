"""Sigmoidal / Hill models and Boltzmann equation.

Models
------
Hill3P    -- 3-parameter log-logistic (Bottom fixed at 0)
Hill4P    -- 4-parameter logistic (4PL); analytic Jacobian provided
Hill5P    -- 5-parameter logistic (5PL / asymmetric 4PL)
Boltzmann -- voltage-dependent activation sigmoid
"""

from __future__ import annotations

import numpy as np

# Minimum x guard: avoids division by zero for 1/x terms in Hill equations.
_X_EPS: float = 1e-300
# Exponent clip: prevents overflow in (EC50/x)^HillSlope.
_EXP_CLIP: float = 700.0


def _hill_ratio_clipped(ec50: float, x: np.ndarray, slope: float) -> np.ndarray:
    """Compute clipped (EC50/x)^HillSlope safely.

    Protects against x=0 (adds epsilon), negative base with fractional exponent
    (takes abs then restores sign where safe), and exponent overflow.
    """
    x_safe = np.where(x == 0.0, _X_EPS, x)
    log_ratio = slope * (np.log(np.abs(ec50)) - np.log(np.abs(x_safe)))
    log_ratio_clipped = np.clip(log_ratio, -_EXP_CLIP, _EXP_CLIP)
    return np.exp(log_ratio_clipped)


def _interp_ec50(x: np.ndarray, y: np.ndarray, bottom: float, top: float) -> float:
    """Estimate EC50 by interpolating x at the midpoint response.

    Works for both increasing (agonist) and decreasing (antagonist) curves.

    Parameters
    ----------
    x : np.ndarray
        Independent-variable values.
    y : np.ndarray
        Observed response values.
    bottom : float
        Estimated bottom asymptote.
    top : float
        Estimated top asymptote.

    Returns
    -------
    float
        Estimated EC50 (x at half-maximum response).
    """
    mid = 0.5 * (top + bottom)
    sort_idx = np.argsort(y)
    y_sorted = y[sort_idx]
    x_sorted = x[sort_idx]
    if len(x_sorted) < 2:
        return float(x[0])
    ec50 = float(np.interp(mid, y_sorted, x_sorted))
    # Fallback: use geometric mean of x range if interpolation returns implausible value.
    x_pos = x[x > 0]
    if x_pos.size > 0 and (ec50 <= 0 or not np.isfinite(ec50)):
        ec50 = float(np.exp(0.5 * (np.log(x_pos.min()) + np.log(x_pos.max()))))
    elif not np.isfinite(ec50):
        ec50 = float(np.median(x))
    return ec50


def _estimate_hillslope(
    x: np.ndarray, y: np.ndarray, ec50: float, bottom: float, top: float
) -> float:
    """Estimate Hill slope from log(y/(top-y)) vs log(x) at the midpoint region.

    Uses points within 1 order of magnitude of EC50.  Falls back to 1.0 if
    fewer than 2 usable points exist.

    Parameters
    ----------
    x : np.ndarray
        Independent-variable values.
    y : np.ndarray
        Observed response values.
    ec50 : float
        Estimated EC50.
    bottom : float
        Estimated bottom asymptote.
    top : float
        Estimated top asymptote.

    Returns
    -------
    float
        Estimated Hill slope (always finite; fallback value is 1.0).
    """
    span = top - bottom
    if abs(span) < 1e-10 or len(x) < 2:
        return 1.0
    # Clip y strictly inside (bottom, top) to avoid log(0) or log(negative).
    y_norm = np.clip((y - bottom) / span, 1e-4, 1.0 - 1e-4)
    log_odds = np.log(y_norm / (1.0 - y_norm))
    x_pos = x[x > 0]
    if x_pos.size < 2 or ec50 <= 0:
        return 1.0
    log_x = np.log(x[x > 0])
    log_odds_pos = log_odds[x > 0]
    # Use only points within 1.5 log-units of EC50.
    mask = np.abs(log_x - np.log(ec50)) < 1.5
    if mask.sum() < 2:
        mask = np.ones(len(log_x), dtype=bool)
    if mask.sum() < 2:
        return 1.0
    lx = log_x[mask]
    lo = log_odds_pos[mask]
    slope, _ = np.polyfit(lx, lo, 1)
    return float(np.clip(slope, 0.1, 20.0))


# ---------------------------------------------------------------------------
# Hill3P
# ---------------------------------------------------------------------------


class Hill3P:
    """3-parameter log-logistic sigmoidal model (Bottom fixed at 0).

    Equation
    --------
    y = Top / (1 + (EC50 / x)^HillSlope)

    Parameters
    ----------
    Top : float
        Upper asymptote.
    EC50 : float
        x value at half-maximum response (>0 for concentration data).
    HillSlope : float
        Steepness of the transition; positive for increasing curves.
    """

    model_id: str = "hill3p"
    param_names: list[str] = ["Top", "EC50", "HillSlope"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate Hill3P at *x*.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include Top, EC50, HillSlope.

        Returns
        -------
        np.ndarray
            Predicted response values.
        """
        top = params["Top"]
        ec50 = params["EC50"]
        slope = params["HillSlope"]
        ratio = _hill_ratio_clipped(ec50, x, slope)
        return np.asarray(top / (1.0 + ratio))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for Hill3P.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        y : np.ndarray
            Observed response values.

        Returns
        -------
        dict[str, float]
            Keys: Top, EC50, HillSlope.
        """
        top = float(np.max(y))
        if top == 0.0:
            top = 1.0
        ec50 = _interp_ec50(x, y, 0.0, top)
        slope = _estimate_hillslope(x, y, ec50, 0.0, top)
        return {"Top": top, "EC50": ec50, "HillSlope": slope}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for Hill3P parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [Top, EC50, HillSlope].
        """
        return (
            [-np.inf, _X_EPS, -np.inf],
            [np.inf, np.inf, np.inf],
        )

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Analytic Jacobian for Hill3P (n_obs x 3).

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include Top, EC50, HillSlope.

        Returns
        -------
        np.ndarray | None
            Jacobian matrix of shape (n, 3).
        """
        return None  # finite-diff fallback; Hill3P is used less frequently than 4PL


# ---------------------------------------------------------------------------
# Hill4P  (4-parameter logistic -- most critical model, analytic Jacobian)
# ---------------------------------------------------------------------------


class Hill4P:
    """4-parameter logistic model (4PL).

    Equation
    --------
    y = Bottom + (Top - Bottom) / (1 + (EC50 / x)^HillSlope)

    Parameters
    ----------
    Bottom : float
        Lower asymptote.
    Top : float
        Upper asymptote.
    EC50 : float
        x value at the inflection point (half-maximum response).
    HillSlope : float
        Steepness of the curve; typically positive for agonist responses.
    """

    model_id: str = "hill4p"
    param_names: list[str] = ["Bottom", "Top", "EC50", "HillSlope"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate Hill4P at *x*.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include Bottom, Top, EC50, HillSlope.

        Returns
        -------
        np.ndarray
            Predicted response values.
        """
        bottom = params["Bottom"]
        top = params["Top"]
        ec50 = params["EC50"]
        slope = params["HillSlope"]
        ratio = _hill_ratio_clipped(ec50, x, slope)
        return np.asarray(bottom + (top - bottom) / (1.0 + ratio))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for Hill4P.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        y : np.ndarray
            Observed response values.

        Returns
        -------
        dict[str, float]
            Keys: Bottom, Top, EC50, HillSlope.
        """
        bottom = float(np.min(y))
        top = float(np.max(y))
        if top == bottom:
            top = bottom + 1.0
        ec50 = _interp_ec50(x, y, bottom, top)
        slope = _estimate_hillslope(x, y, ec50, bottom, top)
        return {"Bottom": bottom, "Top": top, "EC50": ec50, "HillSlope": slope}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for Hill4P parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [Bottom, Top, EC50, HillSlope].
        """
        return (
            [-np.inf, -np.inf, _X_EPS, -np.inf],
            [np.inf, np.inf, np.inf, np.inf],
        )

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Analytic Jacobian for Hill4P (n_obs x 4).

        Partial derivatives:
            dY/dBottom    = R / (1 + R)
            dY/dTop       = 1 / (1 + R)
            dY/dEC50      = -(top-bottom) * HillSlope * R / (EC50 * (1+R)^2)
            dY/dHillSlope = -(top-bottom) * R * ln(EC50/x) / (1+R)^2

        where R = (EC50/x)^HillSlope.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include Bottom, Top, EC50, HillSlope.

        Returns
        -------
        np.ndarray | None
            Jacobian matrix of shape (n, 4); columns ordered as param_names.
        """
        bottom = params["Bottom"]
        top = params["Top"]
        ec50 = params["EC50"]
        slope = params["HillSlope"]
        span = top - bottom

        ratio = _hill_ratio_clipped(ec50, x, slope)
        denom = (1.0 + ratio) ** 2
        inv_1pr = 1.0 / (1.0 + ratio)

        x_safe = np.where(x == 0.0, _X_EPS, x)
        ec50_safe = ec50 if ec50 != 0.0 else _X_EPS
        log_ratio = np.log(np.abs(ec50_safe)) - np.log(np.abs(x_safe))

        d_bottom = ratio * inv_1pr                           # R/(1+R) = 1 - 1/(1+R)
        d_top = inv_1pr
        d_ec50 = -span * slope * ratio / (ec50_safe * denom)
        d_slope = -span * ratio * log_ratio / denom

        return np.column_stack([d_bottom, d_top, d_ec50, d_slope])


# ---------------------------------------------------------------------------
# Hill5P  (5-parameter logistic)
# ---------------------------------------------------------------------------


class Hill5P:
    """5-parameter logistic model (5PL / asymmetric 4PL).

    Equation
    --------
    y = Bottom + (Top - Bottom) / (1 + (EC50 / x)^HillSlope)^Asymmetry

    Parameters
    ----------
    Bottom : float
        Lower asymptote.
    Top : float
        Upper asymptote.
    EC50 : float
        x value at the inflection point.
    HillSlope : float
        Steepness of the curve.
    Asymmetry : float
        Asymmetry factor; 1.0 reduces to 4PL.
    """

    model_id: str = "hill5p"
    param_names: list[str] = ["Bottom", "Top", "EC50", "HillSlope", "Asymmetry"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate Hill5P at *x*.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include Bottom, Top, EC50, HillSlope, Asymmetry.

        Returns
        -------
        np.ndarray
            Predicted response values.
        """
        bottom = params["Bottom"]
        top = params["Top"]
        ec50 = params["EC50"]
        slope = params["HillSlope"]
        asym = params["Asymmetry"]
        ratio = _hill_ratio_clipped(ec50, x, slope)
        inner = np.clip(1.0 + ratio, 1e-300, None)
        # Clip exponent argument to prevent overflow.
        log_inner = np.log(inner)
        log_denom = np.clip(asym * log_inner, -_EXP_CLIP, _EXP_CLIP)
        denom = np.exp(log_denom)
        return np.asarray(bottom + (top - bottom) / denom)

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for Hill5P.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        y : np.ndarray
            Observed response values.

        Returns
        -------
        dict[str, float]
            Keys: Bottom, Top, EC50, HillSlope, Asymmetry.
        """
        bottom = float(np.min(y))
        top = float(np.max(y))
        if top == bottom:
            top = bottom + 1.0
        ec50 = _interp_ec50(x, y, bottom, top)
        slope = _estimate_hillslope(x, y, ec50, bottom, top)
        return {
            "Bottom": bottom,
            "Top": top,
            "EC50": ec50,
            "HillSlope": slope,
            "Asymmetry": 1.0,
        }

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for Hill5P parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [Bottom, Top, EC50, HillSlope, Asymmetry].
        """
        return (
            [-np.inf, -np.inf, _X_EPS, -np.inf, 1e-6],
            [np.inf, np.inf, np.inf, np.inf, np.inf],
        )

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Return None to use finite-difference Jacobian.

        Returns
        -------
        np.ndarray | None
            None.
        """
        return None


# ---------------------------------------------------------------------------
# Boltzmann
# ---------------------------------------------------------------------------


class Boltzmann:
    """Voltage-dependent activation (Boltzmann) sigmoid.

    Equation
    --------
    y = Bottom + (Top - Bottom) / (1 + exp((V50 - x) / Slope))

    Parameters
    ----------
    Bottom : float
        Lower asymptote.
    Top : float
        Upper asymptote.
    V50 : float
        x value at half-activation.
    Slope : float
        Slope factor (positive -> increasing with x; negative -> decreasing).
    """

    model_id: str = "boltzmann"
    param_names: list[str] = ["Bottom", "Top", "V50", "Slope"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate the Boltzmann sigmoid at *x*.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include Bottom, Top, V50, Slope.

        Returns
        -------
        np.ndarray
            Predicted response values.
        """
        bottom = params["Bottom"]
        top = params["Top"]
        v50 = params["V50"]
        slope = params["Slope"]
        slope_safe = slope if slope != 0.0 else 1e-10
        arg = np.clip((v50 - x) / slope_safe, -_EXP_CLIP, _EXP_CLIP)
        return np.asarray(bottom + (top - bottom) / (1.0 + np.exp(arg)))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for Boltzmann.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        y : np.ndarray
            Observed response values.

        Returns
        -------
        dict[str, float]
            Keys: Bottom, Top, V50, Slope.
        """
        bottom = float(np.min(y))
        top = float(np.max(y))
        if top == bottom:
            top = bottom + 1.0
        mid = 0.5 * (top + bottom)
        # Interpolate x at mid-response.
        sort_idx = np.argsort(y)
        v50 = float(np.interp(mid, y[sort_idx], x[sort_idx]))
        if not np.isfinite(v50):
            v50 = float(np.median(x))
        # Rough slope from data range.
        x_range = float(np.ptp(x))
        y_range = top - bottom
        if x_range > 0 and y_range > 0:
            slope_est = x_range / 4.0
        else:
            slope_est = 1.0
        return {"Bottom": bottom, "Top": top, "V50": v50, "Slope": slope_est}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for Boltzmann parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [Bottom, Top, V50, Slope].
        """
        return (
            [-np.inf, -np.inf, -np.inf, -np.inf],
            [np.inf, np.inf, np.inf, np.inf],
        )

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Return None to use finite-difference Jacobian.

        Returns
        -------
        np.ndarray | None
            None.
        """
        return None
