"""Population growth models.

Models
------
Logistic3P       -- 3-parameter logistic (Verhulst)
Logistic4P       -- 4-parameter generalized logistic (simplified, Bottom=0)
Gompertz         -- Gompertz sigmoidal growth curve
AsymmetricGompertz -- asymmetric Gompertz with different left/right growth rates
Richards         -- Richards / generalized logistic (5 parameters)
"""

from __future__ import annotations

import numpy as np

_EXP_CLIP: float = 700.0


def _safe_exp(z: np.ndarray) -> np.ndarray:
    """Evaluate exp(z) with argument clipped to [-700, 700] to prevent overflow."""
    return np.exp(np.clip(z, -_EXP_CLIP, _EXP_CLIP))


def _growth_rate_guess(x: np.ndarray, y: np.ndarray, k: float) -> float:
    """Estimate intrinsic growth rate r from log-linear phase.

    Parameters
    ----------
    x : np.ndarray
        Time values.
    y : np.ndarray
        Population size values.
    k : float
        Estimated carrying capacity.

    Returns
    -------
    float
        Estimated growth rate; positive fallback if estimation fails.
    """
    pos = y[y > 0]
    x_pos = x[y > 0]
    if len(pos) < 2:
        return 1.0
    # Use only the sub-saturation region (y < 0.8 * K).
    sub_k = pos < 0.8 * k
    if sub_k.sum() < 2:
        sub_k = np.ones(len(pos), dtype=bool)
    log_y = np.log(pos[sub_k])
    xv = x_pos[sub_k]
    if len(xv) < 2:
        return 1.0
    slope, _ = np.polyfit(xv, log_y, 1)
    return float(max(abs(slope), 1e-6))


# ---------------------------------------------------------------------------
# Logistic3P
# ---------------------------------------------------------------------------


class Logistic3P:
    """3-parameter logistic (Verhulst) growth model.

    Equation
    --------
    y = K / (1 + ((K - N0) / N0) * exp(-r * x))

    Parameters
    ----------
    K : float
        Carrying capacity (maximum population size).
    N0 : float
        Initial population size at x=0.
    r : float
        Intrinsic growth rate (>0).
    """

    model_id: str = "logistic3p"
    param_names: list[str] = ["K", "N0", "r"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate Logistic3P at *x*.

        Parameters
        ----------
        x : np.ndarray
            Time (or independent-variable) values.
        **params : float
            Must include K, N0, r.

        Returns
        -------
        np.ndarray
            Predicted population values.
        """
        k = params["K"]
        n0 = params["N0"]
        r = params["r"]
        n0_safe = n0 if n0 != 0.0 else 1e-10
        ratio = (k - n0_safe) / n0_safe
        return np.asarray(k / (1.0 + ratio * _safe_exp(-r * x)))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for Logistic3P.

        Parameters
        ----------
        x : np.ndarray
            Time values.
        y : np.ndarray
            Observed population values.

        Returns
        -------
        dict[str, float]
            Keys: K, N0, r.
        """
        k = float(np.max(y)) * 1.05
        if k <= 0:
            k = 1.0
        idx0 = int(np.argmin(np.abs(x)))
        n0 = float(y[idx0])
        if n0 <= 0:
            n0 = k * 0.01
        r = _growth_rate_guess(x, y, k)
        return {"K": k, "N0": n0, "r": r}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for Logistic3P parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [K, N0, r].
        """
        return ([1e-300, 1e-300, 1e-300], [np.inf, np.inf, np.inf])

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Return None to use finite-difference Jacobian.

        Returns
        -------
        np.ndarray | None
            None.
        """
        return None


# ---------------------------------------------------------------------------
# Logistic4P
# ---------------------------------------------------------------------------


class Logistic4P:
    """4-parameter simplified logistic growth model.

    Equation
    --------
    y = K / (1 + exp(-r * (x - x_mid)))

    Parameters
    ----------
    K : float
        Carrying capacity / maximum response.
    r : float
        Steepness / growth rate (>0).
    x_mid : float
        Inflection point (time or x at half-K).
    """

    model_id: str = "logistic4p"
    param_names: list[str] = ["K", "r", "x_mid"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate Logistic4P at *x*.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include K, r, x_mid.

        Returns
        -------
        np.ndarray
            Predicted response values.
        """
        k = params["K"]
        r = params["r"]
        x_mid = params["x_mid"]
        arg = np.clip(-r * (x - x_mid), -_EXP_CLIP, _EXP_CLIP)
        return np.asarray(k / (1.0 + np.exp(arg)))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for Logistic4P.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        y : np.ndarray
            Observed response values.

        Returns
        -------
        dict[str, float]
            Keys: K, r, x_mid.
        """
        k = float(np.max(y)) * 1.05
        if k <= 0:
            k = 1.0
        half_k = k / 2.0
        sort_idx = np.argsort(y)
        x_mid = float(np.interp(half_k, y[sort_idx], x[sort_idx]))
        if not np.isfinite(x_mid):
            x_mid = float(np.median(x))
        r = _growth_rate_guess(x, y, k)
        return {"K": k, "r": r, "x_mid": x_mid}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for Logistic4P parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [K, r, x_mid].
        """
        return ([1e-300, 1e-300, -np.inf], [np.inf, np.inf, np.inf])

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Analytic Jacobian for Logistic4P.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include K, r, x_mid.

        Returns
        -------
        np.ndarray
            Jacobian matrix (n, 3) with columns [dK, dr, dx_mid].
        """
        k = params["K"]
        r = params["r"]
        x_mid = params["x_mid"]
        z = r * (x - x_mid)
        exp_neg_z = _safe_exp(-z)
        denom = 1.0 + exp_neg_z
        dK = 1.0 / denom
        dr = k * (x - x_mid) * exp_neg_z / (denom * denom)
        dx_mid = -k * r * exp_neg_z / (denom * denom)
        J = np.column_stack([dK, dr, dx_mid])
        return J


# ---------------------------------------------------------------------------
# Gompertz
# ---------------------------------------------------------------------------


class Gompertz:
    """Gompertz sigmoidal growth curve.

    Equation
    --------
    y = K * exp(-exp(-r * (x - t_inf)))

    Parameters
    ----------
    K : float
        Asymptotic maximum (carrying capacity).
    r : float
        Growth rate constant (>0).
    t_inf : float
        Inflection point (time of maximum growth rate).
    """

    model_id: str = "gompertz"
    param_names: list[str] = ["K", "r", "t_inf"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate Gompertz growth at *x*.

        Parameters
        ----------
        x : np.ndarray
            Time values.
        **params : float
            Must include K, r, t_inf.

        Returns
        -------
        np.ndarray
            Predicted population values.
        """
        k = params["K"]
        r = params["r"]
        t_inf = params["t_inf"]
        inner = np.clip(-r * (x - t_inf), -_EXP_CLIP, _EXP_CLIP)
        outer = np.clip(-np.exp(inner), -_EXP_CLIP, _EXP_CLIP)
        return np.asarray(k * np.exp(outer))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for Gompertz.

        Parameters
        ----------
        x : np.ndarray
            Time values.
        y : np.ndarray
            Observed population values.

        Returns
        -------
        dict[str, float]
            Keys: K, r, t_inf.
        """
        k = float(np.max(y)) * 1.05
        if k <= 0:
            k = 1.0
        r = _growth_rate_guess(x, y, k)
        # Inflection at ~37% of K for Gompertz.
        inflection_y = k / np.e
        sort_idx = np.argsort(y)
        t_inf = float(np.interp(inflection_y, y[sort_idx], x[sort_idx]))
        if not np.isfinite(t_inf):
            t_inf = float(np.median(x))
        return {"K": k, "r": r, "t_inf": t_inf}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for Gompertz parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [K, r, t_inf].
        """
        return ([1e-300, 1e-300, -np.inf], [np.inf, np.inf, np.inf])

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Analytic Jacobian for Gompertz.

        Parameters
        ----------
        x : np.ndarray
            Time values.
        **params : float
            Must include K, r, t_inf.

        Returns
        -------
        np.ndarray
            Jacobian matrix (n, 3) with columns [dK, dr, dt_inf].
        """
        k = params["K"]
        r = params["r"]
        t_inf = params["t_inf"]
        z = r * (x - t_inf)
        exp_neg_z = _safe_exp(-z)
        inner = np.clip(-z - exp_neg_z, -_EXP_CLIP, _EXP_CLIP)
        common = np.exp(inner)  # exp(-z - exp(-z))
        dK = _safe_exp(-exp_neg_z)  # exp(-exp(-z))
        dr = k * (x - t_inf) * common
        dt_inf = -k * r * common
        J = np.column_stack([dK, dr, dt_inf])
        return J


# ---------------------------------------------------------------------------
# Richards
# ---------------------------------------------------------------------------


class Richards:
    """Richards / 5-parameter generalized logistic growth model.

    Equation
    --------
    y = K / (1 + exp(-r * (x - x_mid)))^(1/v)

    Parameters
    ----------
    K : float
        Carrying capacity / asymptote.
    r : float
        Growth rate (>0).
    x_mid : float
        Inflection x value.
    v : float
        Shape parameter controlling asymmetry (>0); v=1 gives symmetric logistic.
    """

    model_id: str = "richards"
    param_names: list[str] = ["K", "r", "x_mid", "v"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate Richards growth at *x*.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include K, r, x_mid, v.

        Returns
        -------
        np.ndarray
            Predicted response values.
        """
        k = params["K"]
        r = params["r"]
        x_mid = params["x_mid"]
        v = params["v"]
        v_safe = v if v != 0.0 else 1e-10
        arg = np.clip(-r * (x - x_mid), -_EXP_CLIP, _EXP_CLIP)
        base = np.clip(1.0 + np.exp(arg), 1e-300, None)
        # (1 + exp(-r*(x-xm)))^(1/v) using exp(log(base)/v).
        log_pow = np.clip(np.log(base) / v_safe, -_EXP_CLIP, _EXP_CLIP)
        return np.asarray(k / np.exp(log_pow))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for Richards.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        y : np.ndarray
            Observed response values.

        Returns
        -------
        dict[str, float]
            Keys: K, r, x_mid, v.
        """
        k = float(np.max(y)) * 1.05
        if k <= 0:
            k = 1.0
        r = _growth_rate_guess(x, y, k)
        half_k = k / 2.0
        sort_idx = np.argsort(y)
        x_mid = float(np.interp(half_k, y[sort_idx], x[sort_idx]))
        if not np.isfinite(x_mid):
            x_mid = float(np.median(x))
        return {"K": k, "r": r, "x_mid": x_mid, "v": 1.0}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for Richards parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [K, r, x_mid, v].
        """
        return ([1e-300, 1e-300, -np.inf, 1e-10], [np.inf, np.inf, np.inf, np.inf])

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Return None to use finite-difference Jacobian.

        Returns
        -------
        np.ndarray | None
            None.
        """
        return None


# ---------------------------------------------------------------------------
# AsymmetricGompertz
# ---------------------------------------------------------------------------


class AsymmetricGompertz:
    """Asymmetric Gompertz sigmoidal growth curve with different left/right rates.

    Equation
    --------
    y = K * exp(-exp(-r_left  * (x - t_inf)))  for x <= t_inf
    y = K * exp(-exp(-r_right * (x - t_inf)))  for x >  t_inf

    Parameters
    ----------
    K : float
        Asymptotic maximum (carrying capacity).
    r_left : float
        Growth rate constant for the left side (x <= t_inf), >0.
    r_right : float
        Growth rate constant for the right side (x > t_inf), >0.
    t_inf : float
        Inflection point (time of maximum growth rate).
    """

    model_id: str = "gompertz_asym"
    param_names: list[str] = ["K", "r_left", "r_right", "t_inf"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate asymmetric Gompertz at *x*.

        Parameters
        ----------
        x : np.ndarray
            Time values.
        **params : float
            Must include K, r_left, r_right, t_inf.

        Returns
        -------
        np.ndarray
            Predicted population values.
        """
        k = params["K"]
        r_left = params["r_left"]
        r_right = params["r_right"]
        t_inf = params["t_inf"]

        left_mask = x <= t_inf
        inner_left = np.clip(-r_left * (x - t_inf), -_EXP_CLIP, _EXP_CLIP)
        inner_right = np.clip(-r_right * (x - t_inf), -_EXP_CLIP, _EXP_CLIP)
        outer_left = np.clip(-np.exp(inner_left), -_EXP_CLIP, _EXP_CLIP)
        outer_right = np.clip(-np.exp(inner_right), -_EXP_CLIP, _EXP_CLIP)
        y_left = k * np.exp(outer_left)
        y_right = k * np.exp(outer_right)
        return np.asarray(np.where(left_mask, y_left, y_right))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for asymmetric Gompertz.

        Parameters
        ----------
        x : np.ndarray
            Time values.
        y : np.ndarray
            Observed population values.

        Returns
        -------
        dict[str, float]
            Keys: K, r_left, r_right, t_inf.
        """
        k = float(np.max(y)) * 1.05
        if k <= 0:
            k = 1.0
        r = _growth_rate_guess(x, y, k)
        # Inflection at ~37% of K for Gompertz.
        inflection_y = k / np.e
        sort_idx = np.argsort(y)
        t_inf = float(np.interp(inflection_y, y[sort_idx], x[sort_idx]))
        if not np.isfinite(t_inf):
            t_inf = float(np.median(x))
        return {"K": k, "r_left": r, "r_right": r, "t_inf": t_inf}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for asymmetric Gompertz parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [K, r_left, r_right, t_inf].
        """
        return ([1e-300, 1e-300, 1e-300, -np.inf], [np.inf, np.inf, np.inf, np.inf])

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Compute the analytic Jacobian of asymmetric Gompertz at *x*.

        Parameters
        ----------
        x : np.ndarray
            Time values.
        **params : float
            Must include K, r_left, r_right, t_inf.

        Returns
        -------
        np.ndarray
            Jacobian matrix where columns are [K, r_left, r_right, t_inf].
        """
        k = params["K"]
        r_left = params["r_left"]
        r_right = params["r_right"]
        t_inf = params["t_inf"]

        left_mask = x <= t_inf
        inner_left = np.clip(-r_left * (x - t_inf), -_EXP_CLIP, _EXP_CLIP)
        inner_right = np.clip(-r_right * (x - t_inf), -_EXP_CLIP, _EXP_CLIP)
        y_l = k * np.exp(-np.exp(inner_left))
        y_r = k * np.exp(-np.exp(inner_right))

        # dy/dK
        dy_dk = np.where(left_mask, np.exp(-np.exp(inner_left)), np.exp(-np.exp(inner_right)))

        # dy/dr_left
        dy_drl = np.where(left_mask, y_l * np.exp(inner_left) * (x - t_inf), 0.0)

        # dy/dr_right
        dy_drr = np.where(left_mask, 0.0, y_r * np.exp(inner_right) * (x - t_inf))

        # dy/dt_inf
        dy_dt = np.where(
            left_mask, -y_l * np.exp(inner_left) * r_left, -y_r * np.exp(inner_right) * r_right
        )

        return np.column_stack([dy_dk, dy_drl, dy_drr, dy_dt])
