"""Exponential decay and growth models.

Models
------
MonoExp    -- single-exponential decay / rise to plateau
BiExp      -- double-exponential (fast + slow components)
ExpGrowth  -- pure exponential growth from Y0
ExpPlateau -- association to plateau (1 - exp)
ExpDecay   -- simple exponential decay to zero
"""

from __future__ import annotations

import numpy as np

_EXP_CLIP: float = 700.0


def _safe_exp(z: np.ndarray) -> np.ndarray:
    """Evaluate exp(z) with argument clipped to [-700, 700] to prevent overflow."""
    return np.exp(np.clip(z, -_EXP_CLIP, _EXP_CLIP))


def _estimate_rate(x: np.ndarray, y: np.ndarray, plateau: float) -> float:
    """Estimate a first-order rate constant k from (x, y - plateau) by log-linear fit.

    Parameters
    ----------
    x : np.ndarray
        Independent-variable (time) values.
    y : np.ndarray
        Observed response values.
    plateau : float
        Estimated asymptotic value.

    Returns
    -------
    float
        Estimated rate constant; strictly positive fallback if estimation fails.
    """
    resid = y - plateau
    pos_mask = resid > 0
    if pos_mask.sum() < 2:
        # Fallback: 1 / half-range of x.
        x_range = float(np.ptp(x))
        return 1.0 / x_range if x_range > 0 else 1.0
    log_resid = np.log(np.where(pos_mask, resid, np.nan))
    valid = np.isfinite(log_resid)
    if valid.sum() < 2:
        x_range = float(np.ptp(x))
        return 1.0 / x_range if x_range > 0 else 1.0
    slope, _ = np.polyfit(x[valid], log_resid[valid], 1)
    k = -slope  # decay: resid ~ A*exp(-k*x) => slope = -k
    return float(max(abs(k), 1e-10))


# ---------------------------------------------------------------------------
# MonoExp
# ---------------------------------------------------------------------------


class MonoExp:
    """Single-exponential approach to a plateau.

    Equation
    --------
    y = Plateau + (Y0 - Plateau) * exp(-k * x)

    Parameters
    ----------
    Y0 : float
        Response at x=0.
    Plateau : float
        Response as x -> infinity.
    k : float
        Rate constant (>0).
    """

    model_id: str = "monoexp"
    param_names: list[str] = ["Y0", "Plateau", "k"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate MonoExp at *x*.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include Y0, Plateau, k.

        Returns
        -------
        np.ndarray
            Predicted response values.
        """
        y0 = params["Y0"]
        plateau = params["Plateau"]
        k = params["k"]
        return np.asarray(plateau + (y0 - plateau) * _safe_exp(-k * x))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for MonoExp.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        y : np.ndarray
            Observed response values.

        Returns
        -------
        dict[str, float]
            Keys: Y0, Plateau, k.
        """
        idx0 = int(np.argmin(np.abs(x)))
        y0 = float(y[idx0])
        plateau = float(y[np.argmax(np.abs(x))])
        k = _estimate_rate(x, y, plateau)
        return {"Y0": y0, "Plateau": plateau, "k": k}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for MonoExp parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [Y0, Plateau, k].
        """
        return ([-np.inf, -np.inf, 1e-300], [np.inf, np.inf, np.inf])

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Return None to use finite-difference Jacobian.

        Returns
        -------
        np.ndarray | None
            None.
        """
        return None


# ---------------------------------------------------------------------------
# BiExp
# ---------------------------------------------------------------------------


class BiExp:
    """Double-exponential model with fast and slow components.

    Equation
    --------
    y = Plateau + Span_fast * exp(-k_fast * x) + Span_slow * exp(-k_slow * x)

    Parameters
    ----------
    Plateau : float
        Asymptotic baseline.
    Span_fast : float
        Amplitude of the fast component.
    k_fast : float
        Rate constant of the fast component (>0; should be > k_slow).
    Span_slow : float
        Amplitude of the slow component.
    k_slow : float
        Rate constant of the slow component (>0; should be < k_fast).

    Notes
    -----
    The constraint k_fast > k_slow is enforced only in initial_guess (k_fast is
    seeded at 10 * k_slow).  Box bounds only enforce k > 0 for both rates because
    a strict ordering constraint is not expressible as box bounds.
    """

    model_id: str = "biexp"
    param_names: list[str] = ["Plateau", "Span_fast", "k_fast", "Span_slow", "k_slow"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate BiExp at *x*.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include Plateau, Span_fast, k_fast, Span_slow, k_slow.

        Returns
        -------
        np.ndarray
            Predicted response values.
        """
        plateau = params["Plateau"]
        span_fast = params["Span_fast"]
        k_fast = params["k_fast"]
        span_slow = params["Span_slow"]
        k_slow = params["k_slow"]
        return np.asarray(
            plateau + span_fast * _safe_exp(-k_fast * x) + span_slow * _safe_exp(-k_slow * x)
        )

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for BiExp.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        y : np.ndarray
            Observed response values.

        Returns
        -------
        dict[str, float]
            Keys: Plateau, Span_fast, k_fast, Span_slow, k_slow.
        """
        plateau = float(y[-1]) if len(y) > 1 else 0.0
        total_span = float(y[0]) - plateau if len(y) > 0 else 1.0
        k_slow = _estimate_rate(x, y, plateau)
        k_fast = k_slow * 10.0  # seed k_fast > k_slow
        span_fast = total_span * 0.5
        span_slow = total_span * 0.5
        return {
            "Plateau": plateau,
            "Span_fast": span_fast,
            "k_fast": k_fast,
            "Span_slow": span_slow,
            "k_slow": k_slow,
        }

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for BiExp parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [Plateau, Span_fast, k_fast, Span_slow, k_slow].
        """
        return (
            [-np.inf, -np.inf, 1e-300, -np.inf, 1e-300],
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
# ExpGrowth
# ---------------------------------------------------------------------------


class ExpGrowth:
    """Pure exponential growth.

    Equation
    --------
    y = Y0 * exp(k * x)

    Parameters
    ----------
    Y0 : float
        Response at x=0 (must be > 0 for physically meaningful growth).
    k : float
        Growth rate constant (>0).
    """

    model_id: str = "expgrowth"
    param_names: list[str] = ["Y0", "k"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate ExpGrowth at *x*.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include Y0, k.

        Returns
        -------
        np.ndarray
            Predicted response values.
        """
        y0 = params["Y0"]
        k = params["k"]
        return np.asarray(y0 * _safe_exp(k * x))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for ExpGrowth.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        y : np.ndarray
            Observed response values.

        Returns
        -------
        dict[str, float]
            Keys: Y0, k.
        """
        pos = y[y > 0]
        y0 = float(y[0]) if y[0] > 0 else (float(pos[0]) if pos.size > 0 else 1.0)
        x_range = float(np.ptp(x))
        k = float(np.log(pos[-1] / pos[0]) / x_range) if x_range > 0 and pos.size >= 2 else 1.0
        k = max(abs(k), 1e-10)
        return {"Y0": y0, "k": k}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for ExpGrowth parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [Y0, k].
        """
        return ([1e-300, 1e-300], [np.inf, np.inf])

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Return None to use finite-difference Jacobian.

        Returns
        -------
        np.ndarray | None
            None.
        """
        return None


# ---------------------------------------------------------------------------
# ExpPlateau
# ---------------------------------------------------------------------------


class ExpPlateau:
    """Exponential association (approach to plateau from zero).

    Equation
    --------
    y = Plateau * (1 - exp(-k * x))

    Parameters
    ----------
    Plateau : float
        Maximum response (asymptote as x -> infinity).
    k : float
        Association rate constant (>0).
    """

    model_id: str = "expplateau"
    param_names: list[str] = ["Plateau", "k"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate ExpPlateau at *x*.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include Plateau, k.

        Returns
        -------
        np.ndarray
            Predicted response values.
        """
        plateau = params["Plateau"]
        k = params["k"]
        return np.asarray(plateau * (1.0 - _safe_exp(-k * x)))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for ExpPlateau.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        y : np.ndarray
            Observed response values.

        Returns
        -------
        dict[str, float]
            Keys: Plateau, k.
        """
        plateau = float(np.max(y))
        if plateau == 0.0:
            plateau = 1.0
        k = _estimate_rate(x, np.ones_like(y) * plateau - y, 0.0)
        return {"Plateau": plateau, "k": k}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for ExpPlateau parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [Plateau, k].
        """
        return ([-np.inf, 1e-300], [np.inf, np.inf])

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Return None to use finite-difference Jacobian.

        Returns
        -------
        np.ndarray | None
            None.
        """
        return None


# ---------------------------------------------------------------------------
# ExpDecay
# ---------------------------------------------------------------------------


class ExpDecay:
    """Simple exponential decay to zero.

    Equation
    --------
    y = Y0 * exp(-k * x)

    Parameters
    ----------
    Y0 : float
        Response at x=0.
    k : float
        Decay rate constant (>0).
    """

    model_id: str = "expdecay"
    param_names: list[str] = ["Y0", "k"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate ExpDecay at *x*.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include Y0, k.

        Returns
        -------
        np.ndarray
            Predicted response values.
        """
        y0 = params["Y0"]
        k = params["k"]
        return np.asarray(y0 * _safe_exp(-k * x))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for ExpDecay.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        y : np.ndarray
            Observed response values.

        Returns
        -------
        dict[str, float]
            Keys: Y0, k.
        """
        idx0 = int(np.argmin(np.abs(x)))
        y0 = float(y[idx0])
        if y0 == 0.0:
            y0 = float(np.max(np.abs(y))) if np.any(y != 0) else 1.0
        k = _estimate_rate(x, y, 0.0)
        return {"Y0": y0, "k": k}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for ExpDecay parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [Y0, k].
        """
        return ([-np.inf, 1e-300], [np.inf, np.inf])

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Return None to use finite-difference Jacobian.

        Returns
        -------
        np.ndarray | None
            None.
        """
        return None
