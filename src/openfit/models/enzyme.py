"""Enzyme kinetics models.

Models
------
MichaelisMenten     -- hyperbolic saturation kinetics; analytic Jacobian
SubstrateInhibition -- substrate inhibition (uncompetitive)
Allosteric          -- Hill kinetics with cooperativity
"""

from __future__ import annotations

import numpy as np

_X_EPS: float = 1e-300


# ---------------------------------------------------------------------------
# MichaelisMenten
# ---------------------------------------------------------------------------


class MichaelisMenten:
    """Michaelis-Menten enzyme kinetics.

    Equation
    --------
    y = Vmax * x / (Km + x)

    Parameters
    ----------
    Vmax : float
        Maximum reaction velocity (>0).
    Km : float
        Michaelis constant -- substrate concentration at half-Vmax (>0).
    """

    model_id: str = "michaelis_menten"
    param_names: list[str] = ["Vmax", "Km"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate Michaelis-Menten kinetics at *x*.

        Parameters
        ----------
        x : np.ndarray
            Substrate concentration values.
        **params : float
            Must include Vmax, Km.

        Returns
        -------
        np.ndarray
            Predicted reaction velocity values.
        """
        vmax = params["Vmax"]
        km = params["Km"]
        km_safe = km if km != 0.0 else _X_EPS
        return np.asarray(vmax * x / (km_safe + x))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for Michaelis-Menten.

        Parameters
        ----------
        x : np.ndarray
            Substrate concentration values.
        y : np.ndarray
            Observed reaction velocity values.

        Returns
        -------
        dict[str, float]
            Keys: Vmax, Km.
        """
        vmax = float(np.max(y)) * 1.1
        if vmax == 0.0:
            vmax = 1.0
        # Km: x at half-Vmax by interpolation.
        half_vmax = vmax / 2.0
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        if len(x_sorted) >= 2:
            km = float(np.interp(half_vmax, y_sorted, x_sorted))
        else:
            km = float(x[0])
        if km <= 0 or not np.isfinite(km):
            km = float(np.median(x[x > 0])) if np.any(x > 0) else 1.0
        return {"Vmax": vmax, "Km": km}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for Michaelis-Menten parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [Vmax, Km].
        """
        return ([1e-300, 1e-300], [np.inf, np.inf])

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Analytic Jacobian for Michaelis-Menten (n_obs x 2).

        Partial derivatives:
            dY/dVmax = x / (Km + x)
            dY/dKm   = -Vmax * x / (Km + x)^2

        Parameters
        ----------
        x : np.ndarray
            Substrate concentration values.
        **params : float
            Must include Vmax, Km.

        Returns
        -------
        np.ndarray | None
            Jacobian matrix of shape (n, 2).
        """
        vmax = params["Vmax"]
        km = params["Km"]
        km_safe = km if km != 0.0 else _X_EPS
        denom = km_safe + x
        denom2 = denom ** 2
        d_vmax = x / denom
        d_km = -vmax * x / denom2
        return np.column_stack([d_vmax, d_km])


# ---------------------------------------------------------------------------
# SubstrateInhibition
# ---------------------------------------------------------------------------


class SubstrateInhibition:
    """Substrate inhibition model (uncompetitive self-inhibition).

    Equation
    --------
    y = Vmax * x / (Km + x * (1 + x / Ki))

    Parameters
    ----------
    Vmax : float
        Maximum velocity in the absence of inhibition (>0).
    Km : float
        Apparent Michaelis constant (>0).
    Ki : float
        Inhibition constant -- substrate concentration causing half-maximal
        inhibition (>0).
    """

    model_id: str = "substrate_inhibition"
    param_names: list[str] = ["Vmax", "Km", "Ki"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate substrate inhibition kinetics at *x*.

        Parameters
        ----------
        x : np.ndarray
            Substrate concentration values.
        **params : float
            Must include Vmax, Km, Ki.

        Returns
        -------
        np.ndarray
            Predicted reaction velocity values.
        """
        vmax = params["Vmax"]
        km = params["Km"]
        ki = params["Ki"]
        ki_safe = ki if ki != 0.0 else _X_EPS
        denom = km + x * (1.0 + x / ki_safe)
        denom_safe = np.where(denom == 0.0, _X_EPS, denom)
        return np.asarray(vmax * x / denom_safe)

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for SubstrateInhibition.

        Parameters
        ----------
        x : np.ndarray
            Substrate concentration values.
        y : np.ndarray
            Observed reaction velocity values.

        Returns
        -------
        dict[str, float]
            Keys: Vmax, Km, Ki.
        """
        vmax = float(np.max(y)) * 1.2
        if vmax == 0.0:
            vmax = 1.0
        # Km: x at half of apparent maximum.
        half_vmax = np.max(y) / 2.0
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        if len(x_sorted) >= 2:
            km = float(np.interp(half_vmax, y_sorted, x_sorted))
        else:
            km = float(x[0])
        if km <= 0 or not np.isfinite(km):
            km = float(np.median(x[x > 0])) if np.any(x > 0) else 1.0
        # Ki: typically 10x the x value at peak velocity.
        peak_idx = int(np.argmax(y))
        ki = float(x[peak_idx]) * 10.0
        if ki <= 0 or not np.isfinite(ki):
            ki = float(np.max(x)) * 2.0
        return {"Vmax": vmax, "Km": km, "Ki": ki}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for SubstrateInhibition parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [Vmax, Km, Ki].
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
# Allosteric
# ---------------------------------------------------------------------------


class Allosteric:
    """Allosteric enzyme kinetics (Hill equation for cooperativity).

    Equation
    --------
    y = Vmax * x^n / (S_half^n + x^n)

    Parameters
    ----------
    Vmax : float
        Maximum reaction velocity (>0).
    S_half : float
        Substrate concentration at half-Vmax (>0).
    n : float
        Hill coefficient; n>1 positive cooperativity, n<1 negative cooperativity.
    """

    model_id: str = "allosteric"
    param_names: list[str] = ["Vmax", "S_half", "n"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate allosteric kinetics at *x*.

        Parameters
        ----------
        x : np.ndarray
            Substrate concentration values.
        **params : float
            Must include Vmax, S_half, n.

        Returns
        -------
        np.ndarray
            Predicted reaction velocity values.
        """
        vmax = params["Vmax"]
        s_half = params["S_half"]
        n = params["n"]
        s_half_safe = s_half if s_half != 0.0 else _X_EPS
        x_n = np.power(np.clip(x, 0.0, None), n)
        s_n = s_half_safe ** n
        denom = s_n + x_n
        denom_safe = np.where(denom == 0.0, _X_EPS, denom)
        return np.asarray(vmax * x_n / denom_safe)

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for Allosteric.

        Parameters
        ----------
        x : np.ndarray
            Substrate concentration values.
        y : np.ndarray
            Observed reaction velocity values.

        Returns
        -------
        dict[str, float]
            Keys: Vmax, S_half, n.
        """
        vmax = float(np.max(y)) * 1.1
        if vmax == 0.0:
            vmax = 1.0
        half_vmax = vmax / 2.0
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        if len(x_sorted) >= 2:
            s_half = float(np.interp(half_vmax, y_sorted, x_sorted))
        else:
            s_half = float(x[0])
        if s_half <= 0 or not np.isfinite(s_half):
            s_half = float(np.median(x[x > 0])) if np.any(x > 0) else 1.0
        return {"Vmax": vmax, "S_half": s_half, "n": 1.5}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for Allosteric parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [Vmax, S_half, n].
        """
        return ([1e-300, 1e-300, 1e-10], [np.inf, np.inf, np.inf])

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Return None to use finite-difference Jacobian.

        Returns
        -------
        np.ndarray | None
            None.
        """
        return None
