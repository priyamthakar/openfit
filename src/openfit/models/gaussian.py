"""Gaussian / peak-shape models.

Models
------
Gaussian   -- symmetric Gaussian peak
BiGaussian -- asymmetric Gaussian (different half-widths)
Lorentzian -- Cauchy / Lorentz peak
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------


class Gaussian:
    """Symmetric Gaussian (normal) peak.

    Equation
    --------
    y = A * exp(-(x - mu)^2 / (2 * sigma^2))

    Parameters
    ----------
    A : float
        Peak amplitude.
    mu : float
        Peak center (mean).
    sigma : float
        Peak width (standard deviation; >0).
    """

    model_id: str = "gaussian"
    param_names: list[str] = ["A", "mu", "sigma"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate Gaussian peak at *x*.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include A, mu, sigma.

        Returns
        -------
        np.ndarray
            Predicted response values.
        """
        a = params["A"]
        mu = params["mu"]
        sigma = params["sigma"]
        sigma_safe = sigma if sigma != 0.0 else 1e-10
        z = (x - mu) / sigma_safe
        return np.asarray(a * np.exp(-0.5 * z**2))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for Gaussian.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        y : np.ndarray
            Observed response values.

        Returns
        -------
        dict[str, float]
            Keys: A, mu, sigma.
        """
        a = float(np.max(y))
        if a == 0.0:
            a = 1.0
        mu = float(x[np.argmax(y)])
        # Estimate sigma as 1/4 of x range (heuristic half-width).
        x_range = float(np.ptp(x))
        sigma = x_range / 4.0 if x_range > 0 else 1.0
        return {"A": a, "mu": mu, "sigma": sigma}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for Gaussian parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [A, mu, sigma].
        """
        return ([-np.inf, -np.inf, 1e-10], [np.inf, np.inf, np.inf])

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Analytic Jacobian for Gaussian (n_obs x 3).

        Partial derivatives:
            z = (x - mu) / sigma,   g = exp(-0.5*z^2)
            dY/dA     = g
            dY/dmu    = A * z * g / sigma
            dY/dsigma = A * z^2 * g / sigma

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include A, mu, sigma.

        Returns
        -------
        np.ndarray | None
            Jacobian matrix of shape (n, 3).
        """
        a = params["A"]
        mu = params["mu"]
        sigma = params["sigma"]
        sigma_safe = sigma if sigma != 0.0 else 1e-10
        z = (x - mu) / sigma_safe
        g = np.exp(-0.5 * z**2)
        da = g
        dmu = a * z * g / sigma_safe
        dsigma = a * z**2 * g / sigma_safe
        return np.column_stack([da, dmu, dsigma])


# ---------------------------------------------------------------------------
# BiGaussian
# ---------------------------------------------------------------------------


class BiGaussian:
    """Asymmetric Gaussian with different widths on each side of the peak.

    Equation
    --------
    y = A * exp(-(x - mu)^2 / (2 * sigma_l^2))   for x <= mu
    y = A * exp(-(x - mu)^2 / (2 * sigma_r^2))   for x >  mu

    Parameters
    ----------
    A : float
        Peak amplitude.
    mu : float
        Peak center.
    sigma_l : float
        Left (descending) half-width (>0).
    sigma_r : float
        Right (ascending) half-width (>0).
    """

    model_id: str = "bigaussian"
    param_names: list[str] = ["A", "mu", "sigma_l", "sigma_r"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate BiGaussian at *x*.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include A, mu, sigma_l, sigma_r.

        Returns
        -------
        np.ndarray
            Predicted response values.
        """
        a = params["A"]
        mu = params["mu"]
        sigma_l = params["sigma_l"]
        sigma_r = params["sigma_r"]
        sl_safe = sigma_l if sigma_l != 0.0 else 1e-10
        sr_safe = sigma_r if sigma_r != 0.0 else 1e-10

        left_mask = x <= mu
        z_l = (x - mu) / sl_safe
        z_r = (x - mu) / sr_safe
        y_left = a * np.exp(-0.5 * z_l**2)
        y_right = a * np.exp(-0.5 * z_r**2)
        return np.asarray(np.where(left_mask, y_left, y_right))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for BiGaussian.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        y : np.ndarray
            Observed response values.

        Returns
        -------
        dict[str, float]
            Keys: A, mu, sigma_l, sigma_r.
        """
        a = float(np.max(y))
        if a == 0.0:
            a = 1.0
        peak_idx = int(np.argmax(y))
        mu = float(x[peak_idx])
        half_width = float(np.ptp(x)) / 4.0 if float(np.ptp(x)) > 0 else 1.0
        return {"A": a, "mu": mu, "sigma_l": half_width, "sigma_r": half_width}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for BiGaussian parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [A, mu, sigma_l, sigma_r].
        """
        return ([-np.inf, -np.inf, 1e-10, 1e-10], [np.inf, np.inf, np.inf, np.inf])

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Return None to use finite-difference Jacobian.

        Returns
        -------
        np.ndarray | None
            None.
        """
        return None


# ---------------------------------------------------------------------------
# Lorentzian
# ---------------------------------------------------------------------------


class Lorentzian:
    """Lorentzian (Cauchy) peak.

    Equation
    --------
    y = A / (1 + ((x - x0) / gamma)^2)

    Parameters
    ----------
    A : float
        Peak amplitude.
    x0 : float
        Peak center.
    gamma : float
        Half-width at half-maximum (>0).
    """

    model_id: str = "lorentzian"
    param_names: list[str] = ["A", "x0", "gamma"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate Lorentzian at *x*.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include A, x0, gamma.

        Returns
        -------
        np.ndarray
            Predicted response values.
        """
        a = params["A"]
        x0 = params["x0"]
        gamma = params["gamma"]
        gamma_safe = gamma if gamma != 0.0 else 1e-10
        z = (x - x0) / gamma_safe
        return np.asarray(a / (1.0 + z**2))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for Lorentzian.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        y : np.ndarray
            Observed response values.

        Returns
        -------
        dict[str, float]
            Keys: A, x0, gamma.
        """
        a = float(np.max(y))
        if a == 0.0:
            a = 1.0
        x0 = float(x[np.argmax(y)])
        # Gamma: half the x range as a conservative starting estimate.
        x_range = float(np.ptp(x))
        gamma = x_range / 4.0 if x_range > 0 else 1.0
        return {"A": a, "x0": x0, "gamma": gamma}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for Lorentzian parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [A, x0, gamma].
        """
        return ([1e-300, -np.inf, 1e-300], [np.inf, np.inf, np.inf])

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Analytic Jacobian for Lorentzian (n_obs x 3).

        Partial derivatives:
            z = (x - x0) / gamma,   denom = (1 + z^2)^2
            dY/dA     = 1 / (1 + z^2)
            dY/dx0    = 2*A*z / (gamma * denom)
            dY/dgamma = 2*A*z^2 / (gamma * denom)

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include A, x0, gamma.

        Returns
        -------
        np.ndarray | None
            Jacobian matrix of shape (n, 3).
        """
        a = params["A"]
        x0 = params["x0"]
        gamma = params["gamma"]
        gamma_safe = gamma if gamma != 0.0 else 1e-10
        z = (x - x0) / gamma_safe
        z2 = z**2
        inv_denom = 1.0 / (1.0 + z2)
        denom2 = (1.0 + z2) ** 2
        da = inv_denom
        dx0 = 2.0 * a * z / (gamma_safe * denom2)
        dgamma = 2.0 * a * z2 / (gamma_safe * denom2)
        return np.column_stack([da, dx0, dgamma])
