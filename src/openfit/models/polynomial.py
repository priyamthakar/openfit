"""Polynomial models (degree 1 through 6).

Each PolyN model fits y = a0 + a1*x + a2*x^2 + ... + aN*x^N.
Coefficients are stored in ascending degree order matching param_names.

Analytic Jacobians are provided for all polynomial models (trivial: column j is x^j).
"""

from __future__ import annotations

import numpy as np


def _poly_equation(x: np.ndarray, coeffs: list[float]) -> np.ndarray:
    """Evaluate a polynomial at *x* using ascending-degree coefficients.

    Parameters
    ----------
    x : np.ndarray
        Independent-variable values.
    coeffs : list[float]
        Polynomial coefficients in ascending degree order [a0, a1, ..., aN].

    Returns
    -------
    np.ndarray
        Evaluated polynomial values.
    """
    result = np.zeros_like(x, dtype=float)
    for j, c in enumerate(coeffs):
        result = result + c * x ** j
    return result


def _poly_jacobian(x: np.ndarray, degree: int) -> np.ndarray:
    """Compute the analytic Jacobian of a polynomial of the given degree.

    Parameters
    ----------
    x : np.ndarray
        Independent-variable values.
    degree : int
        Polynomial degree (number of parameters = degree + 1).

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (n, degree+1); column j is x^j.
    """
    cols = [x ** j for j in range(degree + 1)]
    return np.column_stack(cols)


def _poly_initial_guess(x: np.ndarray, y: np.ndarray, degree: int) -> dict[str, float]:
    """Compute least-squares polynomial initial guesses via np.polyfit.

    Parameters
    ----------
    x : np.ndarray
        Independent-variable values.
    y : np.ndarray
        Observed response values.
    degree : int
        Polynomial degree.

    Returns
    -------
    dict[str, float]
        Ascending-degree coefficients keyed as a0, a1, ..., aN.
    """
    n = min(degree, len(x) - 1)  # polyfit requires len(x) > degree
    n = max(n, 1)
    coeffs_desc = np.polyfit(x, y, n)  # highest degree first
    coeffs_asc = list(coeffs_desc[::-1])  # convert to ascending
    # Pad with zeros if degree > n.
    while len(coeffs_asc) < degree + 1:
        coeffs_asc.append(0.0)
    return {f"a{j}": float(coeffs_asc[j]) for j in range(degree + 1)}


# ---------------------------------------------------------------------------
# Concrete PolyN classes
# ---------------------------------------------------------------------------


class Poly1:
    """First-degree polynomial (linear).

    Equation: y = a0 + a1*x
    """

    model_id: str = "poly1"
    param_names: list[str] = ["a0", "a1"]
    _degree: int = 1

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate Poly1 at *x*.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Must include a0, a1.

        Returns
        -------
        np.ndarray
            Predicted response values.
        """
        coeffs = [params[f"a{j}"] for j in range(self._degree + 1)]
        return _poly_equation(x, coeffs)

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute polynomial initial guesses via least-squares fit.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        y : np.ndarray
            Observed response values.

        Returns
        -------
        dict[str, float]
            Coefficient initial estimates.
        """
        return _poly_initial_guess(x, y, self._degree)

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return unbounded box bounds for polynomial coefficients.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds (all unconstrained).
        """
        n = self._degree + 1
        return ([-np.inf] * n, [np.inf] * n)

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Analytic Jacobian for the polynomial (n_obs x n_params).

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Polynomial coefficients (unused; Jacobian depends only on x).

        Returns
        -------
        np.ndarray | None
            Jacobian matrix of shape (n, degree+1).
        """
        return _poly_jacobian(x, self._degree)


class Poly2(Poly1):
    """Second-degree polynomial (quadratic).

    Equation: y = a0 + a1*x + a2*x^2
    """

    model_id: str = "poly2"
    param_names: list[str] = ["a0", "a1", "a2"]
    _degree: int = 2


class Poly3(Poly1):
    """Third-degree polynomial (cubic).

    Equation: y = a0 + a1*x + a2*x^2 + a3*x^3
    """

    model_id: str = "poly3"
    param_names: list[str] = ["a0", "a1", "a2", "a3"]
    _degree: int = 3


class Poly4(Poly1):
    """Fourth-degree polynomial (quartic).

    Equation: y = a0 + a1*x + ... + a4*x^4
    """

    model_id: str = "poly4"
    param_names: list[str] = ["a0", "a1", "a2", "a3", "a4"]
    _degree: int = 4


class Poly5(Poly1):
    """Fifth-degree polynomial (quintic).

    Equation: y = a0 + a1*x + ... + a5*x^5
    """

    model_id: str = "poly5"
    param_names: list[str] = ["a0", "a1", "a2", "a3", "a4", "a5"]
    _degree: int = 5


class Poly6(Poly1):
    """Sixth-degree polynomial.

    Equation: y = a0 + a1*x + ... + a6*x^6
    """

    model_id: str = "poly6"
    param_names: list[str] = ["a0", "a1", "a2", "a3", "a4", "a5", "a6"]
    _degree: int = 6
