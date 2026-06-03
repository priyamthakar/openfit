"""BaseModel protocol and model registration.

Every openfit built-in model satisfies this Protocol.  Third-party models can
also satisfy it by implementing the same four methods and two class variables.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BaseModel(Protocol):
    """Protocol that every openfit model must satisfy.

    Attributes
    ----------
    model_id : str
        Unique lowercase identifier for the model, e.g. ``"hill4p"``.
    param_names : list[str]
        Ordered list of parameter names, e.g. ``["Bottom", "Top", "EC50", "HillSlope"]``.
        The order here governs the order of the parameter vector passed to
        ``scipy.optimize.least_squares``.
    """

    model_id: str
    param_names: list[str]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate the model at *x* given named parameter values.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.  Shape (n,).
        **params : float
            Keyword arguments whose keys match ``param_names``.

        Returns
        -------
        np.ndarray
            Predicted response values.  Shape (n,).
        """
        ...

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial parameter estimates.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.  Shape (n,).
        y : np.ndarray
            Observed response values.  Shape (n,).

        Returns
        -------
        dict[str, float]
            Mapping from parameter name to initial estimate.
            Must never contain NaN, Inf, or random values.
        """
        ...

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for all parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            ``(lower_bounds, upper_bounds)`` where each list has one entry
            per parameter in the same order as ``param_names``.
            Use ``-np.inf`` / ``np.inf`` for unbounded parameters.
        """
        ...

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Compute the analytic Jacobian of the model.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.  Shape (n,).
        **params : float
            Keyword arguments whose keys match ``param_names``.

        Returns
        -------
        np.ndarray | None
            Matrix of shape ``(n_obs, n_params)`` where column *j* is the
            partial derivative of the model with respect to ``param_names[j]``.
            Return ``None`` to fall back to finite-difference approximation.
        """
        ...
