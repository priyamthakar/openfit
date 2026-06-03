"""User-defined custom model.

CustomModel wraps any user-supplied callable or expression string, making it
compatible with the openfit Fit engine without requiring a new class.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np


class CustomModel:
    """User-defined model from a callable or expression string.

    The model is registered like any built-in model, but its equation, parameter
    names, and initial-guess logic come from the caller.

    Parameters
    ----------
    model_id : str
        User-chosen identifier string.
    func : Callable[..., np.ndarray]
        Callable with signature ``func(x, param1, param2, ...) -> np.ndarray``.
        The first positional argument must be ``x`` (the independent variable
        array); all remaining arguments are the named model parameters.
    param_names : list[str] | None, optional
        Ordered list of parameter names.  If ``None``, inferred from the
        function signature by inspecting all positional parameters after ``x``.
        ``*args`` and ``**kwargs`` are not supported for auto-inference.
    initial_guess_func : Callable[[np.ndarray, np.ndarray], dict[str, float]] | None
        Optional function ``(x, y) -> dict[str, float]`` returning data-driven
        initial guesses.  If ``None``, all parameters are initialised to 1.0.
    bounds_dict : dict[str, tuple[float, float]] | None, optional
        Per-parameter bounds as ``{name: (lower, upper)}``.  Parameters not
        listed are treated as unbounded (``-inf`` / ``inf``).

    Raises
    ------
    TypeError
        If *func* is not callable.
    ValueError
        If *param_names* is empty or cannot be inferred from the signature.

    Examples
    --------
    Define a simple power-law model:

    >>> import numpy as np
    >>> from openfit.models.custom import CustomModel
    >>> def power_law(x, A, n):
    ...     return A * x ** n
    >>> model = CustomModel(model_id="power_law", func=power_law)
    >>> model.param_names
    ['A', 'n']
    """

    def __init__(
        self,
        model_id: str,
        func: Callable[..., np.ndarray],
        param_names: list[str] | None = None,
        initial_guess_func: Callable[[np.ndarray, np.ndarray], dict[str, float]] | None = None,
        bounds_dict: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func)!r}")

        self.model_id: str = model_id
        self._func = func
        self._initial_guess_func = initial_guess_func
        self._bounds_dict: dict[str, tuple[float, float]] = bounds_dict or {}

        if param_names is not None:
            if len(param_names) == 0:
                raise ValueError("param_names must not be empty.")
            self.param_names: list[str] = list(param_names)
        else:
            self.param_names = self._infer_param_names(func)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_param_names(func: Callable[..., np.ndarray]) -> list[str]:
        """Infer parameter names from the function signature.

        Parameters
        ----------
        func : Callable[..., np.ndarray]
            The model function.

        Returns
        -------
        list[str]
            Parameter names (all positional parameters after the first one).

        Raises
        ------
        ValueError
            If fewer than 2 positional parameters exist (needs at least x + 1 param)
            or if the signature uses *args / **kwargs exclusively.
        """
        sig = inspect.signature(func)
        params = [
            name
            for name, p in sig.parameters.items()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(params) < 2:
            raise ValueError(
                "Cannot infer param_names: the function must have at least two "
                "positional parameters (x plus at least one model parameter)."
            )
        # Skip the first parameter (assumed to be x).
        return params[1:]

    # ------------------------------------------------------------------
    # BaseModel Protocol implementation
    # ------------------------------------------------------------------

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate the custom model at *x*.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        **params : float
            Keyword arguments corresponding to the model parameters.

        Returns
        -------
        np.ndarray
            Predicted response values.
        """
        ordered_vals = [params[name] for name in self.param_names]
        return np.asarray(self._func(x, *ordered_vals))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute initial parameter estimates.

        Uses the caller-supplied function if provided; otherwise returns 1.0
        for every parameter.

        Parameters
        ----------
        x : np.ndarray
            Independent-variable values.
        y : np.ndarray
            Observed response values.

        Returns
        -------
        dict[str, float]
            Mapping from parameter name to initial estimate.
        """
        if self._initial_guess_func is not None:
            return dict(self._initial_guess_func(x, y))
        return {name: 1.0 for name in self.param_names}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for all parameters.

        Uses per-parameter bounds from *bounds_dict*; unbounded (-inf/inf)
        for parameters not listed.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds in the same order as param_names.
        """
        lower = []
        upper = []
        for name in self.param_names:
            if name in self._bounds_dict:
                lo, hi = self._bounds_dict[name]
            else:
                lo, hi = -np.inf, np.inf
            lower.append(lo)
            upper.append(hi)
        return (lower, upper)

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray | None:
        """Return None to use finite-difference Jacobian.

        Returns
        -------
        np.ndarray | None
            None (custom models do not provide analytic Jacobians by default).
        """
        return None
