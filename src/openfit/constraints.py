# src/openfit/constraints.py
"""Algebraic parameter mapping and safe expression parsing for constraints."""

from __future__ import annotations

import re

import numpy as np

_ALLOWED_CHARS = re.compile(r"^[a-zA-Z0-9_\s\+\-\*\/\(\)\.,]+$")
_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
_MATH_HELPERS = {"abs", "min", "max"}


def safe_eval(expr: str, variables: dict[str, float]) -> float:
    """Evaluate a mathematical expression in a restricted safe context.

    Parameters
    ----------
    expr : str
        Expression to evaluate (e.g. "2 * EC50").
    variables : dict[str, float]
        Context containing variable values.

    Returns
    -------
    float
        Evaluation result.

    Raises
    ------
    ValueError
        If expression contains invalid characters or evaluation fails.
    """
    expr = expr.strip()
    if not _ALLOWED_CHARS.match(expr):
        raise ValueError(f"Invalid characters in expression: {expr!r}")

    # Restrict execution globals
    safe_globals = {
        "__builtins__": None,
        "abs": abs,
        "min": min,
        "max": max,
    }

    try:
        return float(eval(expr, safe_globals, variables))
    except Exception as exc:
        raise ValueError(f"Failed to evaluate constraint expression {expr!r}: {exc}") from exc


class ParameterMapper:
    """Map active optimizer parameter vector to full model parameters.

    Supports fixed values and algebraic constraints (e.g. linked variables).
    """

    def __init__(
        self,
        param_names: list[str],
        fixed: dict[str, float] | None = None,
        constraints: dict[str, str] | None = None,
    ) -> None:
        self.param_names = param_names
        self.fixed = fixed or {}
        self.constraints = constraints or {}

        # Validate parameter name inputs
        all_names_set = set(param_names)
        for name in self.fixed:
            if name not in all_names_set:
                raise ValueError(
                    f"Fixed parameter '{name}' is not in model parameters {param_names}."
                )
        for name in self.constraints:
            if name not in all_names_set:
                raise ValueError(
                    f"Constraint parameter '{name}' is not in model parameters {param_names}."
                )

        overlap = set(self.fixed.keys()) & set(self.constraints.keys())
        if overlap:
            raise ValueError(f"Parameters cannot be both fixed and constrained: {overlap}")

        # Active parameters optimized by solver
        self.free_names = [
            name for name in param_names if name not in self.fixed and name not in self.constraints
        ]

    def to_full(self, p_active: np.ndarray) -> dict[str, float]:
        """Map active parameter array back to full parameter dictionary.

        Parameters
        ----------
        p_active : np.ndarray
            Values of free parameters in free_names order.

        Returns
        -------
        dict[str, float]
            Full dictionary of all parameter values.
        """
        resolved = dict(zip(self.free_names, p_active, strict=False))
        resolved.update(self.fixed)

        pending = dict(self.constraints)
        n_params = len(self.param_names)

        # Topological sorting: resolve dependencies in steps
        for _ in range(n_params):
            if not pending:
                break
            made_progress = False
            for target, expr in list(pending.items()):
                tokens = _TOKEN_PATTERN.findall(expr)
                vars_needed = [t for t in tokens if t not in _MATH_HELPERS]
                if all(v in resolved for v in vars_needed):
                    resolved[target] = safe_eval(expr, resolved)
                    pending.pop(target)
                    made_progress = True

            if not made_progress:
                break

        if pending:
            raise ValueError(f"Unresolvable or cyclic dependencies in constraints: {pending}")

        return resolved

    def jacobian_mapping(self, p_active: np.ndarray) -> np.ndarray:
        """Calculate derivative matrix of full parameters w.r.t active ones.

        Returns matrix of shape (n_full, n_active).
        """
        n_full = len(self.param_names)
        n_active = len(self.free_names)
        if n_active == 0:
            return np.empty((n_full, 0))

        J = np.zeros((n_full, n_active))

        # 1. Fill exact derivatives for free parameters
        for j, name in enumerate(self.free_names):
            i = self.param_names.index(name)
            J[i, j] = 1.0

        # 2. Fill exact derivatives for fixed parameters (remain 0.0)

        # 3. For constrained parameters, use finite differences
        constrained_indices = [
            (i, name) for i, name in enumerate(self.param_names) if name in self.constraints
        ]

        if constrained_indices:
            h = 1e-8
            p_full_base = self.to_full(p_active)
            p_full_base_arr = np.array([p_full_base[name] for name in self.param_names])

            for j in range(n_active):
                p_active_step = p_active.copy()
                p_active_step[j] += h
                p_full_step = self.to_full(p_active_step)
                p_full_step_arr = np.array([p_full_step[name] for name in self.param_names])

                for i, _ in constrained_indices:
                    J[i, j] = (p_full_step_arr[i] - p_full_base_arr[i]) / h

        return J
