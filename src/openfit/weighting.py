"""openfit.weighting -- Weight schemes for weighted nonlinear least squares.

Weights scale the residuals: the optimizer minimizes sum(w_i * (y_i - f(x_i))**2).
Each apply_weights() call returns w_i (not sqrt(w_i)).  Downstream code in Fit()
multiplies by sqrt(w_i) before passing to least_squares.

Interpretation: w_i = 1 / sigma_i**2, so a measurement with large variance gets a
small weight and contributes less to the objective.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

class WeightScheme(str, Enum):
    """Named weight schemes for nonlinear fitting.

    Values are the canonical string aliases accepted by parse_weight_scheme().
    """

    UNIFORM = "uniform"
    ONE_OVER_Y = "1/y"
    ONE_OVER_Y2 = "1/y2"
    ONE_OVER_SD2 = "1/sd2"
    POISSON = "poisson"


# ---------------------------------------------------------------------------
# Alias table
# ---------------------------------------------------------------------------

_ALIASES: dict[str, WeightScheme] = {
    # UNIFORM
    "uniform": WeightScheme.UNIFORM,
    "none": WeightScheme.UNIFORM,
    # ONE_OVER_Y
    "1/y": WeightScheme.ONE_OVER_Y,
    "one_over_y": WeightScheme.ONE_OVER_Y,
    # ONE_OVER_Y2
    "1/y2": WeightScheme.ONE_OVER_Y2,
    "1/y^2": WeightScheme.ONE_OVER_Y2,
    "one_over_y2": WeightScheme.ONE_OVER_Y2,
    "one_over_y_squared": WeightScheme.ONE_OVER_Y2,
    # ONE_OVER_SD2
    "1/sd2": WeightScheme.ONE_OVER_SD2,
    "1/sd^2": WeightScheme.ONE_OVER_SD2,
    "one_over_sd2": WeightScheme.ONE_OVER_SD2,
    # POISSON
    "poisson": WeightScheme.POISSON,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_weight_scheme(s: str | WeightScheme) -> WeightScheme:
    """Parse a string or WeightScheme into a canonical WeightScheme member.

    Parameters
    ----------
    s : str | WeightScheme
        Canonical name or accepted alias.  Case-insensitive and strip-tolerant.
        Accepted aliases:
        - "uniform", "none" -> UNIFORM
        - "1/y", "one_over_y" -> ONE_OVER_Y
        - "1/y2", "1/y^2", "one_over_y2", "one_over_y_squared" -> ONE_OVER_Y2
        - "1/sd2", "1/sd^2", "one_over_sd2" -> ONE_OVER_SD2
        - "poisson" -> POISSON

    Returns
    -------
    WeightScheme
        Resolved enum member.

    Raises
    ------
    ValueError
        If the string does not match any known scheme or alias.
    """
    if isinstance(s, WeightScheme):
        return s

    key = str(s).strip().lower()
    result = _ALIASES.get(key)
    if result is None:
        valid = ", ".join(sorted(_ALIASES.keys()))
        raise ValueError(
            f"Unknown weight scheme: {s!r}. "
            f"Valid values and aliases are: {valid}"
        )
    return result


def apply_weights(
    y: np.ndarray,
    scheme: WeightScheme | str,
    sd: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the weight array for a given scheme.

    Returns w_i such that the weighted residual used by the optimizer is
    sqrt(w_i) * (y_i - f(x_i)).  Under the 1/sigma^2 interpretation,
    w_i = 1 / sigma_i**2.

    Parameters
    ----------
    y : np.ndarray
        Observed response values.  Shape (n,).  Must be finite; NaN or Inf
        raises ValueError regardless of scheme.
    scheme : WeightScheme | str
        Weight scheme.  Accepts WeightScheme enum members or string aliases
        (see parse_weight_scheme for the full alias list).
    sd : np.ndarray | None
        Standard deviations required for ONE_OVER_SD2.  Shape must match y.
        Must be strictly positive and finite.  Ignored for all other schemes.

    Returns
    -------
    np.ndarray
        Weight array of shape (n,).  All values are strictly positive and finite.

    Raises
    ------
    ValueError
        If y contains NaN or Inf values.
        If scheme is ONE_OVER_Y or ONE_OVER_Y2 and any y_i <= 0.
        If scheme is POISSON and any y_i <= 0.
        If scheme is ONE_OVER_SD2 and sd is None.
        If scheme is ONE_OVER_SD2 and sd contains non-positive or non-finite values.
        If scheme is ONE_OVER_SD2 and sd.shape != y.shape.
        If computed weights contain NaN, Inf, or non-positive values (backstop).
    """
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(
            "y contains NaN or Inf values. Clean the input data before fitting."
        )

    resolved = parse_weight_scheme(scheme)

    if resolved is WeightScheme.UNIFORM:
        weights = np.ones_like(y)

    elif resolved is WeightScheme.ONE_OVER_Y:
        if np.any(y <= 0):
            raise ValueError(
                "Weight scheme '1/y' requires all y values to be strictly positive, "
                "but y contains values <= 0.  Remove or replace non-positive observations "
                "before using this scheme."
            )
        weights = 1.0 / y

    elif resolved is WeightScheme.ONE_OVER_Y2:
        if np.any(y <= 0):
            raise ValueError(
                "Weight scheme '1/y2' requires all y values to be strictly positive, "
                "but y contains values <= 0.  Remove or replace non-positive observations "
                "before using this scheme."
            )
        weights = 1.0 / (y ** 2)

    elif resolved is WeightScheme.ONE_OVER_SD2:
        if sd is None:
            raise ValueError(
                "Weight scheme '1/sd2' requires replicate standard deviations. "
                "Pass sd=<array> to apply_weights()."
            )
        sd = np.asarray(sd, dtype=float)
        if sd.shape != y.shape:
            raise ValueError(
                f"sd.shape {sd.shape} does not match y.shape {y.shape}. "
                "sd must have one standard deviation per observation."
            )
        if not np.isfinite(sd).all():
            raise ValueError(
                "sd contains NaN or Inf values. Provide finite standard deviations."
            )
        if np.any(sd <= 0):
            raise ValueError(
                "Weight scheme '1/sd2' requires all sd values to be strictly positive, "
                "but sd contains values <= 0.  Zero or negative standard deviations "
                "are not physically meaningful."
            )
        weights = 1.0 / (sd ** 2)

    elif resolved is WeightScheme.POISSON:
        if np.any(y <= 0):
            raise ValueError(
                "Weight scheme 'poisson' requires all y values to be strictly positive, "
                "but y contains values <= 0.  Poisson weighting assumes variance equals "
                "the mean (count data), which requires positive counts."
            )
        weights = 1.0 / y

    else:
        # Unreachable given the enum definition, but keeps mypy and linters happy.
        raise ValueError(f"Unhandled weight scheme: {resolved!r}")

    # Backstop: ensure the result is valid regardless of floating-point edge cases.
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError(
            f"Computed weights for scheme '{resolved.value}' contain non-positive or "
            "non-finite values.  Check the input data for zeros, near-zeros, or "
            "extreme outliers."
        )

    return weights
