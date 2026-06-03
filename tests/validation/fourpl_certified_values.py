"""Synthetic 4PL and 5PL certified datasets with known exact parameters.

Generates noise-free and noisy sigmoidal dose-response datasets at import time.
The exact (certified) parameter values are known by construction, enabling
validation of parameter recovery by the fitting engine.

4PL equation: y = Bottom + (Top - Bottom) / (1 + (EC50 / x)^HillSlope)
5PL equation: y = Bottom + (Top - Bottom) / (1 + (EC50 / x)^HillSlope)^Asymmetry

Datasets
--------
- fourpl_noise0 : 4PL, noise-free
- fourpl_noise1 : 4PL, 1% Gaussian noise (std = 1% of response range)
- fourpl_noise5 : 4PL, 5% Gaussian noise (std = 5% of response range)
- fivepl_noise0 : 5PL, noise-free (Asymmetry=0.5, genuinely asymmetric)
- fivepl_noise1 : 5PL, 1% Gaussian noise
- fivepl_noise5 : 5PL, 5% Gaussian noise

All datasets use:
- 50 x-points log-spaced from 0.001 to 1000
- random_seed = 42 for reproducibility
- 4PL: Bottom=0, Top=100, EC50=1.0, HillSlope=1.0
- 5PL: Bottom=0, Top=100, EC50=1.0, HillSlope=1.0, Asymmetry=0.5

Usage
-----
    from tests.validation.fourpl_certified_values import SYNTH_DATASETS
    ds = SYNTH_DATASETS["fourpl_noise1"]
    x, y = ds["x"], ds["y"]
    certified = ds["certified_params"]
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Common parameters
# ---------------------------------------------------------------------------
BOTTOM = 0.0
TOP = 100.0
EC50 = 1.0
HILLSLOPE = 1.0
ASYMMETRY = 0.5  # Non-unity so 5PL is genuinely asymmetric (not degenerate with 4PL)
N_POINTS = 50
X_MIN = 0.001
X_MAX = 1000.0
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Model functions (pure numpy, used only for data generation)
# ---------------------------------------------------------------------------


def _hill4p_eval(
    x: np.ndarray, bottom: float, top: float, ec50: float, hillslope: float
) -> np.ndarray:
    """Evaluate 4PL model at x."""
    x_safe = np.where(x == 0.0, 1e-300, x)
    log_ratio = hillslope * (np.log(np.abs(ec50)) - np.log(np.abs(x_safe)))
    log_ratio = np.clip(log_ratio, -700.0, 700.0)
    ratio = np.exp(log_ratio)
    return bottom + (top - bottom) / (1.0 + ratio)


def _hill5p_eval(
    x: np.ndarray,
    bottom: float,
    top: float,
    ec50: float,
    hillslope: float,
    asymmetry: float,
) -> np.ndarray:
    """Evaluate 5PL model at x."""
    x_safe = np.where(x == 0.0, 1e-300, x)
    log_ratio = hillslope * (np.log(np.abs(ec50)) - np.log(np.abs(x_safe)))
    log_ratio = np.clip(log_ratio, -700.0, 700.0)
    ratio = np.exp(log_ratio)
    inner = np.clip(1.0 + ratio, 1e-300, None)
    log_inner = np.log(inner)
    log_denom = np.clip(asymmetry * log_inner, -700.0, 700.0)
    denom = np.exp(log_denom)
    return bottom + (top - bottom) / denom


# ---------------------------------------------------------------------------
# Generate datasets
# ---------------------------------------------------------------------------

# Shared x-axis: 50 points log-spaced from 0.001 to 1000
_x = np.geomspace(X_MIN, X_MAX, N_POINTS)

# Noise-free 4PL y-values
_y_4pl_exact = _hill4p_eval(_x, BOTTOM, TOP, EC50, HILLSLOPE)

# Noise-free 5PL y-values
_y_5pl_exact = _hill5p_eval(_x, BOTTOM, TOP, EC50, HILLSLOPE, ASYMMETRY)

# Generate noise vectors (seeded for reproducibility)
_rng = np.random.default_rng(RANDOM_SEED)
_y_range = TOP - BOTTOM  # 100.0

_noise_1pct = _rng.normal(0.0, 0.01 * _y_range, N_POINTS)
_noise_5pct = _rng.normal(0.0, 0.05 * _y_range, N_POINTS)

# Noisy 4PL datasets
_y_4pl_1pct = _y_4pl_exact + _noise_1pct
_y_4pl_5pct = _y_4pl_exact + _noise_5pct

# Noisy 5PL datasets (use independent noise draws for 5PL)
_noise_5pl_1pct = _rng.normal(0.0, 0.01 * _y_range, N_POINTS)
_noise_5pl_5pct = _rng.normal(0.0, 0.05 * _y_range, N_POINTS)
_y_5pl_1pct = _y_5pl_exact + _noise_5pl_1pct
_y_5pl_5pct = _y_5pl_exact + _noise_5pl_5pct

# ---------------------------------------------------------------------------
# Certified parameter dicts
# ---------------------------------------------------------------------------
_FOURPL_CERTIFIED = {
    "Bottom": BOTTOM,
    "Top": TOP,
    "EC50": EC50,
    "HillSlope": HILLSLOPE,
}

_FIVEPL_CERTIFIED = {
    "Bottom": BOTTOM,
    "Top": TOP,
    "EC50": EC50,
    "HillSlope": HILLSLOPE,
    "Asymmetry": ASYMMETRY,
}

# Initial guesses for fitting (exact values -- tests optimizer convergence,
# not initial-guess quality; 5PL parameters are confounded so the optimizer
# needs a reasonable starting point to converge to the global minimum)
_FOURPL_P0 = dict(_FOURPL_CERTIFIED)
_FIVEPL_P0 = dict(_FIVEPL_CERTIFIED)

# ---------------------------------------------------------------------------
# Public dataset registry
# ---------------------------------------------------------------------------

SYNTH_DATASETS: dict[str, dict] = {
    "fourpl_noise0": {
        "x": _x.tolist(),
        "y": _y_4pl_exact.tolist(),
        "certified_params": dict(_FOURPL_CERTIFIED),
        "p0": dict(_FOURPL_P0),
        "noise_level": 0.0,
        "model_type": "hill4p",
    },
    "fourpl_noise1": {
        "x": _x.tolist(),
        "y": _y_4pl_1pct.tolist(),
        "certified_params": dict(_FOURPL_CERTIFIED),
        "p0": dict(_FOURPL_P0),
        "noise_level": 0.01,
        "model_type": "hill4p",
    },
    "fourpl_noise5": {
        "x": _x.tolist(),
        "y": _y_4pl_5pct.tolist(),
        "certified_params": dict(_FOURPL_CERTIFIED),
        "p0": dict(_FOURPL_P0),
        "noise_level": 0.05,
        "model_type": "hill4p",
    },
    "fivepl_noise0": {
        "x": _x.tolist(),
        "y": _y_5pl_exact.tolist(),
        "certified_params": dict(_FIVEPL_CERTIFIED),
        "p0": dict(_FIVEPL_P0),
        "noise_level": 0.0,
        "model_type": "hill5p",
    },
    "fivepl_noise1": {
        "x": _x.tolist(),
        "y": _y_5pl_1pct.tolist(),
        "certified_params": dict(_FIVEPL_CERTIFIED),
        "p0": dict(_FIVEPL_P0),
        "noise_level": 0.01,
        "model_type": "hill5p",
    },
    "fivepl_noise5": {
        "x": _x.tolist(),
        "y": _y_5pl_5pct.tolist(),
        "certified_params": dict(_FIVEPL_CERTIFIED),
        "p0": dict(_FIVEPL_P0),
        "noise_level": 0.05,
        "model_type": "hill5p",
    },
}
