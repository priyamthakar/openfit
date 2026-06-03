"""Synthetic certified values for receptor-ligand binding models.

No public database of certified binding parameters exists (analogous to the NIST
StRD for polynomial/nonlinear models). These datasets are generated analytically
from known exact parameters, following the same strategy used for 4PL/5PL
certified values in fourpl_certified_values.py.

Mathematical basis
------------------
OneSiteBinding
    Y = Bmax * X / (Kd + X)                        [Langmuir isotherm]
    Identity: Y = Bmax/2 at X = Kd (half-saturation at Kd).

TwoSiteBinding
    Y = Bmax1*X/(Kd1+X) + Bmax2*X/(Kd2+X)         [additive superposition]
    Two independent Langmuir sites; reduces to OneSiteBinding when one site
    dominates (Kd2 >> Kd1 and Bmax2 -> 0).

CompetitiveBinding
    Y = Bmax * X / (Kd_app + X)                    [Cheng-Prusoff]
    where Kd_app = Kd * (1 + I/Ki)
    Identity: Y = Bmax/2 at X = Kd_app.

These identities are verifiable by substitution. Tests that recover the exact
parameters from noise-free data prove the equation is correctly implemented;
tests at 1% and 5% noise probe numerical robustness of the optimizer.

References
----------
Langmuir, I. (1918). The adsorption of gases on plane surfaces of glass, mica
and platinum. J. Am. Chem. Soc., 40(9), 1361-1403.

Cheng, Y. & Prusoff, W.H. (1973). Relationship between the inhibition constant
(K_i) and the concentration of inhibitor which causes 50 per cent inhibition
(I50) of an enzymatic reaction. Biochem. Pharmacol., 22, 3099-3108.

Motulsky, H. & Christopoulos, A. (2003). Fitting Models to Biological Data
Using Linear and Nonlinear Regression. GraphPad Software. Chapters 7-9 cover
saturation radioligand binding analysis using the Langmuir isotherm and its
competitive-binding extension.

Datasets
--------
one_site_noise0     : OneSiteBinding, noise-free
one_site_noise1     : OneSiteBinding, 1% noise (sigma = 1% of Bmax)
one_site_noise5     : OneSiteBinding, 5% noise (sigma = 5% of Bmax)
two_site_noise0     : TwoSiteBinding, noise-free
two_site_noise1     : TwoSiteBinding, 1% noise (sigma = 1% of total Bmax)
two_site_noise5     : TwoSiteBinding, 5% noise (sigma = 5% of total Bmax)
competitive_noise0  : CompetitiveBinding, noise-free (I=10, Kd_app=30)
competitive_noise1  : CompetitiveBinding, 1% noise
competitive_noise5  : CompetitiveBinding, 5% noise

Usage
-----
    from tests.validation.binding_certified_values import BINDING_DATASETS
    ds = BINDING_DATASETS["one_site_noise1"]
    x, y = ds["x"], ds["y"]
    certified = ds["certified_params"]
    inhibitor_conc = ds.get("inhibitor_conc", 0.0)   # for CompetitiveBinding only
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# OneSiteBinding parameters
# ---------------------------------------------------------------------------
_ONE_SITE_BMAX: float = 100.0
_ONE_SITE_KD: float = 10.0

# X spans 3 decades below and above Kd (0.01 to 10000): 24 log-spaced points.
_ONE_SITE_X = np.geomspace(0.01, 10000.0, 24)


def _one_site(x: np.ndarray, bmax: float, kd: float) -> np.ndarray:
    return bmax * x / (kd + x)


_one_site_exact = _one_site(_ONE_SITE_X, _ONE_SITE_BMAX, _ONE_SITE_KD)

# ---------------------------------------------------------------------------
# TwoSiteBinding parameters
# ---------------------------------------------------------------------------
_TWO_SITE_BMAX1: float = 60.0
_TWO_SITE_KD1: float = 1.0
_TWO_SITE_BMAX2: float = 40.0
_TWO_SITE_KD2: float = 100.0
_TWO_SITE_TOTAL_BMAX: float = _TWO_SITE_BMAX1 + _TWO_SITE_BMAX2  # 100.0

# X spans both Kd1=1 and Kd2=100: 0.01 to 10000, 28 log-spaced points.
_TWO_SITE_X = np.geomspace(0.01, 10000.0, 28)


def _two_site(x: np.ndarray, bmax1: float, kd1: float, bmax2: float, kd2: float) -> np.ndarray:
    return bmax1 * x / (kd1 + x) + bmax2 * x / (kd2 + x)


_two_site_exact = _two_site(
    _TWO_SITE_X, _TWO_SITE_BMAX1, _TWO_SITE_KD1, _TWO_SITE_BMAX2, _TWO_SITE_KD2
)

# ---------------------------------------------------------------------------
# CompetitiveBinding parameters
# ---------------------------------------------------------------------------
_COMP_BMAX: float = 100.0
_COMP_KD: float = 5.0
_COMP_KI: float = 2.0
_COMP_I: float = 10.0  # fixed inhibitor concentration

# Apparent Kd = Kd*(1+I/Ki) = 5*(1+10/2) = 30.0  (Cheng-Prusoff 1973)
_COMP_KD_APP: float = _COMP_KD * (1.0 + _COMP_I / _COMP_KI)  # = 30.0

# X spans from well below to well above the apparent Kd: 0.1 to 3000.
_COMP_X = np.geomspace(0.1, 3000.0, 20)


def _competitive(
    x: np.ndarray, bmax: float, kd: float, ki: float, inhibitor_conc: float
) -> np.ndarray:
    kd_app = kd * (1.0 + inhibitor_conc / ki)
    return bmax * x / (kd_app + x)


_comp_exact = _competitive(_COMP_X, _COMP_BMAX, _COMP_KD, _COMP_KI, _COMP_I)

# ---------------------------------------------------------------------------
# Noise generation (seeded for reproducibility)
# ---------------------------------------------------------------------------
_rng = np.random.default_rng(43)  # distinct seed from fourpl_certified_values (42)

# OneSiteBinding noise
_one_noise_1pct = _rng.normal(0.0, 0.01 * _ONE_SITE_BMAX, len(_ONE_SITE_X))
_one_noise_5pct = _rng.normal(0.0, 0.05 * _ONE_SITE_BMAX, len(_ONE_SITE_X))

# TwoSiteBinding noise
_two_noise_1pct = _rng.normal(0.0, 0.01 * _TWO_SITE_TOTAL_BMAX, len(_TWO_SITE_X))
_two_noise_5pct = _rng.normal(0.0, 0.05 * _TWO_SITE_TOTAL_BMAX, len(_TWO_SITE_X))

# CompetitiveBinding noise
_comp_noise_1pct = _rng.normal(0.0, 0.01 * _COMP_BMAX, len(_COMP_X))
_comp_noise_5pct = _rng.normal(0.0, 0.05 * _COMP_BMAX, len(_COMP_X))

# ---------------------------------------------------------------------------
# Public dataset registry
# ---------------------------------------------------------------------------

BINDING_DATASETS: dict[str, dict] = {
    # OneSiteBinding
    "one_site_noise0": {
        "model_type": "one_site_binding",
        "x": _ONE_SITE_X.tolist(),
        "y": _one_site_exact.tolist(),
        "certified_params": {"Bmax": _ONE_SITE_BMAX, "Kd": _ONE_SITE_KD},
        "p0": {"Bmax": _ONE_SITE_BMAX, "Kd": _ONE_SITE_KD},
        "noise_level": 0.0,
        "inhibitor_conc": None,
    },
    "one_site_noise1": {
        "model_type": "one_site_binding",
        "x": _ONE_SITE_X.tolist(),
        "y": (_one_site_exact + _one_noise_1pct).tolist(),
        "certified_params": {"Bmax": _ONE_SITE_BMAX, "Kd": _ONE_SITE_KD},
        "p0": {"Bmax": _ONE_SITE_BMAX, "Kd": _ONE_SITE_KD},
        "noise_level": 0.01,
        "inhibitor_conc": None,
    },
    "one_site_noise5": {
        "model_type": "one_site_binding",
        "x": _ONE_SITE_X.tolist(),
        "y": (_one_site_exact + _one_noise_5pct).tolist(),
        "certified_params": {"Bmax": _ONE_SITE_BMAX, "Kd": _ONE_SITE_KD},
        "p0": {"Bmax": _ONE_SITE_BMAX, "Kd": _ONE_SITE_KD},
        "noise_level": 0.05,
        "inhibitor_conc": None,
    },
    # TwoSiteBinding
    "two_site_noise0": {
        "model_type": "two_site_binding",
        "x": _TWO_SITE_X.tolist(),
        "y": _two_site_exact.tolist(),
        "certified_params": {
            "Bmax1": _TWO_SITE_BMAX1,
            "Kd1": _TWO_SITE_KD1,
            "Bmax2": _TWO_SITE_BMAX2,
            "Kd2": _TWO_SITE_KD2,
        },
        "p0": {
            "Bmax1": _TWO_SITE_BMAX1,
            "Kd1": _TWO_SITE_KD1,
            "Bmax2": _TWO_SITE_BMAX2,
            "Kd2": _TWO_SITE_KD2,
        },
        "noise_level": 0.0,
        "inhibitor_conc": None,
    },
    "two_site_noise1": {
        "model_type": "two_site_binding",
        "x": _TWO_SITE_X.tolist(),
        "y": (_two_site_exact + _two_noise_1pct).tolist(),
        "certified_params": {
            "Bmax1": _TWO_SITE_BMAX1,
            "Kd1": _TWO_SITE_KD1,
            "Bmax2": _TWO_SITE_BMAX2,
            "Kd2": _TWO_SITE_KD2,
        },
        "p0": {
            "Bmax1": _TWO_SITE_BMAX1,
            "Kd1": _TWO_SITE_KD1,
            "Bmax2": _TWO_SITE_BMAX2,
            "Kd2": _TWO_SITE_KD2,
        },
        "noise_level": 0.01,
        "inhibitor_conc": None,
    },
    "two_site_noise5": {
        "model_type": "two_site_binding",
        "x": _TWO_SITE_X.tolist(),
        "y": (_two_site_exact + _two_noise_5pct).tolist(),
        "certified_params": {
            "Bmax1": _TWO_SITE_BMAX1,
            "Kd1": _TWO_SITE_KD1,
            "Bmax2": _TWO_SITE_BMAX2,
            "Kd2": _TWO_SITE_KD2,
        },
        "p0": {
            "Bmax1": _TWO_SITE_BMAX1,
            "Kd1": _TWO_SITE_KD1,
            "Bmax2": _TWO_SITE_BMAX2,
            "Kd2": _TWO_SITE_KD2,
        },
        "noise_level": 0.05,
        "inhibitor_conc": None,
    },
    # CompetitiveBinding (I=10, apparent Kd=30)
    "competitive_noise0": {
        "model_type": "competitive_binding",
        "x": _COMP_X.tolist(),
        "y": _comp_exact.tolist(),
        "certified_params": {"Bmax": _COMP_BMAX, "Kd": _COMP_KD, "Ki": _COMP_KI},
        "p0": {"Bmax": _COMP_BMAX, "Kd": _COMP_KD, "Ki": _COMP_KI},
        "noise_level": 0.0,
        "inhibitor_conc": _COMP_I,
    },
    "competitive_noise1": {
        "model_type": "competitive_binding",
        "x": _COMP_X.tolist(),
        "y": (_comp_exact + _comp_noise_1pct).tolist(),
        "certified_params": {"Bmax": _COMP_BMAX, "Kd": _COMP_KD, "Ki": _COMP_KI},
        "p0": {"Bmax": _COMP_BMAX, "Kd": _COMP_KD, "Ki": _COMP_KI},
        "noise_level": 0.01,
        "inhibitor_conc": _COMP_I,
    },
    "competitive_noise5": {
        "model_type": "competitive_binding",
        "x": _COMP_X.tolist(),
        "y": (_comp_exact + _comp_noise_5pct).tolist(),
        "certified_params": {"Bmax": _COMP_BMAX, "Kd": _COMP_KD, "Ki": _COMP_KI},
        "p0": {"Bmax": _COMP_BMAX, "Kd": _COMP_KD, "Ki": _COMP_KI},
        "noise_level": 0.05,
        "inhibitor_conc": _COMP_I,
    },
}
