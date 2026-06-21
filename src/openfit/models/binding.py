"""Receptor-ligand binding models.

Models
------
OneSiteBinding     -- single-site Langmuir isotherm (2 parameters)
TwoSiteBinding     -- two independent binding sites (4 parameters)
CompetitiveBinding -- one-site with competitive inhibitor (3 parameters + I)
"""

from __future__ import annotations

import numpy as np

_X_EPS: float = 1e-300


# ---------------------------------------------------------------------------
# OneSiteBinding
# ---------------------------------------------------------------------------


class OneSiteBinding:
    """Single-site receptor-ligand binding (Langmuir isotherm).

    Equation
    --------
    y = Bmax * x / (Kd + x)

    Parameters
    ----------
    Bmax : float
        Maximum binding capacity (>0).
    Kd : float
        Dissociation constant -- ligand concentration at half-Bmax (>0).
    """

    model_id: str = "one_site_binding"
    param_names: list[str] = ["Bmax", "Kd"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate one-site binding at *x*.

        Parameters
        ----------
        x : np.ndarray
            Ligand concentration values.
        **params : float
            Must include Bmax, Kd.

        Returns
        -------
        np.ndarray
            Predicted bound-ligand values.
        """
        bmax = params["Bmax"]
        kd = params["Kd"]
        kd_safe = kd if kd != 0.0 else _X_EPS
        return np.asarray(bmax * x / (kd_safe + x))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for one-site binding.

        Parameters
        ----------
        x : np.ndarray
            Ligand concentration values.
        y : np.ndarray
            Observed bound-ligand values.

        Returns
        -------
        dict[str, float]
            Keys: Bmax, Kd.
        """
        bmax = float(np.max(y)) * 1.1
        if bmax == 0.0:
            bmax = 1.0
        kd = float(np.median(x))
        if kd <= 0 or not np.isfinite(kd):
            kd = 1.0
        return {"Bmax": bmax, "Kd": kd}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for one-site binding parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [Bmax, Kd].
        """
        return ([_X_EPS, _X_EPS], [np.inf, np.inf])

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Compute the analytic Jacobian of one-site binding at *x*.

        Parameters
        ----------
        x : np.ndarray
            Ligand concentration values.
        **params : float
            Must include Bmax, Kd.

        Returns
        -------
        np.ndarray
            Jacobian matrix of shape (n_obs, n_params) where columns are [Bmax, Kd].
        """
        bmax = params["Bmax"]
        kd = params["Kd"]
        kd_safe = kd if kd != 0.0 else _X_EPS
        D = kd_safe + x
        dy_dbmax = x / D
        dy_dkd = -bmax * x / (D**2)
        return np.column_stack([dy_dbmax, dy_dkd])


# ---------------------------------------------------------------------------
# TwoSiteBinding
# ---------------------------------------------------------------------------


class TwoSiteBinding:
    """Two independent binding sites (sum of two Langmuir isotherms).

    Equation
    --------
    y = Bmax1 * x / (Kd1 + x) + Bmax2 * x / (Kd2 + x)

    Parameters
    ----------
    Bmax1 : float
        Binding capacity of site 1 (>0).
    Kd1 : float
        Dissociation constant of site 1 (>0).
    Bmax2 : float
        Binding capacity of site 2 (>0).
    Kd2 : float
        Dissociation constant of site 2 (>0, typically > Kd1).
    """

    model_id: str = "two_site_binding"
    param_names: list[str] = ["Bmax1", "Kd1", "Bmax2", "Kd2"]

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate two-site binding at *x*.

        Parameters
        ----------
        x : np.ndarray
            Ligand concentration values.
        **params : float
            Must include Bmax1, Kd1, Bmax2, Kd2.

        Returns
        -------
        np.ndarray
            Predicted bound-ligand values.
        """
        bmax1 = params["Bmax1"]
        kd1 = params["Kd1"]
        bmax2 = params["Bmax2"]
        kd2 = params["Kd2"]
        kd1_safe = kd1 if kd1 != 0.0 else _X_EPS
        kd2_safe = kd2 if kd2 != 0.0 else _X_EPS
        site1 = bmax1 * x / (kd1_safe + x)
        site2 = bmax2 * x / (kd2_safe + x)
        return np.asarray(site1 + site2)

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for two-site binding.

        Parameters
        ----------
        x : np.ndarray
            Ligand concentration values.
        y : np.ndarray
            Observed bound-ligand values.

        Returns
        -------
        dict[str, float]
            Keys: Bmax1, Kd1, Bmax2, Kd2.
        """
        bmax_total = float(np.max(y)) * 1.1
        if bmax_total == 0.0:
            bmax_total = 1.0
        # Split total binding capacity between the two sites.
        bmax1 = bmax_total * 0.6
        bmax2 = bmax_total * 0.4
        x_sorted = np.sort(x)
        if len(x_sorted) >= 4:
            # Kd1 from the lower half of x, Kd2 from the upper half.
            kd1 = float(np.median(x_sorted[: len(x_sorted) // 2]))
            kd2 = float(np.median(x_sorted[len(x_sorted) // 2 :]))
        else:
            kd1 = max(float(x_sorted[0]) * 0.1, _X_EPS) if len(x_sorted) > 0 else 1.0
            kd2 = float(x_sorted[-1]) * 10.0 if len(x_sorted) > 0 else 100.0
        if kd1 <= 0 or not np.isfinite(kd1):
            kd1 = 1.0
        if kd2 <= 0 or not np.isfinite(kd2):
            kd2 = 100.0
        # Ensure Kd1 < Kd2.
        if kd1 >= kd2:
            kd1, kd2 = kd2 * 0.1, kd1 * 10.0
        return {"Bmax1": bmax1, "Kd1": kd1, "Bmax2": bmax2, "Kd2": kd2}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for two-site binding parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [Bmax1, Kd1, Bmax2, Kd2].
        """
        return (
            [_X_EPS, _X_EPS, _X_EPS, _X_EPS],
            [np.inf, np.inf, np.inf, np.inf],
        )

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Compute the analytic Jacobian of two-site binding at *x*.

        Parameters
        ----------
        x : np.ndarray
            Ligand concentration values.
        **params : float
            Must include Bmax1, Kd1, Bmax2, Kd2.

        Returns
        -------
        np.ndarray
            Jacobian matrix where columns are [Bmax1, Kd1, Bmax2, Kd2].
        """
        bmax1 = params["Bmax1"]
        kd1 = params["Kd1"]
        bmax2 = params["Bmax2"]
        kd2 = params["Kd2"]
        kd1_safe = kd1 if kd1 != 0.0 else _X_EPS
        kd2_safe = kd2 if kd2 != 0.0 else _X_EPS
        D1 = kd1_safe + x
        D2 = kd2_safe + x
        dy_dbmax1 = x / D1
        dy_dkd1 = -bmax1 * x / (D1**2)
        dy_dbmax2 = x / D2
        dy_dkd2 = -bmax2 * x / (D2**2)
        return np.column_stack([dy_dbmax1, dy_dkd1, dy_dbmax2, dy_dkd2])


# ---------------------------------------------------------------------------
# CompetitiveBinding
# ---------------------------------------------------------------------------


class CompetitiveBinding:
    """One-site binding in the presence of a competitive inhibitor.

    Equation
    --------
    y = Bmax * x / (Kd * (1 + I / Ki) + x)

    Parameters
    ----------
    Bmax : float
        Maximum binding capacity in the absence of inhibitor (>0).
    Kd : float
        Dissociation constant (>0).
    Ki : float
        Inhibition constant (>0).

    Construction
    ------------
    CompetitiveBinding(inhibitor_conc=0.0)

        *inhibitor_conc* (``I``) is a fixed concentration of the competing
        ligand.  With ``I=0`` the model reduces to the one-site isotherm.
    """

    model_id: str = "competitive_binding"
    param_names: list[str] = ["Bmax", "Kd", "Ki"]

    def __init__(self, inhibitor_conc: float = 0.0) -> None:
        self.I = inhibitor_conc

    def equation(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate competitive binding at *x*.

        Parameters
        ----------
        x : np.ndarray
            Ligand concentration values.
        **params : float
            Must include Bmax, Kd, Ki.

        Returns
        -------
        np.ndarray
            Predicted bound-ligand values.
        """
        bmax = params["Bmax"]
        kd = params["Kd"]
        ki = params["Ki"]
        kd_safe = kd if kd != 0.0 else _X_EPS
        ki_safe = ki if ki != 0.0 else _X_EPS
        apparent_kd = kd_safe * (1.0 + self.I / ki_safe)
        return np.asarray(bmax * x / (apparent_kd + x))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute data-driven initial estimates for competitive binding.

        Parameters
        ----------
        x : np.ndarray
            Ligand concentration values.
        y : np.ndarray
            Observed bound-ligand values.

        Returns
        -------
        dict[str, float]
            Keys: Bmax, Kd, Ki.
        """
        bmax = float(np.max(y)) * 1.1
        if bmax == 0.0:
            bmax = 1.0
        kd = float(np.median(x))
        if kd <= 0 or not np.isfinite(kd):
            kd = 1.0
        # Assume Ki = Kd as starting point (equal affinity).
        return {"Bmax": bmax, "Kd": kd, "Ki": kd}

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return box bounds for competitive binding parameters.

        Returns
        -------
        tuple[list[float], list[float]]
            Lower and upper bounds for [Bmax, Kd, Ki].
        """
        return ([_X_EPS, _X_EPS, _X_EPS], [np.inf, np.inf, np.inf])

    def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Compute the analytic Jacobian of competitive binding at *x*.

        Parameters
        ----------
        x : np.ndarray
            Ligand concentration values.
        **params : float
            Must include Bmax, Kd, Ki.

        Returns
        -------
        np.ndarray
            Jacobian matrix where columns are [Bmax, Kd, Ki].
        """
        bmax = params["Bmax"]
        kd = params["Kd"]
        ki = params["Ki"]
        kd_safe = kd if kd != 0.0 else _X_EPS
        ki_safe = ki if ki != 0.0 else _X_EPS
        app_kd = kd_safe * (1.0 + self.I / ki_safe)
        D = app_kd + x
        dy_dbmax = x / D
        dy_dkd = -bmax * x * (1.0 + self.I / ki_safe) / (D**2)
        dy_dki = bmax * x * kd_safe * self.I / (ki_safe**2 * D**2)
        return np.column_stack([dy_dbmax, dy_dkd, dy_dki])
