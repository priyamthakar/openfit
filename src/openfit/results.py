# src/openfit/results.py
"""FitResult: the output of a completed openfit fit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure

    from openfit.models.base import BaseModel
    from openfit.outliers import ROUTResult
    from openfit.spec import FitSpec


@dataclass
class FitResult:
    """Container for a completed nonlinear least-squares fit.

    Parameters
    ----------
    params : dict[str, float]
        Fitted parameter values keyed by parameter name.
    se : dict[str, float]
        Asymptotic standard errors keyed by parameter name.
    ci : dict[str, tuple[float, float]]
        95% confidence intervals keyed by parameter name (asymptotic by default).
    covariance : np.ndarray
        Full parameter covariance matrix of shape (n_params, n_params).
        Row/column order matches ``model.param_names`` (same order as
        ``params`` dict iteration).  Computed as ``(J^T J)^{-1} * s^2``
        where ``J`` is the weighted Jacobian at the solution and
        ``s^2 = weighted_RSS / (n_obs - n_params)``.  Diagonal elements
        equal ``se[name]**2``.
    r_squared : float
        Coefficient of determination R^2 = 1 - SS_res/SS_tot.
        For weighted fits both SS terms use the weight array.
    aic : float
        Akaike Information Criterion: n*ln(RSS/n) + 2k  (unweighted RSS).
    bic : float
        Bayesian Information Criterion: n*ln(RSS/n) + k*ln(n)  (unweighted RSS).
    aicc : float
        Bias-corrected AIC: AIC + 2k(k+1)/(n-k-1).
    rss : float
        Unweighted residual sum of squares.
    x : np.ndarray
        Independent-variable values used in the fit.
    y : np.ndarray
        Observed response values.
    y_fitted : np.ndarray
        Model-predicted values f(x, **params).
    residuals : np.ndarray
        Raw residuals y - y_fitted.
    weighted_residuals : np.ndarray
        Weighted residuals sqrt(w) * (y - y_fitted).
    standardized_residuals : np.ndarray
        Residuals divided by their standard deviation.
    n_obs : int
        Number of observations.
    n_params : int
        Number of fitted parameters.
    model_id : str
        Model identifier string (e.g. "hill4p").
    weight_scheme : str
        Weight scheme string (e.g. "1/y2", "uniform").
    spec : FitSpec
        Full reproducibility manifest.
    _model : BaseModel
        Model instance (private; used by plot helpers and uncertainty methods).
    _weights : np.ndarray
        Weight array w_i (private; used by uncertainty methods).
    """

    # Core fitted values
    params: dict[str, float]
    se: dict[str, float]
    ci: dict[str, tuple[float, float]]
    covariance: np.ndarray

    # Goodness of fit
    r_squared: float
    aic: float
    bic: float
    aicc: float
    rss: float

    # Data and residuals
    x: np.ndarray
    y: np.ndarray
    y_fitted: np.ndarray
    residuals: np.ndarray
    weighted_residuals: np.ndarray
    standardized_residuals: np.ndarray

    # Metadata
    n_obs: int
    n_params: int
    model_id: str
    weight_scheme: str
    spec: FitSpec

    # Private fields used by uncertainty and plotting helpers
    _model: BaseModel = field(repr=False, compare=False)
    _weights: np.ndarray = field(repr=False, compare=False)

    # Optional ROUT outlier detection result
    rout_result: ROUTResult | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # summary()
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return an ASCII summary table of the fit result.

        Returns
        -------
        str
            Multi-line ASCII string with parameter table and GOF statistics.
        """
        sep = "-" * 64
        lines: list[str] = []

        header = (
            f"Model: {self.model_id:<12}  "
            f"Weights: {self.weight_scheme:<8}  "
            f"n={self.n_obs}  params={self.n_params}"
        )
        lines.append(header)
        lines.append(sep)

        col_fmt = "{:<16}  {:>12}  {:>10}  {:>20}"
        lines.append(col_fmt.format("Parameter", "Value", "SE", "95% CI"))

        for name in self.params:
            val = self.params[name]
            se_val = self.se.get(name, float("nan"))
            lo, hi = self.ci.get(name, (float("nan"), float("nan")))
            ci_str = f"[{lo:>9.3f}, {hi:>9.3f}]"
            lines.append(col_fmt.format(name, f"{val:.6g}", f"{se_val:.3g}", ci_str))

        lines.append(sep)

        gof = (
            f"R^2={self.r_squared:.4f}  "
            f"AICc={self.aicc:.2f}  "
            f"BIC={self.bic:.2f}  "
            f"RSS={self.rss:.3e}"
        )
        lines.append(gof)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # plot()
    # ------------------------------------------------------------------

    def plot(
        self,
        log_x: bool = False,
        xlabel: str = "x",
        ylabel: str = "y",
    ) -> matplotlib.figure.Figure:
        """Return a fit overlay figure (observed data + fitted curve).

        Parameters
        ----------
        log_x : bool
            Use log10 x-axis scale.  Default False.
        xlabel : str
            X-axis label.  Default "x".
        ylabel : str
            Y-axis label.  Default "y".

        Returns
        -------
        matplotlib.figure.Figure
            The overlay figure.
        """
        from openfit.plotting import fit_overlay_plot

        return fit_overlay_plot(
            self,
            log_x=log_x,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    # ------------------------------------------------------------------
    # report()
    # ------------------------------------------------------------------

    def report(self, path: str, fmt: str = "html") -> None:
        """Write a fit report to a file.

        Parameters
        ----------
        path : str
            Output file path (e.g. "fit_result.html").
        fmt : str
            Output format: "html", "markdown", "pdf", or "docx". Default "html".

        Raises
        ------
        ImportError
            If the openfit.report module has not been installed (it requires the
            [reports] optional-dependency group).
        ValueError
            If fmt is not "html", "markdown", "pdf", or "docx".
        """
        _valid = {"html", "markdown", "pdf", "docx"}
        if fmt not in _valid:
            raise ValueError(f"fmt must be one of {sorted(_valid)}, got {fmt!r}.")

        try:
            from openfit.report import report_fit  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "openfit.report is not available. "
                "Install the reports extras with: pip install 'openfit[reports]'"
            ) from exc

        report_fit(self, path=path, fmt=fmt)
