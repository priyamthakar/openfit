"""Publication-quality plots for openfit FitResult objects.

All plot functions return a live matplotlib Figure.  To embed in HTML reports,
pass the returned figure to figure_to_base64().  figure_to_base64() closes the
figure after encoding so that long-running processes do not accumulate memory.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend -- safe for server/report use

if TYPE_CHECKING:
    from openfit.outliers import ROUTResult
    from openfit.results import FitResult

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STYLE = "seaborn-v0_8-whitegrid"
_COLOR_DATA = "#2563EB"  # blue
_COLOR_FIT = "#DC2626"  # red
_COLOR_OUTLIER = "#DC2626"  # red for outliers
_COLOR_NORMAL = "#2563EB"  # blue for normal points
_COLOR_POS = "#2563EB"  # positive residuals -- blue
_COLOR_NEG = "#DC2626"  # negative residuals -- red
_COLOR_ZERO = "#6B7280"  # dashed zero line -- gray


def _make_axes(
    ax: matplotlib.axes.Axes | None,
    figsize: tuple[float, float],
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Return (fig, ax), creating a new figure only when ax is None."""
    if ax is None:
        fig, new_ax = plt.subplots(figsize=figsize)
        return fig, new_ax
    return ax.figure, ax  # type: ignore[return-value]


def _try_style() -> object:
    """Return a style context manager, falling back to default if unavailable."""
    try:
        return plt.style.context(_STYLE)
    except OSError:
        return plt.style.context("default")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_overlay_plot(
    result: FitResult,
    ax: matplotlib.axes.Axes | None = None,
    x_smooth_points: int = 200,
    log_x: bool = False,
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    show_ci: bool = False,
    figsize: tuple[float, float] = (7, 5),
) -> matplotlib.figure.Figure:
    """Plot observed data + fitted curve overlay.

    Parameters
    ----------
    result : FitResult
        Completed fit result.
    ax : matplotlib.axes.Axes | None
        Axes to draw on.  If None, creates a new figure.
    x_smooth_points : int
        Number of points for smooth curve interpolation.  Default 200.
    log_x : bool
        Use log10 scale for x axis.  Default False.
    title : str | None
        Plot title.  If None, uses "model_id fit (R^2 = {r2:.4f})".
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    show_ci : bool
        If True and result.ci is populated, draw 95% CI band.  Default False.
    figsize : tuple[float, float]
        Figure size in inches.  Default (7, 5).

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    Raises
    ------
    ValueError
        If log_x is True and x contains non-positive values.
    """
    x: np.ndarray = np.asarray(result.x, dtype=float)
    y: np.ndarray = np.asarray(result.y, dtype=float)

    if log_x and np.any(x <= 0):
        raise ValueError(
            "log_x=True requires all x values to be strictly positive, but x contains values <= 0."
        )

    # Build smooth x range for the fitted curve.
    if log_x:
        x_smooth = np.logspace(np.log10(x.min()), np.log10(x.max()), x_smooth_points)
    else:
        x_smooth = np.linspace(x.min(), x.max(), x_smooth_points)

    y_smooth: np.ndarray = result._model.equation(x_smooth, **result.params)

    with _try_style():
        fig, current_ax = _make_axes(ax, figsize)

        # Observed data points.
        current_ax.scatter(
            x,
            y,
            color=_COLOR_DATA,
            alpha=0.85,
            s=40,
            zorder=3,
            label="Observed",
        )

        # Fitted curve.
        current_ax.plot(
            x_smooth,
            y_smooth,
            color=_COLOR_FIT,
            linewidth=2,
            zorder=2,
            label="Fit",
        )

        # Optional CI band.
        if show_ci:
            ci = getattr(result, "ci", None)
            if ci is not None:
                # ci is expected to be a dict or object with lower/upper arrays at x.
                # Only draw if lower and upper keys are present.
                lower = ci.get("lower") if isinstance(ci, dict) else getattr(ci, "lower", None)
                upper = ci.get("upper") if isinstance(ci, dict) else getattr(ci, "upper", None)
                if lower is not None and upper is not None:
                    current_ax.fill_between(
                        x,
                        np.asarray(lower, dtype=float),
                        np.asarray(upper, dtype=float),
                        alpha=0.20,
                        color=_COLOR_FIT,
                        label="95% CI",
                    )

        # Axis scales and labels.
        if log_x:
            current_ax.set_xscale("log")

        if title is None:
            title = f"{result.model_id} fit (R^2 = {result.r_squared:.4f})"
        current_ax.set_title(title)
        current_ax.set_xlabel(xlabel)
        current_ax.set_ylabel(ylabel)
        current_ax.legend(loc="best", framealpha=0.7)

        # Annotation box: R^2 and AICc.
        annotation = f"R^2 = {result.r_squared:.4f}\nAICc = {result.aicc:.1f}"
        current_ax.text(
            0.97,
            0.97,
            annotation,
            transform=current_ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
        )

        fig.tight_layout()

    return fig


def residual_plot(
    result: FitResult,
    ax: matplotlib.axes.Axes | None = None,
    plot_type: str = "vs_fitted",
    figsize: tuple[float, float] = (7, 4),
) -> matplotlib.figure.Figure:
    """Plot residuals.

    Parameters
    ----------
    result : FitResult
        Completed fit result.
    ax : matplotlib.axes.Axes | None
        Axes to draw on.  If None, creates a new figure.
    plot_type : str
        "vs_fitted" -- residuals vs fitted values.
        "vs_x"      -- residuals vs x (predictor).
    figsize : tuple[float, float]
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If plot_type is not "vs_fitted" or "vs_x".
    """
    _valid_plot_types = {"vs_fitted", "vs_x"}
    if plot_type not in _valid_plot_types:
        raise ValueError(
            f"plot_type must be one of {sorted(_valid_plot_types)}, got {plot_type!r}."
        )

    residuals: np.ndarray = np.asarray(result.residuals, dtype=float)
    y_fitted: np.ndarray = np.asarray(result.y_fitted, dtype=float)
    x: np.ndarray = np.asarray(result.x, dtype=float)

    if plot_type == "vs_fitted":
        xvals = y_fitted
        xlabel_str = "Fitted values"
    else:
        xvals = x
        xlabel_str = "x"

    pos_mask = residuals >= 0

    with _try_style():
        fig, current_ax = _make_axes(ax, figsize)

        # Positive residuals in blue, negative in red.
        if np.any(pos_mask):
            current_ax.scatter(
                xvals[pos_mask],
                residuals[pos_mask],
                color=_COLOR_POS,
                alpha=0.85,
                s=40,
                zorder=3,
                label="Positive",
            )
        if np.any(~pos_mask):
            current_ax.scatter(
                xvals[~pos_mask],
                residuals[~pos_mask],
                color=_COLOR_NEG,
                alpha=0.85,
                s=40,
                zorder=3,
                label="Negative",
            )

        # Horizontal zero line.
        current_ax.axhline(
            0,
            color=_COLOR_ZERO,
            linestyle="--",
            linewidth=1.2,
            zorder=1,
        )

        current_ax.set_title("Residuals")
        current_ax.set_xlabel(xlabel_str)
        current_ax.set_ylabel("Residual (y - fitted)")
        current_ax.legend(loc="best", framealpha=0.7)

        fig.tight_layout()

    return fig


def qq_plot(
    result: FitResult,
    ax: matplotlib.axes.Axes | None = None,
    figsize: tuple[float, float] = (5, 5),
) -> matplotlib.figure.Figure:
    """Normal Q-Q plot of standardized residuals.

    Uses scipy.stats.probplot for theoretical quantiles.

    Parameters
    ----------
    result : FitResult
        Completed fit result.
    ax : matplotlib.axes.Axes | None
        Axes to draw on.
    figsize : tuple[float, float]
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import scipy.stats  # local import -- keeps top-level imports lean

    std_resid: np.ndarray = np.asarray(result.standardized_residuals, dtype=float)

    with _try_style():
        fig, current_ax = _make_axes(ax, figsize)

        # probplot returns (osm, osr), (slope, intercept, r) and draws onto the axes.
        scipy.stats.probplot(std_resid, dist="norm", plot=current_ax)

        # Override the default probplot styling to match openfit palette.
        # probplot adds a Line2D for the reference line and a scatter for the data.
        for line in current_ax.get_lines():
            line.set_color(_COLOR_FIT)
            line.set_linewidth(1.5)
        for collection in current_ax.collections:
            collection.set_facecolor(_COLOR_DATA)
            collection.set_alpha(0.85)

        current_ax.set_title("Normal Q-Q Plot of Standardized Residuals")
        current_ax.set_xlabel("Theoretical Quantiles")
        current_ax.set_ylabel("Standardized Residuals")

        fig.tight_layout()

    return fig


def rout_outlier_plot(
    x: np.ndarray,
    y: np.ndarray,
    rout_result: ROUTResult,
    model_equation=None,
    model_params: dict | None = None,
    ax: matplotlib.axes.Axes | None = None,
    x_smooth_points: int = 200,
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    figsize: tuple[float, float] = (7, 5),
) -> matplotlib.figure.Figure:
    """Plot data with ROUT-flagged outliers highlighted.

    Displays the fitted curve with normal points in blue circles and
    ROUT-flagged outliers as red X markers.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    y : np.ndarray
        Observed response values.
    rout_result : ROUTResult
        ROUT outlier detection result.
    model_equation : callable | None
        Model equation function f(x, **params) -> y. If provided along with
        model_params, draws the fitted curve.
    model_params : dict | None
        Model parameters for the fitted curve.
    ax : matplotlib.axes.Axes | None
        Axes to draw on. If None, creates a new figure.
    x_smooth_points : int
        Number of points for smooth curve interpolation. Default 200.
    title : str | None
        Plot title. If None, uses "ROUT Outlier Detection (Q={Q})".
    xlabel : str
        X-axis label. Default "x".
    ylabel : str
        Y-axis label. Default "y".
    figsize : tuple[float, float]
        Figure size in inches. Default (7, 5).

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    outlier_mask = rout_result.outlier_mask
    normal_mask = ~outlier_mask

    with _try_style():
        fig, current_ax = _make_axes(ax, figsize)

        # Fitted curve (if model provided)
        if model_equation is not None and model_params is not None:
            x_smooth = np.linspace(x.min(), x.max(), x_smooth_points)
            y_smooth = model_equation(x_smooth, **model_params)
            current_ax.plot(
                x_smooth,
                y_smooth,
                color=_COLOR_FIT,
                linewidth=2,
                zorder=2,
                label="Fit",
            )

        # Normal points (blue circles)
        if np.any(normal_mask):
            current_ax.scatter(
                x[normal_mask],
                y[normal_mask],
                color=_COLOR_NORMAL,
                marker="o",
                alpha=0.85,
                s=40,
                zorder=3,
                label=f"Normal ({normal_mask.sum()})",
            )

        # Outlier points (red X markers)
        if np.any(outlier_mask):
            current_ax.scatter(
                x[outlier_mask],
                y[outlier_mask],
                color=_COLOR_OUTLIER,
                marker="x",
                s=100,
                linewidth=2.5,
                zorder=4,
                label=f"Outlier (ROUT Q={rout_result.Q * 100:.0f}%)",
            )

        if title is None:
            title = f"ROUT Outlier Detection (Q={rout_result.Q * 100:.0f}%)"
        current_ax.set_title(title)
        current_ax.set_xlabel(xlabel)
        current_ax.set_ylabel(ylabel)
        current_ax.legend(loc="best", framealpha=0.7)

        # Annotation box with outlier count
        annotation = f"Outliers: {rout_result.n_outliers}/{len(x)}"
        current_ax.text(
            0.97,
            0.97,
            annotation,
            transform=current_ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
        )

        fig.tight_layout()

    return fig


def figure_to_base64(
    fig: matplotlib.figure.Figure,
    dpi: int = 150,
    fmt: str = "png",
) -> str:
    """Encode a matplotlib figure as a base64 data URI string.

    Closes the figure after encoding to prevent memory leaks.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to encode.
    dpi : int
        Resolution.  Default 150 (suitable for web reports).
    fmt : str
        Image format: "png" or "svg".

    Returns
    -------
    str
        Base64-encoded data URI string for embedding in HTML:
        "data:image/png;base64,<data>"    for PNG,
        "data:image/svg+xml;base64,<data>" for SVG.

    Raises
    ------
    ValueError
        If fmt is not "png" or "svg".
    """
    _valid_fmts = {"png", "svg"}
    if fmt not in _valid_fmts:
        raise ValueError(f"fmt must be one of {sorted(_valid_fmts)}, got {fmt!r}.")

    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    encoded = base64.b64encode(buf.read()).decode("ascii")

    if fmt == "png":
        return f"data:image/png;base64,{encoded}"
    return f"data:image/svg+xml;base64,{encoded}"
