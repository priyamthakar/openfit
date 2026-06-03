"""HTML report renderer for openfit."""

from __future__ import annotations

import html as html_module
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openfit.results import FitResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISCLAIMER: str = (
    "This report was generated using openfit (open-source). "
    "Results should be independently verified for regulatory or "
    "clinical decision-making."
)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_FIT_TEMPLATE_NAME = "fit_report.html.j2"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    """Format a float to 5 significant figures, or 'N/A' for missing/non-finite.

    Parameters
    ----------
    value : float | None
        Value to format.

    Returns
    -------
    str
        Formatted string safe for HTML output.
    """
    if value is None:
        return "N/A"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(f):
        return str(f)
    return f"{f:.5g}"


def _decide_log_x(x: Any) -> bool:
    """Return True when log x scale is appropriate.

    Parameters
    ----------
    x : array-like
        X values from the fit result.

    Returns
    -------
    bool
        True when all x > 0 and max(x)/min(x) > 100.
    """
    import numpy as np

    arr = np.asarray(x, dtype=float)
    x_min = float(arr.min())
    if x_min <= 0:
        return False
    x_max = float(arr.max())
    return (x_max / x_min) > 100.0


# ---------------------------------------------------------------------------
# Context builder (shared between Jinja2 and fallback paths)
# ---------------------------------------------------------------------------


def _build_context(result: "FitResult") -> dict[str, Any]:
    """Build the template context dict from a FitResult.

    Parameters
    ----------
    result : FitResult
        Completed fit result.

    Returns
    -------
    dict[str, Any]
        Context dict consumed by both the Jinja2 template and the
        pure-Python fallback renderer.
    """
    from openfit.plotting import figure_to_base64, fit_overlay_plot, qq_plot, residual_plot, rout_outlier_plot

    log_x = _decide_log_x(result.x)

    # --- plots ---
    try:
        overlay_fig = fit_overlay_plot(result, log_x=log_x)
        overlay_img: str = figure_to_base64(overlay_fig)
    except Exception:
        overlay_img = ""

    try:
        resid_fig = residual_plot(result)
        residual_img: str = figure_to_base64(resid_fig)
    except Exception:
        residual_img = ""

    try:
        qq_fig = qq_plot(result)
        qq_img: str = figure_to_base64(qq_fig)
    except Exception:
        qq_img = ""

    # --- ROUT outlier plot (if available) ---
    rout_img: str = ""
    rout_summary: dict[str, Any] = {}
    rout_result = getattr(result, "rout_result", None)
    if rout_result is not None:
        try:
            model_obj = getattr(result, "_model", None)
            model_equation = model_obj.equation if model_obj else None
            model_params = result.params if model_obj else None
            
            rout_fig = rout_outlier_plot(
                result.x,
                result.y,
                rout_result,
                model_equation=model_equation,
                model_params=model_params,
            )
            rout_img = figure_to_base64(rout_fig)
        except Exception:
            rout_img = ""
        
        # Build ROUT summary
        rout_summary = {
            "n_total": len(result.x),
            "n_outliers": int(rout_result.n_outliers),
            "Q": float(rout_result.Q),
            "outlier_indices": rout_result.outlier_indices.tolist() if hasattr(rout_result.outlier_indices, 'tolist') else list(rout_result.outlier_indices),
            "outlier_x_values": result.x[rout_result.outlier_mask].tolist() if rout_result.n_outliers > 0 else [],
        }

    # --- spec JSON ---
    try:
        spec_json: str = result.spec.to_json()
    except Exception:
        spec_json = "{}"

    # --- GOF ---
    n_obs: int = int(result.n_obs)
    n_params: int = int(result.n_params)
    df: int = n_obs - n_params

    # --- timestamp from spec ---
    timestamp: str = getattr(result.spec, "timestamp", "N/A")

    return {
        "model_id": str(result.model_id),
        "timestamp": timestamp,
        "params": dict(result.params),
        "se": dict(result.se),
        "ci": dict(result.ci),
        "r_squared": result.r_squared,
        "aic": result.aic,
        "aicc": result.aicc,
        "bic": result.bic,
        "rss": result.rss,
        "n_obs": n_obs,
        "n_params": n_params,
        "df": df,
        "weight_scheme": str(result.weight_scheme),
        "overlay_img": overlay_img,
        "residual_img": residual_img,
        "qq_img": qq_img,
        "rout_img": rout_img,
        "rout_summary": rout_summary,
        "spec_json": spec_json,
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Inline CSS (shared by the fallback renderer)
# ---------------------------------------------------------------------------

_CSS = """
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 14px;
    color: #1a202c;
    background: #ffffff;
    line-height: 1.6;
  }
  header {
    background: #1e3a5f;
    color: #ffffff;
    padding: 24px 40px;
  }
  header h1 { font-size: 1.5rem; font-weight: 700; margin-bottom: 4px; }
  header .subtitle { font-size: 0.875rem; opacity: 0.8; }
  main { max-width: 960px; margin: 32px auto; padding: 0 24px; }
  h2 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e3a5f;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px;
    margin: 32px 0 16px;
  }
  .plots { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 8px; }
  .plots img { max-width: 100%; border: 1px solid #e2e8f0; border-radius: 6px; }
  .plot-full { flex: 1 1 100%; }
  .plot-half { flex: 1 1 calc(50% - 8px); }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-bottom: 8px;
  }
  th {
    background: #1e3a5f;
    color: #ffffff;
    text-align: left;
    padding: 8px 12px;
    font-weight: 600;
  }
  td { padding: 7px 12px; border-bottom: 1px solid #e2e8f0; }
  tr:nth-child(even) td { background: #f8fafc; }
  tr:last-child td { border-bottom: none; }
  details { margin-top: 8px; }
  summary {
    cursor: pointer;
    font-weight: 600;
    color: #2d6a9f;
    padding: 4px 0;
    user-select: none;
  }
  pre {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 16px;
    overflow-x: auto;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 12px;
    margin-top: 8px;
    white-space: pre-wrap;
    word-break: break-all;
  }
  .disclaimer {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    padding: 12px 16px;
    font-size: 12px;
    color: #78350f;
    margin: 32px 0 16px;
    border-radius: 0 4px 4px 0;
  }
  footer {
    text-align: center;
    font-size: 12px;
    color: #718096;
    padding: 24px 0 32px;
    border-top: 1px solid #e2e8f0;
    margin-top: 40px;
  }
"""


# ---------------------------------------------------------------------------
# Pure-Python fallback renderer
# ---------------------------------------------------------------------------


def _render_html_fallback(ctx: dict[str, Any]) -> str:
    """Render the fit report as HTML without Jinja2.

    Parameters
    ----------
    ctx : dict[str, Any]
        Context dict from _build_context().

    Returns
    -------
    str
        Complete HTML5 document as a string.
    """

    def esc(s: str) -> str:
        return html_module.escape(str(s), quote=True)

    model_id = esc(ctx["model_id"])
    timestamp = esc(ctx["timestamp"])
    weight_scheme = esc(ctx["weight_scheme"])
    spec_json_escaped = esc(ctx["spec_json"])
    disclaimer = esc(ctx["disclaimer"])

    # --- parameter table rows ---
    param_rows: list[str] = []
    for name, val in ctx["params"].items():
        se_val = ctx["se"].get(name) if isinstance(ctx["se"], dict) else None
        ci_entry = ctx["ci"].get(name) if isinstance(ctx["ci"], dict) else None
        ci_lo = _fmt(ci_entry[0]) if ci_entry is not None else "N/A"
        ci_hi = _fmt(ci_entry[1]) if ci_entry is not None else "N/A"
        param_rows.append(
            f"<tr>"
            f"<td>{esc(name)}</td>"
            f"<td>{_fmt(val)}</td>"
            f"<td>{_fmt(se_val)}</td>"
            f"<td>{ci_lo}</td>"
            f"<td>{ci_hi}</td>"
            f"</tr>"
        )
    param_table_body = (
        "\n          ".join(param_rows)
        if param_rows
        else "<tr><td colspan='5'>No parameters</td></tr>"
    )

    # --- GOF table rows ---
    gof_rows_data: list[tuple[str, str]] = [
        ("R^2", _fmt(ctx["r_squared"])),
        ("AICc", _fmt(ctx["aicc"])),
        ("BIC", _fmt(ctx["bic"])),
        ("AIC", _fmt(ctx["aic"])),
        ("RSS", _fmt(ctx["rss"])),
        ("Observations (n)", str(ctx["n_obs"])),
        ("Parameters (p)", str(ctx["n_params"])),
        ("Degrees of freedom", str(ctx["df"])),
        ("Weight scheme", weight_scheme),
    ]
    gof_rows = "\n          ".join(
        f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in gof_rows_data
    )

    # --- plot img tags ---
    def img_tag(uri: str, alt: str, css_class: str) -> str:
        if not uri:
            return f'<p class="{esc(css_class)}">[Plot unavailable]</p>'
        return f'<img src="{uri}" alt="{esc(alt)}" class="{esc(css_class)}">'

    overlay_tag = img_tag(ctx["overlay_img"], "Fit overlay", "plot-full")
    residual_tag = img_tag(ctx["residual_img"], "Residuals", "plot-half")
    qq_tag = img_tag(ctx["qq_img"], "Normal Q-Q", "plot-half")

    # --- ROUT outlier section ---
    rout_section = ""
    rout_img = ctx.get("rout_img", "")
    rout_summary = ctx.get("rout_summary", {})
    if rout_summary:
        rout_img_tag = img_tag(rout_img, "ROUT Outlier Detection", "plot-full")
        
        # Build outlier summary table
        n_total = rout_summary.get("n_total", 0)
        n_outliers = rout_summary.get("n_outliers", 0)
        q_param = rout_summary.get("Q", 0.0)
        outlier_indices = rout_summary.get("outlier_indices", [])
        outlier_x_values = rout_summary.get("outlier_x_values", [])
        
        rout_table_rows = f"<tr><td>Total points</td><td>{n_total}</td></tr>"
        rout_table_rows += f"<tr><td>Flagged outliers</td><td>{n_outliers}</td></tr>"
        rout_table_rows += f"<tr><td>Q parameter</td><td>{q_param*100:.1f}%</td></tr>"
        
        if outlier_indices:
            idx_str = ", ".join(str(i) for i in outlier_indices[:20])
            if len(outlier_indices) > 20:
                idx_str += f"... ({len(outlier_indices)} total)"
            rout_table_rows += f"<tr><td>Outlier indices</td><td>{idx_str}</td></tr>"
            
            x_str = ", ".join(f"{x:.3g}" for x in outlier_x_values[:20])
            if len(outlier_x_values) > 20:
                x_str += f"... ({len(outlier_x_values)} total)"
            rout_table_rows += f"<tr><td>Outlier x-values</td><td>{x_str}</td></tr>"
        
        rout_section = f"""
    <h2>ROUT Outlier Detection</h2>
    <div class="plots">
      {rout_img_tag}
    </div>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
          {rout_table_rows}
      </tbody>
    </table>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>openfit Fit Report -- {model_id}</title>
  <style>
{_CSS}
  </style>
</head>
<body>
  <header>
    <h1>openfit Fit Report</h1>
    <div class="subtitle">Model: {model_id} &nbsp;|&nbsp; {timestamp}</div>
  </header>
  <main>

    <h2>Fit Overlay</h2>
    <div class="plots">
      {overlay_tag}
    </div>

    <h2>Residual Diagnostics</h2>
    <div class="plots">
      {residual_tag}
      {qq_tag}
    </div>

    <h2>Parameter Estimates</h2>
    <table>
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Value</th>
          <th>SE</th>
          <th>95% CI Lower</th>
          <th>95% CI Upper</th>
        </tr>
      </thead>
      <tbody>
          {param_table_body}
      </tbody>
    </table>

    <h2>Goodness of Fit</h2>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
          {gof_rows}
      </tbody>
    </table>
{rout_section}
    <h2>Reproducibility Spec</h2>
    <details>
      <summary>FitSpec JSON (click to expand)</summary>
      <pre>{spec_json_escaped}</pre>
    </details>

    <div class="disclaimer">{disclaimer}</div>

  </main>
  <footer>Generated by openfit &nbsp;|&nbsp; {timestamp}</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_html_report(result: "FitResult") -> str:
    """Render a complete HTML fit report for a FitResult.

    Attempts to use the Jinja2 template if Jinja2 is installed and the
    template file is present.  Falls back to a pure-Python renderer when
    Jinja2 is unavailable or the template cannot be located.

    Parameters
    ----------
    result : FitResult
        Completed fit result.

    Returns
    -------
    str
        Full HTML5 document as a UTF-8 string.
    """
    ctx = _build_context(result)

    try:
        import jinja2  # type: ignore[import-not-found]
    except ImportError:
        # Jinja2 is not installed -- use pure-Python fallback.
        return _render_html_fallback(ctx)

    try:
        loader = jinja2.FileSystemLoader(str(_TEMPLATE_DIR))
        env = jinja2.Environment(
            loader=loader,
            autoescape=jinja2.select_autoescape(["html", "j2"]),
        )
        template = env.get_template(_FIT_TEMPLATE_NAME)
        return template.render(**ctx)
    except jinja2.TemplateNotFound:
        # Template file is absent (e.g. package installed without data files).
        return _render_html_fallback(ctx)
