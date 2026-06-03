"""Global fit report renderer for openfit."""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

if TYPE_CHECKING:
    from openfit.global_fit import GlobalFitResult

from openfit.report.html import DISCLAIMER

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    """Format a float to 5 significant figures, or 'N/A' for missing/non-finite."""
    if value is None:
        return "N/A"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(f):
        return str(f)
    return f"{f:.5g}"


def _figure_to_base64(fig, dpi: int = 150) -> str:
    """Convert matplotlib figure to base64 PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def _global_overlay_plot(
    result: "GlobalFitResult",
    x_smooth_points: int = 200,
) -> str:
    """Generate overlay plot with all datasets, shared curve, and local curves."""
    colors = plt.cm.Set1.colors[:len(result._datasets)]
    markers = ["o", "s", "^", "D", "v", "p", "h", "*", "X", "P"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each dataset with its color and marker
    for i, (x, y) in enumerate(result._datasets):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.scatter(
            x, y,
            color=color,
            marker=marker,
            alpha=0.7,
            s=50,
            label=f"Dataset {i+1}",
            zorder=3,
        )
        
        # Plot local curve for this dataset
        x_smooth = np.linspace(x.min(), x.max(), x_smooth_points)
        params = result.all_params_per_dataset[i]
        y_smooth = result._model.equation(x_smooth, **params)
        ax.plot(
            x_smooth, y_smooth,
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
            label=f"Local fit {i+1}",
            zorder=2,
        )
    
    # Plot shared curve (using shared params + first dataset's local params as example)
    # Note: shared curve is conceptual - we show it using dataset 0's local params
    x_all = np.concatenate([x for x, y in result._datasets])
    x_smooth_shared = np.linspace(x_all.min(), x_all.max(), x_smooth_points)
    shared_params_with_local = {**result.shared_params}
    # Add local params from first dataset for demonstration
    if result.local_params:
        shared_params_with_local.update(result.local_params[0])
    y_smooth_shared = result._model.equation(x_smooth_shared, **shared_params_with_local)
    ax.plot(
        x_smooth_shared, y_smooth_shared,
        color="black",
        linestyle="-",
        linewidth=2.5,
        label="Shared params",
        zorder=4,
    )
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Global Fit: {result.model_id} (n={result.n_datasets} datasets)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return _figure_to_base64(fig)


def _residual_plots(result: "GlobalFitResult") -> list[str]:
    """Generate per-dataset residual plots."""
    plots = []
    for i, (x, y) in enumerate(result._datasets):
        fig, ax = plt.subplots(figsize=(7, 4))
        
        params = result.all_params_per_dataset[i]
        y_pred = result._model.equation(x, **params)
        residuals = y - y_pred
        
        colors = plt.cm.Set1.colors
        color = colors[i % len(colors)]
        
        ax.scatter(x, residuals, color=color, alpha=0.7, s=40, zorder=3)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1.2, zorder=1)
        
        ax.set_xlabel("x")
        ax.set_ylabel("Residual (y - fitted)")
        ax.set_title(f"Residuals: Dataset {i+1}")
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        plots.append(_figure_to_base64(fig))
    
    return plots


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------


def _render_html(result: "GlobalFitResult") -> str:
    """Render global fit report as HTML."""
    import html as html_module
    
    def esc(s: str) -> str:
        return html_module.escape(str(s), quote=True)
    
    # Generate plots
    overlay_img = _global_overlay_plot(result)
    residual_imgs = _residual_plots(result)
    
    # Shared parameters table
    shared_rows = []
    for name in result.shared_param_names:
        val = result.shared_params[name]
        shared_rows.append(
            f"<tr><td>{esc(name)}</td><td>{_fmt(val)}</td></tr>"
        )
    shared_table = "\n          ".join(shared_rows) if shared_rows else "<tr><td colspan='2'>None</td></tr>"
    
    # Local parameters table
    local_rows = []
    for i, lp in enumerate(result.local_params):
        for name in result.local_param_names:
            val = lp[name]
            local_rows.append(
                f"<tr><td>Dataset {i+1}</td><td>{esc(name)}</td><td>{_fmt(val)}</td></tr>"
            )
    local_table = "\n          ".join(local_rows) if local_rows else "<tr><td colspan='3'>None</td></tr>"
    
    # Per-dataset statistics
    stats_rows = []
    for i in range(result.n_datasets):
        r2 = result.r_squared_per_dataset[i]
        rss = result.rss_per_dataset[i]
        n = result.n_obs_per_dataset[i]
        stats_rows.append(
            f"<tr><td>Dataset {i+1}</td><td>{n}</td><td>{_fmt(rss)}</td><td>{_fmt(r2)}</td></tr>"
        )
    stats_table = "\n          ".join(stats_rows)
    
    # F-test result
    if result.f_test_sharing is not None:
        ft = result.f_test_sharing
        verdict = "Sharing justified" if ft.sharing_justified else "Sharing questionable"
        f_test_html = f"""
    <h2>F-test: Is Sharing Statistically Justified?</h2>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
          <tr><td>F statistic</td><td>{_fmt(ft.f_statistic)}</td></tr>
          <tr><td>df (numerator)</td><td>{ft.df_numerator}</td></tr>
          <tr><td>df (denominator)</td><td>{ft.df_denominator}</td></tr>
          <tr><td>p-value</td><td>{_fmt(ft.p_value)}</td></tr>
          <tr><td>RSS (shared)</td><td>{_fmt(ft.rss_shared)}</td></tr>
          <tr><td>RSS (independent)</td><td>{_fmt(ft.rss_independent)}</td></tr>
          <tr><td>Conclusion</td><td><strong>{esc(verdict)}</strong> (p {'>' if ft.sharing_justified else '<='} 0.05)</td></tr>
      </tbody>
    </table>
"""
    else:
        f_test_html = """
    <h2>F-test: Is Sharing Statistically Justified?</h2>
    <p>F-test not computed.</p>
"""
    
    # Residual plots HTML
    residual_html = "\n".join([
        f'<div class="plots"><img src="{img}" alt="Residuals Dataset {i+1}" class="plot-half"></div>'
        for i, img in enumerate(residual_imgs)
    ])
    
    disclaimer = esc(DISCLAIMER)
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>openfit Global Fit Report -- {esc(result.model_id)}</title>
  <style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 14px;
    color: #1a202c;
    background: #ffffff;
    line-height: 1.6;
  }}
  header {{
    background: #1e3a5f;
    color: #ffffff;
    padding: 24px 40px;
  }}
  header h1 {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 4px; }}
  header .subtitle {{ font-size: 0.875rem; opacity: 0.8; }}
  main {{ max-width: 960px; margin: 32px auto; padding: 0 24px; }}
  h2 {{
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e3a5f;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px;
    margin: 32px 0 16px;
  }}
  .plots {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 8px; }}
  .plots img {{ max-width: 100%; border: 1px solid #e2e8f0; border-radius: 6px; }}
  .plot-full {{ flex: 1 1 100%; }}
  .plot-half {{ flex: 1 1 calc(50% - 8px); }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-bottom: 8px;
  }}
  th {{
    background: #1e3a5f;
    color: #ffffff;
    text-align: left;
    padding: 8px 12px;
    font-weight: 600;
  }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #e2e8f0; }}
  tr:nth-child(even) td {{ background: #f8fafc; }}
  tr:last-child td {{ border-bottom: none; }}
  .disclaimer {{
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    padding: 12px 16px;
    font-size: 12px;
    color: #78350f;
    margin: 32px 0 16px;
    border-radius: 0 4px 4px 0;
  }}
  footer {{
    text-align: center;
    font-size: 12px;
    color: #718096;
    padding: 24px 0 32px;
    border-top: 1px solid #e2e8f0;
    margin-top: 40px;
  }}
  </style>
</head>
<body>
  <header>
    <h1>openfit Global Fit Report</h1>
    <div class="subtitle">Model: {esc(result.model_id)} &nbsp;|&nbsp; Datasets: {result.n_datasets} &nbsp;|&nbsp; Weights: {esc(result.weight_scheme)}</div>
  </header>
  <main>

    <h2>Fit Overlay: All Datasets with Shared and Local Curves</h2>
    <div class="plots">
      <img src="{overlay_img}" alt="Global fit overlay" class="plot-full">
    </div>

    <h2>Per-Dataset Residual Plots</h2>
    {residual_html}

    <h2>Shared Parameters (Same Value Across All Datasets)</h2>
    <table>
      <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
      <tbody>
          {shared_table}
      </tbody>
    </table>

    <h2>Local Parameters (Different Per Dataset)</h2>
    <table>
      <thead><tr><th>Dataset</th><th>Parameter</th><th>Value</th></tr></thead>
      <tbody>
          {local_table}
      </tbody>
    </table>

    <h2>Per-Dataset Goodness of Fit</h2>
    <table>
      <thead><tr><th>Dataset</th><th>Observations</th><th>RSS</th><th>R^2</th></tr></thead>
      <tbody>
          {stats_table}
      </tbody>
    </table>

    {f_test_html}

    <div class="disclaimer">{disclaimer}</div>

  </main>
  <footer>Generated by openfit</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def _render_markdown(result: "GlobalFitResult") -> str:
    """Render global fit report as Markdown."""
    lines = []
    
    lines.append(f"# openfit Global Fit Report: {result.model_id}")
    lines.append("")
    lines.append(f"**Datasets:** {result.n_datasets}  ")
    lines.append(f"**Weight scheme:** {result.weight_scheme}")
    lines.append("")
    
    # Shared parameters
    lines.append("## Shared Parameters (Same Value Across All Datasets)")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    for name in result.shared_param_names:
        val = result.shared_params[name]
        lines.append(f"| {name} | {_fmt(val)} |")
    lines.append("")
    
    # Local parameters
    lines.append("## Local Parameters (Different Per Dataset)")
    lines.append("")
    lines.append("| Dataset | Parameter | Value |")
    lines.append("|---------|-----------|-------|")
    for i, lp in enumerate(result.local_params):
        for name in result.local_param_names:
            val = lp[name]
            lines.append(f"| Dataset {i+1} | {name} | {_fmt(val)} |")
    lines.append("")
    
    # Per-dataset statistics
    lines.append("## Per-Dataset Goodness of Fit")
    lines.append("")
    lines.append("| Dataset | Observations | RSS | R^2 |")
    lines.append("|---------|--------------|-----|-----|")
    for i in range(result.n_datasets):
        r2 = result.r_squared_per_dataset[i]
        rss = result.rss_per_dataset[i]
        n = result.n_obs_per_dataset[i]
        lines.append(f"| Dataset {i+1} | {n} | {_fmt(rss)} | {_fmt(r2)} |")
    lines.append("")
    
    # F-test
    if result.f_test_sharing is not None:
        ft = result.f_test_sharing
        verdict = "Sharing justified" if ft.sharing_justified else "Sharing questionable"
        lines.append("## F-test: Is Sharing Statistically Justified?")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| F statistic | {_fmt(ft.f_statistic)} |")
        lines.append(f"| df (numerator) | {ft.df_numerator} |")
        lines.append(f"| df (denominator) | {ft.df_denominator} |")
        lines.append(f"| p-value | {_fmt(ft.p_value)} |")
        lines.append(f"| RSS (shared) | {_fmt(ft.rss_shared)} |")
        lines.append(f"| RSS (independent) | {_fmt(ft.rss_independent)} |")
        lines.append(f"| Conclusion | **{verdict}** (p {'>' if ft.sharing_justified else '<='} 0.05) |")
        lines.append("")
    else:
        lines.append("## F-test: Is Sharing Statistically Justified?")
        lines.append("")
        lines.append("F-test not computed.")
        lines.append("")
    
    # Disclaimer
    lines.append("---")
    lines.append("")
    lines.append(f"> {DISCLAIMER}")
    lines.append("")
    
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def report_global_fit(result: "GlobalFitResult", path: str, fmt: str = "html") -> None:
    """Write a global fit report to a file.

    Parameters
    ----------
    result : GlobalFitResult
        Completed global fit result.
    path : str
        Output file path.
    fmt : str
        "html" or "markdown". Default "html".

    Raises
    ------
    ValueError
        If fmt is not "html" or "markdown".
    """
    if fmt == "html":
        content = _render_html(result)
    elif fmt in ("markdown", "md"):
        content = _render_markdown(result)
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use 'html' or 'markdown'.")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


__all__ = ["report_global_fit"]
