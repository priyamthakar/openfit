"""Markdown report renderer for openfit."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openfit.results import FitResult

from openfit.report.html import DISCLAIMER

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
        Formatted string.
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_markdown_report(result: "FitResult") -> str:
    """Render a Markdown fit report for a FitResult.

    Parameters
    ----------
    result : FitResult
        Completed fit result.

    Returns
    -------
    str
        Markdown-formatted report as a string.  Does not include embedded
        images; embed externally if needed.
    """
    model_id: str = str(result.model_id)
    timestamp: str = getattr(result.spec, "timestamp", "N/A")
    weight_scheme: str = str(result.weight_scheme)
    n_obs: int = int(result.n_obs)
    n_params: int = int(result.n_params)
    df: int = n_obs - n_params

    lines: list[str] = []

    # --- header ---
    lines.append(f"# openfit Fit Report: {model_id}")
    lines.append("")
    lines.append(f"**Timestamp:** {timestamp}")
    lines.append("")

    # --- parameter table ---
    lines.append("## Parameter Estimates")
    lines.append("")
    lines.append("| Parameter | Value | SE | 95% CI Lower | 95% CI Upper |")
    lines.append("|-----------|-------|----|--------------|--------------|")

    for name, val in result.params.items():
        se_val = result.se.get(name) if isinstance(result.se, dict) else None
        ci_entry = result.ci.get(name) if isinstance(result.ci, dict) else None
        ci_lo = _fmt(ci_entry[0]) if ci_entry is not None else "N/A"
        ci_hi = _fmt(ci_entry[1]) if ci_entry is not None else "N/A"
        lines.append(f"| {name} | {_fmt(val)} | {_fmt(se_val)} | {ci_lo} | {ci_hi} |")

    lines.append("")

    # --- GOF table ---
    lines.append("## Goodness of Fit")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    gof_entries: list[tuple[str, str]] = [
        ("R^2", _fmt(result.r_squared)),
        ("AICc", _fmt(result.aicc)),
        ("BIC", _fmt(result.bic)),
        ("AIC", _fmt(result.aic)),
        ("RSS", _fmt(result.rss)),
        ("Observations (n)", str(n_obs)),
        ("Parameters (p)", str(n_params)),
        ("Degrees of freedom", str(df)),
        ("Weight scheme", weight_scheme),
    ]
    for label, value in gof_entries:
        lines.append(f"| {label} | {value} |")

    lines.append("")

    # --- ROUT outlier section (if available) ---
    rout_result = getattr(result, "rout_result", None)
    if rout_result is not None:
        lines.append("## ROUT Outlier Detection")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total points | {len(result.x)} |")
        lines.append(f"| Flagged outliers | {rout_result.n_outliers} |")
        lines.append(f"| Q parameter | {rout_result.Q*100:.1f}% |")
        
        if rout_result.n_outliers > 0:
            outlier_indices = rout_result.outlier_indices
            outlier_x_values = result.x[rout_result.outlier_mask]
            
            # Show first 20 outliers
            if len(outlier_indices) > 20:
                idx_str = ", ".join(str(i) for i in outlier_indices[:20]) + f"... ({len(outlier_indices)} total)"
                x_str = ", ".join(f"{x:.3g}" for x in outlier_x_values[:20]) + f"... ({len(outlier_x_values)} total)"
            else:
                idx_str = ", ".join(str(i) for i in outlier_indices)
                x_str = ", ".join(f"{x:.3g}" for x in outlier_x_values)
            
            lines.append(f"| Outlier indices | {idx_str} |")
            lines.append(f"| Outlier x-values | {x_str} |")
        
        lines.append("")

    # --- FitSpec ---
    lines.append("## Reproducibility Spec")
    lines.append("")
    try:
        spec_json: str = result.spec.to_json()
    except Exception:
        spec_json = "{}"

    lines.append("```json")
    lines.append(spec_json)
    lines.append("```")
    lines.append("")

    # --- disclaimer ---
    lines.append("---")
    lines.append("")
    lines.append(f"> {DISCLAIMER}")
    lines.append("")

    return "\n".join(lines)
