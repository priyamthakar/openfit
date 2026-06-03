"""PDF report renderer for openfit."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

if TYPE_CHECKING:
    import matplotlib.figure

    from openfit.results import FitResult


DISCLAIMER: str = (
    "This report was generated using openfit (open-source). "
    "Results should be independently verified for regulatory or "
    "clinical decision-making."
)


def _fmt(value: float | None) -> str:
    """Format a float to 5 significant figures, or 'N/A'."""
    import math

    if value is None:
        return "N/A"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(f):
        return str(f)
    return f"{f:.5g}"


def _fig_to_image(fig: matplotlib.figure.Figure, width: float = 450) -> Image:
    """Convert a matplotlib Figure to a ReportLab Image.

    Renders the figure to a BytesIO PNG buffer and returns a platypus Image
    with proportional height.  The figure is closed after rendering.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to convert.
    width : float
        Width in points for the ReportLab Image.

    Returns
    -------
    reportlab.platypus.Image
    """
    import matplotlib.pyplot as plt

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    # Let ReportLab compute height from width to preserve aspect ratio.
    img = Image(buf, width=width)
    return img


def render_pdf_report(result: FitResult) -> bytes:
    """Generate a PDF fit report as bytes.

    Parameters
    ----------
    result : FitResult
        A completed fit result.

    Returns
    -------
    bytes
        PDF file contents as bytes.
    """
    from openfit.plotting import fit_overlay_plot, residual_plot
    from openfit.report.html import _decide_log_x

    buf = BytesIO()

    # Page setup — single page, clean professional layout.
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title=f"openfit Fit Report - {result.model_id}",
    )

    styles = getSampleStyleSheet()
    style_h1 = styles["Heading1"]
    style_h2 = styles["Heading2"]
    style_normal = styles["Normal"]

    # --- Metadata ---
    timestamp = getattr(result.spec, "timestamp", "N/A")
    n_obs = int(result.n_obs)
    n_params = int(result.n_params)
    df = n_obs - n_params

    story = []

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    story.append(Paragraph("openfit Fit Report", style_h1))
    story.append(Paragraph(f"Model: {result.model_id}  |  {timestamp}", style_normal))
    story.append(Spacer(1, 6 * mm))

    # ------------------------------------------------------------------
    # Parameter Estimates table
    # ------------------------------------------------------------------
    story.append(Paragraph("Parameter Estimates", style_h2))
    story.append(Spacer(1, 3 * mm))

    param_data = [["Parameter", "Value", "SE", "95% CI"]]
    for name in result.params:
        val = _fmt(result.params[name])
        se_val = _fmt(result.se.get(name) if isinstance(result.se, dict) else None)
        ci_entry = result.ci.get(name) if isinstance(result.ci, dict) else None
        ci_str = f"[{_fmt(ci_entry[0])}, {_fmt(ci_entry[1])}]" if ci_entry is not None else "N/A"
        param_data.append([name, val, se_val, ci_str])

    param_table = Table(param_data, colWidths=[120, 90, 90, 160])
    param_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#e2e8f0"),
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 0),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f8fafc")],
                ),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(param_table)
    story.append(Spacer(1, 6 * mm))

    # ------------------------------------------------------------------
    # Goodness of Fit table
    # ------------------------------------------------------------------
    story.append(Paragraph("Goodness of Fit", style_h2))
    story.append(Spacer(1, 3 * mm))

    gof_data = [
        ["Metric", "Value"],
        ["R^2", _fmt(result.r_squared)],
        ["AICc", _fmt(result.aicc)],
        ["BIC", _fmt(result.bic)],
        ["AIC", _fmt(result.aic)],
        ["RSS", _fmt(result.rss)],
        ["Observations (n)", str(n_obs)],
        ["Parameters (p)", str(n_params)],
        ["Degrees of freedom", str(df)],
        ["Weight scheme", str(result.weight_scheme)],
    ]

    gof_table = Table(gof_data, colWidths=[200, 260])
    gof_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#e2e8f0"),
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 0),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f8fafc")],
                ),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(gof_table)
    story.append(Spacer(1, 6 * mm))

    # ------------------------------------------------------------------
    # Optional ROUT outlier table
    # ------------------------------------------------------------------
    rout_result = getattr(result, "rout_result", None)
    if rout_result is not None:
        story.append(Paragraph("ROUT Outlier Detection", style_h2))
        story.append(Spacer(1, 3 * mm))

        rout_data = [
            ["Metric", "Value"],
            ["Total points", str(len(result.x))],
            ["Flagged outliers", str(rout_result.n_outliers)],
            ["Q parameter", f"{rout_result.Q * 100:.1f}%"],
        ]

        rout_table = Table(rout_data, colWidths=[200, 260])
        rout_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#1e3a5f"),
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.HexColor("#e2e8f0"),
                    ),
                    (
                        "ROWBACKGROUNDS",
                        (0, 0),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f8fafc")],
                    ),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(rout_table)
        story.append(Spacer(1, 6 * mm))

    # ------------------------------------------------------------------
    # Fit Overlay Plot
    # ------------------------------------------------------------------
    story.append(Paragraph("Fit Overlay", style_h2))
    story.append(Spacer(1, 3 * mm))

    log_x = _decide_log_x(result.x)

    try:
        overlay_fig = fit_overlay_plot(result, log_x=log_x)
        overlay_img = _fig_to_image(overlay_fig, width=460)
        story.append(overlay_img)
    except Exception:
        story.append(Paragraph("[Plot unavailable]", style_normal))

    story.append(Spacer(1, 6 * mm))

    # ------------------------------------------------------------------
    # Residual Plot
    # ------------------------------------------------------------------
    story.append(Paragraph("Residuals", style_h2))
    story.append(Spacer(1, 3 * mm))

    try:
        resid_fig = residual_plot(result)
        resid_img = _fig_to_image(resid_fig, width=460)
        story.append(resid_img)
    except Exception:
        story.append(Paragraph("[Plot unavailable]", style_normal))

    story.append(Spacer(1, 10 * mm))

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    story.append(Paragraph(DISCLAIMER, style_normal))
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            f"Generated by openfit  |  {timestamp}",
            styles["Italic"],
        )
    )

    doc.build(story)

    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes
