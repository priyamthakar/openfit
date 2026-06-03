"""Report generation for openfit fits."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openfit.report.global_fit import report_global_fit
from openfit.report.html import render_html_report
from openfit.report.markdown import render_markdown_report

if TYPE_CHECKING:
    from openfit.results import FitResult


def report_fit(result: FitResult, path: str, fmt: str = "html") -> None:
    """Write a fit report to a file.

    Parameters
    ----------
    result : FitResult
        A completed fit result.
    path : str
        Output file path.
    fmt : str
        "html", "markdown", or "pdf". Default "html".

    Raises
    ------
    ValueError
        If fmt is not one of "html", "markdown", or "pdf".
    """
    if fmt == "html":
        html = render_html_report(result)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    elif fmt in ("markdown", "md"):
        md = render_markdown_report(result)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
    elif fmt == "pdf":
        from openfit.report.pdf import render_pdf_report

        pdf_bytes = render_pdf_report(result)
        with open(path, "wb") as f:
            f.write(pdf_bytes)
    elif fmt == "docx":
        from openfit.report.docx import render_docx_report

        docx_bytes = render_docx_report(result)
        with open(path, "wb") as f:
            f.write(docx_bytes)
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use 'html', 'markdown', 'pdf', or 'docx'.")


__all__ = ["report_fit", "render_html_report", "render_markdown_report", "report_global_fit"]
