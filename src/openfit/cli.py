"""openfit command-line interface."""

from __future__ import annotations

from pathlib import Path

import rich.box
import typer
from rich.console import Console
from rich.table import Table

from openfit import __version__
from openfit.compare import compare_models
from openfit.fit import Fit
from openfit.io import load_csv, load_excel
from openfit.models import get_model, list_models
from openfit.weighting import WeightScheme, parse_weight_scheme

app = typer.Typer(
    name="openfit",
    help="Reproducible, open-source nonlinear curve fitting.",
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm"}


def _parse_col(s: str) -> str | int:
    """Convert a CLI column specifier to int (0-based index) or str (name).

    Parameters
    ----------
    s : str
        Raw CLI value, e.g. "0", "2", or "concentration".

    Returns
    -------
    str | int
        Integer index when *s* is a non-negative decimal digit string,
        otherwise the original string (column name).
    """
    try:
        val = int(s)
        if val >= 0:
            return val
        return s
    except ValueError:
        return s


def _load_data(
    data_file: Path,
    x_col: str | int,
    y_col: str | int,
    sd_col: str | int | None,
) -> tuple:
    """Load data using load_csv or load_excel depending on file suffix.

    Parameters
    ----------
    data_file : Path
        Path to the data file.
    x_col : str | int
        Column specifier for x values.
    y_col : str | int
        Column specifier for y values.
    sd_col : str | int | None
        Column specifier for SD values, or None.

    Returns
    -------
    tuple
        (x, y, sd) numpy arrays; sd may be None.

    Raises
    ------
    FileNotFoundError
        If data_file does not exist.
    ValueError
        If the data contains NaN, Inf, or fewer than 3 points.
    """
    if data_file.suffix.lower() in _EXCEL_SUFFIXES:
        return load_excel(data_file, x_col=x_col, y_col=y_col, sd_col=sd_col)
    return load_csv(data_file, x_col=x_col, y_col=y_col, sd_col=sd_col)


def _validate_model_id(model_id: str) -> None:
    """Print an error and exit 1 if model_id is not registered.

    Parameters
    ----------
    model_id : str
        Model ID to check.
    """
    known = list_models()
    if model_id not in known:
        err_console.print(
            f"[red]Error:[/red] Unknown model ID: {model_id!r}. "
            "Run 'openfit models' to see available models."
        )
        raise typer.Exit(code=1)


def _validate_weights(weights: str) -> WeightScheme:
    """Parse and validate a weight scheme string, exiting 1 on failure.

    Parameters
    ----------
    weights : str
        Weight scheme alias string.

    Returns
    -------
    WeightScheme
        Parsed enum member.
    """
    try:
        return parse_weight_scheme(weights)
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _print_param_table(result: object) -> None:
    """Print a Rich ASCII table of fitted parameters, SE, and CI.

    Parameters
    ----------
    result : FitResult
        A fitted result object with .params, .se, .ci attributes.
    """
    tbl = Table(box=rich.box.ASCII, show_header=True, header_style="bold")
    tbl.add_column("Parameter", style="cyan", no_wrap=True)
    tbl.add_column("Value", justify="right")
    tbl.add_column("SE", justify="right")
    tbl.add_column("95% CI", justify="right")

    params: dict = getattr(result, "params", {})
    se: dict | None = getattr(result, "se", None)
    ci: dict | None = getattr(result, "ci", None)

    for name, value in params.items():
        se_str = "n/a"
        ci_str = "n/a"

        if se is not None:
            se_val = se.get(name)
            if se_val is not None:
                try:
                    se_str = f"{float(se_val):.6g}"
                except (TypeError, ValueError):
                    se_str = str(se_val)

        if ci is not None:
            ci_val = ci.get(name)
            if ci_val is not None:
                try:
                    lo, hi = float(ci_val[0]), float(ci_val[1])
                    ci_str = f"[{lo:.6g}, {hi:.6g}]"
                except (TypeError, ValueError, IndexError):
                    ci_str = str(ci_val)

        tbl.add_row(name, f"{value:.6g}", se_str, ci_str)

    console.print(tbl)


def _print_gof(result: object) -> None:
    """Print goodness-of-fit statistics from a FitResult.

    Parameters
    ----------
    result : FitResult
        Fitted result with .r_squared and .aic attributes.
    """
    r2 = getattr(result, "r_squared", None)
    aic = getattr(result, "aic", None)
    if r2 is not None:
        console.print(f"  R-squared : {float(r2):.6f}")
    if aic is not None:
        console.print(f"  AICc      : {float(aic):.4f}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the openfit version."""
    console.print(f"openfit {__version__}")


@app.command()
def models() -> None:
    """List all available built-in models."""
    tbl = Table(
        title="openfit built-in models",
        box=rich.box.ASCII,
        show_header=True,
        header_style="bold",
    )
    tbl.add_column("Model ID", style="cyan", no_wrap=True)
    tbl.add_column("Parameters")

    for model_id in list_models():
        mdl = get_model(model_id)
        param_str = ", ".join(mdl.param_names)
        tbl.add_row(model_id, param_str)

    console.print(tbl)


@app.command()
def fit(
    data_file: Path = typer.Argument(..., help="Path to CSV or Excel data file."),  # noqa: B008
    model: str = typer.Option(..., "--model", "-m", help="Model ID (e.g. 'hill4p')."),
    weights: str = typer.Option(
        "uniform",
        "--weights",
        "-w",
        help="Weight scheme: uniform, 1/y, 1/y2, 1/sd2, poisson.",
    ),
    x_col: str = typer.Option("0", "--x-col", help="X column name or 0-based index."),
    y_col: str = typer.Option("1", "--y-col", help="Y column name or 0-based index."),
    sd_col: str | None = typer.Option(  # noqa: B008
        None, "--sd-col", help="SD column for 1/sd2 weighting."
    ),
    report: Path | None = typer.Option(  # noqa: B008
        None, "--report", "-r", help="Output report file (.html or .md)."
    ),
    fmt: str = typer.Option("html", "--format", "-f", help="Report format: html or markdown."),
    log_x: bool = typer.Option(False, "--log-x", help="Use log scale for x axis in plot."),
) -> None:
    """Fit a model to data from a CSV or Excel file."""
    # -- Validate model and weights before touching disk ----------------------
    _validate_model_id(model)
    _validate_weights(weights)

    # -- Resolve column specifiers --------------------------------------------
    x_col_resolved: str | int = _parse_col(x_col)
    y_col_resolved: str | int = _parse_col(y_col)
    sd_col_resolved: str | int | None = _parse_col(sd_col) if sd_col is not None else None

    # -- Load data ------------------------------------------------------------
    try:
        x, y, sd = _load_data(data_file, x_col_resolved, y_col_resolved, sd_col_resolved)
    except FileNotFoundError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # -- Run fit --------------------------------------------------------------
    try:
        fit_kwargs: dict = {"weights": weights}
        if sd is not None:
            fit_kwargs["sd"] = sd
        if log_x:
            fit_kwargs["log_x"] = log_x
        result = Fit(model, x, y, **fit_kwargs).run()
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Fit failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # -- Print summary --------------------------------------------------------
    console.print(f"\nopenfit -- {model}")
    console.print(f"  Data file : {data_file}")
    console.print(f"  n obs     : {len(x)}")
    console.print(f"  Weights   : {weights}")

    summary_method = getattr(result, "summary", None)
    if callable(summary_method):
        console.print(summary_method())
    else:
        _print_param_table(result)
        _print_gof(result)

    # -- Write report ---------------------------------------------------------
    if report is not None:
        try:
            report_method = getattr(result, "report", None)
            if callable(report_method):
                report_method(str(report), fmt=fmt)
            else:
                err_console.print(
                    "[yellow]Warning:[/yellow] FitResult has no report() method; "
                    "report not written."
                )
        except Exception as exc:  # noqa: BLE001
            err_console.print(f"[red]Error writing report:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        else:
            console.print(f"  Report    : {report}")


@app.command()
def compare(
    data_file: Path = typer.Argument(..., help="Path to CSV or Excel data file."),  # noqa: B008
    models_arg: str = typer.Option(..., "--models", help="Comma-separated model IDs."),
    weights: str = typer.Option(
        "uniform",
        "--weights",
        "-w",
        help="Weight scheme: uniform, 1/y, 1/y2, 1/sd2, poisson.",
    ),
    x_col: str = typer.Option("0", "--x-col", help="X column name or 0-based index."),
    y_col: str = typer.Option("1", "--y-col", help="Y column name or 0-based index."),
    sd_col: str | None = typer.Option(  # noqa: B008
        None, "--sd-col", help="SD column for 1/sd2 weighting."
    ),
    report: Path | None = typer.Option(  # noqa: B008
        None, "--report", "-r", help="Output report file (.html or .md)."
    ),
    fmt: str = typer.Option("html", "--format", "-f", help="Report format: html or markdown."),
) -> None:
    """Fit and compare multiple models on the same dataset."""
    # -- Parse and validate model IDs -----------------------------------------
    model_ids = [m.strip() for m in models_arg.split(",") if m.strip()]
    if len(model_ids) < 2:
        err_console.print(
            "[red]Error:[/red] --models requires at least two comma-separated model IDs."
        )
        raise typer.Exit(code=1)

    failed = False
    for mid in model_ids:
        if mid not in list_models():
            err_console.print(
                f"[red]Error:[/red] Unknown model ID: {mid!r}. "
                "Run 'openfit models' to see available models."
            )
            failed = True
    if failed:
        raise typer.Exit(code=1)

    _validate_weights(weights)

    # -- Resolve column specifiers --------------------------------------------
    x_col_resolved: str | int = _parse_col(x_col)
    y_col_resolved: str | int = _parse_col(y_col)
    sd_col_resolved: str | int | None = _parse_col(sd_col) if sd_col is not None else None

    # -- Load data ------------------------------------------------------------
    try:
        x, y, sd = _load_data(data_file, x_col_resolved, y_col_resolved, sd_col_resolved)
    except FileNotFoundError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # -- Fit each model -------------------------------------------------------
    fit_kwargs: dict = {"weights": weights}
    if sd is not None:
        fit_kwargs["sd"] = sd

    results = []
    for mid in model_ids:
        try:
            result = Fit(mid, x, y, **fit_kwargs).run()
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            err_console.print(f"[red]Fit failed for {mid!r}:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    # -- Compare models -------------------------------------------------------
    try:
        comparison = compare_models(results)
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Comparison failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # -- Print comparison table to console ------------------------------------
    console.print(comparison.summary)

    # -- Write report ---------------------------------------------------------
    if report is not None:
        try:
            _write_comparison_report(comparison, report, fmt)
        except Exception as exc:  # noqa: BLE001
            err_console.print(f"[red]Error writing report:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        else:
            console.print(f"  Report written to: {report}")


# ---------------------------------------------------------------------------
# Report helper for comparison (ComparisonResult has no .report() method)
# ---------------------------------------------------------------------------


def _write_comparison_report(comparison: object, path: Path, fmt: str) -> None:
    """Write a ComparisonResult summary to disk.

    ComparisonResult exposes a plain-text ``.summary`` string.  For HTML output
    the text is wrapped in a minimal ``<pre>`` block; for markdown it is fenced.

    Parameters
    ----------
    comparison : ComparisonResult
        Object with a ``.summary`` str attribute.
    path : Path
        Destination file path.
    fmt : str
        Either ``"html"`` or ``"markdown"`` (``"md"`` also accepted).
    """
    summary: str = getattr(comparison, "summary", "")
    fmt_lower = fmt.lower().strip()

    if fmt_lower in {"html"}:
        content = (
            "<!DOCTYPE html>\n<html>\n<head>\n"
            "<meta charset='utf-8'>\n"
            "<title>openfit Model Comparison</title>\n"
            "</head>\n<body>\n"
            "<pre style='font-family: monospace; white-space: pre;'>\n"
            f"{summary}\n"
            "</pre>\n"
            "<p>This report was generated using openfit (open-source). "
            "Results should be independently verified for regulatory or "
            "clinical decision-making.</p>\n"
            "</body>\n</html>\n"
        )
    else:
        content = (
            "# openfit Model Comparison\n\n"
            "```\n"
            f"{summary}\n"
            "```\n\n"
            "> This report was generated using openfit (open-source). "
            "Results should be independently verified for regulatory or "
            "clinical decision-making.\n"
        )

    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point registered in pyproject.toml as openfit = openfit.cli:main."""
    app()


if __name__ == "__main__":
    main()
