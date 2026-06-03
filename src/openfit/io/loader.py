"""CSV and Excel data loaders for openfit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _validate_arrays(x: np.ndarray, y: np.ndarray) -> None:
    """Raise ValueError if x or y contain NaN or Inf, or have mismatched lengths.

    Parameters
    ----------
    x : np.ndarray
        Array of x values.
    y : np.ndarray
        Array of y values.

    Raises
    ------
    ValueError
        If x or y contain NaN or Inf.
        If x and y have different lengths.
    """
    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length. Got len(x)={len(x)}, len(y)={len(y)}."
        )
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN or Inf values. Clean the data before fitting.")
    if not np.isfinite(y).all():
        raise ValueError("y contains NaN or Inf values. Clean the data before fitting.")


def _extract(
    df: pd.DataFrame,
    x_col: str | int,
    y_col: str | int,
    sd_col: str | int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract x, y, and optional sd columns from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    x_col : str | int
        Column name or 0-based integer index for x values.
    y_col : str | int
        Column name or 0-based integer index for y values.
    sd_col : str | int | None
        Column name or 0-based integer index for sd values. None = no sd column.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None]
        (x, y, sd) as float64 arrays. sd is None if sd_col is None.

    Raises
    ------
    ValueError
        If x or y contain NaN or Inf, or fewer than 3 data points.
    """
    x_series = df.iloc[:, x_col] if isinstance(x_col, int) else df[x_col]
    y_series = df.iloc[:, y_col] if isinstance(y_col, int) else df[y_col]

    x = np.asarray(x_series, dtype=np.float64)
    y = np.asarray(y_series, dtype=np.float64)

    _validate_arrays(x, y)

    if len(x) < 3:
        raise ValueError("x and y must contain at least 3 data points")

    sd: np.ndarray | None = None
    if sd_col is not None:
        sd_series = df.iloc[:, sd_col] if isinstance(sd_col, int) else df[sd_col]
        sd = np.asarray(sd_series, dtype=np.float64)

    return x, y, sd


def load_csv(
    path: str | Path,
    x_col: str | int = 0,
    y_col: str | int = 1,
    sheet: str | int = 0,
    skip_rows: int = 0,
    comment_char: str | None = "#",
    sd_col: str | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load x, y (and optionally SD) columns from a CSV file.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    x_col : str | int
        Column name or 0-based index for x values. Default 0.
    y_col : str | int
        Column name or 0-based index for y values. Default 1.
    sheet : str | int
        Sheet name or index (accepted for API symmetry with load_excel, unused for CSV).
        Default 0.
    skip_rows : int
        Number of rows to skip before the header. Default 0.
    comment_char : str | None
        Lines starting with this character are treated as comments. Default '#'.
    sd_col : str | int | None
        Column name or 0-based index for SD values. None = no SD column.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None]
        (x, y, sd) arrays. sd is None if sd_col is None.

    Raises
    ------
    ValueError
        If x or y contain NaN or Inf.
        If x or y have fewer than 3 data points.
    FileNotFoundError
        If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(
        path,
        skiprows=skip_rows,
        comment=comment_char,
    )
    return _extract(df, x_col, y_col, sd_col)


def load_excel(
    path: str | Path,
    x_col: str | int = 0,
    y_col: str | int = 1,
    sheet: str | int = 0,
    skip_rows: int = 0,
    sd_col: str | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load x, y (and optionally SD) from an Excel file.

    Accepts .xlsx, .xls, and .xlsm formats. Requires the openpyxl package for
    .xlsx and .xlsm files.

    Parameters
    ----------
    path : str | Path
        Path to the Excel file.
    x_col : str | int
        Column name or 0-based index for x values. Default 0.
    y_col : str | int
        Column name or 0-based index for y values. Default 1.
    sheet : str | int
        Sheet name or 0-based index to read. Default 0.
    skip_rows : int
        Number of rows to skip before the header. Default 0.
    sd_col : str | int | None
        Column name or 0-based index for SD values. None = no SD column.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None]
        (x, y, sd) arrays. sd is None if sd_col is None.

    Raises
    ------
    ValueError
        If x or y contain NaN or Inf.
        If x or y have fewer than 3 data points.
    FileNotFoundError
        If path does not exist.
    ImportError
        If openpyxl is not installed (required for .xlsx/.xlsm files).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "openpyxl is required to read .xlsx/.xlsm files. "
                "Install it with: pip install openpyxl"
            ) from exc

    df = pd.read_excel(
        path,
        sheet_name=sheet,
        skiprows=skip_rows,
    )
    return _extract(df, x_col, y_col, sd_col)
