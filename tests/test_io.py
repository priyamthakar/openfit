"""Tests for openfit.io.loader: CSV/Excel loading and array validation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import csv
import tempfile

import numpy as np
import pytest

from openfit.io.loader import _validate_arrays, load_csv


# ---------------------------------------------------------------------------
# _validate_arrays
# ---------------------------------------------------------------------------


def test_validate_arrays_passes_clean():
    """Clean finite arrays of equal length pass _validate_arrays without error."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    _validate_arrays(x, y)  # should not raise


def test_validate_arrays_raises_on_nan():
    """NaN in x raises ValueError."""
    x = np.array([1.0, float("nan"), 3.0])
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="NaN"):
        _validate_arrays(x, y)


def test_validate_arrays_raises_on_inf():
    """Inf in y raises ValueError."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, float("inf"), 3.0])
    with pytest.raises(ValueError):
        _validate_arrays(x, y)


def test_validate_arrays_mismatched_lengths():
    """Mismatched array lengths raise ValueError."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        _validate_arrays(x, y)


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------


def _write_csv(rows, tmp_path):
    """Write rows (list of lists) to a temp CSV and return its path."""
    csv_file = tmp_path / "test_data.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return csv_file


def test_load_csv_basic(tmp_path):
    """load_csv returns correct x and y arrays from a simple two-column CSV."""
    rows = [["x", "y"], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]
    csv_file = _write_csv(rows, tmp_path)
    x, y, sd = load_csv(csv_file)
    np.testing.assert_allclose(x, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(y, [2.0, 4.0, 6.0, 8.0])
    assert sd is None


def test_load_csv_with_missing_values_raises(tmp_path):
    """CSV with NaN in y raises ValueError (rule 8: no silent NaN)."""
    rows = [["x", "y"], [1.0, 2.0], [2.0, ""], [3.0, 6.0], [4.0, 8.0]]
    csv_file = _write_csv(rows, tmp_path)
    # load_csv calls _validate_arrays which raises on NaN
    with pytest.raises(ValueError):
        load_csv(csv_file)


def test_load_excel_basic(tmp_path):
    """load_excel returns correct arrays from a simple .xlsx file (skipped if openpyxl missing)."""
    openpyxl = pytest.importorskip("openpyxl")
    from openfit.io.loader import load_excel

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["x", "y"])
    for xi, yi in [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0)]:
        ws.append([xi, yi])

    xlsx_file = tmp_path / "test_data.xlsx"
    wb.save(xlsx_file)

    x, y, sd = load_excel(xlsx_file)
    np.testing.assert_allclose(x, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(y, [2.0, 4.0, 6.0, 8.0])
    assert sd is None
