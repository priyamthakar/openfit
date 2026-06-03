"""Tests for DOCX report generation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tempfile

import numpy as np
import pytest

from openfit import Fit
from openfit.report import report_fit
from openfit.report.docx import render_docx_report


@pytest.fixture
def hill4p_fit_result():
    """Run a simple Hill4P fit and return the FitResult."""
    x = np.logspace(-2, 2, 20)
    y_true = 0.0 + (100.0 - 0.0) / (1.0 + (1.0 / x) ** 1.0)
    y = y_true * (1.0 + 0.01 * np.random.default_rng(42).standard_normal(len(x)))
    y = np.maximum(y, 0.01)
    return Fit("hill4p", x, y, weights="uniform").run()


def test_render_docx_report_returns_bytes(hill4p_fit_result) -> None:
    """render_docx_report returns non-empty bytes."""
    docx_bytes = render_docx_report(hill4p_fit_result)
    assert isinstance(docx_bytes, bytes)
    assert len(docx_bytes) > 0


def test_docx_is_valid(hill4p_fit_result) -> None:
    """DOCX output starts with 'PK' magic bytes (ZIP container)."""
    docx_bytes = render_docx_report(hill4p_fit_result)
    # DOCX files are ZIP archives, which start with PK
    assert docx_bytes.startswith(b"PK")


def test_report_fit_with_docx_writes_file(hill4p_fit_result) -> None:
    """report_fit with fmt='docx' writes a valid DOCX file."""
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        report_fit(hill4p_fit_result, tmp_path, fmt="docx")

        # Verify file was created and is non-empty
        file_path = Path(tmp_path)
        assert file_path.exists()
        file_bytes = file_path.read_bytes()
        assert len(file_bytes) > 0
        assert file_bytes.startswith(b"PK")
    finally:
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)


def test_report_docx_with_pathlib(hill4p_fit_result) -> None:
    """report_fit with fmt='docx' accepts a pathlib.Path via str()."""
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        report_fit(hill4p_fit_result, str(tmp_path), fmt="docx")
        assert tmp_path.exists()
        assert tmp_path.stat().st_size > 0
    finally:
        tmp_path.unlink(missing_ok=True)
