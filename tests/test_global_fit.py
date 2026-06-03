"""Tests for openfit.global_fit.GlobalFit."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from openfit import GlobalFit


def _make_datasets_shared_top_bottom(n=20, seed=0):
    """3 Hill4P datasets with same Top/Bottom (100/0), different EC50."""
    rng = np.random.default_rng(seed)
    x = np.logspace(-2, 2, n)
    datasets = []
    ec50_vals = [0.5, 1.0, 2.0]
    for ec50 in ec50_vals:
        y_true = 0.0 + (100.0 - 0.0) / (1.0 + (ec50 / x) ** 1.0)
        y = y_true * (1.0 + 0.005 * rng.standard_normal(n))
        y = np.maximum(y, 0.01)
        datasets.append((x, y))
    return datasets


def _make_datasets_different_top(n=20, seed=1):
    """3 Hill4P datasets with different Top values — sharing Top is NOT justified."""
    rng = np.random.default_rng(seed)
    x = np.logspace(-2, 2, n)
    datasets = []
    top_vals = [60.0, 100.0, 140.0]
    for top in top_vals:
        y_true = 0.0 + (top - 0.0) / (1.0 + (1.0 / x) ** 1.0)
        y = y_true * (1.0 + 0.01 * rng.standard_normal(n))
        y = np.maximum(y, 0.01)
        datasets.append((x, y))
    return datasets


# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------


def test_global_fit_result_has_shared_and_local():
    """GlobalFitResult has .shared_params and .local_params attributes."""
    datasets = _make_datasets_shared_top_bottom()
    gf = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top", "Bottom", "HillSlope"],
        local=["EC50"],
        weights="uniform",
    ).run()
    assert hasattr(gf, "shared_params")
    assert hasattr(gf, "local_params")


def test_global_fit_local_params_count():
    """len(local_params) equals the number of datasets."""
    datasets = _make_datasets_shared_top_bottom()
    gf = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top", "Bottom", "HillSlope"],
        local=["EC50"],
        weights="uniform",
    ).run()
    assert len(gf.local_params) == len(datasets)


def test_global_fit_shared_top_bottom():
    """Shared Top and Bottom are recovered within 10% of true values (0 and 100)."""
    datasets = _make_datasets_shared_top_bottom()
    gf = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top", "Bottom", "HillSlope"],
        local=["EC50"],
        weights="uniform",
    ).run()
    assert abs(gf.shared_params["Top"] - 100.0) / 100.0 < 0.10
    assert abs(gf.shared_params["Bottom"] - 0.0) < 5.0  # absolute tolerance near zero


# ---------------------------------------------------------------------------
# F-test (sharing justified vs. not)
# ---------------------------------------------------------------------------


def test_global_fit_f_test_justifies_sharing():
    """F-test p_value > 0.05 when datasets truly share Top and Bottom."""
    datasets = _make_datasets_shared_top_bottom()
    gf = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top", "Bottom"],
        local=["EC50", "HillSlope"],
        weights="uniform",
        run_f_test=True,
    ).run()
    assert gf.f_test_sharing is not None
    assert gf.f_test_sharing.sharing_justified, (
        f"Expected sharing justified, p={gf.f_test_sharing.p_value:.4f}"
    )


def test_global_fit_f_test_rejects_sharing():
    """F-test marks sharing_justified=False when datasets have very different Top."""
    datasets = _make_datasets_different_top()
    gf = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top"],
        local=["Bottom", "EC50", "HillSlope"],
        weights="uniform",
        run_f_test=True,
    ).run()
    assert gf.f_test_sharing is not None
    # With dramatically different Top values (60, 100, 140), sharing should be rejected.
    assert not gf.f_test_sharing.sharing_justified, (
        f"Expected sharing rejected, p={gf.f_test_sharing.p_value:.4f}"
    )


# ---------------------------------------------------------------------------
# Global fit report tests
# ---------------------------------------------------------------------------


def test_global_fit_report_html_generation(tmp_path):
    """Test HTML report generation for 3-dataset global fit."""
    datasets = _make_datasets_shared_top_bottom()
    gf = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top", "Bottom", "HillSlope"],
        local=["EC50"],
        weights="uniform",
    ).run()
    
    report_path = tmp_path / "global_fit_report.html"
    gf.report(str(report_path), fmt="html")
    
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    
    # Check that all datasets are mentioned
    assert "Dataset 1" in content
    assert "Dataset 2" in content
    assert "Dataset 3" in content
    
    # Check that shared parameters are listed
    assert "Top" in content
    assert "Bottom" in content
    assert "HillSlope" in content
    
    # Check that local parameter is listed
    assert "EC50" in content
    
    # Check that R^2 values are present
    assert "R^2" in content
    
    # Check for disclaimer
    assert "independently verified" in content


def test_global_fit_report_markdown_generation(tmp_path):
    """Test Markdown report generation for global fit."""
    datasets = _make_datasets_shared_top_bottom()
    gf = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top", "Bottom"],
        local=["EC50", "HillSlope"],
        weights="uniform",
    ).run()
    
    report_path = tmp_path / "global_fit_report.md"
    gf.report(str(report_path), fmt="markdown")
    
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    
    # Check that all datasets are mentioned
    assert "Dataset 1" in content
    assert "Dataset 2" in content
    assert "Dataset 3" in content
    
    # Check for parameter tables
    assert "Shared Parameters" in content
    assert "Local Parameters" in content
    
    # Check for disclaimer
    assert "independently verified" in content


def test_global_fit_report_contains_f_test(tmp_path):
    """Test that F-test results are included in the report."""
    datasets = _make_datasets_shared_top_bottom()
    gf = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top", "Bottom"],
        local=["EC50", "HillSlope"],
        weights="uniform",
        run_f_test=True,
    ).run()
    
    report_path = tmp_path / "global_fit_with_ftest.html"
    gf.report(str(report_path), fmt="html")
    
    content = report_path.read_text(encoding="utf-8")
    
    # Check F-test section exists
    assert "F-test" in content
    assert "Sharing" in content
    
    # Check F-test metrics
    assert "F statistic" in content
    assert "p-value" in content
    assert "RSS" in content


def test_global_fit_report_shared_local_tables(tmp_path):
    """Test that shared and local parameter tables are correctly populated."""
    datasets = _make_datasets_shared_top_bottom()
    gf = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top", "Bottom"],
        local=["EC50", "HillSlope"],
        weights="uniform",
    ).run()
    
    report_path = tmp_path / "global_fit_tables.html"
    gf.report(str(report_path), fmt="html")
    
    content = report_path.read_text(encoding="utf-8")
    
    # Check shared parameter section
    assert "Shared Parameters" in content
    # Shared params should appear once (same value for all datasets)
    assert content.count(">Top<") >= 1
    assert content.count(">Bottom<") >= 1
    
    # Check local parameter section
    assert "Local Parameters" in content
    # Local params should appear for each dataset
    assert "Dataset 1" in content
    assert "Dataset 2" in content
    assert "Dataset 3" in content
    
    # EC50 and HillSlope should be in local parameters
    assert "EC50" in content
    assert "HillSlope" in content


def test_global_fit_report_invalid_format(tmp_path):
    """Test that invalid format raises ValueError."""
    datasets = _make_datasets_shared_top_bottom()
    gf = GlobalFit(
        datasets,
        "hill4p",
        shared=["Top", "Bottom"],
        local=["EC50", "HillSlope"],
        weights="uniform",
    ).run()
    
    report_path = tmp_path / "global_fit_report.txt"
    with pytest.raises(ValueError, match="fmt must be one of"):
        gf.report(str(report_path), fmt="txt")

