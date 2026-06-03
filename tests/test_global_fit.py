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
