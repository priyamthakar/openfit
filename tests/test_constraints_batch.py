# tests/test_constraints_batch.py
"""Tests for advanced fitting: constraints, fixed values, penalties, and batch fits."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from openfit import BatchFit, Fit
from openfit.constraints import ParameterMapper, safe_eval

# ---------------------------------------------------------------------------
# Unit tests for ParameterMapper and safe_eval
# ---------------------------------------------------------------------------


def test_safe_eval() -> None:
    """safe_eval evaluates simple expressions and blocks malicious input."""
    assert safe_eval("2 * x", {"x": 5.0}) == 10.0
    assert safe_eval("abs(x - 5)", {"x": 2.0}) == 3.0
    assert safe_eval("min(a, b)", {"a": 2.0, "b": 4.0}) == 2.0

    # Malicious inputs should fail regex validation or eval
    with pytest.raises(ValueError, match="Invalid characters"):
        safe_eval("import os; os.system('echo')", {})

    with pytest.raises(ValueError, match="Failed to evaluate"):
        safe_eval("nonexistent * 2", {})


def test_parameter_mapper_free_only() -> None:
    """Mapper returns free parameters as-is when no fixed/constraints exist."""
    mapper = ParameterMapper(["a", "b", "c"])
    assert mapper.free_names == ["a", "b", "c"]

    p_act = np.array([1.0, 2.0, 3.0])
    full = mapper.to_full(p_act)
    assert full == {"a": 1.0, "b": 2.0, "c": 3.0}

    # Jacobian is identity matrix
    J = mapper.jacobian_mapping(p_act)
    np.testing.assert_allclose(J, np.eye(3))


def test_parameter_mapper_fixed_and_constrained() -> None:
    """Mapper maps free parameters, injects fixed values, and resolves constraints."""
    mapper = ParameterMapper(
        param_names=["Bottom", "Top", "EC50", "HillSlope"],
        fixed={"Bottom": 2.0},
        constraints={"Top": "Bottom + 90.0", "HillSlope": "2.0 * EC50"},
    )
    assert mapper.free_names == ["EC50"]

    # Top is constrained by Bottom (fixed), HillSlope is constrained by EC50 (free)
    p_act = np.array([3.0])
    full = mapper.to_full(p_act)
    assert full["Bottom"] == 2.0
    assert full["Top"] == 92.0
    assert full["EC50"] == 3.0
    assert full["HillSlope"] == 6.0

    # Jacobian mapping: d(full)/d(free)
    # [Bottom, Top, EC50, HillSlope] w.r.t [EC50]
    # d(Bottom)/d(EC50) = 0
    # d(Top)/d(EC50) = 0 (since Top depends on Bottom only)
    # d(EC50)/d(EC50) = 1
    # d(HillSlope)/d(EC50) = 2.0
    J = mapper.jacobian_mapping(p_act)
    expected = np.array([[0.0], [0.0], [1.0], [2.0]])
    np.testing.assert_allclose(J, expected)


# ---------------------------------------------------------------------------
# Fit integration tests
# ---------------------------------------------------------------------------


def test_fit_fixed_parameter() -> None:
    """Fit with fixed parameter enforces the fixed value and SE = 0."""
    x = np.array([0.1, 1.0, 10.0, 100.0])
    # Ground truth parameters: Bottom=5, Top=95, EC50=2, HillSlope=1.5
    y = 5.0 + 90.0 / (1.0 + (2.0 / x) ** 1.5)

    # Fit with fixed Bottom = 5.0
    res = Fit("hill4p", x, y, weights="uniform", fixed={"Bottom": 5.0}).run()

    assert res.params["Bottom"] == 5.0
    assert res.se["Bottom"] == 0.0
    assert res.covariance[0, 0] == 0.0  # covariance row/col for fixed is 0
    assert res.params["Top"] == pytest.approx(95.0, rel=1e-3)
    assert res.params["EC50"] == pytest.approx(2.0, rel=1e-3)


def test_fit_parameter_expression() -> None:
    """Fit with constraint expressions enforces linked parameter relationships."""
    x = np.array([0.1, 1.0, 10.0, 100.0])
    # Model: y = Bottom + (Top - Bottom) / (1 + (EC50/x)^HillSlope)
    # Force Top to be exactly Bottom + 100.0
    y = 0.0 + 100.0 / (1.0 + (2.0 / x) ** 1.0)

    res = Fit(
        "hill4p",
        x,
        y,
        weights="uniform",
        fixed={"Bottom": 0.0},
        constraints={"Top": "Bottom + 100.0"},
    ).run()

    assert res.params["Bottom"] == 0.0
    assert res.params["Top"] == 100.0
    assert res.se["Top"] == 0.0  # Top derived from fixed, so SE is 0
    assert res.params["EC50"] == pytest.approx(2.0, rel=1e-3)


def test_fit_l2_penalties() -> None:
    """Fit with L2 penalty pulls the parameter estimate towards target value."""
    x = np.array([0.1, 0.5, 2.0, 10.0, 50.0, 200.0])
    # Generates a standard curve peaking at 100
    y = 0.0 + 100.0 / (1.0 + (2.0 / x) ** 1.0)

    # 1. Standard fit (no penalties)
    res_unpen = Fit("hill4p", x, y, weights="uniform").run()

    # 2. Penalized fit pulling Top towards 80
    # penalties format: {name: (type, target, weight)}
    res_pen = Fit(
        "hill4p",
        x,
        y,
        weights="uniform",
        penalties={"Top": ("l2", 80.0, 10.0)},
    ).run()

    # Penalized Top should be pulled below the unpenalized Top
    assert res_pen.params["Top"] < res_unpen.params["Top"]
    assert res_pen.params["Top"] > 80.0


# ---------------------------------------------------------------------------
# BatchFit tests
# ---------------------------------------------------------------------------


def test_batch_fit_sequential() -> None:
    """BatchFit fits multiple datasets and outputs a clean summary DataFrame."""
    x_val = np.array([0.1, 0.5, 2.0, 10.0, 50.0, 200.0])
    datasets = [
        # Dataset 0: EC50 = 2.0
        {"x": x_val, "y": 2.0 + 98.0 / (1.0 + (2.0 / x_val) ** 1.0)},
        # Dataset 1: EC50 = 10.0
        {"x": x_val, "y": 2.0 + 98.0 / (1.0 + (10.0 / x_val) ** 1.0)},
    ]

    batch = BatchFit("hill4p", datasets, weights="uniform")
    results = batch.run()

    assert len(results) == 2
    assert results[0].params["EC50"] == pytest.approx(2.0, rel=1e-3)
    assert results[1].params["EC50"] == pytest.approx(10.0, rel=1e-3)

    # DataFrame summary check
    df = batch.summary_df(results)
    assert len(df) == 2
    assert "dataset_index" in df.columns
    assert "r_squared" in df.columns
    assert "EC50" in df.columns
    assert "EC50_se" in df.columns

    # Check values in DataFrame
    assert df.loc[0, "EC50"] == pytest.approx(2.0, rel=1e-3)
    assert df.loc[1, "EC50"] == pytest.approx(10.0, rel=1e-3)
