"""Tests for openfit.spec: FitSpec reproducibility manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from openfit.spec import FitSpec, compute_data_hash, build_spec


# ---------------------------------------------------------------------------
# compute_data_hash
# ---------------------------------------------------------------------------


def test_data_hash_deterministic():
    """Same x, y arrays always produce the same SHA-256 hash."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    h1 = compute_data_hash(x, y)
    h2 = compute_data_hash(x.copy(), y.copy())
    assert h1 == h2
    assert len(h1) == 64  # 256-bit hex digest


def test_data_hash_changes_on_data_change():
    """Mutating any value in the arrays changes the hash."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    h_orig = compute_data_hash(x, y)

    y2 = y.copy()
    y2[0] = 999.0
    h_modified = compute_data_hash(x, y2)
    assert h_orig != h_modified


# ---------------------------------------------------------------------------
# FitSpec serialisation
# ---------------------------------------------------------------------------


def test_to_json_roundtrip():
    """FitSpec -> to_json() -> from_json() reconstructs identical object."""
    spec = build_spec(
        model_id="hill4p",
        param_values={"Bottom": 0.0, "Top": 100.0, "EC50": 1.0, "HillSlope": 1.5},
        weights="1/y2",
        x=np.array([0.1, 1.0, 10.0]),
        y=np.array([10.0, 50.0, 90.0]),
        random_seed=42,
    )
    json_str = spec.to_json()
    spec2 = FitSpec.from_json(json_str)

    assert spec2.model_id == spec.model_id
    assert spec2.weights == spec.weights
    assert spec2.data_hash == spec.data_hash
    assert spec2.random_seed == spec.random_seed
    assert spec2.param_values == spec.param_values


def test_param_values_float_lossless():
    """A float param value survives JSON round-trip with exact precision (repr trick)."""
    tricky_float = 1.0 / 3.0  # not exactly representable in decimal
    spec = FitSpec(
        model_id="test",
        param_values={"a": tricky_float},
        weights="uniform",
        data_hash="deadbeef" * 8,
        openfit_version="0.0.0",
        scipy_version="1.0",
        numpy_version="1.0",
        random_seed=0,
    )
    json_str = spec.to_json()
    spec2 = FitSpec.from_json(json_str)
    # float(repr(x)) is the exact inverse of repr() for IEEE 754 doubles.
    assert spec2.param_values["a"] == tricky_float


def test_from_json_invalid_raises():
    """Malformed JSON raises an error (JSONDecodeError or similar)."""
    with pytest.raises(Exception):
        FitSpec.from_json("{not valid json }")
