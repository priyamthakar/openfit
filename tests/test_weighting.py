"""Tests for openfit.weighting: weight schemes and parsing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from openfit.weighting import WeightScheme, apply_weights, parse_weight_scheme


# ---------------------------------------------------------------------------
# apply_weights correctness
# ---------------------------------------------------------------------------


def test_uniform_weights_are_ones():
    """apply_weights(y, 'uniform') returns an array of all 1.0."""
    y = np.array([1.0, 2.0, 5.0, 10.0])
    w = apply_weights(y, "uniform")
    np.testing.assert_array_equal(w, np.ones_like(y))


def test_1_over_y_weights():
    """apply_weights(y, '1/y') returns 1/y elementwise."""
    y = np.array([1.0, 2.0, 4.0, 8.0])
    w = apply_weights(y, "1/y")
    np.testing.assert_allclose(w, 1.0 / y)


def test_1_over_y2_weights():
    """apply_weights(y, '1/y2') returns 1/y^2 elementwise."""
    y = np.array([1.0, 2.0, 4.0])
    w = apply_weights(y, "1/y2")
    np.testing.assert_allclose(w, 1.0 / (y ** 2))


def test_poisson_weights():
    """apply_weights(y, 'poisson') returns 1/y (Poisson variance = mean)."""
    y = np.array([3.0, 6.0, 12.0])
    w = apply_weights(y, "poisson")
    np.testing.assert_allclose(w, 1.0 / y)


# ---------------------------------------------------------------------------
# parse_weight_scheme aliases
# ---------------------------------------------------------------------------


def test_string_aliases_accepted():
    """Case-insensitive aliases resolve to the correct WeightScheme member."""
    assert parse_weight_scheme("1/Y") == WeightScheme.ONE_OVER_Y
    assert parse_weight_scheme("1/Y2") == WeightScheme.ONE_OVER_Y2
    assert parse_weight_scheme("1/Y^2") == WeightScheme.ONE_OVER_Y2
    assert parse_weight_scheme("UNIFORM") == WeightScheme.UNIFORM
    assert parse_weight_scheme("Poisson") == WeightScheme.POISSON
    assert parse_weight_scheme("none") == WeightScheme.UNIFORM
    assert parse_weight_scheme("one_over_y") == WeightScheme.ONE_OVER_Y


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------


def test_negative_y_raises_for_1_over_y():
    """Negative y values raise ValueError for '1/y'."""
    y = np.array([1.0, -0.5, 2.0])
    with pytest.raises(ValueError, match="strictly positive"):
        apply_weights(y, "1/y")


def test_zero_y_raises_for_1_over_y2():
    """Zero in y raises ValueError for '1/y2' (non-positive guard)."""
    y = np.array([1.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="strictly positive"):
        apply_weights(y, "1/y2")


def test_unknown_scheme_raises():
    """An unrecognised scheme string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown weight scheme"):
        parse_weight_scheme("not_a_scheme")
