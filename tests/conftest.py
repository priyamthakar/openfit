"""Shared pytest fixtures for openfit tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path whether or not the package is installed editably.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest


@pytest.fixture
def hill4p_data():
    """20-point 4PL dataset: Bottom=0, Top=100, EC50=1, HillSlope=1 + 1% noise."""
    rng = np.random.default_rng(42)
    x = np.logspace(-2, 2, 20)
    y_true = 0.0 + (100.0 - 0.0) / (1.0 + (1.0 / x) ** 1.0)
    # Multiplicative noise so all y remain positive (needed for 1/y2 weight tests).
    y = y_true * (1.0 + 0.01 * rng.standard_normal(len(x)))
    # Clamp to strictly positive to guarantee 1/y2 weights are valid.
    y = np.maximum(y, 0.01)
    return x, y


@pytest.fixture
def mono_exp_data():
    """30-point MonoExp dataset: Y0=10, Plateau=0, k=0.5 + 2% noise."""
    rng = np.random.default_rng(7)
    x = np.linspace(0, 5, 30)
    # y = Plateau + (Y0 - Plateau)*exp(-k*x) = 0 + 10*exp(-0.5*x)
    y_true = 0.0 + (10.0 - 0.0) * np.exp(-0.5 * x)
    y = y_true * (1.0 + 0.02 * rng.standard_normal(len(x)))
    y = np.maximum(y, 1e-6)
    return x, y


@pytest.fixture
def mm_data():
    """20-point Michaelis-Menten dataset: Vmax=50, Km=10 + 2% noise."""
    rng = np.random.default_rng(13)
    x = np.linspace(1, 100, 20)
    y_true = 50.0 * x / (10.0 + x)
    y = y_true * (1.0 + 0.02 * rng.standard_normal(len(x)))
    y = np.maximum(y, 0.01)
    return x, y
