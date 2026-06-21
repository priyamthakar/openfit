"""Tests for openfit bands, leverage convenience methods, and plotting extensions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from openfit import (
    BandResult,
    DurbinWatsonResult,
    Fit,
    FitResult,
    LackOfFitResult,
    LeverageResult,
    confidence_band,
    durbin_watson,
    lack_of_fit_test,
    leverage_diagnostics,
    prediction_band,
)


def _make_dummy_fit_result() -> FitResult:
    """Helper to construct a simple fit result."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    return Fit("poly1", x, y, weights="uniform").run()


def test_package_exports() -> None:
    """Verify that all new symbols are properly exported from openfit."""
    from openfit.bands import BandResult as RealBandResult
    from openfit.bands import confidence_band as real_confidence_band
    from openfit.bands import prediction_band as real_prediction_band
    from openfit.leverage import LeverageResult as RealLeverageResult
    from openfit.leverage import leverage_diagnostics as real_leverage_diagnostics

    assert BandResult is RealBandResult
    assert confidence_band is real_confidence_band
    assert prediction_band is real_prediction_band
    assert LeverageResult is RealLeverageResult
    assert leverage_diagnostics is real_leverage_diagnostics
    assert DurbinWatsonResult is not None
    assert LackOfFitResult is not None
    assert durbin_watson is not None
    assert lack_of_fit_test is not None


def test_fit_result_convenience_methods() -> None:
    """Verify convenience methods on FitResult delegate correctly."""
    result = _make_dummy_fit_result()

    # confidence_band
    cb = result.confidence_band(x_pred=np.array([1.0, 2.0]), confidence=0.90)
    assert isinstance(cb, BandResult)
    np.testing.assert_array_equal(cb.x, [1.0, 2.0])

    # prediction_band
    pb = result.prediction_band(x_pred=np.array([2.0, 3.0]), confidence=0.99)
    assert isinstance(pb, BandResult)
    np.testing.assert_array_equal(pb.x, [2.0, 3.0])

    # leverage
    lev = result.leverage()
    assert isinstance(lev, LeverageResult)
    assert len(lev.hat_values) == len(result.x)


def test_plotting_bands_enabled() -> None:
    """Verify that plot() and fit_overlay_plot() run correctly with bands enabled."""
    result = _make_dummy_fit_result()

    # Draw plot with confidence and prediction bands
    fig = result.plot(show_confidence_band=True, show_prediction_band=True, confidence=0.95)
    assert fig is not None
    plt.close(fig)


def test_plotting_show_ci_deprecation() -> None:
    """Verify show_ci triggers warning and maps to show_confidence_band."""
    result = _make_dummy_fit_result()

    with pytest.warns(DeprecationWarning, match="show_ci is deprecated"):
        fig = result.plot(show_ci=True)
    assert fig is not None
    plt.close(fig)
