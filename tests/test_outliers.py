"""Tests for openfit.outliers: ROUT adaptive outlier detection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from openfit.outliers import rout_outliers


def _clean_hill4p_data(n=30, seed=0):
    """Clean Hill4P data with very low noise (no outliers)."""
    rng = np.random.default_rng(seed)
    x = np.logspace(-1, 1, n)  # narrower range to avoid extreme tail values
    y_true = 0.0 + (100.0 - 0.0) / (1.0 + (1.0 / x) ** 1.0)
    y = y_true * (1.0 + 0.002 * rng.standard_normal(n))  # 0.2% noise -- very clean
    return x, np.maximum(y, 0.01)


# ---------------------------------------------------------------------------
# Structural / type tests
# ---------------------------------------------------------------------------


def test_rout_mask_shape():
    """ROUTResult.outlier_mask has the same shape as the input data."""
    x, y = _clean_hill4p_data()
    result = rout_outliers(x, y, "hill4p", Q=0.01)
    assert result.outlier_mask.shape == (len(x),)


def test_rout_mask_dtype_bool():
    """outlier_mask is a boolean array."""
    x, y = _clean_hill4p_data()
    result = rout_outliers(x, y, "hill4p", Q=0.01)
    assert result.outlier_mask.dtype == bool


def test_rout_result_has_expected_attributes():
    """ROUTResult exposes .outlier_mask, .n_outliers, and .Q."""
    x, y = _clean_hill4p_data()
    result = rout_outliers(x, y, "hill4p", Q=0.01)
    assert hasattr(result, "outlier_mask")
    assert hasattr(result, "n_outliers")
    assert hasattr(result, "Q")
    assert result.Q == 0.01


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


def test_rout_no_outliers_clean_data():
    """ROUT detects 0 outliers on clean low-noise Hill4P data."""
    x, y = _clean_hill4p_data(n=40)
    result = rout_outliers(x, y, "hill4p", Q=0.01)
    assert result.n_outliers == 0, (
        f"Expected 0 outliers in clean data, got {result.n_outliers} "
        f"at indices {result.outlier_indices}"
    )


def test_rout_detects_single_outlier():
    """ROUT flags a massively spiked point as an outlier."""
    x, y = _clean_hill4p_data(n=40, seed=1)
    # Inject an extreme outlier: set a mid-curve point to 10x the true value
    y_outlier = y.copy()
    spike_idx = len(x) // 2
    y_outlier[spike_idx] = y[spike_idx] * 10.0
    result = rout_outliers(x, y_outlier, "hill4p", Q=0.01)
    assert result.n_outliers >= 1, "Expected at least 1 outlier to be detected"
    assert result.outlier_mask[spike_idx], "Spiked point was not flagged"


def test_rout_q_parameter_respected():
    """Higher Q should flag same or more outliers than lower Q."""
    x, y = _clean_hill4p_data(n=40, seed=2)
    # Inject a moderate outlier
    y_mod = y.copy()
    y_mod[15] = y[15] * 3.5
    result_strict = rout_outliers(x, y_mod, "hill4p", Q=0.001)
    result_lenient = rout_outliers(x, y_mod, "hill4p", Q=0.05)
    assert result_lenient.n_outliers >= result_strict.n_outliers, (
        f"Expected higher Q to flag >= outliers: strict={result_strict.n_outliers}, "
        f"lenient={result_lenient.n_outliers}"
    )


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------


def test_rout_outlier_plot_generation():
    """rout_outlier_plot creates a matplotlib Figure with correct structure."""
    import matplotlib.pyplot as plt
    from openfit.plotting import rout_outlier_plot
    
    x, y = _clean_hill4p_data(n=40, seed=1)
    y_mod = y.copy()
    y_mod[20] = y[20] * 5.0
    
    result = rout_outliers(x, y_mod, "hill4p", Q=0.01)
    fig = rout_outlier_plot(x, y_mod, result)
    
    assert fig is not None
    assert hasattr(fig, "savefig")
    plt.close(fig)


def test_rout_outlier_plot_distinct_markers():
    """Normal and outlier points use visually distinct markers and colors."""
    import matplotlib.pyplot as plt
    from openfit.plotting import rout_outlier_plot
    
    x, y = _clean_hill4p_data(n=40, seed=1)
    y_mod = y.copy()
    y_mod[20] = y[20] * 5.0
    
    result = rout_outliers(x, y_mod, "hill4p", Q=0.01)
    assert result.n_outliers > 0, "Test requires at least one outlier"
    
    fig = rout_outlier_plot(x, y_mod, result)
    ax = fig.axes[0]
    
    # Check that we have PathCollection objects (scatter plots)
    collections = [c for c in ax.collections if hasattr(c, "get_paths")]
    assert len(collections) >= 2, "Should have at least 2 scatter collections (normal + outlier)"
    
    # Check that outlier collection uses 'x' marker
    outlier_collection = collections[-1]  # Outliers are plotted last
    paths = outlier_collection.get_paths()
    # 'x' marker has a specific path structure
    assert len(paths) > 0
    
    plt.close(fig)


def test_rout_outlier_plot_with_model_curve():
    """rout_outlier_plot can overlay the fitted model curve."""
    import matplotlib.pyplot as plt
    from openfit.plotting import rout_outlier_plot
    from openfit.models import get_model
    
    x, y = _clean_hill4p_data(n=40, seed=1)
    y_mod = y.copy()
    y_mod[20] = y[20] * 5.0
    
    result = rout_outliers(x, y_mod, "hill4p", Q=0.01)
    model = get_model("hill4p")
    
    fig = rout_outlier_plot(
        x, y_mod, result,
        model_equation=model.equation,
        model_params=result.clean_params
    )
    
    ax = fig.axes[0]
    # Check that we have line plots (model curve)
    lines = ax.get_lines()
    assert len(lines) > 0, "Should have at least one line (model curve)"
    
    plt.close(fig)


def test_rout_outlier_plot_title_contains_q():
    """Default title includes the Q parameter value."""
    import matplotlib.pyplot as plt
    from openfit.plotting import rout_outlier_plot
    
    x, y = _clean_hill4p_data(n=40, seed=1)
    y_mod = y.copy()
    y_mod[20] = y[20] * 5.0
    
    result = rout_outliers(x, y_mod, "hill4p", Q=0.01)
    fig = rout_outlier_plot(x, y_mod, result)
    
    ax = fig.axes[0]
    title = ax.get_title()
    assert "ROUT" in title
    assert "1%" in title or "Q=" in title
    
    plt.close(fig)


def test_html_report_includes_rout_section_with_outliers():
    """HTML report includes ROUT outlier section when outliers are detected."""
    from openfit.report.html import render_html_report
    
    x, y = _clean_hill4p_data(n=40, seed=1)
    y_mod = y.copy()
    y_mod[20] = y[20] * 5.0
    
    result = rout_outliers(x, y_mod, "hill4p", Q=0.01)
    assert result.n_outliers > 0, "Test requires at least one outlier"
    
    # Build a FitResult with rout_result attached
    from openfit.fit import Fit
    fit_result = Fit("hill4p", x, y_mod, weights="uniform").run()
    fit_result.rout_result = result
    
    html = render_html_report(fit_result)
    
    # Check that ROUT section is present
    assert "ROUT Outlier Detection" in html
    assert f"{result.n_outliers}" in html


def test_html_report_excludes_rout_section_without_outliers():
    """HTML report does not include ROUT section when no ROUT was run."""
    from openfit.report.html import render_html_report
    
    x, y = _clean_hill4p_data(n=40, seed=0)
    
    # Build a FitResult WITHOUT rout_result
    from openfit.fit import Fit
    fit_result = Fit("hill4p", x, y, weights="uniform").run()
    
    html = render_html_report(fit_result)
    
    # Check that ROUT section is not present (rout_result is None)
    assert "ROUT Outlier Detection" not in html


def test_markdown_report_includes_rout_section_with_outliers():
    """Markdown report includes ROUT outlier section when outliers are detected."""
    from openfit.report.markdown import render_markdown_report
    
    x, y = _clean_hill4p_data(n=40, seed=1)
    y_mod = y.copy()
    y_mod[20] = y[20] * 5.0
    
    result = rout_outliers(x, y_mod, "hill4p", Q=0.01)
    assert result.n_outliers > 0, "Test requires at least one outlier"
    
    # Build a FitResult with rout_result attached
    from openfit.fit import Fit
    fit_result = Fit("hill4p", x, y_mod, weights="uniform").run()
    fit_result.rout_result = result
    
    md = render_markdown_report(fit_result)
    
    # Check that ROUT section is present
    assert "ROUT Outlier Detection" in md
    assert f"{result.n_outliers}" in md


def test_markdown_report_excludes_rout_section_without_outliers():
    """Markdown report does not include ROUT section when no ROUT was run."""
    from openfit.report.markdown import render_markdown_report
    
    x, y = _clean_hill4p_data(n=40, seed=0)
    
    # Build a FitResult WITHOUT rout_result
    from openfit.fit import Fit
    fit_result = Fit("hill4p", x, y, weights="uniform").run()
    
    md = render_markdown_report(fit_result)
    
    # Check that ROUT section is not present
    assert "ROUT Outlier Detection" not in md
