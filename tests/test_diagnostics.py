"""Tests for openfit.diagnostics: residual analysis, runs test, normality test."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from openfit import Fit
from openfit.diagnostics import (
    normality_test,
    replicates_test,
    residual_analysis,
    runs_test,
)

# ---------------------------------------------------------------------------
# runs_test (standalone)
# ---------------------------------------------------------------------------


def test_runs_test_random_residuals() -> None:
    """Uniformly random residuals pass the runs test (p > 0.05)."""
    rng = np.random.default_rng(0)
    residuals = rng.standard_normal(100)
    p, n_runs = runs_test(residuals)
    # White noise should not be rejected by the runs test at p=0.05
    assert p > 0.05, f"Expected p>0.05 for random residuals, got {p:.4f}"


def test_runs_test_systematic_residuals() -> None:
    """Perfectly alternating +/- residuals produce a p-value < 0.05."""
    # Alternating signs: maximum number of runs -> too many, statistically significant
    n = 40
    residuals = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
    p, n_runs = runs_test(residuals)
    # Alternating gives n_runs = n which is far above the expected value
    assert p < 0.05, f"Expected p<0.05 for alternating residuals, got {p:.4f}"


# ---------------------------------------------------------------------------
# normality_test (standalone)
# ---------------------------------------------------------------------------


def test_normality_test_normal_data() -> None:
    """Gaussian residuals pass the normality test (p > 0.05)."""
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal(40)
    p, name = normality_test(residuals)
    assert p > 0.05, f"Expected p>0.05 for normal data, got {p:.4f}"
    assert name == "Shapiro-Wilk"


def test_normality_test_non_normal() -> None:
    """Highly skewed (exponential) residuals fail the normality test (p < 0.05)."""
    rng = np.random.default_rng(7)
    # Exponential distribution is strongly right-skewed
    residuals = rng.exponential(scale=1.0, size=40)
    p, name = normality_test(residuals)
    assert p < 0.05, f"Expected p<0.05 for skewed data, got {p:.4f}"


def test_normality_test_large_uses_dagostino() -> None:
    """n > 50 uses D'Agostino-Pearson."""
    rng = np.random.default_rng(0)
    residuals = rng.standard_normal(60)
    p, name = normality_test(residuals)
    assert name == "D'Agostino-Pearson"


# ---------------------------------------------------------------------------
# replicates_test (standalone)
# ---------------------------------------------------------------------------


def test_replicates_test_returns_none_without_replicates() -> None:
    """All unique x values: replicates_test returns None."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_fitted = np.array([1.1, 1.9, 3.1, 3.9])
    result = replicates_test(x, y, y_fitted)
    assert result is None


def test_replicates_test_pvalue_with_replicates() -> None:
    """x has duplicates: replicates_test returns a float p-value."""
    x = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    y = np.array([1.0, 1.1, 2.0, 2.1, 3.0, 3.1])
    y_fitted = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    result = replicates_test(x, y, y_fitted)
    assert result is not None
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# residual_analysis (full integration via FitResult)
# ---------------------------------------------------------------------------


def _make_fit_result():
    rng = np.random.default_rng(11)
    x = np.logspace(-2, 2, 30)
    y_true = 0.0 + (100.0 - 0.0) / (1.0 + (1.0 / x) ** 1.0)
    y = y_true * (1.0 + 0.01 * rng.standard_normal(30))
    y = np.maximum(y, 0.01)
    return Fit("hill4p", x, y, weights="uniform").run()


def test_outlier_flags_shape() -> None:
    """DiagnosticsResult.outlier_flags has shape (n_obs,)."""
    result = _make_fit_result()
    diag = residual_analysis(result)
    assert diag.outlier_flags.shape == (result.n_obs,)


def test_outlier_flags_false_for_clean_data() -> None:
    """Clean Hill4P fit: no points flagged as outliers at 3-sigma."""
    result = _make_fit_result()
    diag = residual_analysis(result)
    assert not np.any(diag.outlier_flags), (
        f"Expected 0 outliers in clean data, got {diag.outlier_flags.sum()}"
    )


def test_summary_is_string() -> None:
    """DiagnosticsResult.summary is a non-empty string."""
    result = _make_fit_result()
    diag = residual_analysis(result)
    assert isinstance(diag.summary, str)
    assert len(diag.summary) > 0
