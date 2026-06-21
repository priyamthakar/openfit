"""Tests for openfit.diagnostics: residual analysis, runs test, normality test."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from openfit import Fit
from openfit.diagnostics import (
    DurbinWatsonResult,
    LackOfFitResult,
    durbin_watson,
    lack_of_fit_test,
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


# ---------------------------------------------------------------------------
# Durbin-Watson tests
# ---------------------------------------------------------------------------


def test_durbin_watson_uncorrelated() -> None:
    """Test that Durbin-Watson statistic is near 2 for uncorrelated residuals."""
    x = np.linspace(0, 10, 100)
    # Generate linear data with independent normal noise
    rng = np.random.default_rng(42)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.5, len(x))

    result = Fit("poly1", x, y, weights="uniform").run()
    dw_res = durbin_watson(result)

    assert isinstance(dw_res, DurbinWatsonResult)
    # DW should be close to 2.0 (no autocorrelation)
    assert 1.5 <= dw_res.statistic <= 2.5
    assert dw_res.interpretation == "no autocorrelation"


def test_durbin_watson_positive_autocorrelation() -> None:
    """Test that Durbin-Watson statistic is << 2 for positive autocorrelation."""
    x = np.linspace(0, 10, 100)
    # Generate positively autocorrelated noise using an AR(1) process
    rng = np.random.default_rng(42)
    noise = np.zeros(len(x))
    for i in range(1, len(x)):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.2)

    y = 2.0 * x + 1.0 + noise
    result = Fit("poly1", x, y, weights="uniform").run()
    dw_res = durbin_watson(result)

    assert isinstance(dw_res, DurbinWatsonResult)
    # DW should be small (positive autocorrelation)
    assert dw_res.statistic < 1.5
    assert dw_res.interpretation == "positive autocorrelation"


def test_durbin_watson_negative_autocorrelation() -> None:
    """Test that Durbin-Watson statistic is > 2.5 for negative autocorrelation."""
    x = np.linspace(0, 10, 100)
    # Generate negatively autocorrelated noise (alternating signs)
    rng = np.random.default_rng(42)
    noise = np.zeros(len(x))
    for i in range(1, len(x)):
        noise[i] = -0.9 * noise[i - 1] + rng.normal(0, 0.2)

    y = 2.0 * x + 1.0 + noise
    result = Fit("poly1", x, y, weights="uniform").run()
    dw_res = durbin_watson(result)

    assert isinstance(dw_res, DurbinWatsonResult)
    assert dw_res.statistic > 2.5
    assert dw_res.interpretation == "negative autocorrelation"


def test_durbin_watson_insufficient_data() -> None:
    """Test Durbin-Watson with very few residuals."""

    class DummyResult:
        residuals = np.array([1.0])

    dw_res = durbin_watson(DummyResult())  # type: ignore
    assert dw_res.statistic == 2.0
    assert dw_res.interpretation == "no autocorrelation"


# ---------------------------------------------------------------------------
# Lack-of-Fit tests
# ---------------------------------------------------------------------------


def test_lack_of_fit_correct_model() -> None:
    """Lack-of-Fit test should be non-significant for a correct model on replicated data."""
    # Define replicated x values: 5 levels, each repeated 4 times
    x = np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 4)
    # Generate data using a linear equation + normal noise
    rng = np.random.default_rng(100)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.1, len(x))

    # Fit a linear (correct) model
    result = Fit("poly1", x, y, weights="uniform").run()
    lof_res = lack_of_fit_test(result)

    assert isinstance(lof_res, LackOfFitResult)
    assert lof_res.statistic is not None
    assert lof_res.p_value is not None
    assert lof_res.passed is True
    assert lof_res.p_value > 0.05
    assert lof_res.df_lof == 3  # M - K = 5 - 2 = 3
    assert lof_res.df_pe == 15  # N - M = 20 - 5 = 15


def test_lack_of_fit_incorrect_model() -> None:
    """Lack-of-Fit test should be significant for an incorrect model on replicated data."""
    # Define replicated x values
    x = np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 4)
    # Generate quadratic data (nonlinear) with small noise
    rng = np.random.default_rng(100)
    y = 1.5 * x**2 + 0.5 * x + 2.0 + rng.normal(0, 0.1, len(x))

    # Fit a linear (incorrect) model
    result = Fit("poly1", x, y, weights="uniform").run()
    lof_res = lack_of_fit_test(result)

    assert isinstance(lof_res, LackOfFitResult)
    assert lof_res.statistic is not None
    assert lof_res.p_value is not None
    assert lof_res.passed is False
    assert lof_res.p_value <= 0.05
    assert lof_res.df_lof == 3  # M - K = 5 - 2 = 3
    assert lof_res.df_pe == 15  # N - M = 20 - 5 = 15


def test_lack_of_fit_custom_x_groups() -> None:
    """Lack-of-Fit test with custom x_groups should partition correctly."""
    # 6 points, two groups defined by x_groups
    x = np.array([1.0, 1.1, 1.2, 2.0, 2.1, 2.2])
    x_groups = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([2.0, 2.1, 1.9, 4.0, 4.1, 3.9])

    result = Fit("poly1", x, y, weights="uniform").run()

    # Pass custom x_groups
    lof_res = lack_of_fit_test(result, x_groups=x_groups)

    assert isinstance(lof_res, LackOfFitResult)
    assert lof_res.df_lof == 2 - 2  # M - K = 2 - 2 = 0
    assert lof_res.df_pe == 6 - 2  # N - M = 6 - 2 = 4
    # Since df_lof <= 0, lack_of_fit_test should return p_value=None
    assert lof_res.p_value is None
    assert lof_res.statistic is None
    assert lof_res.passed is None

    # Test mismatched length raises ValueError
    import pytest

    with pytest.raises(ValueError, match="x_groups must have the same length"):
        lack_of_fit_test(result, x_groups=np.array([0, 0, 1]))


def test_lack_of_fit_no_replicates() -> None:
    """If there are no replicates, the test cannot be performed."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.1, 1.9, 3.1, 3.9])
    result = Fit("poly1", x, y, weights="uniform").run()
    lof_res = lack_of_fit_test(result)

    assert lof_res.statistic is None
    assert lof_res.p_value is None
    assert lof_res.passed is None
    assert lof_res.df_pe == 0  # N - M = 4 - 4 = 0
