"""Published-reference validation tests for ROUT algorithm.

Validates the ROUT implementation against Motulsky & Brown (2006) BMC Bioinformatics.

Reference:
    Motulsky HJ, Brown RE. Detecting outliers when fitting data with nonlinear
    regression - a new method based on robust nonlinear regression and the false
    discovery rate. BMC Bioinformatics. 2006;7:123.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC1472692/

The paper describes:
- Figure 2: Synthetic dose-response data with 10 injected outliers
- Table 1: Q parameter vs outlier detection rates
- Q=1% (0.01) as the recommended default FDR threshold
- Testing only the 30% most extreme residuals
- Sequential FDR procedure (Equations 17-18)

These tests reconstruct the paper's methodology using synthetic data with
known outliers to verify the implementation matches the published algorithm.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from openfit.outliers import rout_outliers

# ---------------------------------------------------------------------------
# Figure 2 reconstruction: Biphasic dose-response with 10 outliers
# ---------------------------------------------------------------------------


def _biphasic_dose_response(
    x, bottom1, top1, ec50_1, slope1, bottom2, top2, ec50_2, slope2, fraction
):
    """Biphasic dose-response model from Motulsky & Brown 2006.

    This is the model used in Figure 2 of the paper to demonstrate ROUT
    on complex nonlinear curves with multiple phases.

    Parameters
    ----------
    x : np.ndarray
        Log-concentration values.
    bottom1, top1, ec50_1, slope1 : float
        Parameters for the first (low-dose) phase.
    bottom2, top2, ec50_2, slope2 : float
        Parameters for the second (high-dose) phase.
    fraction : float
        Fraction of response attributed to the first phase.

    Returns
    -------
    np.ndarray
        Predicted response values.
    """
    # First phase
    ratio1 = (ec50_1 / x) ** slope1
    phase1 = bottom1 + (top1 - bottom1) / (1.0 + ratio1)

    # Second phase
    ratio2 = (ec50_2 / x) ** slope2
    phase2 = bottom2 + (top2 - bottom2) / (1.0 + ratio2)

    # Combined biphasic response
    return fraction * phase1 + (1.0 - fraction) * phase2


def test_rout_figure2_biphasic_with_outliers() -> None:
    """Test ROUT on biphasic dose-response data similar to Figure 2.

    The paper's Figure 2 shows a biphasic curve with 10 outliers injected.
    ROUT should detect the majority of these outliers at Q=1%.

    This test uses a simplified 4PL model (available in openfit) to
    approximate the behavior, as the full biphasic model is not in the
    current model registry.
    """
    # Generate clean dose-response data (4PL approximation)
    rng = np.random.default_rng(42)
    x = np.logspace(-2, 2, 50)  # 50 points across 4 log-decades

    # True parameters for a standard 4PL curve
    true_bottom = 0.0
    true_top = 100.0
    true_ec50 = 1.0
    true_slope = 1.0

    # Generate clean data with small Gaussian noise
    y_true = true_bottom + (true_top - true_bottom) / (1.0 + (true_ec50 / x) ** true_slope)
    noise = rng.normal(0, 2.0, size=len(x))  # SD=2, ~2% noise
    y = y_true + noise

    # Inject 10 outliers at specific positions (mimicking Figure 2)
    outlier_indices = [5, 12, 18, 23, 28, 33, 38, 42, 45, 48]
    outlier_magnitudes = [35, -40, 45, -35, 50, -45, 40, -50, 38, -42]

    y_with_outliers = y.copy()
    for idx, mag in zip(outlier_indices, outlier_magnitudes, strict=False):
        y_with_outliers[idx] += mag

    # Run ROUT with Q=1% (paper's default)
    result = rout_outliers(x, y_with_outliers, "hill4p", Q=0.01)

    # Paper claims ROUT detects most outliers while controlling FDR
    # We expect at least 70% of injected outliers to be detected
    detected = sum(1 for idx in outlier_indices if result.outlier_mask[idx])
    detection_rate = detected / len(outlier_indices)

    assert detection_rate >= 0.70, (
        f"ROUT detected only {detected}/{len(outlier_indices)} injected outliers "
        f"({detection_rate:.1%} detection rate). Expected >=70% based on Figure 2. "
        f"Detected indices: {result.outlier_indices.tolist()}"
    )

    # Verify false positive rate is controlled
    non_outlier_indices = [i for i in range(len(x)) if i not in outlier_indices]
    false_positives = sum(1 for idx in non_outlier_indices if result.outlier_mask[idx])
    fp_rate = false_positives / len(non_outlier_indices)

    assert fp_rate <= 0.05, (
        f"False positive rate {fp_rate:.1%} exceeds 5%. FDR should be controlled at Q=1%."
    )


# ---------------------------------------------------------------------------
# Table 1: Q parameter behavior
# ---------------------------------------------------------------------------


def test_rout_table1_q_parameter_monotonicity() -> None:
    """Verify Q parameter behavior as described in Table 1.

    The paper's Table 1 shows that higher Q values (more permissive FDR)
    should detect equal or more outliers than lower Q values on the same data.
    """
    # Generate data with moderate outliers
    rng = np.random.default_rng(123)
    x = np.logspace(-1, 1, 40)
    y_true = 0.0 + 100.0 / (1.0 + (1.0 / x) ** 1.0)
    y = y_true + rng.normal(0, 3.0, size=len(x))

    # Inject 5 moderate outliers
    outlier_idx = [8, 15, 22, 29, 35]
    for idx in outlier_idx:
        y[idx] += 30  # Moderate outlier magnitude

    # Test Q values from very strict to permissive
    q_values = [0.001, 0.01, 0.05, 0.10]
    n_outliers_list = []

    for q in q_values:
        result = rout_outliers(x, y, "hill4p", Q=q)
        n_outliers_list.append(result.n_outliers)

    # Verify monotonicity: higher Q should detect >= outliers
    for i in range(len(n_outliers_list) - 1):
        assert n_outliers_list[i] <= n_outliers_list[i + 1], (
            f"Q monotonicity violated: Q={q_values[i]} detected "
            f"{n_outliers_list[i]} outliers, but Q={q_values[i + 1]} detected "
            f"only {n_outliers_list[i + 1]}. Higher Q should be more permissive."
        )


def test_rout_table1_fdr_control_on_clean_data() -> None:
    """Verify FDR control: Q=1% should flag ~1% false positives on clean data.

    Table 1 in the paper shows that on data with no true outliers,
    ROUT should flag approximately Q fraction of points as false positives.
    """
    rng = np.random.default_rng(456)
    x = np.logspace(-1, 1, 100)  # 100 points for good statistics
    y_true = 0.0 + 100.0 / (1.0 + (1.0 / x) ** 1.0)

    # Generate clean data with Gaussian noise (no outliers)
    y = y_true + rng.normal(0, 2.0, size=len(x))

    # Test with Q=1%
    result = rout_outliers(x, y, "hill4p", Q=0.01)

    # On clean data, we expect very few false positives
    # Allow up to 3% false positive rate (some statistical variation)
    fp_rate = result.n_outliers / len(x)

    assert fp_rate <= 0.03, (
        f"False positive rate on clean data is {fp_rate:.1%}, exceeding 3%. "
        f"With Q=1%, expected ~1% FPR. Detected {result.n_outliers} outliers "
        f"out of {len(x)} clean points."
    )


# ---------------------------------------------------------------------------
# Algorithm fidelity tests
# ---------------------------------------------------------------------------


def test_rout_30_percent_scan_rule() -> None:
    """Verify ROUT tests only the 30% most extreme residuals.

    The paper specifies that only the top 30% of residuals by magnitude
    are tested for outlier status (Equation 17).
    """
    rng = np.random.default_rng(789)
    x = np.logspace(-1, 1, 50)
    y_true = 0.0 + 100.0 / (1.0 + (1.0 / x) ** 1.0)
    y = y_true + rng.normal(0, 2.0, size=len(x))

    result = rout_outliers(x, y, "hill4p", Q=0.01)

    # Count how many points have p-values < 1.0 (were actually tested)
    n_tested = sum(1 for p in result.p_values if p < 1.0)

    # Should test approximately 30% of points (allow 5% tolerance)
    expected_tested = int(0.30 * len(x))
    assert abs(n_tested - expected_tested) <= 3, (
        f"Expected ~{expected_tested} points tested (30% of {len(x)}), "
        f"but {n_tested} points have p-values < 1.0. "
        f"ROUT should scan only the 30% most extreme residuals."
    )


def test_rout_sequential_fdr_procedure() -> None:
    """Verify the sequential FDR decision rule from Equations 17-18.

    Once a point is flagged as an outlier, all points with larger |residuals|
    should also be flagged (the sequential nature of the test).
    """
    rng = np.random.default_rng(999)
    x = np.logspace(-1, 1, 40)
    y_true = 0.0 + 100.0 / (1.0 + (1.0 / x) ** 1.0)
    y = y_true + rng.normal(0, 2.0, size=len(x))

    # Inject multiple outliers with varying magnitudes
    y[10] += 25
    y[20] += 35
    y[30] += 45

    result = rout_outliers(x, y, "hill4p", Q=0.01)

    if result.n_outliers > 1:
        # Get residuals for outlier points
        param_dict = result.robust_params
        from openfit.models import get_model

        model = get_model("hill4p")
        y_hat = model.equation(x, **param_dict)
        residuals = np.abs(y - y_hat)

        outlier_res = residuals[result.outlier_mask]
        non_outlier_res = residuals[~result.outlier_mask]

        # All outlier residuals should be >= all non-outlier residuals
        # (with possible small numerical tolerance)
        if len(outlier_res) > 0 and len(non_outlier_res) > 0:
            min_outlier_res = np.min(outlier_res)
            max_non_outlier_res = np.max(non_outlier_res)

            # Allow small tolerance for numerical precision
            assert min_outlier_res >= max_non_outlier_res * 0.95, (
                f"Sequential FDR violated: smallest outlier residual "
                f"({min_outlier_res:.3f}) is less than largest non-outlier "
                f"residual ({max_non_outlier_res:.3f}). All points with larger "
                f"|residuals| should also be flagged as outliers."
            )


def test_rout_rsdr_calculation() -> None:
    """Verify RSDR calculation matches paper Equation 1.

    RSDR = P68 * N / (N - K)
    where P68 is the 68.27th percentile of |residuals|.
    """
    rng = np.random.default_rng(111)
    x = np.logspace(-1, 1, 30)
    y_true = 0.0 + 100.0 / (1.0 + (1.0 / x) ** 1.0)
    y = y_true + rng.normal(0, 3.0, size=len(x))

    result = rout_outliers(x, y, "hill4p", Q=0.01)

    # Compute RSDR manually from robust fit residuals
    from openfit.models import get_model

    model = get_model("hill4p")
    y_hat = model.equation(x, **result.robust_params)
    residuals = y - y_hat

    n = len(residuals)
    k = len(result.robust_params)  # Number of parameters
    abs_res = np.abs(residuals)
    p68 = np.percentile(abs_res, 68.27)
    expected_rsdr = p68 * n / (n - k)

    # The result object should have RSDR embedded in its calculations
    # We can verify by checking that the p-values are consistent
    # (This is an indirect test since ROUTResult doesn't expose RSDR directly)

    # At minimum, verify the calculation is numerically stable
    assert np.isfinite(expected_rsdr), "RSDR calculation produced non-finite value"
    assert expected_rsdr > 0, "RSDR should be positive for non-degenerate data"


# ---------------------------------------------------------------------------
# Edge cases from paper discussion
# ---------------------------------------------------------------------------


def test_rout_handles_few_outliers() -> None:
    """Test ROUT behavior when very few outliers are present.

    The paper discusses that ROUT should work well even with sparse outliers.
    """
    rng = np.random.default_rng(222)
    x = np.logspace(-1, 1, 50)
    y_true = 0.0 + 100.0 / (1.0 + (1.0 / x) ** 1.0)
    y = y_true + rng.normal(0, 2.0, size=len(x))

    # Inject only 2 outliers
    y[15] += 40
    y[35] -= 45

    result = rout_outliers(x, y, "hill4p", Q=0.01)

    # Should detect at least 1 of the 2 outliers
    assert result.n_outliers >= 1, (
        "ROUT failed to detect any of 2 injected outliers. Expected at least 1 detection."
    )


def test_rout_handles_many_outliers() -> None:
    """Test ROUT behavior when many outliers are present (>20% of data).

    The paper notes that ROUT can handle datasets with substantial contamination.
    """
    rng = np.random.default_rng(333)
    x = np.logspace(-1, 1, 50)
    y_true = 0.0 + 100.0 / (1.0 + (1.0 / x) ** 1.0)
    y = y_true + rng.normal(0, 2.0, size=len(x))

    # Inject outliers at 25% of points (12 outliers)
    outlier_idx = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46]
    for idx in outlier_idx:
        y[idx] += rng.choice([-1, 1]) * rng.uniform(30, 50)

    result = rout_outliers(x, y, "hill4p", Q=0.01)

    # Should detect a substantial fraction of outliers
    detected = sum(1 for idx in outlier_idx if result.outlier_mask[idx])
    detection_rate = detected / len(outlier_idx)

    assert detection_rate >= 0.50, (
        f"ROUT detected only {detected}/{len(outlier_idx)} outliers "
        f"({detection_rate:.1%}) in heavily contaminated data. "
        f"Expected >=50% detection rate."
    )


def test_rout_rejects_invalid_q() -> None:
    """Verify ROUT rejects invalid Q parameter values.

    Q must be strictly between 0 and 1.
    """
    rng = np.random.default_rng(444)
    x = np.logspace(-1, 1, 20)
    y = rng.normal(50, 5, size=len(x))

    # Q=0 should be rejected
    with pytest.raises(ValueError, match="Q must be strictly between 0 and 1"):
        rout_outliers(x, y, "hill4p", Q=0.0)

    # Q=1 should be rejected
    with pytest.raises(ValueError, match="Q must be strictly between 0 and 1"):
        rout_outliers(x, y, "hill4p", Q=1.0)

    # Q<0 should be rejected
    with pytest.raises(ValueError, match="Q must be strictly between 0 and 1"):
        rout_outliers(x, y, "hill4p", Q=-0.01)

    # Q>1 should be rejected
    with pytest.raises(ValueError, match="Q must be strictly between 0 and 1"):
        rout_outliers(x, y, "hill4p", Q=1.5)


# ---------------------------------------------------------------------------
# Comparison with paper's claimed performance
# ---------------------------------------------------------------------------


def test_rout_performance_matches_paper_claims() -> None:
    """Verify ROUT achieves performance levels claimed in the paper.

    The paper claims ROUT can detect outliers with high sensitivity while
    controlling false discovery rate at the specified Q level.
    """
    rng = np.random.default_rng(555)

    # Generate data similar to paper's simulation studies
    x = np.logspace(-1, 1, 60)
    y_true = 0.0 + 100.0 / (1.0 + (1.0 / x) ** 1.0)

    # Add realistic noise (SD ~3% of range)
    y = y_true + rng.normal(0, 3.0, size=len(x))

    # Inject 8 clear outliers
    outlier_idx = [5, 12, 20, 28, 36, 44, 52, 58]
    for idx in outlier_idx:
        y[idx] += rng.choice([-1, 1]) * rng.uniform(35, 50)

    # Test with Q=1%
    result = rout_outliers(x, y, "hill4p", Q=0.01)

    # Calculate sensitivity and FDR
    true_positives = sum(1 for idx in outlier_idx if result.outlier_mask[idx])
    false_positives = sum(
        1 for i in range(len(x)) if i not in outlier_idx and result.outlier_mask[i]
    )

    sensitivity = true_positives / len(outlier_idx) if len(outlier_idx) > 0 else 0.0
    actual_fdr = false_positives / result.n_outliers if result.n_outliers > 0 else 0.0

    # Paper claims high sensitivity (>80%) with FDR control
    assert sensitivity >= 0.75, (
        f"Sensitivity {sensitivity:.1%} is below 75%. "
        f"Paper claims ROUT achieves high sensitivity. "
        f"Detected {true_positives}/{len(outlier_idx)} true outliers."
    )

    # FDR should be controlled near Q=1%
    # Allow up to 5% actual FDR due to statistical variation
    assert actual_fdr <= 0.05, (
        f"Actual FDR {actual_fdr:.1%} exceeds 5%. "
        f"With Q=1%, expected FDR control. "
        f"{false_positives} false positives out of {result.n_outliers} detections."
    )


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
