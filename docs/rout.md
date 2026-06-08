# ROUT Outlier Detection

## What Is ROUT?

ROUT (Robust nonlinear regression followed by Outlier Detection) is a statistically principled method for identifying outliers in curve-fitting data. It was introduced by Motulsky & Brown in:

> Motulsky HJ, Brown RE. Detecting outliers when fitting data with nonlinear regression — a new method based on robust nonlinear regression and the false discovery rate. *BMC Bioinformatics* 2006, **7**:123.

openfit provides the first Python implementation of this algorithm.

The key advantage of ROUT over simple residual cutoffs (e.g. "flag points more than 3 SD from the fit") is that it controls the **false discovery rate (FDR)**: you choose the maximum acceptable fraction of non-outlier points that will be incorrectly flagged, and the method enforces that bound statistically.

---

## The Algorithm

ROUT proceeds in four steps:

**Step 1 — Robust regression.** Fit the model using a Lorentzian (Cauchy) loss function instead of ordinary least squares. The Lorentzian loss down-weights points with large residuals, so genuine outliers have little influence on the fitted curve. This is critical: ordinary least squares is distorted by the very points you are trying to detect.

**Step 2 — Compute RSDR.** Calculate the Robust Standard Deviation of Residuals (RSDR) from the robust fit. The RSDR is a resistant measure of scatter that is not inflated by outliers, analogous to the standard deviation of residuals from an ordinary fit but computed in a way that ignores extreme values.

**Step 3 — Flag candidates.** For each data point, compute the absolute residual from the robust fit and compare it to a threshold derived from Q (the user-specified FDR). Points whose residuals exceed this threshold are candidate outliers.

**Step 4 — Benjamini-Hochberg FDR correction.** Apply the Benjamini-Hochberg procedure to the set of candidate flags to control the false discovery rate at level Q across all points simultaneously. Only points that survive this correction are reported as outliers.

---

## The `rout_outliers` API

```python
from openfit import rout_outliers

result = rout_outliers("hill4p", x, y, weights="1/y2", Q=0.01)
print(result.n_outliers)       # number of flagged points
print(result.outlier_mask)     # boolean array, True = outlier
print(result.outlier_indices)  # integer indices of flagged points

# Refit without outliers
x_clean = x[~result.outlier_mask]
y_clean = y[~result.outlier_mask]
clean_fit = Fit("hill4p", x_clean, y_clean, weights="1/y2").run()
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `model` | str | Model name, e.g. `"hill4p"`. Same names as `Fit()`. |
| `x` | array-like | Independent variable values. |
| `y` | array-like | Dependent variable values. |
| `weights` | str | Weight scheme string. Should match what you intend to use for the final fit. |
| `Q` | float | False discovery rate (FDR) threshold. Default `0.01`. Range: `(0, 1)`. |

### Result Attributes

| Attribute | Type | Description |
|---|---|---|
| `n_outliers` | int | Number of points flagged as outliers. |
| `outlier_mask` | ndarray (bool) | Boolean array of length `len(x)`. `True` at index `i` means point `i` is an outlier. |
| `outlier_indices` | ndarray (int) | Integer indices of flagged points. Equivalent to `np.where(result.outlier_mask)[0]`. |
| `rsdr` | float | The Robust Standard Deviation of Residuals from the robust fit. |
| `robust_params` | dict | Parameter estimates from the intermediate robust fit (for diagnostic use). |

---

## What Q Means

Q is the maximum acceptable **false discovery rate**: the fraction of flagged points that are expected to be genuine non-outliers incorrectly flagged.

- `Q=0.01` (1%): The recommended default. Strict. Among all flagged points, at most 1% are expected to be false positives.
- `Q=0.05` (5%): More permissive. Standard significance level analogy.
- `Q=0.10` (10%): Aggressive. Flags more points but accepts more false positives.
- Lower Q means stricter flagging (fewer points flagged, fewer false positives).
- Higher Q means more permissive flagging (more points flagged, more false positives possible).

Setting Q very low (e.g. `Q=0.001`) near the floor of numerical precision is not recommended; the method is not designed to operate at extremely conservative FDR levels with small datasets.

---

## Practical Guidance

**Always inspect flagged points visually before removing them.** ROUT flags statistically unusual points — points whose residuals are inconsistent with the fitted model and the observed scatter. That is a mathematical criterion, not a scientific one. A flagged point might represent:

- A genuine measurement error (pipetting error, instrument glitch, contaminated well) — appropriate to exclude.
- A real biological effect at that concentration (hook effect, cell toxicity, aggregation) — should not be excluded; the model may need to be changed.
- An outlier in the X dimension (concentration error) — requires experimental investigation.

Inspect each flagged point in the context of the experiment before deciding whether to remove it.

**ROUT outliers are shown highlighted in fit reports.** When you call `report_fit()` (or `result.report()` after a `Fit` run that included ROUT), flagged points are displayed in a distinct colour on the plot, with a table of their indices, coordinates, and residuals in the report body.

**Match the weight scheme.** The Q-threshold derivation assumes that the weight scheme used in `rout_outliers()` matches the weight scheme you will use for the final clean fit. Using `"1/y2"` weights for ROUT and `"uniform"` weights for the final fit may produce inconsistent results.

---

> **Warning:** Do not use ROUT as an automatic data-cleaning step in a pipeline without human review. Automated outlier removal without inspection can suppress real biological signals, inflate apparent goodness-of-fit metrics, and introduce reporting bias. ROUT is a tool to assist expert judgment, not to replace it.
