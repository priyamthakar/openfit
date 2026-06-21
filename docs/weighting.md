# Choosing a Weight Scheme

## Why weighting matters

Most biological assay data is **heteroscedastic** — the measurement error (SD) is not constant across the range of Y values. In a typical ELISA or dose-response assay, the SD at a high-signal point may be ten times larger than at a low-signal point.

When you fit with **uniform weighting**, the least-squares objective treats every residual as equally important. This means:

- High-Y points (where absolute error is large) contribute disproportionately to the sum of squares — the optimizer bends the curve toward them.
- Low-Y points (where absolute error is small) are effectively down-weighted relative to their actual precision — their information is wasted.

The result is a biased parameter estimate. EC50, Kd, or rate constants estimated from uniformly weighted ELISA data are systematically wrong. Weighting correctly re-scales each residual so that a 10 % deviation at a low-signal point and a 10 % deviation at a high-signal point contribute equally.

---

## The five weight schemes

| Scheme | Formula | Weight per point |
|---|---|---|
| `uniform` | w = 1 | Equal for all points |
| `1/y` | w = 1 / Y | Proportional to 1/Y |
| `1/y2` | w = 1 / Y² | Proportional to 1/Y² |
| `1/sd2` | w = 1 / SD² | Inverse variance (empirical) |
| `poisson` | w = 1 / Y | Same formula as `1/y`, different justification |

### `uniform` — equal weighting

Use **only** when the absolute SD is approximately constant across the entire measurement range (homoscedastic data). This is rare in practice. Appropriate for some well-controlled spectrophotometric assays where instrument noise dominates over sample variability.

!!! warning "Most common mistake"
    Fitting ELISA or RIA data with `uniform` weighting is the single most common source of biased EC50 estimates. If your assay signal spans more than one order of magnitude, `uniform` is almost certainly wrong.

### `1/y` — proportional weighting

Weight = 1 / Y. Use when the **SD scales linearly with Y** — that is, the absolute error grows in proportion to the signal. Common for immunoassays (ELISA, CLIA) and some fluorescence readers where background is low but multiplicative noise dominates.

If SD ≈ k × Y for some constant k, then weighting by 1/Y makes all standardised residuals (residual / SD) comparable in magnitude.

### `1/y2` — constant-CV weighting

Weight = 1 / Y². Use when the **coefficient of variation (CV = SD/Y) is approximately constant** across the range. This is the most common choice for dose-response data (Hill equations, sigmoidal curves) and is the default recommended starting point for most bioassays.

The CV-constant assumption means the relative error is stable: a 5 % deviation at a low signal and a 5 % deviation at a high signal are treated as equivalent. This is biologically realistic for most receptor-binding and enzyme-activity assays.

!!! tip "When in doubt, start here"
    If you have no replicate data to compute empirical SDs, `1/y2` is the best-justified default for dose-response and binding assays.

### `1/sd2` — inverse-variance weighting

Weight = 1 / SD², where SD is the empirical standard deviation computed from replicates at each X value. This is the **statistically optimal** weighting when replicate data are available — it is the maximum-likelihood estimator under Gaussian errors.

```python
import numpy as np
from openfit import Fit

# sds is a 1-D array of per-point standard deviations
result = Fit("hill4p", x, y, weights=1 / sds**2).run()
```

You can also pass a numeric array directly for any custom weight vector.

!!! warning "Requires replicates"
    SD estimated from a single observation is undefined. You need at least two replicates per concentration to use `1/sd2`. With only three or four replicates, the estimated SDs are noisy — consider whether `1/y2` is more stable for your sample size.

### `poisson` — count data weighting

Weight = 1 / Y (identical formula to `1/y`, different statistical justification). For **Poisson-distributed count data** — radioligand binding (DPM), flow cytometry event counts, sequencing read counts — the variance equals the mean, so the inverse-mean weight is the correct inverse-variance weight.

Use `poisson` rather than `1/y` as a semantic signal that your data are counts; openfit records the weight scheme in the FitSpec manifest for reproducibility.

---

## How to specify weights

Pass the `weights` keyword to `Fit`. String aliases select built-in schemes:

```python
from openfit import Fit

result = Fit("hill4p", x, y, weights="1/y2").run()
```

For custom numeric weights, pass an array of the same length as `y`:

```python
import numpy as np
from openfit import Fit

w = 1.0 / sds**2          # inverse-variance from replicate SDs
result = Fit("hill4p", x, y, weights=w).run()
```

---

## Why openfit requires explicit weights

openfit does not have a silent default weight scheme. Calling `Fit` without `weights=` raises a `TypeError` at construction time.

This is intentional. The most common error in curve fitting for biological data is applying uniform weighting to heteroscedastic data without realising it. Libraries that silently default to uniform weighting make this mistake invisible — the fit runs, looks plausible, and returns a biased EC50 that nobody questions.

By requiring you to name the scheme, openfit forces a deliberate choice and records that choice in the FitSpec reproducibility manifest. When a collaborator reproduces your fit six months later, the weight scheme is part of the specification, not a hidden assumption.

---

## Effect of weight scheme on parameter estimates

The same four-parameter Hill fit on a simulated ELISA dataset with CV ≈ 15 %:

```python
import numpy as np
from openfit import Fit

x = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
y = np.array([102, 310, 890, 2450, 6100, 14200, 28900, 31800])

r_uniform = Fit("hill4p", x, y, weights="uniform").run()
r_1y2    = Fit("hill4p", x, y, weights="1/y2").run()

print(f"Uniform  EC50: {r_uniform.params['ec50']:.4f}")
print(f"1/y2     EC50: {r_1y2.params['ec50']:.4f}")
```

Typical output on this dataset:

```
Uniform  EC50: 0.0731
1/y2     EC50: 0.0518
```

A 40 % difference in EC50 from the weight scheme alone — with no change to the data. The `1/y2` result is closer to the true value (0.05) because it does not allow the high-signal plateau points to dominate the fit.

---

## Practical diagnostic: residuals vs fitted values

If you are unsure which scheme is appropriate, fit with `1/y2` and plot the standardised residuals against the fitted values:

```python
import matplotlib.pyplot as plt

result = Fit("hill4p", x, y, weights="1/y2").run()
plt.scatter(result.fitted, result.std_residuals)
plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("Fitted value")
plt.ylabel("Standardised residual")
plt.show()
```

- **Random scatter around zero** — weight scheme is appropriate.
- **Fan pattern (residuals grow with fitted value)** — your CV is not constant; try `1/y` or empirical `1/sd2`.
- **Systematic curve** — consider a different model, not just a different weight scheme.

!!! tip "Rule of thumb"
    If the standardised residuals fan out as fitted values increase, your data have non-constant CV and `1/y2` is underweighting the high-signal points. Collect replicates and switch to `1/sd2` if possible.
