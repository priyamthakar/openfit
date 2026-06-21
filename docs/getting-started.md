# Getting Started

This guide walks you through installing openfit, running your first fit, understanding FitSpec, and using the CLI.

---

## Installation

openfit requires Python 3.10 or later.

### Core library

```bash
pip install openfit
```

The core package includes the fitting engine, all 29 built-in models, profile-likelihood confidence intervals, ROUT outlier detection, and global fitting.

### Optional extras

```bash
# Adds the `openfit` command-line tool
pip install "openfit[cli]"

# Adds HTML, PDF, and DOCX report generation
pip install "openfit[reports]"
```

You can install both at once:

```bash
pip install "openfit[cli,reports]"
```

---

## Your First Fit

### 1. Prepare your data

openfit works with NumPy arrays. Load your x and y data however you like — from a CSV, a DataFrame column, or directly in Python:

```python
import numpy as np

x = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
y = np.array([2.1, 4.5, 11.3, 28.7, 51.2, 73.4, 88.6, 95.1])
```

### 2. Run the fit

Pass the model name, x, y, and a weights expression to `Fit`:

```python
from openfit import Fit

result = Fit("hill4p", x, y, weights="1/y2")
```

!!! note "Weights are always required"
    openfit does not fall back to ordinary least squares. The `weights` argument is mandatory.
    `"1/y2"` means each point is weighted by `1/y²`, which is appropriate when measurement
    error scales with the response (relative/proportional weighting). See [Choosing Weights](weighting.md).

### 3. Inspect the results

```python
result.summary()
```

This prints a table of fitted parameters, standard errors, and profile-likelihood confidence intervals:

```
Model   : hill4p
Engine  : LM (Levenberg-Marquardt)
R²      : 0.9994

Parameter  Estimate    SE       CI 95% (lower)  CI 95% (upper)
---------  ----------  -------  --------------  --------------
Bottom     1.843       0.412    0.962           2.724
Top        96.21       1.03     94.10           98.32
EC50       1.127       0.041    1.046           1.212
HillSlope  1.812       0.089    1.631           1.998
```

### 4. Generate a report

```python
result.report("fit.html")
```

This produces a self-contained HTML file with the fit plot, residuals panel, and parameter table. To generate PDF or DOCX instead:

```python
result.report("fit.pdf")
result.report("fit.docx")
```

!!! tip
    Report generation requires `pip install "openfit[reports]"`.

### 5. Save the FitSpec

```python
result.spec.save("fit_spec.json")
```

---

## What Is FitSpec?

FitSpec is openfit's reproducibility manifest — a JSON record that captures everything needed to reproduce a fit exactly:

- A SHA-256 hash of your input data
- The model name and parameter initial values
- The weights expression or array used
- The solver (LM or TRF) and its tolerances
- The versions of openfit, scipy, and numpy used

```json
{
  "openfit_version": "1.0.0",
  "model": "hill4p",
  "weights": "1/y2",
  "solver": "lm",
  "data_hash": "sha256:a3f1c...",
  "scipy_version": "1.13.0",
  "numpy_version": "1.26.4"
}
```

To re-run a fit from a saved spec:

```python
from openfit import Fit

result = Fit.from_spec("fit_spec.json", x, y)
result.summary()
```

FitSpec makes it straightforward to include a link to your manifest in a paper's supplementary materials so reviewers and readers can reproduce your curve fits from raw data.

See [Reproducibility](reproducibility.md) for the full FitSpec guide.

---

## Using the CLI

If you installed the `[cli]` extra, the `openfit` command is available in your shell.

### Fit a CSV file

```bash
openfit fit data.csv --model hill4p --weights 1/y2 --report fit.html
```

The CSV must have at minimum an `x` column and a `y` column. Additional columns (e.g. `sd` for standard-deviation weights) are supported.

### Save a spec

```bash
openfit fit data.csv --model hill4p --weights 1/y2 --save-spec fit_spec.json
```

### Re-run from a spec

```bash
openfit rerun fit_spec.json data.csv --report fit2.html
```

### List available models

```bash
openfit models
```

---

## Next Steps

- [Choosing Weights](weighting.md) — understand proportional, absolute, and custom weighting
- [Confidence Intervals](confidence-intervals.md) — profile-likelihood vs. asymptotic SE
- [Model Comparison](model-comparison.md) — AIC, BIC, and F-test between nested models
- [ROUT Outlier Detection](rout.md) — identify and exclude outliers before fitting
- [Global Fitting](global-fit.md) — share parameters across multiple datasets
