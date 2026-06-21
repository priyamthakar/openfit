# openfit

**Reproducible, open-source nonlinear curve fitting. Every fit you can rerun, every result you can cite.**

openfit is a Python library for nonlinear curve fitting built on scipy, designed for scientists who need results that are fully reproducible, citable, and publication-ready. With 29 built-in models, profile-likelihood confidence intervals, ROUT outlier detection, and export to HTML, PDF, and DOCX, openfit covers the full fitting workflow from raw data to polished report.

---

## Install

```bash
# Core library
pip install openfit

# With CLI support
pip install "openfit[cli]"

# With report generation (HTML, PDF, DOCX)
pip install "openfit[reports]"
```

---

## Quick Example

```python
import numpy as np
from openfit import Fit

# Example dose-response data
x = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
y = np.array([2.1, 4.5, 11.3, 28.7, 51.2, 73.4, 88.6, 95.1])

# Fit a 4-parameter Hill / sigmoidal model
# Explicit weights are required — here: 1/y^2 (relative/proportional weighting)
result = Fit("hill4p", x, y, weights="1/y2")

# Print a summary table of parameters, SE, and confidence intervals
result.summary()

# Export a self-contained HTML report
result.report("fit.html")

# Save a FitSpec reproducibility manifest to JSON
result.spec.save("fit_spec.json")
```

!!! note "Weights are required"
    openfit does not assume unweighted (OLS) fitting. You must supply a `weights` argument.
    See [Choosing Weights](weighting.md) for guidance.

---

## Features

| Feature | Details |
|---|---|
| 29 built-in models | Sigmoidal, exponential, polynomial, power law, Michaelis-Menten, Gaussian, and more |
| Explicit weighting | Proportional (`1/y2`), absolute (`1/SD2`), uniform, or custom arrays |
| Profile-likelihood CI | Asymmetric, statistically rigorous confidence intervals |
| ROUT outlier detection | First Python implementation of the ROUT method (Motulsky & Brown 2006) |
| Global fitting | Shared parameters across multiple datasets |
| FitSpec manifest | JSON reproducibility record: data hash, model, weights, solver settings, versions |
| NIST StRD validated | All 27 NIST Statistical Reference Datasets pass to certified precision |
| Publication reports | HTML, PDF, and DOCX output with parameter tables, fit plots, and residuals |

---

## Next Steps

Ready to fit your first dataset? Head to [Getting Started](getting-started.md).
