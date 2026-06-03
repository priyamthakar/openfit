# openfit

![CI](https://github.com/priyamthakar/openfit/actions/workflows/ci.yml/badge.svg)

Reproducible, open-source nonlinear curve fitting with publication-quality reports.
Every fit you can rerun, every result you can cite.

> **Status:** v0.1.2 (in progress). 368 tests collected (368 passed, 5 skipped;
> NIST StRD all 27 datasets, synthetic 4PL/5PL and binding-model certified,
> published-reference validation). PyPI publish planned for v0.1.2.

---

## What is openfit?

openfit is a Python library for nonlinear curve fitting that combines the convenience
of GraphPad Prism with open-source transparency and full reproducibility.

Every fit emits a `FitSpec` -- a JSON manifest containing the model, fitted parameters,
weight scheme, data hash, and exact software versions. Anyone with the spec and the
data can reproduce the identical result.

```python
import numpy as np
from openfit import Fit

x = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
y = np.array([2.0, 5.0, 20.0, 65.0, 90.0, 97.0, 99.0])

result = Fit("hill4p", x, y, weights="1/y2").run()
print(result.summary())
result.report("fit.html")
result.spec.to_json("fit_spec.json")  # reproducibility manifest
```

---

## Install

```bash
pip install openfit                   # core engine + models + reports
pip install "openfit[cli]"            # + CLI (typer, rich)
pip install "openfit[reports]"        # + PDF & DOCX reports (reportlab, jinja2, python-docx)
pip install "openfit[dev]"            # + development tools (pytest, ruff, mypy)
```

---

## Features

### Fitting engine
- `Fit(model, x, y, *, weights)` wraps `scipy.optimize.least_squares`
  (Levenberg-Marquardt and Trust Region Reflective)
- `weights=` is required -- no silent defaults (prevents wrong results on
  heteroscedastic data like ELISA plates)
- Weight schemes: `uniform`, `1/y`, `1/y2`, `1/sd2`, `poisson`
- Smart per-model initial guesses from data-driven heuristics
- Analytic Jacobian where available; finite-difference fallback

### Models (29 built-in)

| Family | Models |
|--------|--------|
| Sigmoidal | Hill3P, Hill4P (4PL), Hill5P (5PL), Boltzmann |
| Exponential | MonoExp, BiExp, ExpGrowth, ExpPlateau, ExpDecay |
| Enzyme | MichaelisMenten, SubstrateInhibition, Allosteric |
| Growth | Logistic3P, Logistic4P, Gompertz, AsymmetricGompertz, Richards |
| Gaussian | Gaussian, BiGaussian, Lorentzian |
| Polynomial | Poly1 through Poly6 |
| Binding | OneSiteBinding, TwoSiteBinding, CompetitiveBinding |
| Custom | CustomModel (any callable; param names inferred from signature) |

### Statistical output
- Fitted parameters + asymptotic standard errors
- Full parameter covariance matrix via `FitResult.covariance` in
  `model.param_names` order; singular Jacobians return a NaN-filled matrix
  with infinite standard errors and undefined (`nan`) asymptotic CIs
- Confidence intervals: asymptotic, profile-likelihood, bootstrap (BCa)
- R^2 (correct nonlinear definition), AICc, BIC, residuals (raw, weighted, standardized)
- `FitSpec` reproducibility manifest: data SHA-256 hash, version pins, random seed

### Model comparison
- `compare_models(results)` -- AICc, BIC, evidence ratios, Akaike weights
- F-test (extra sum-of-squares) for nested model pairs only
- Non-nested models: AICc only (F-test not valid)

### Residual diagnostics
- Wald-Wolfowitz runs test (tests for systematic patterns)
- Shapiro-Wilk / D'Agostino-Pearson normality test
- Replicates / lack-of-fit F-test
- 3-sigma outlier flags

### Global/shared-parameter fitting
```python
from openfit import GlobalFit

# Fit multiple datasets sharing Top and Bottom, independent EC50 per dataset
gf = GlobalFit(datasets=[(x1,y1), (x2,y2), (x3,y3)],
               model="hill4p",
               shared=["Top","Bottom"],
               local=["EC50","HillSlope"])
result = gf.run()
print(result.shared_params)    # {"Top": ..., "Bottom": ...}
print(result.local_params)     # [{"EC50": ..., "HillSlope": ...}, ...]
print(result.f_test.p_value)   # is sharing statistically justified?
```

### ROUT outlier detection
```python
from openfit import rout_outliers

rout = rout_outliers("hill4p", x, y, weights="1/y2", Q=0.01)
print(rout.n_outliers)         # number flagged
print(rout.outlier_mask)       # bool array
```

First Python implementation of the Motulsky & Brown (BMC Bioinformatics, 2006) algorithm.

### CLI
```bash
openfit version
openfit models
openfit fit data.csv --model hill4p --weights 1/y2 --report fit.html
openfit compare data.csv --models hill3,hill4,hill5 --report comparison.html
```

### Prism migration
```python
from openfit.io import load_pzfx

tables = load_pzfx("existing_analysis.pzfx")  # read-only, migration only
```

---

## Validation

openfit is validated against the NIST Statistical Reference Datasets (StRD)
for nonlinear regression. All 27 certified datasets are tested; parameter
recovery matches NIST certified values (128-bit precision) to at least
6 significant figures.

| Difficulty | Datasets | Tests | Status |
|------------|---------|-------|--------|
| Lower | 8 | 32 param + 8 RSS | All pass |
| Average | 11 | 44 param + 11 RSS | All pass |
| Higher | 8 | 32 param + 8 RSS | All pass |

Notable:
- Hahn1 (ill-conditioned cubic/cubic rational): passes with analytic Jacobian
- BoxBOD (far start basin): passes with physical bounds b2 in (0, 10]
- Lanczos1 RSS: skipped (certified RSS = 1.43e-25, below 64-bit floor); params pass

NIST data source: https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml (public domain)

---

## Reproducibility

```python
# Run a fit and save the reproducibility manifest
result = Fit("hill4p", x, y, weights="1/y2").run()
result.spec.to_json("my_fit.spec.json")

# Anyone can reproduce the exact result
from openfit import FitSpec
spec = FitSpec.from_json(open("my_fit.spec.json").read())
# spec.data_hash -- verify against your data
# spec.param_values -- the exact fitted values
# spec.openfit_version, spec.scipy_version, spec.numpy_version -- version pins
```

---

## Comparison

| Feature | openfit | scipy curve_fit | lmfit |
|---------|---------|-----------------|-------|
| 4PL/5PL built-in | Yes | No | No |
| Smart initial guesses | Yes (per model) | No | Partial |
| 1/Y, 1/Y^2 weighting | Yes (required) | Manual | Manual |
| Global/shared fitting | Yes (declarative) | No | Manual |
| Profile-likelihood CI | Yes | No | Yes |
| ROUT outlier removal | Yes | No | No |
| Reproducibility spec | Yes (unique) | No | No |
| NIST validation | Published | No | Not published |
| Publication report | HTML + Markdown + PDF + DOCX | No | No |

---

## Philosophy

openfit is domain-agnostic. It knows mathematics, not biology or pharmacology.
Domain-specific logic (ELISA back-calculation, IC50 interpretation, PK parameters)
belongs in downstream packages that import openfit.

See also: [`openassay`](https://github.com/priyamthakar/openassay) -- the
separate immunoassay and ligand-binding-assay workflow package for standard
curves, back-calculation, LLOQ/ULOQ, acceptance criteria, plate workflows, ADA
cut points, and regulatory-style reports. openassay imports openfit; openfit
never imports openassay.

---

## Development

```bash
git clone https://github.com/priyamthakar/openfit
cd openfit
pip install -e ".[dev]"
pytest                              # 339 tests (338 passed, 1 skipped)
pytest tests/validation/ -v        # NIST StRD suite
ruff check src/ tests/ --fix
mypy src/openfit
```

---

## License

MIT. Author: Priyam Thakar <priyamthakar1@gmail.com>

Reproducible, open-source nonlinear curve fitting with publication-quality reports.
Every fit you can rerun, every result you can cite.
