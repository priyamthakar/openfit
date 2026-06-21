# Global (Shared-Parameter) Fitting

## What Is Global Fitting?

Global fitting means fitting multiple datasets simultaneously, with some parameters constrained to be equal across all datasets (shared) and others allowed to vary independently per dataset (local).

The canonical use case is a dose-response experiment where you measure multiple compounds in the same assay. The assay top and bottom signals are properties of the assay plate, not of the compounds, so it is reasonable to require `Top` and `Bottom` to be identical across all curves. Each compound has its own potency, so `EC50` is local. By fitting all curves at once under this constraint, you reduce the uncertainty on the shared parameters compared to estimating them separately from each curve.

### When to Use Global Fitting

- Multiple datasets share a physically meaningful common property (same assay background, same receptor Bmax, same protein stability baseline).
- Individual datasets are too noisy to pin down the shared parameter reliably on their own.
- You want to test formally whether a parameter really is consistent across conditions.

### When Not to Use Global Fitting

Do not use global fitting when datasets were collected under different assay conditions — different plates, different labs, different instruments. Systematic offsets between datasets will corrupt the shared parameters, and the F-test will falsely indicate that sharing is unjustified even when the biology is the same. Resolve inter-batch offsets by normalisation before global fitting, or by treating the offset itself as a local parameter.

---

## The `GlobalFit` API

```python
from openfit import GlobalFit

# Three dose-response curves: shared Top and Bottom, independent EC50 per curve
gf = GlobalFit(
    datasets=[(x1, y1, "1/y2"), (x2, y2, "1/y2"), (x3, y3, "1/y2")],
    model="hill4p",
    shared=["Top", "Bottom"],
    local=["EC50", "HillSlope"],
)
result = gf.run()

print(result.shared_params)     # {"Top": 100.0, "Bottom": 0.5}
print(result.local_params)      # [{"EC50": 1.2, ...}, {"EC50": 4.7, ...}, ...]
print(result.f_test.p_value)    # is sharing statistically justified?
result.report("global_fit.html")
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `datasets` | list of tuples | Each tuple is `(x, y, weight_scheme)`. Every dataset must be compatible with the chosen model. |
| `model` | str | Model name as used in `Fit()`, e.g. `"hill4p"`. |
| `shared` | list of str | Parameter names that are constrained to be equal across all datasets. |
| `local` | list of str | Parameter names that are estimated independently for each dataset. |

Every parameter in the model must appear in exactly one of `shared` or `local`. `GlobalFit` will raise `ValueError` if any parameter is missing or duplicated.

### The Dataset Tuple and Weights

Each dataset tuple includes its own weight scheme string. This allows different datasets to use different weighting strategies if the measurement error structure differs between them, while still sharing biological parameters. Weight scheme strings follow the same syntax as `Fit()` — see the main fitting guide for the full list (`"1/y2"`, `"1/y"`, `"uniform"`, etc.).

---

## How the Joint Objective Works

Internally, `GlobalFit` constructs a single objective function that sums the weighted squared residuals across all datasets:

```
SSQ_joint = sum over datasets i of: sum over points j of: w_ij * (y_ij - f(x_ij; shared, local_i))^2
```

Shared parameters appear once in this expression and are optimised globally. Local parameters for dataset `i` only affect the residuals for dataset `i`. The optimizer (Levenberg-Marquardt or Trust Region Reflective, depending on bounds) minimises `SSQ_joint` over all shared and all local parameters simultaneously.

This joint optimisation is what distinguishes global fitting from fitting each curve separately and then averaging the shared parameters — the latter ignores the cross-dataset constraint during optimisation and gives wider confidence intervals.

---

## The F-Test

`result.f_test` contains the outcome of a formal hypothesis test comparing:

- **Null model (shared):** The joint fit with the shared parameters constrained to be equal.
- **Alternative model (independent):** Each dataset fitted separately with all parameters free.

The F-statistic is:

```
F = ((SSQ_shared - SSQ_independent) / df_numerator) / (SSQ_independent / df_denominator)
```

where the degrees of freedom account for the number of extra parameters freed when you allow each dataset its own copy of the shared parameters.

| `f_test` attribute | Description |
|---|---|
| `f_statistic` | The computed F value. |
| `df_numerator` | Degrees of freedom in the numerator (number of shared parameters × (n_datasets - 1)). |
| `df_denominator` | Residual degrees of freedom of the independent fit. |
| `p_value` | Two-tailed p-value from the F distribution. |

A **p-value below 0.05** means the data are inconsistent with the shared constraint: the independent fit is significantly better, and you should consider releasing that parameter to local status. A p-value above 0.05 means the sharing constraint is statistically defensible.

Reference: Motulsky & Christopoulos 2003, *Fitting Models to Biological Data*, Chapter 25.

---

## Practical Workflow

A reliable strategy is to start permissive and constrain incrementally:

1. Fit all datasets independently (all parameters local). Inspect the per-dataset parameter estimates. Parameters whose estimates are numerically similar across datasets are candidates for sharing.
2. Move one candidate parameter to `shared` and re-run `GlobalFit`. Check `result.f_test.p_value`.
3. If the F-test p-value is above 0.05, the constraint is supported. Move to the next candidate.
4. If the F-test p-value is below 0.05, the shared constraint is inconsistent with the data. Investigate whether there is a systematic difference between datasets before proceeding.
5. Repeat until you have constrained all parameters that the data justify sharing.

This stepwise approach prevents overly aggressive sharing from biasing your local parameter estimates.

---

## Result Object

`gf.run()` returns a `GlobalFitResult` with the following attributes:

| Attribute | Description |
|---|---|
| `shared_params` | `dict` of shared parameter name to fitted value. |
| `local_params` | List of `dict`, one per dataset, containing local parameter names and fitted values. |
| `shared_ci` | `dict` of shared parameter name to `(lower, upper)` 95% confidence interval. |
| `local_ci` | List of `dict`, one per dataset, containing local parameter CIs. |
| `f_test` | `FTestResult` object (see above). |
| `aic` | Akaike Information Criterion for the joint fit. |
| `bic` | Bayesian Information Criterion for the joint fit. |

`result.report("global_fit.html")` generates a self-contained HTML report with per-dataset overlaid plots, shared parameter table, local parameter table, and the F-test summary.
