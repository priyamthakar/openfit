# Confidence Intervals

openfit provides three methods for computing confidence intervals on fitted parameters. They differ in speed, assumptions, and reliability when the fit is poorly constrained or the likelihood surface is non-parabolic.

---

## Overview

| Method | Speed | Symmetric? | Assumptions | Best for |
|---|---|---|---|---|
| Asymptotic (Wald) | Fast | Yes | Parabolic likelihood surface | Well-constrained fits, large n |
| Profile-likelihood | Moderate | No | Correct likelihood shape | EC50 near data edge, few df |
| Bootstrap BCa | Slow | No | None beyond model structure | Small n, irregular residuals |

---

## Asymptotic (Wald) CI

The asymptotic CI is computed from the covariance matrix of the parameter estimates, which openfit derives from the Jacobian matrix evaluated at the optimum:

```
CI = param ± t(α/2, dof) × SE
```

where `SE` is the square root of the corresponding diagonal element of the covariance matrix, `t(α/2, dof)` is the two-tailed t-critical value at the desired confidence level, and `dof = n - p` (number of observations minus number of fitted parameters).

This method is exact when the likelihood surface is perfectly parabolic near the optimum (i.e., when the model is linear in its parameters near the solution). For nonlinear models, it is an approximation that degrades as:

- Sample size decreases.
- The parameter is constrained near a boundary.
- The likelihood surface is skewed or flat in one direction.

Asymptotic CIs are always available in `result.ci` after a successful fit:

```python
from openfit import Fit

result = Fit("hill4p", x, y, weights="1/y2").run()
print(result.ci)          # dict of {param: (lower, upper)}
print(result.se)          # dict of {param: SE}
```

!!! warning "Singular Jacobian"
    If the fit is poorly determined — too few data points, redundant parameters, or a flat likelihood surface — the Jacobian may be singular. In that case `result.se` will contain `inf` and `result.ci` will contain `nan` for the affected parameters. Switch to profile-likelihood CI, which does not require an invertible Jacobian.

---

## Profile-Likelihood CI

Profile-likelihood CI walks each parameter across a grid, re-optimising all other parameters at each step, until the profile log-likelihood drops by the critical threshold corresponding to the desired confidence level (χ² with 1 df for 95 % CI: Δ log L = 1.92).

The result is an **asymmetric interval** that correctly reflects the true shape of the likelihood surface. If the uncertainty in EC50 is larger on the high side (because the upper plateau is poorly determined), the profile-likelihood CI will show a wider upper arm.

This is the preferred method when:

- EC50, Kd, or another key parameter is near the edge of the data range.
- The fit has few degrees of freedom (fewer than ~5).
- The asymptotic CI for any parameter spans more than one order of magnitude.
- Parameters are correlated (off-diagonal covariance terms are large relative to variances).

Reference: Motulsky & Christopoulos (2003), *Fitting Models to Biological Data*, Table 22.1.

```python
from openfit import Fit
from openfit.uncertainty import profile_ci

result = Fit("hill4p", x, y, weights="1/y2").run()
pci = profile_ci(result)
print(pci)                # dict of {param: (lower, upper)}
```

!!! tip "Asymmetry as a diagnostic"
    If the profile-likelihood CI is highly asymmetric (one arm more than twice the width of the other), the asymptotic CI is unreliable for that parameter. Report the profile-likelihood interval in any publication.

---

## Bootstrap BCa CI

Bootstrap BCa (bias-corrected accelerated) CI uses residual resampling: residuals from the original fit are resampled with replacement, added back to the fitted values to generate synthetic datasets, and the model is re-fit to each. The BCa correction adjusts for bias in the bootstrap distribution and for the acceleration (change in SE with parameter value).

This is the most assumption-free method. It makes no claim about the shape of the likelihood surface and is robust to:

- Non-Gaussian residual distributions.
- Small sample sizes where the t-approximation is poor.
- Models where the parameter space has irregular geometry.

The tradeoff is computational cost: `n_boot=1000` requires fitting the model 1000 additional times.

```python
from openfit import Fit
from openfit.uncertainty import bootstrap_ci

result = Fit("hill4p", x, y, weights="1/y2").run()
bci = bootstrap_ci(result, n_boot=1000, random_seed=42)
print(bci)                # dict of {param: (lower, upper)}
```

!!! tip "Reproducibility"
    Always specify `random_seed`. openfit records the seed in the FitSpec reproducibility manifest so that collaborators can reproduce the exact bootstrap sample. Omitting the seed means the intervals will differ between runs.

!!! warning "Minimum sample size"
    BCa CI requires at least `n_boot` successful re-fits. If many bootstrap datasets produce failed fits (e.g., because the synthetic data do not span the transition region), the effective bootstrap sample shrinks and the CI becomes unreliable. Examine `bci.n_failed` and increase `n_boot` or switch to profile-likelihood if failures exceed 5 %.

---

## Complete example

```python
from openfit import Fit
from openfit.uncertainty import profile_ci, bootstrap_ci

result = Fit("hill4p", x, y, weights="1/y2").run()

# Asymptotic CI — already computed, no extra cost
print("Asymptotic CI:", result.ci)

# Profile-likelihood CI — preferred for publication
pci = profile_ci(result)
print("Profile-likelihood CI:", pci)

# Bootstrap BCa CI — most robust, slowest
bci = bootstrap_ci(result, n_boot=1000, random_seed=42)
print("Bootstrap BCa CI:", bci)
```

---

## Choosing a method

**Use asymptotic CI** when:

- You have more than ~10 data points per fitted parameter.
- The fit is well-constrained (all SEs are finite and small relative to the parameter values).
- You need a quick check during exploratory analysis.

**Use profile-likelihood CI** when:

- Any asymptotic SE is large or the CI spans more than an order of magnitude.
- EC50 or another key parameter is near the boundary of your concentration range.
- You are preparing results for publication.
- The Jacobian is near-singular (asymptotic CI returns `nan`).

**Use bootstrap BCa CI** when:

- You have reason to believe the residuals are non-Gaussian (e.g., after a Shapiro-Wilk test).
- Sample size is small (n < 10 total observations).
- You want a method-of-last-resort check against the profile-likelihood result.

!!! tip "Quick decision rule"
    For most dose-response work: compute asymptotic CI first. If any CI arm is more than 3× the parameter value, or if any SE is `inf`, switch to profile-likelihood. Reserve bootstrap for n < 10 or clearly non-Gaussian residuals.
