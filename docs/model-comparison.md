# Comparing Models

## When to compare models

Model comparison answers the question: does adding a parameter improve the fit enough to justify its cost in complexity?

Common comparisons in dose-response and binding analysis:

- **Hill3P vs Hill4P** — is the bottom of the curve fixed at zero, or does it float?
- **Hill4P vs Hill5P** — does the curve have a symmetric Hill slope, or does an asymmetry parameter improve it?
- **MonoExp vs BiExp** — is one exponential component sufficient, or are two needed?
- **Straight line vs quadratic** — is the curvature real or noise?

openfit's `compare_models()` function computes AICc, Akaike weights, BIC, and (where valid) an F-test for a list of fitted results, and returns a ranked summary.

---

## The `compare_models()` API

```python
from openfit import Fit, compare_models

r3 = Fit("hill3p", x, y, weights="1/y2").run()
r4 = Fit("hill4p", x, y, weights="1/y2").run()
r5 = Fit("hill5p", x, y, weights="1/y2").run()

comp = compare_models([r3, r4, r5])
print(comp.summary())
```

`comp.summary()` returns a formatted table:

```
Model     k   AICc    ΔAICc   Weight   BIC     F vs prev   p
hill3p    3   -84.1   4.3     0.10     -81.6   —           —
hill4p    4   -88.4   0.0     0.81     -84.9   8.21        0.012
hill5p    5   -87.0   1.4     0.09     -82.3   1.44        0.261
```

Individual attributes:

```python
comp.table          # pandas DataFrame
comp.best           # FitResult with lowest AICc
comp.weights        # dict of {model_name: Akaike weight}
comp.evidence_ratio("hill4p", "hill3p")   # weight ratio
```

---

## Information criteria

### AICc — corrected Akaike Information Criterion

AICc penalises the log-likelihood for the number of free parameters k, with a finite-sample correction for n observations:

```
AICc = -2 × log L + 2k + (2k² + 2k) / (n - k - 1)
```

The correction term is important when n/k < 40 — which covers most dose-response experiments with 8–16 concentrations and 3–5 parameters. **The model with the lowest AICc is preferred.**

The correction approaches zero as n → ∞, so AICc converges to AIC for large datasets and is always at least as conservative.

!!! tip "Interpreting ΔAICc"
    - ΔAICc < 2: both models have substantial support; prefer the simpler one.
    - ΔAICc 2–7: moderate evidence for the lower-AICc model.
    - ΔAICc > 10: strong evidence; the higher-AICc model has essentially no support.

### Akaike weights

The Akaike weight for model i is the normalised relative likelihood:

```
w_i = exp(-ΔAICc_i / 2) / Σ exp(-ΔAICc_j / 2)
```

Weights sum to 1 across the comparison set and can be interpreted as the **probability that model i is the best model given the data and the set of candidates considered**.

The **evidence ratio** w_i / w_j quantifies how much more support model i has than model j:

```python
comp.evidence_ratio("hill4p", "hill3p")   # e.g., 8.1 — Hill4P is 8× more likely
```

### BIC — Bayesian Information Criterion

BIC replaces the 2k penalty in AIC with k × ln(n):

```
BIC = -2 × log L + k × ln(n)
```

BIC imposes a stronger penalty for parameters than AICc when n > 8 (which is almost always), so it tends to prefer simpler models. Use BIC as a secondary check: if AICc and BIC agree, the evidence is consistent. If they disagree, report both and prefer the simpler model unless there is strong mechanistic reason to include the extra parameter.

---

## F-test for nested models

The F-test (extra sum-of-squares test) is valid **only for nested model pairs** — that is, when the simpler model is a special case of the complex model obtained by fixing one or more parameters to a specific value.

Examples of valid nested pairs:

- Hill3P (Bottom fixed at 0) nested in Hill4P (Bottom free).
- Hill4P (symmetric slope, Asym = 1) nested in Hill5P (asymmetry free).
- MonoExp nested in BiExp (second amplitude fixed at 0).

openfit detects nestedness automatically by checking whether the parameter sets of the two models have the required subset relationship. For a valid nested pair:

```
F = [(SS_simple - SS_complex) / (df_complex - df_simple)] / [SS_complex / df_complex]
```

The p-value is the upper tail probability from the F distribution with (df_complex - df_simple, df_complex) degrees of freedom. A p-value below 0.05 indicates that the reduction in residual sum-of-squares from adding the parameter is larger than expected by chance.

For **non-nested pairs** (e.g., Hill4P vs MonoExp), `compare_models()` returns `nan` for the F statistic and reports AICc only. The F-test cannot be applied because the two models do not share the same constrained parameter space.

!!! warning "Do not F-test non-nested models"
    Applying the F-test to models that are not nested (same data, same weights, parameter subset relationship) produces a statistic with an undefined distribution. The p-value is meaningless. openfit will not compute an F-test for non-nested pairs and will display a warning if you attempt to force one.

---

## Practical guidance

### When F-test and AICc agree

If F-test p < 0.05 **and** ΔAICc > 2 in favour of the complex model, the extra parameter is well justified. Report the complex model.

If F-test p > 0.05 **and** ΔAICc < 2, both criteria prefer the simpler model. Report the simpler model.

### When F-test and AICc disagree

This can occur because the F-test is sensitive to the absolute reduction in SS while AICc weights the improvement against the cost of an extra parameter:

- F-test significant but ΔAICc < 2: the extra parameter reduces SS but the improvement is small relative to its complexity cost. **Prefer the simpler model** unless there is a mechanistic reason to include the parameter.
- F-test not significant but ΔAICc > 2: rare in practice; recheck whether the models are truly nested and whether n is large enough for the asymptotic F approximation to hold.

!!! tip "Practical rule"
    F-test p < 0.05 AND ΔAICc > 2 → extra parameter justified, use the complex model.
    Otherwise → prefer the simpler model. Biological interpretability should always be a tiebreaker.

---

## Comparing models fit with different weight schemes

This is **not valid**. The log-likelihood, AIC, and BIC values are only comparable across models when all models were fit to the same data with the same weight scheme. Mixing weight schemes changes the objective function, so a lower AICc under `1/y2` cannot be compared with a lower AICc under `uniform`.

!!! warning "Always use identical weights"
    Always pass the same `weights=` argument to every `Fit` call whose results you intend to compare with `compare_models()`. openfit will raise a `WeightMismatchError` if you pass results with differing weight schemes to `compare_models()`.

---

## Complete example

```python
import numpy as np
from openfit import Fit, compare_models

x = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
y = np.array([98, 315, 870, 2400, 6050, 14100, 28700, 31600, 32100])

r3 = Fit("hill3p", x, y, weights="1/y2").run()
r4 = Fit("hill4p", x, y, weights="1/y2").run()
r5 = Fit("hill5p", x, y, weights="1/y2").run()

comp = compare_models([r3, r4, r5])
print(comp.summary())

# Evidence ratio: how much more support does Hill4P have than Hill3P?
print(f"Evidence ratio Hill4P/Hill3P: {comp.evidence_ratio('hill4p', 'hill3p'):.1f}")

# Access the best model directly
best = comp.best
print(f"Best model: {best.model_name}, EC50 = {best.params['ec50']:.4f}")
```

---

## Reference

Motulsky H and Christopoulos A (2003). *Fitting Models to Biological Data using Linear and Nonlinear Regression: A Practical Guide to Curve Fitting.* GraphPad Software, San Diego. Chapter 22: Comparing models.
