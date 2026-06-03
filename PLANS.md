# PLANS.md -- openfit future work

This document tracks what comes next, in priority order, informed by the
"validation before features" principle in ROADMAP.md.

The short version: the build ran ahead of the roadmap. The v0.2-v0.5 features
(compare, diagnostics, profile CI, bootstrap, global fit, ROUT) are already
implemented. The next work is validation catch-up, not new features.

---

## Immediate: v0.1.1 completion (DONE)

**Status:** Complete. NIST parameter recovery done, synthetic 4PL/5PL
datasets done, published-reference validation done, engine fix done,
reports extended. 264 tests passing, 1 skip.

### 1. Synthetic 4PL/5PL certified datasets [DONE]

No NIST-equivalent exists for logistic models (confirmed by the NIST StRD and
the openassayflow validation research). We generated our own ground truth.

Done:
- Generated synthetic 4PL datasets with known exact parameters (Bottom=0,
  Top=100, EC50=1.0, HillSlope=1.0) plus controlled noise levels (0%, 1%, 5%)
- Generated 5PL datasets similarly (add HillAsym=0.5)
- Stored as `tests/validation/fourpl_certified_values.py`
- Created `tests/validation/test_fourpl_synth.py` (16 tests)

### 2. R drda cross-validation [MOVED TO v0.1.2]

Cross-validate openfit Hill4P/Hill5P on a shared dataset against R's `drda`
package (Marasini et al., J Stat Softw 2023, DOI: 10.18637/jss.v106.i04).

Plan:
- Use the `voropm2` dataset from R `drda` (publicly documented)
- Run both R drda and openfit on identical data
- Assert parameter agreement within 6 sig figs
- Document the R command and openfit equivalent side-by-side in the test file
- Do NOT copy R source code; study behavior and reference outputs only

Deliverable: `tests/validation/test_drda_crossvalidation.py`

---

## Near-term: v0.1.2 -- PyPI publish

1. **Trusted Publishing via GitHub Actions**
   - Create `.github/workflows/publish.yml`
   - Trigger on version tags (v0.x.y)
   - Use PyPI Trusted Publishing (no API key required)
   - Build and upload wheel + sdist

2. **README validation badge**
   - CI matrix showing NIST dataset pass/fail (GitHub Actions badge)
   - "200 tests / 27 NIST datasets" claim in README

3. **Version bump to 0.1.2**
   - Update `pyproject.toml`
   - Add CHANGELOG entry
   - Tag: `git tag v0.1.2`

---

## Validation catch-up: published reference checks for already-built modules

The unit tests prove correctness in the sense of "runs + basic math."
The ROADMAP.md ties each feature to a specific published example that must match.
These cross-checks are not yet written:

### compare.py -- Motulsky & Christopoulos (2003) tables

The F-test and AICc comparison logic is implemented. Validate it against
"Fitting Models to Biological Data Using Linear and Nonlinear Regression"
(GraphPad Software, 2003). Key table to target: F-test examples in Ch. 15-16
(comparing one-site vs. two-site binding models).

File: `tests/test_compare_reference.py`
Citation: Motulsky & Christopoulos 2003, specific table/equation numbers in test.

### uncertainty.py -- Profile CI asymmetric intervals

Profile-likelihood CI is implemented. Validate against M&C Table 22.1 (known
asymmetric CI example for a 4PL parameter).

File: `tests/test_uncertainty_reference.py`
Acceptance: the profile CI on the textbook dataset matches the published bounds
within tolerance.

### outliers.py -- ROUT paper Figure 2 / Table 1

ROUT is implemented using Motulsky & Brown 2006 (BMC Bioinformatics 7:123).
Validate against Figure 2 (the 10-outlier synthetic dataset) and Table 1
(Q vs detection rate). The paper is open access at PMC.

File: `tests/test_rout_reference.py`
Note: requires digitizing the paper data -- the paper contains exact numerical
examples in the appendix.

### global_fit.py -- M&C Chapter 25 examples

Global fitting is implemented. Validate against the shared-parameter fitting
examples in Motulsky & Christopoulos Ch. 25. These are the dose-response
examples with shared Top/Bottom across multiple cell lines.

File: `tests/test_global_fit_reference.py`

---

## Engine improvement: silent argument dropping

**Issue surfaced during NIST testing:** When `Fit()` is called with
`method='lm'` (the default when no bounds), the `x_scale` and `diff_method`
parameters are silently accepted but never passed to scipy. This violates the
ROADMAP's "explicit over implicit" principle.

**Options:**
1. Warn when `x_scale` or `diff_method` is provided with `method='lm'`
2. Auto-switch to TRF when `x_scale='jac'` is requested
3. Accept that the validation test explicitly passes `method='trf'` -- document
   the limitation, do nothing to the engine

**Recommendation:** Option 1 (warn). The user asked for x_scale; silently
ignoring it is a footgun. A `UserWarning` on construction costs nothing.

File to change: `src/openfit/fit.py` (lines 258-260)

---

## Hahn1 and BoxBOD: test-only workarounds

The NIST validation tests for Hahn1 and BoxBOD use workarounds that live only
in the test file, not in the core engine:

- **Hahn1:** `_AnalyticJacModel` subclass providing an analytic Jacobian.
  The shipped `CustomModel` returns `None` (finite-difference only). The
  analytic Jacobian is needed because the cubic/cubic rational objective has a
  flat ridge that numerical Jacobians traverse incorrectly.

- **BoxBOD:** `bounds_dict={"b2": (0.0, 10.0)}` prevents the b2 ~ 88 spurious
  basin. A user fitting BoxBOD through the default CustomModel with no bounds
  would hit this issue from a bad starting point.

These are not bugs in the engine -- they are correct behaviors (an analytic
Jacobian is always better; physical bounds are a user responsibility). But the
README and docs should not imply the default engine handles these automatically.

Future work (optional, low priority): document in the CustomModel docstring
that analytic Jacobians significantly improve convergence for ill-conditioned
rational-polynomial models.

---

## Phase 2+ features (from ROADMAP.md)

These remain as-is in ROADMAP.md. No changes to the sequence. Key note:
**do not start Phase 6 (model library expansion) until the validation catch-up
above is complete.** Every existing model needs its published-reference test
before new models are added.

Phases at a glance:
- v0.2.x-v0.3.x: Documentation only (code already shipped ahead of schedule)
- v0.6.x: Full model library (~30 equations) -- blocked by validation catch-up
- v0.7.x: PDF + Word reports + Prism import
- v0.8.x+: Constraint fitting, batch fitting, stable v1.0.0

---

## openassayflow (downstream package)

`openassayflow` (D:/openassay/OPENASSAYFLOW_CLAUDE.md) is blocked until openfit
v0.1.1 ships the synthetic 4PL/5PL certified datasets (item 1 above). Those
datasets will be openassayflow's primary validation reference for its
StandardCurve module.

Do not start openassayflow until:
1. openfit v0.1.1 is tagged
2. 4PL/5PL synthetic certified values are committed to openfit

---

## What NOT to do next

- Do not build new models (Gaussian2, binding isotherm, etc.) before validation
  catch-up is complete.
- Do not start openassayflow before openfit v0.1.1 is tagged.
- Do not implement Bayesian fitting (out of scope per CLAUDE.md).
- Do not add a GUI (out of scope per ROADMAP.md).
