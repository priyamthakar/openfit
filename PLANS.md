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
reports extended. 343 tests passing (342 passed, 1 skipped as of 2026-06-03).

### 1. Synthetic 4PL/5PL certified datasets [DONE]

No NIST-equivalent exists for logistic models (confirmed by the NIST StRD and
the openassayflow validation research). We generated our own ground truth.

Done:
- Generated synthetic 4PL datasets with known exact parameters (Bottom=0,
  Top=100, EC50=1.0, HillSlope=1.0) plus controlled noise levels (0%, 1%, 5%)
- Generated 5PL datasets similarly (add HillAsym=0.5)
- Stored as `tests/validation/fourpl_certified_values.py`
- Created `tests/validation/test_fourpl_synth.py` (16 tests)

### 2. R drda cross-validation [IN PROGRESS -- see issue #4 in AGENT_GUIDANCE]

Cross-validate openfit Hill4P/Hill5P on a shared dataset against R's `drda`
package (Marasini et al., J Stat Softw 2023, DOI: 10.18637/jss.v106.i04).

Current state: `tests/validation/test_drda_crossvalidation.py` checks
convergence, parameter plausibility, AICc improvement, RSS improvement, and
the parameter-mapping math. It does NOT assert against stored R `drda`
coefficient values from a reproducible R run. That gap must be closed before
claiming drda cross-validation is complete.

Plan for the remaining work:
- Run R `drda` on `voropm2`, record `coef(fit_l4)` and `AIC(fit_l4)` with
  package version (drda >= 2.0.4) as comments.
- Assert openfit Hill4P parameters match (via alpha/delta/eta/phi mapping)
  within tolerance (rtol=1e-3 or better, accounting for weighting differences).
- For Hill5P vs drda logistic5: document parameterization differences clearly
  and limit assertions to directional checks (AICc improvement).
- Do NOT copy R source code; reference outputs and document the R command only.

Deliverable: update `tests/validation/test_drda_crossvalidation.py`

---

## Near-term: v0.1.2 -- PyPI publish

Before tagging v0.1.2, resolve in order:

1. **Resolve binding model validation gap (BLOCKING -- see issue #3 in AGENT_GUIDANCE)**
   - OneSiteBinding, TwoSiteBinding, CompetitiveBinding: committed to master
     and listed under "29 built-in" in README, but no published-reference
     validation test exists.
   - Option A: Add published-reference test (e.g., saturation binding data from
     Motulsky & Christopoulos, or a pharmacology textbook dataset).
   - Option B: Mark all three as `experimental` in the registry, rename the
     README count to "25 validated + 3 experimental binding models", and add
     a disclaimer.
   - Do not ship v0.1.2 without resolving this.

2. **Triage scratch R files**
   - `drda.R`, `drda_data.R`, `drda_main.R`, `drda_vignette.R`, `extract_drda.py`,
     `testdata.R`, `voropm2.Rd`, `voropm2.rda` are tracked in git.
   - Move any needed test fixtures to `tests/validation/reference/`.
   - Remove the rest. Update `.gitignore`. Verify the wheel excludes them.

3. **Complete drda coefficient cross-check (see item 2 above)**

4. **Fix README test count** (currently stale: says 339/338, actual 343/342)
   Update only after the final test inventory is stable.

5. **README validation badge**
   - CI matrix showing NIST dataset pass/fail (GitHub Actions badge)
   - Verify the badge points to a real passing workflow before publishing

6. **Version bump to 0.1.2**
   - Update `pyproject.toml`
   - Add CHANGELOG entry
   - Align README and ROADMAP status text
   - Tag: `git tag v0.1.2`

---

## Validation catch-up: published reference checks for already-built modules

The unit tests prove correctness in the sense of "runs + basic math."
The ROADMAP.md ties each feature to a specific published example that must match.

### compare.py -- Motulsky & Christopoulos (2003) tables [DONE]

F-test validation against M&C 2003 formula: `tests/test_compare_reference.py`

### uncertainty.py -- Profile CI asymmetric intervals [DONE]

Profile-likelihood CI validation against M&C Table 22.1:
`tests/test_uncertainty_reference.py` (6 tests)

### outliers.py -- ROUT paper Figure 2 / Table 1 [DONE]

ROUT validation against Motulsky & Brown 2006:
`tests/test_rout_reference.py` (10 tests)

### global_fit.py -- M&C Chapter 25 examples [DONE]

Global fit validation against textbook shared-parameter examples:
`tests/test_global_fit_reference.py`

### binding.py -- published saturation-binding reference [PENDING -- see above]

No published-reference test exists. Blocked until resolved.

---

## Engine improvement: silent argument dropping [DONE]

`UserWarning` when `x_scale` or `diff_method` is provided with `method='lm'`
was added in v0.1.1. Confirmed in CHANGELOG.

---

## Hahn1 and BoxBOD: test-only workarounds [DOCUMENTED, ACCEPTABLE]

The NIST validation tests for Hahn1 and BoxBOD use workarounds that live only
in the test file, not in the core engine:

- **Hahn1:** `_AnalyticJacModel` subclass providing an analytic Jacobian.
- **BoxBOD:** `bounds_dict={"b2": (0.0, 10.0)}` prevents a spurious basin.

These are correct behaviors (analytic Jacobian is always better; physical
bounds are a user responsibility). The CustomModel docstring should note that
analytic Jacobians significantly improve convergence for ill-conditioned
rational-polynomial models.

---

## Phase 2+ features (from ROADMAP.md)

These remain as-is in ROADMAP.md. No changes to the sequence. Key note:
**do not start Phase 6 (model library expansion) until the binding model
validation is resolved.** Every existing model needs its published-reference
test before new models are added.

Phases at a glance:
- v0.2.x-v0.3.x: Documentation only (code already shipped ahead of schedule)
- v0.6.x: Full model library (~30 equations) -- blocked by validation catch-up
- v0.7.x: PDF + Word reports + Prism import
- v0.8.x+: Constraint fitting, batch fitting, stable v1.0.0

---

## openassayflow (downstream package)

`openassayflow` is blocked until openfit v0.1.2 is tagged and CI-clean.

Do not start openassayflow until:
1. openfit v0.1.2 is tagged
2. Binding model validation is resolved
3. CI badge is green

---

## What NOT to do next

- Do not add new models before resolving the binding model validation gap.
- Do not start openassayflow before openfit v0.1.2 is tagged.
- Do not implement Bayesian fitting (out of scope per CLAUDE.md).
- Do not add a GUI (out of scope per ROADMAP.md).
- Do not tag v0.1.2 with scratch R files still committed at the repo root.
