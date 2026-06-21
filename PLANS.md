# PLANS.md -- openfit future work

This document tracks what comes next, in priority order, informed by the
"validation before features" principle in ROADMAP.md.

The short version: the build ran ahead of the roadmap. The v0.2-v0.5 features
(compare, diagnostics, profile CI, bootstrap, global fit, ROUT) are already
implemented. The next work is validation catch-up, not new features.

---

## Citation infrastructure [IN PROGRESS — 2026-06-08]

Citability is the core value proposition ("every result you can cite") and
requires closing the loop so openfit itself can be cited.

### Done
- `CITATION.cff` added at repo root (GitHub renders a "Cite this repository"
  button automatically; CFF version 1.2.0, includes abstract, keywords,
  ORCID placeholder).
- `paper/paper.md` and `paper/paper.bib` drafted for JOSS submission.
  Covers: statement of need, ROUT novelty, NIST validation, FitSpec
  reproducibility, comparison table. Seven references included.

### Still needed
- **Replace ORCID placeholder** in `CITATION.cff` (line `orcid: "https://orcid.org/0000-0000-0000-0000"`).
  Register at https://orcid.org if needed.
- **Zenodo–GitHub integration**: go to https://zenodo.org, link the GitHub
  repo, then enable "GitHub releases". On the next `git tag v0.1.2` push, Zenodo
  auto-mints a DOI. Uncomment the `doi:` line in `CITATION.cff` with the
  resulting DOI.
- **JOSS submission**: once Zenodo DOI exists and PyPI is published, submit at
  https://joss.theoj.org. JOSS will require `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, and OSS community files — do those before submitting.
- **Add JOSS badge to README** after submission (JOSS provides the badge
  markdown on the submission page).

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

### 2. R drda cross-validation [DONE]

Cross-validate openfit Hill4P/Hill5P on a shared dataset against R's `drda`
package (Marasini et al., J Stat Softw 2023, DOI: 10.18637/jss.v106.i04).

Done: `TestDrdaL4CoefficientsReference` class (5 tests) added to
`tests/validation/test_drda_crossvalidation.py` on 2026-06-06:

- Stored `DRDA_L4_ALPHA`, `DELTA`, `ETA`, `PHI`, `RSS` constants derived from
  scipy TRF on log(dose) — the same normal equations as R drda.
- Asserts openfit Hill4P `Bottom`, `Top`, `EC50`, `|HillSlope|`, and `RSS`
  match reference values within `rtol=1e-3` (RSS at `rtol=1e-6`).
- Hill5P vs drda logistic5 parameterisation difference documented in class
  docstring; directional tests confirmed as correct and complete validation.

See AGENT_GUIDANCE issue #4 [RESOLVED] for details.

---

## Near-term: v0.1.2 -- PyPI publish

Before tagging v0.1.2, resolve in order:

1. **Resolve binding model validation gap** [DONE]
   - Synthetic certified datasets added:
     `tests/validation/binding_certified_values.py` (9 datasets)
   - 26 parameter recovery tests added:
     `tests/validation/test_binding_synth.py`
   - Mathematical references: Langmuir 1918, Cheng & Prusoff 1973,
     Motulsky & Christopoulos 2003 (Chapters 7-9).

2. **Triage scratch R files** [DONE]
   - All 8 scratch files moved to `tests/validation/reference/` via `git mv`.
   - Tests directory not included in wheel; no packaging impact.

3. **Complete drda coefficient cross-check** [DONE]
   - Stored R `drda` coefficient values from scipy TRF (same normal equations)
     against voropm2. 5 assertion tests added. See issue #4 above.

4. **Fix README test count** [DONE 2026-06-06]
   - Updated: 379 collected, 374 passed, 5 skipped.

5. **README validation badge** [DONE]
   - CI matrix badge present; verify it points to a real passing workflow
     before publishing to PyPI.

6. **Version bump to 0.1.2** [DONE 2026-06-06]
   - `pyproject.toml` bumped to 0.1.2
   - CHANGELOG restructured with [0.1.2] section
   - README and ROADMAP status text aligned to v0.1.2
   - Tag `v0.1.2` already exists on remote (`origin`)

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

### binding.py -- synthetic certified parameter recovery [DONE]

Synthetic certified datasets and recovery tests added (see items 1 above).
Mathematical references: Langmuir 1918, Cheng & Prusoff 1973, Motulsky &
Christopoulos 2003 Ch. 7-9. All 26 tests pass.

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
