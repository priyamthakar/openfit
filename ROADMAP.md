# ROADMAP.md -- openfit

Reproducible, open-source nonlinear curve fitting with publication-quality reports.

---

## Guiding Principles

1. **Validation before features.** No model ships without NIST StRD or published-reference tests.
2. **Reproducibility is the moat.** Every fit emits a FitSpec. Anyone with the spec and the data gets the identical result.
3. **Domain-agnostic engine.** openfit knows math, not biology. Domain interpretation lives in downstream packages (openassayflow, openpkflow).
4. **Report-first.** The deliverable is a shareable document, not a number in a terminal.
5. **Explicit over implicit.** Weights, initial guesses, and CI methods are never silently defaulted.

---

## Release Ladder

### Phase 1 -- Foundation (v0.1.x)

```
v0.1.0  Core engine + essential models + reports                [SHIPPED 2026-06-03]
        -----------------------------------------------
        [x] Fit() engine wrapping scipy.optimize.least_squares (LM + TRF)
        [x] 25 models: Hill3P, Hill4P (4PL), Hill5P (5PL), Boltzmann,
            MonoExp, BiExp, ExpGrowth, ExpPlateau, ExpDecay,
            MichaelisMenten, SubstrateInhibition, Allosteric,
            Logistic3P, Logistic4P, Gompertz, Richards,
            Gaussian, BiGaussian, Lorentzian, Poly1-Poly6, CustomModel
        [x] Smart initial_guess(x, y) for every model (data-driven heuristics)
        [x] Weighting: uniform, 1/Y, 1/Y^2, 1/SD^2, Poisson
        [x] FitResult: params, asymptotic SE, CI, R^2, AICc, BIC, residuals
        [x] FitSpec: reproducibility manifest (.to_json() / .from_json())
        [x] HTML and Markdown fit reports (Jinja2 + matplotlib)
        [x] CLI: openfit fit, openfit compare, openfit models, openfit version
        [x] Input validation: NaN/Inf rejection
        [x] Tests: 93 unit tests, all passing
        [x] Wheel builds clean: openfit-0.1.0-py3-none-any.whl
        NOTE: Shipped ahead of schedule -- also includes compare, diagnostics,
              profile CI, bootstrap CI, global fit, ROUT (v0.2-v0.5 features).

v0.1.1  NIST StRD validation suite + 4PL/5PL synthetic certified data [COMPLETE]
        -----------------------------------------------
        [x] All 27 NIST .dat files downloaded to tests/validation/nist_data/
        [x] nist_certified_values.py: all 27 datasets with certified params + data
        [x] Parametrized suite: both Start I and Start II converge for all 27
        [x] Parameter recovery to >= 6 sig digits (200 tests, 0 failures, 1 skip*)
        [x] Higher-difficulty all pass: BoxBOD, Eckerle4, Rat42, Rat43, MGH09,
            MGH10, Thurber, Bennett5
        [x] Fit() extended: diff_method, xtol, ftol, gtol, x_scale parameters
        [x] CustomModel extended: bounds_dict parameter
        [x] 4PL/5PL synthetic certified datasets (0%, 1%, 5% noise)
        [x] Engine fix: UserWarning when x_scale/diff_method dropped with lm
        [x] Published-reference validation: compare, uncertainty, outliers, global_fit
        [x] Global fit report + ROUT outlier visualization in reports
        [ ] Cross-validate 4PL/5PL against R drda (Marasini et al., JSS 2023)
        [ ] CI badge in README showing NIST pass/fail matrix

        * Lanczos1 RSS check skipped: certified RSS = 1.43e-25 (128-bit);
          at 64-bit, floating-point cancellation makes relative comparison
          meaningless. Lanczos1 parameter tests still pass.

        Definition of done: all 27 NIST datasets pass + 4PL/5PL synthetic recovery.
        >>> CORE DONE. drda cross-validation and CI badge moved to v0.1.2.

v0.1.2  Cross-validation & reference alignment                 [COMPLETE]
        -----------------------------------------------
        [x] R drda cross-validation (coefficient-level Hill4P vs logistic4)
        [x] Zero-RSS guard for perfect fits
        [x] Expose parameter covariance matrix for downstream packages
        [x] Tag: git tag v0.1.2

v0.1.3  PyPI publish, citation & docs                         [IN PROGRESS]
        -----------------------------------------------
        [x] GitHub Actions CI workflow (3 OS x 3 Python versions)
        [x] PyPI Trusted Publishing workflow (GitHub Actions on version tags)
        [x] py.typed already present (PEP 561)
        [x] Documentation site built with mkdocs
        [x] JOSS paper draft and Zenodo citation CFF
        [ ] PyPI project registration & publish
```

### Phase 2 -- Statistical Depth (v0.2.x - v0.3.x)

NOTE: Code for v0.2 and v0.3 was shipped in v0.1.0 (ahead of schedule).
      What remains is validation catch-up: published-reference tests to confirm
      the implementations match the cited methods, not just "run without error."

```
v0.2.0  Model comparison + GOF diagnostics                    [CODE DONE, VALIDATION DONE]
        -----------------------------------------------
        [x] compare_models(): AICc, BIC, evidence ratio, Akaike weights
        [x] F-test (extra sum-of-squares) for nested model pairs
        [x] Nestedness detection: auto-check if model A is a special case of model B
        [x] Residual analysis: runs test (Wald-Wolfowitz), replicates test, Shapiro-Wilk
        [x] QQ plot, residuals-vs-predicted plot
        [x] Comparison HTML report: side-by-side fit overlays + criteria table
        [x] VALIDATION: F-test result matches M&C 2003 formula (tests/test_compare_reference.py)

v0.3.0  Profile-likelihood CI + bootstrap CI                  [CODE DONE, VALIDATION DONE]
        -----------------------------------------------
        [x] profile_likelihood_ci(): walk each parameter, find likelihood ratio boundary
        [x] Detect non-unimodal profiles, warn user
        [x] bootstrap_ci(): residual resampling, BCa correction
        [x] Fixed random seed in FitSpec for bootstrap reproducibility
        [x] VALIDATION: profile CI on Hill4P produces asymmetric intervals
            (tests/test_uncertainty_reference.py -- 6 tests)
```

### Phase 3 -- The Moat Features (v0.4.x - v0.5.x)

NOTE: Both v0.4 and v0.5 were shipped in v0.1.0 (ahead of schedule).
      Same caveat: published-reference validation is pending.

```
v0.4.0  Global/shared-parameter fitting                       [CODE DONE, VALIDATION DONE]
        -----------------------------------------------
        [x] GlobalFit(datasets, model, shared=[...], local=[...])
        [x] Joint optimization: shared params constrained equal across datasets
        [x] Per-dataset local params fitted independently within the joint objective
        [x] F-test: is sharing justified? Compare joint vs. independent RSS
        [x] Global fit report: overlay all datasets with shared curve + local curves
        [x] VALIDATION: textbook shared-fitting examples (tests/test_global_fit_reference.py)

v0.5.0  ROUT outlier detection
        -----------------------------------------------
        [x] rout_outliers(): Motulsky & Brown (BMC Bioinformatics 2006) -- first Python impl
        [x] Lorentzian merit function, RSDR = P68*N/(N-K), BH FDR control
        [x] Q parameter user-configurable, default 1%
        [x] Report: flagged points highlighted on fit plot
        [x] VALIDATION: reproduce Figure 2 / Table 1 from the 2006 paper
            (tests/test_rout_reference.py -- 10 tests)
```

### Phase 4 -- Model Library Expansion (v0.6.x)

NOTE: Blocked until validation catch-up (Phase 2-3) is complete per
      "validation before features" principle. Many models from this list were
      already shipped in v0.1.0; the remaining ones and binding models are new.

```
v0.6.0  Full model library (~30 equations)              [PARTIALLY DONE -- SEE PLANS.md]
        -----------------------------------------------
        Already shipped in v0.1.0: Boltzmann, ExpGrowth, ExpPlateau, ExpDecay,
          Gompertz, Richards, BiGaussian, Lorentzian, SubstrateInhibition,
          Allosteric, Poly4-Poly6.
        Still needed:
        - Binding: one-site specific, two-site, competitive (new)
        - Asymmetric Gompertz-sigmoid (new)
        - Each new model: equation, smart initial_guess, analytic Jacobian, >= 2 tests
        Definition of done: every model has >= 2 tests (degenerate + published reference).
```

### Phase 5 -- Reports + Migration (v0.7.x)

```
v0.7.0  PDF + Word reports + Prism import
        -----------------------------------------------
        - ReportLab PDF report: publication-quality, embeddable in papers
        - python-docx Word report: for collaborators who need .docx
        - Prism .pzfx XML import (read-only): parse data tables and model selections
          so users can migrate existing Prism analyses to openfit
        - Plot export: SVG, PNG, PDF (individual plots, not just in reports)
        Definition of done: a user can import a Prism file, rerun the fit, and get a
        reproducible spec + report that matches the Prism results.
```

### Phase 6 -- Advanced Features (v0.8.x+)

```
v0.8.0  Constraint fitting + parameter expressions
        -----------------------------------------------
        - Parameter bounds (already in scipy, surface it cleanly)
        - Parameter expressions: "Top = 100" (fixed), "EC50_B = 2 * EC50_A" (linked)
        - Penalty functions for soft constraints

v0.9.0  Batch fitting
        -----------------------------------------------
        - Fit the same model to 100+ datasets (e.g., plate reader rows)
        - Summary table: parameter estimates across all fits
        - Batch report: heatmap of R^2, flagged poor fits

v1.0.0  Stable public release
        -----------------------------------------------
        - All NIST StRD datasets passing
        - Full model library (30+)
        - Global fitting, profile CI, bootstrap CI, ROUT
        - HTML + PDF + Markdown + DOCX reports
        - Prism import
        - FitSpec reproducibility for every fit
        - Comprehensive docs (mkdocs)
        - conda-forge recipe
```

---

## Validation Matrix

Every release must maintain the following:

| Validation tier | Source | What it proves |
|----------------|--------|----------------|
| NIST StRD (27 datasets) | NIST public domain | Parameter recovery to 6+ sig digits |
| Motulsky & Christopoulos textbook | Published tables | F-test, AICc, profile CI correctness |
| Motulsky & Brown 2006 | BMC Bioinformatics | ROUT outlier detection correctness |
| R `drc` cross-validation | Ritz et al. 2015, PLOS ONE | 4PL/5PL parameter agreement on shared data |
| Degenerate cases | Mathematical identity | Edge case robustness (zero variance, flat data, single point) |

No release ships if any NIST test regresses.

---

## Downstream Packages (planned)

openfit is the engine. Domain packages add interpretation:

```
openfit                     -- domain-agnostic curve fitting engine
  |
  +-- openassayflow         -- ELISA 4PL/5PL, standard curves, back-calculation,
  |                            LLOQ/ULOQ, parallelism, relative potency,
  |                            ADA cut points, FDA BMV compliance
  |
  +-- openpkflow            -- may adopt openfit for dissolution model fitting
  |                            (Weibull, Korsmeyer-Peppas, etc.)
  |
  +-- (future)              -- environmental dose-response, agricultural assays, etc.
```

Each downstream package:
- Imports openfit for the fitting engine
- Adds domain-specific interpretation, acceptance criteria, and reports
- Has its own validation suite against domain-specific published references
- Has its own CLAUDE.md with domain-specific correctness rules

---

## Non-Goals (things we will never do)

- Replace scipy as a general optimization library
- Build a GUI (Prism's GUI is their moat; our moat is reproducibility + transparency)
- Bayesian inference in core (optional extension only)
- Real-time / streaming fitting
- GPU acceleration (our datasets are small; scipy on CPU is fast enough)

---

## Key References

- Motulsky, H. & Christopoulos, A. (2003). *Fitting Models to Biological Data Using Linear and Nonlinear Regression.* GraphPad Software. -- The textbook behind Prism's methods.
- Motulsky, H.J. & Brown, R.E. (2006). Detecting outliers when fitting data with nonlinear regression: a new method based on robust nonlinear regression and the false discovery rate. *BMC Bioinformatics*, 7, 123. -- The ROUT paper.
- NIST StRD: https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml -- Certified reference datasets.
- Ritz, C. et al. (2015). Dose-Response Analysis Using R. *PLOS ONE*, 10(12), e0146021. -- The R `drc` package paper.
- DeLean, A., Munson, P.J. & Rodbard, D. (1978). Simultaneous analysis of families of sigmoidal curves. *Am. J. Physiol.*, 235(2), E97-E102. -- The original 4PL/ALLFIT paper.
