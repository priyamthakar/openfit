# Changelog

All notable changes to openfit are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.1.2] - 2026-06-06

### Added
- R drda 2.0.4 coefficient-level reference values for Hill4P on voropm2:
  stored `DRDA_L4_ALPHA`, `DELTA`, `ETA`, `PHI`, `RSS` constants in
  `tests/validation/test_drda_crossvalidation.py`; Hill4P parameters now
  asserted to match drda logistic4 within `rtol=1e-3` (5 new tests). Closes issue #4.
- Hill5P vs drda logistic5 parameterisation difference documented explicitly;
  directional AICc/RSS improvement tests confirmed as correct cross-validation.
- Zero-RSS guard in `src/openfit/fit.py`: `AIC`, `BIC`, and `AICc` are defined
  as `-inf` when `RSS == 0`, without emitting `RuntimeWarning`. New test in
  `tests/test_fit.py` asserts this behaviour.

### Fixed
- `AGENT_GUIDANCE.md`: stale test count updated (343 → 374 passed, 5 skipped);
  issues #4 and #6 marked resolved; periodic review dates refreshed.

### Changed
- Version bump to 0.1.2.
- README development section test count corrected (339 → 379 collected, 374 passed, 5 skipped).
- `ROADMAP.md`: R drda cross-validation marked done `[x]`.

---

## [0.1.1] - 2026-06-03

### Added
- `FitResult.covariance` exposes the full parameter covariance matrix in
  `model.param_names` order; singular Jacobians return a NaN-filled covariance
  matrix, infinite standard errors, and undefined (`nan`) asymptotic CIs.
- Uncertainty helpers now reject non-finite standard errors before computing
  asymptotic or profile-likelihood intervals.
- NIST StRD nonlinear regression suite: all 27 datasets, 200 tests passing.
  Both NIST Start I (far) and Start II (close) recover certified parameters
  to >= 6 significant digits for all 27 datasets. RSS matches certified values
  to >= 6 significant digits (26/27; Lanczos1 RSS check skipped: certified
  RSS = 1.43e-25, below double-precision floor).
- `Fit()` extended with `diff_method`, `xtol`, `ftol`, `gtol`, `x_scale` parameters.
- `CustomModel` extended with `bounds_dict` parameter.
- Synthetic 4PL/5PL certified datasets with known exact parameters (0%, 1%, 5% noise).
- Parameter recovery tests for 4PL and 5PL (16 tests).
- Engine fix: `UserWarning` when `x_scale`/`diff_method` silently dropped with `method='lm'`.
- Published-reference validation tests:
  - F-test comparison validation (Motulsky & Christopoulos 2003)
  - Profile-likelihood CI validation (asymmetric CI for 4PL)
  - ROUT outlier detection validation (Motulsky & Brown 2006)
  - Global fit validation (shared-parameter fitting)
- Global fit report: overlay all datasets with shared + local curves (HTML/Markdown).
- ROUT outlier visualization: flagged points highlighted in fit reports.
- GitHub Actions CI workflow (3 OS, 3 Python versions).
- PyPI Trusted Publishing workflow (GitHub Actions on version tags).

---

## [0.1.0] - 2026-06-03

Initial release. Full implementation of the core engine and all
v0.1.0 roadmap features plus several v0.2-v0.5 features built ahead of schedule.

### Added

**Core fitting engine**
- `Fit(model, x, y, *, weights, ...)` -- nonlinear least-squares via
  scipy.optimize.least_squares (Levenberg-Marquardt + Trust Region Reflective)
- `weights=` is keyword-only and required: no silent default (CLAUDE.md rule 1)
- NaN/Inf in input data raises ValueError (CLAUDE.md rule 8)
- Auto-selects TRF when any parameter bounds are finite
- `p0=`, `bounds=`, `method=`, `x_scale=`, `diff_method=`, `max_nfev=`,
  `xtol=`, `ftol=`, `gtol=`, `random_seed=` parameters
- Analytic Jacobian used when model provides one (finite-difference fallback)

**FitResult**
- Fitted `params` dict, asymptotic `se`, `ci` (t-distribution at 95%)
- `r_squared` (correct nonlinear definition: 1 - SS_res/SS_tot)
- `aic`, `bic`, `aicc` information criteria
- `rss`, `residuals`, `weighted_residuals`, `standardized_residuals`
- `n_obs`, `n_params`, `dof`, `model_id`, `weight_scheme`
- `spec` (FitSpec reproducibility manifest)
- `summary()` -- plain ASCII table of all fit statistics
- `plot()` -- fit overlay + residual panel (Matplotlib)
- `report()` -- HTML or Markdown report

**FitSpec (reproducibility manifest)**
- SHA-256 hash of input data (float64 LE bytes)
- Fitted parameter values (lossless repr() serialization)
- Weight scheme, openfit/scipy/numpy versions, random seed
- `.to_json()` / `.from_json()` roundtrip

**Weight schemes**
- `uniform`, `1/y`, `1/y2`, `1/sd2`, `poisson`
- 13 string aliases (case-insensitive)
- Strict positive-y validation before division

**Models (25 built-in)**
- Sigmoidal: Hill3P, Hill4P (4PL), Hill5P (5PL), Boltzmann
- Exponential: MonoExp, BiExp, ExpGrowth, ExpPlateau, ExpDecay
- Enzyme: MichaelisMenten, SubstrateInhibition, Allosteric
- Growth: Logistic3P, Logistic4P, Gompertz, Richards
- Gaussian: Gaussian, BiGaussian, Lorentzian
- Polynomial: Poly1, Poly2, Poly3, Poly4, Poly5, Poly6
- Custom: CustomModel (wraps any callable, infers param names from signature)
- All models: `equation()`, `initial_guess(x, y)`, `bounds()`
- Hill4P, MichaelisMenten: analytic Jacobian
- All polynomials: analytic Jacobian
- Registry: `get_model(name)` (case-insensitive), `list_models()`,
  `register_model()`

**Statistical inference**
- Asymptotic SE from scaled Jacobian covariance matrix
- Profile-likelihood CI (walk parameter, find likelihood ratio boundary;
  warns on non-unimodal profile)
- Bootstrap CI with BCa correction (residual resampling; fixed seed in FitSpec)

**Model comparison**
- `compare_models(results)` -> ComparisonResult
- AICc, BIC, evidence ratio, Akaike weights
- F-test (extra sum-of-squares) for nested model pairs only
  (non-nested pairs: AICc only, per CLAUDE.md rule 4)
- Nestedness detection via parameter intersection

**Residual diagnostics**
- `residual_analysis(result)` -> DiagnosticsResult
- Wald-Wolfowitz runs test (Wald & Wolfowitz 1940)
- Replicates / lack-of-fit F-test (Draper & Smith 1998)
- Normality test: Shapiro-Wilk (n <= 50), D'Agostino-Pearson (n > 50)
- 3-sigma outlier flags

**Global/shared-parameter fitting (ahead of v0.4.0 schedule)**
- `GlobalFit(datasets, model, shared=[...], local=[...])` -> GlobalFitResult
- Joint optimization: shared parameters constrained equal across all datasets
- Per-dataset local parameters fitted independently within joint objective
- F-test: compare joint vs. independent fits (is sharing statistically justified?)

**ROUT outlier detection (ahead of v0.5.0 schedule)**
- `rout_outliers(model, x, y, weights, Q=0.01)` -> ROUTResult
- Implements Motulsky & Brown (BMC Bioinformatics, 2006) algorithm exactly:
  Lorentzian merit function, RSDR = P68*N/(N-K), BH FDR control
- First Python implementation of the ROUT algorithm

**Reports**
- `report_fit(result, path)` -- dispatches to HTML or Markdown based on extension
- HTML: Jinja2 template with embedded base64 plots; pure-Python fallback
- Markdown: ASCII-safe table format
- All reports include CLAUDE.md-required disclaimer

**I/O**
- `load_csv(path, x_col, y_col)` -- flexible column mapping
- `load_excel(path, ...)` -- same API for .xlsx
- `load_pzfx(path)` -- read-only Prism XML import (migration helper)
- `_validate_arrays(x, y)` -- NaN/Inf/length checks (public)

**CLI**
- `openfit version` -- show versions
- `openfit models` -- list all registered models
- `openfit fit data.csv --model hill4p --weights 1/y2 --report out.html`
- `openfit compare data.csv --models hill3,hill4,hill5`
- ASCII-safe output (no Unicode punctuation, cp1252-compatible)

**Plotting**
- `fit_overlay_plot()` -- fit curve over data, auto log-x when max/min > 100
- `residual_plot()` -- residuals vs. predicted
- `qq_plot()` -- normal Q-Q plot of standardized residuals
- `figure_to_base64()` -- PNG for HTML embedding

**Packaging**
- `src/` layout (PEP 517/518), hatchling build backend
- `py.typed` marker (PEP 561, enables downstream type checking)
- Optional dep groups: `[cli]`, `[reports]`, `[dev]`
- Python >= 3.10

**Tests**
- 93 unit tests across: spec, weighting, models, fit, uncertainty, compare,
  diagnostics, global_fit, outliers, io
- All tests pass: 200 total, 0 failures, 1 skip (Lanczos1 RSS precision limit)

---

[Unreleased]: https://github.com/priyamthakar/openfit/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/priyamthakar/openfit/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/priyamthakar/openfit/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/priyamthakar/openfit/releases/tag/v0.1.0
