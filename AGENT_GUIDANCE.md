# AGENT_GUIDANCE.md -- openfit / openassay agent handoff

Last evaluated: 2026-06-03 (re-audited 2026-06-03)

This repository contains the `openfit` Python package. The working directory is
`D:\openfit`. Treat the package identity as `openfit` unless the user explicitly
asks to start the downstream `openassayflow` package.

## Current Snapshot

- Branch: `master`
- Package: `openfit`
- Declared version in `pyproject.toml`: `0.1.1`
- README status text: `v0.1.2 (in progress)`
- Test status (verified 2026-06-03): `343 collected, 342 passed, 1 skipped`
- Lint status (verified 2026-06-03): PASSING (`ruff check src tests: All checks passed!`)
- Format status (verified 2026-06-03): PASSING (`58 files already formatted`)
- Build status (verified 2026-06-03): `openfit-0.1.1.tar.gz` and `.whl` build clean
- Working tree status (verified 2026-06-03): CLEAN

Do not describe this checkout as release-ready. CI badge status is unverified
remotely. PyPI publication has not occurred.

## Agent Rules

1. Re-read `CLAUDE.md`, `ROADMAP.md`, `PLANS.md`, and this file before making
   major changes.
2. Validation work outranks feature work. Do not add new models unless their
   tests include both mathematical sanity checks and external or published
   reference validation.
3. Keep `openfit` domain-agnostic. ELISA, ADA, LLOQ/ULOQ, bioanalytical
   acceptance criteria, and regulatory interpretation belong in downstream
   `openassayflow`.
4. Preserve the explicit-weighting rule. A user must choose `weights=...`; no
   silent unweighted default.
5. Do not claim PyPI publication, release completion, CI success, or validation
   parity unless verified in the current session.
6. Do not revert existing changes unless the user explicitly asks.
   Work with the current state.

## Verification Commands

Run these from `D:\openfit`.

```powershell
git status --short --branch
python -m pytest
python -m ruff check src tests
python -m ruff format --check src tests
python -m build
```

Use the test result and Ruff result separately in status reports. Passing tests
do not imply CI is passing because `.github/workflows/ci.yml` runs Ruff before
pytest.

---

## Known Issues

Items marked [FIXED] were confirmed resolved in the 2026-06-03 audit.
Items marked [OPEN] remain unresolved.

### 1. README test count is stale [OPEN]

README says `339 tests collected (338 passed, 1 skipped)`. Verified count on
2026-06-03: `343 collected, 342 passed, 1 skipped`.

Fix: update README test count after the intended test inventory is stable.
Do not update during intermediate development.

### 2. Version/status language is inconsistent [OPEN]

`pyproject.toml` declares `0.1.1`; README says `v0.1.2 (in progress)`. That
may be intentional during development, but agents must not tag or publish
without aligning `pyproject.toml`, `CHANGELOG.md`, README, and `ROADMAP.md`.

### 3. Binding models have no published-reference validation [OPEN -- BLOCKING]

`src/openfit/models/binding.py` (OneSiteBinding, TwoSiteBinding,
CompetitiveBinding) is committed to master and listed in README under "29
built-in." `src/openfit/models/growth.py` contains AsymmetricGompertz.

The ROADMAP's iron rule: "No model ships without NIST StRD or
published-reference tests." `tests/test_binding_models.py` contains only:
- model_id and param_names checks
- equation shape and finiteness
- mathematical identities (midpoint at Kd, additive superposition, zero-inhibitor limit)
- initial_guess / bounds structure

No assertion cites a published binding isotherm dataset or textbook reference.
No assertion checks against a stored externally-derived coefficient.

Fix options:
  a. Add a test asserting published saturation-binding parameters from a
     pharmacology textbook or FDA reference dataset.
  b. Mark binding models as `experimental` in the registry and README, exclude
     from the "29 built-in" count, and document validation is pending.

Do not increment the ROADMAP model count or remove the "validation pending"
flag until at least one published-reference test exists per binding model type.

### 4. drda cross-validation is not a true coefficient cross-check [OPEN]

`tests/validation/test_drda_crossvalidation.py` validates convergence,
parameter plausibility (range checks), AICc improvement of 5PL over 4PL, RSS
improvement, and internal parameter-mapping consistency. It does NOT assert
against stored R `drda` coefficient values from a reproducible R run.

Safe status language: "drda-related tests exist; coefficient-level R
cross-validation still needs to be made explicit."

Fix: run R `drda` v2.x on `voropm2`, store the exact `coef(fit_l4)` values and
`AIC(fit_l4)` with package version and R command as comments, then assert
openfit Hill4P parameters agree (via the alpha/delta/eta/phi mapping) within
tolerance. For Hill5P vs drda logistic5, document clearly that the
parameterizations differ and what tolerance is justified.

### 5. Optional report dependencies imported eagerly [FIXED]

`src/openfit/report/__init__.py` now lazy-loads PDF and DOCX renderers inside
the `if fmt == "pdf"` and `elif fmt == "docx"` branches. A core-only install
that calls `report_fit(result, "out.html")` will not trigger an ImportError
for missing `reportlab` or `python-docx`.

### 6. report_fit() docstring is stale [OPEN -- minor]

`FitResult.report()` docstring (in `results.py`) correctly lists all four
formats: `"html", "markdown", "pdf", or "docx"`.

`report_fit()` docstring (in `report/__init__.py`, line ~23) still reads:
`"html", "markdown", or "pdf"` -- does not mention "docx". The Raises text
also omits "docx". The implementation correctly handles "docx".

Fix: update the `report_fit()` docstring and its Raises text.

### 7. CI was failing because of Ruff [FIXED]

`ruff check src tests` and `ruff format --check src tests` both pass as of
the 2026-06-03 audit. CI should be unblocked on Ruff.

### 8. Zero-RSS synthetic tests emit divide-by-zero warnings [UNVERIFIED]

The 29 warnings in the test run on 2026-06-03 are overflow/invalid-value
warnings from NIST MGH17 (ill-conditioned exponential sum). Whether zero-RSS
synthetic fits still emit divide-by-zero for AIC/BIC was not separately
verified. If future test runs show this warning type, define explicit behavior:
`AIC = BIC = AICc = -inf` for `rss == 0`, and add a direct test.

### 9. Scratch files are tracked in git [OPEN]

The following files are committed to the repository:

```
drda.R
drda_data.R
drda_main.R
drda_vignette.R
extract_drda.py
testdata.R
voropm2.Rd
voropm2.rda
```

These are scratch artifacts from the drda cross-validation work. They should
either be moved to `tests/validation/reference/` with clear provenance
comments, or removed and replaced by the hard-coded data already in
`test_drda_crossvalidation.py`. They should not ship in the package tarball.

Fix: decide whether any of these are needed as test fixtures. Move needed ones
to `tests/validation/reference/` with attribution. Remove the rest, then run
`python -m build` and verify they are excluded from the wheel.

---

## Immediate Plan

Priority order, informed by "validation before features" principle.

1. **Fix report_fit() docstring (issue #6).** One-line edit; no risk.
2. **Decide binding model status (issue #3 -- blocking).** Either add a
   published-reference test for OneSiteBinding (saturation-binding table from
   Motulsky & Christopoulos or equivalent), or mark binding models experimental
   and remove them from the "29 built-in" README claim. Do not ship v0.1.2 with
   the current ambiguity.
3. **Triage scratch files (issue #9).** Move or delete. Update `.gitignore`.
4. **Update README test count (issue #1).** After test inventory is stable.
5. **Make drda tests explicit (issue #4).** Add stored R coefficient values.
6. **Align version metadata (issue #2).** Align pyproject.toml, CHANGELOG,
   README, and ROADMAP before tagging.
7. **Tag v0.1.2 and set up PyPI Trusted Publishing.**

---

## Future Plans

### v0.1.2 release hardening

- CI green across Python 3.10, 3.11, 3.12 and Windows/macOS/Linux.
- PyPI Trusted Publishing workflow verified on a test tag or dry run.
- README badge points to a real passing workflow.
- `python -m build` and `twine check dist/*` pass from a clean checkout.
- README installation instructions match actual optional dependency behavior.
- Binding model status resolved (experimental vs validated).

### Validation expansion

- True R `drda` coefficient reference for Hill4P/logistic4 equivalence.
- External validation or documented mathematical references for binding models.
- Published or textbook reference for AsymmetricGompertz, or mark experimental.
- Tests for report generation under both core-only and `[reports]` installs.

### Downstream openassayflow

Start only after openfit is CI-clean and tagged. `openassayflow` should import
openfit rather than copy its fitting logic. Its first modules should be:

- Standard curve fitting and back-calculation.
- Calibrator, QC, and anchor-point handling.
- FDA/EMA bioanalytical method validation acceptance criteria.
- Assay run reports that clearly separate computed values from final regulated
  acceptance decisions.

---

## Periodic Review Protocol

After 7 days (by 2026-06-10):
- Re-run tests, Ruff, build, and `git status`.
- Check whether README test counts and status text still match the live suite.
- Verify CI badge status on GitHub Actions.

After 30 days (by 2026-07-03):
- Re-check package metadata, CI badge, PyPI status, and release tags.
- Re-evaluate all validation claims against actual tests, not roadmap text.
- Move or remove scratch files with explicit user approval.

After 90 days (by 2026-09-03):
- Revisit dependency versions and Python support.
- Re-run validation against current scipy/numpy versions.
- Re-check external reference package versions for R `drda`, `drc`, and `nplr`
  before making current-comparison claims.

---

## Safe Status Language

Use:

> Tests pass locally (343 collected, 342 passed, 1 skipped). Ruff lint and
> format pass. Build produces a clean wheel. Remote CI status unverified.

Use:

> drda-related tests exist but coefficient-level R cross-validation still
> needs to be made explicit.

Use:

> Binding models (OneSiteBinding, TwoSiteBinding, CompetitiveBinding) are
> implemented and pass mathematical-identity tests; published-reference
> validation is pending.

Do not use:

> Release-ready.

Do not use:

> Fully cross-validated against R drda.

Do not use:

> PyPI publish complete.

Do not use:

> Binding models are fully validated.
