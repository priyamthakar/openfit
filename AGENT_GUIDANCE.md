# AGENT_GUIDANCE.md -- openfit / openassay agent handoff

Last evaluated: 2026-06-03

This repository currently contains the `openfit` Python package, not a finished
`openassayflow` domain package. Treat `D:\openassay` as the working directory,
but treat the package identity as `openfit` unless the user explicitly asks to
start the downstream `openassayflow` package.

## Current Snapshot

- Branch: `master`
- Package: `openfit`
- Declared version in `pyproject.toml`: `0.1.1`
- README status text: `v0.1.2 (in progress)`
- Test status on 2026-06-03: `335 passed, 1 skipped`
- Lint status on 2026-06-03: failing
- Format status on 2026-06-03: failing
- Working tree status on 2026-06-03: dirty, with modified tracked files and
  untracked model/report/validation files

Do not describe this checkout as release-ready. It is functionally strong by
tests, but not CI-clean or packaging-clean.

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
6. Do not revert existing dirty-tree changes unless the user explicitly asks.
   Work with the current state.

## Verification Commands

Run these from `D:\openassay`.

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

## Known Mistakes And Drift Points

### 1. README test count is stale

README says `324 tests collected`, but local pytest collected 336 tests on
2026-06-03 and reported `335 passed, 1 skipped`.

Fix: update README only after the intended test inventory is stable.

### 2. Version/status language is inconsistent

`pyproject.toml` declares `0.1.1`, while README says `v0.1.2 (in progress)`.
That may be intentional during development, but future agents must not tag or
publish without aligning `pyproject.toml`, `CHANGELOG.md`, README, and release
notes.

### 3. Roadmap says model expansion is blocked, but new models were added

`PLANS.md` says not to add binding models or additional model-library features
before validation catch-up. The working tree includes untracked
`src/openfit/models/binding.py` and a new `AsymmetricGompertz` in
`src/openfit/models/growth.py`.

Fix: either add real published-reference validation for these models or mark
them experimental and keep them out of release claims.

### 4. drda cross-validation is not a true coefficient cross-check yet

`tests/validation/test_drda_crossvalidation.py` documents R `drda`, but the
tests mainly check convergence, plausibility, AICc improvement, and internal
mapping consistency. They do not assert against stored R `drda` coefficient
values from a reproducible R run.

Fix: run R `drda` on `voropm2`, store the exact coefficients/AIC values with
package version and command provenance, then assert openfit agreement where the
model parameterizations are genuinely equivalent. For 5PL, document clearly
that openfit Hill5P and drda logistic5 are not the same parameterization.

### 5. Optional report dependencies are imported eagerly

`src/openfit/report/__init__.py` imports PDF and DOCX renderers at module import
time. This can make `from openfit.report import report_fit` fail in a core-only
install without `reportlab` or `python-docx`, even for HTML/Markdown reports.

Fix: import `render_pdf_report` only inside the `fmt == "pdf"` branch and
`render_docx_report` only inside the `fmt == "docx"` branch, or make the base
package depend on report backends and update packaging claims accordingly.

### 6. Report docs are stale

`FitResult.report()` docstring still says output format is `"html" or
"markdown"` even though code accepts `"pdf"` and `"docx"`.

Fix: update the docstring and error text after deciding whether PDF/DOCX are
core or optional.

### 7. CI will fail until Ruff is handled

Local `python -m pytest` passes, but `python -m ruff check src tests` reports
many errors and `python -m ruff format --check src tests` reports 40 files that
would be reformatted. Because CI runs both Ruff commands, GitHub CI should be
assumed failing until proven otherwise.

Fix: run `python -m ruff format src tests`, then address `ruff check` errors in
small batches. Avoid using broad unsafe fixes without review.

### 8. Zero-RSS synthetic tests emit divide-by-zero warnings

Perfect synthetic fits produce `RuntimeWarning: divide by zero encountered in
log` for AIC/BIC calculations. Tests pass, but this is noisy and can hide more
important warnings.

Fix: define explicit behavior for `rss == 0`, such as `AIC = -inf` and
`BIC = -inf`, and test it directly.

### 9. Stray root files and malformed directories should be triaged

The root contains extracted R files (`drda*.R`, `testdata.R`, `voropm2.*`,
`extract_drda.py`, `test_out.html`) and malformed directory names such as
`D:openfit...` rendered with replacement characters. These may be useful
scratch artifacts, but they should not ship in the package.

Fix: decide which artifacts belong under `tests/validation/reference/` and
remove or ignore scratch files only after confirming they are not user-needed.

## Immediate Plan

1. Freeze the current dirty-tree intent.
   - Decide whether binding models, asymmetric Gompertz, PDF/DOCX reports, and
     drda validation are part of v0.1.2 or should remain experimental.
2. Make CI honest.
   - Fix Ruff format first.
   - Fix Ruff lint in code touched by current work first, then repository-wide.
3. Make validation claims precise.
   - Update README test count and status from a fresh pytest run.
   - Replace "drda cross-validation" claims with either real R coefficient
     comparisons or a weaker "voropm2 logistic smoke validation" label.
4. Make optional dependencies sane.
   - Lazy-load PDF/DOCX dependencies or promote them to core dependencies.
5. Clean release metadata.
   - Align `pyproject.toml`, `CHANGELOG.md`, README, `ROADMAP.md`, and `PLANS.md`.
6. Only then consider tagging or PyPI publishing.

## Future Plans

### v0.1.2 release hardening

- CI green across Python 3.10, 3.11, 3.12 and Windows/macOS/Linux.
- PyPI Trusted Publishing workflow verified on a test tag or dry run.
- README badge points to a real passing workflow.
- `python -m build` and `twine check dist/*` pass from a clean checkout.
- README installation instructions match actual optional dependency behavior.

### Validation expansion

- True R `drda` coefficient reference for Hill4P/logistic4 equivalence.
- External validation or documented mathematical references for binding models.
- Published or textbook reference for asymmetric Gompertz, or mark experimental.
- Tests for report generation under both core-only and `[reports]` installs.

### Downstream openassayflow

Start only after openfit is CI-clean and tagged. `openassayflow` should import
openfit rather than copy its fitting logic. Its first modules should be:

- Standard curve fitting and back-calculation.
- Calibrator, QC, and anchor-point handling.
- FDA/EMA bioanalytical method validation acceptance criteria.
- Assay run reports that clearly separate computed values from final regulated
  acceptance decisions.

## Periodic Review Protocol

After 7 days:
- Re-run tests, Ruff, build, and `git status`.
- Check whether README test counts and status text still match the live suite.
- Record any new mismatch in this file or a follow-up handoff.

After 30 days:
- Re-check package metadata, CI badge, PyPI status, and release tags.
- Re-evaluate all validation claims against actual tests, not roadmap text.
- Move scratch files into durable test fixtures or remove them with explicit
  user approval.

After 90 days:
- Revisit dependency versions and Python support.
- Re-run validation against current scipy/numpy versions.
- Re-check external reference package versions for R `drda`, `drc`, and `nplr`
  before making current-comparison claims.

## Safe Status Language

Use:

> Tests pass locally, but CI is not clean until Ruff passes.

Use:

> drda-related tests exist, but coefficient-level R cross-validation still
> needs to be made explicit.

Do not use:

> Release-ready.

Do not use:

> Fully cross-validated against R drda.

Do not use:

> PyPI publish complete.
