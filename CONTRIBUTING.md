# Contributing to openfit

Thank you for your interest in contributing. openfit follows a
**validation-before-features** principle: every model and statistical method
must have a published-reference or certified-dataset test before it ships.
Please read this document before opening a pull request.

---

## Quick start

```bash
git clone https://github.com/priyamthakar/openfit
cd openfit
pip install -e ".[dev]"
pytest                          # 379 tests (374 passed, 5 skipped)
ruff check src tests            # lint
ruff format --check src tests   # format
mypy src/openfit                # type check
python -m build                 # wheel + sdist
```

All four commands must pass before a PR is mergeable. CI runs them in order
on Python 3.10, 3.11, 3.12 across Ubuntu, macOS, and Windows.

---

## What to work on

Good starting points are issues labelled **`good first issue`**. Before
starting anything large, open an issue to discuss the approach.

The highest-value contributions right now (in priority order):

1. **Validation tests** for existing models against published datasets.
2. **Documentation** — narrative guides for the mkdocs site.
3. **Example notebooks** — runnable Jupyter demos.
4. **Bug fixes** with a regression test.

Things explicitly out of scope for openfit (see ROADMAP.md):

- Domain-specific logic (ELISA, IC50 interpretation, PK parameters) — belongs
  in downstream packages.
- GUI or interactive widgets.
- Bayesian fitting.
- New models without published-reference validation tests.

---

## Adding a new model

Every new model must include:

1. The model class in `src/openfit/models/` implementing `equation()`,
   `initial_guess(x, y)`, and `bounds()`.
2. A unit test in `tests/test_models.py` (sanity check: non-NaN output,
   correct asymptotes).
3. A validation test in `tests/validation/` that recovers known parameters
   from either:
   - A NIST StRD certified dataset, or
   - A synthetic certified dataset generated from the exact equation (with
     noise-free recovery to `rtol=1e-6`), or
   - Published parameter values from a peer-reviewed paper or textbook.

PRs that add a model without a validation test will not be merged.

---

## Code style

- **Line length:** 100 characters (configured in `pyproject.toml`).
- **Formatter:** `ruff format` (run before committing).
- **Linter:** `ruff check` with rules E, F, W, I, UP, B, SIM.
- **Types:** `mypy --strict` must pass on `src/openfit`.
- **Docstrings:** NumPy style. Every public function, class, and method needs
  a one-line summary at minimum.
- **No silent defaults:** the `weights=` argument to `Fit()` is required by
  design. Do not add silent defaults to any fitting API.

---

## Tests

- Tests live in `tests/`. Validation tests live in `tests/validation/`.
- Use `pytest` fixtures and parametrize where sensible.
- Aim for full branch coverage on new code (`pytest --cov=src/openfit`).
- Do not commit generated files (reports, plots) to the test directory.

---

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(models): add WeibullGrowth model with synthetic certified tests
fix(uncertainty): reject non-finite SE before profile CI walk
docs: update CONTRIBUTING with model validation requirements
test(validation): add Thurber NIST Start I / Start II tests
chore: bump version to 0.1.3
```

---

## Pull request checklist

Before marking a PR as ready for review:

- [ ] `pytest` passes (no new failures, no skipped tests without explanation)
- [ ] `ruff check src tests` passes
- [ ] `ruff format --check src tests` passes
- [ ] `mypy src/openfit` passes
- [ ] New public API has docstrings
- [ ] New or changed behaviour has tests
- [ ] New model has a validation test (see above)
- [ ] CHANGELOG.md updated under `[Unreleased]`

---

## Reporting bugs

Use the bug report issue template. Please include:

- openfit version (`openfit version` or `python -c "import openfit; print(openfit.__version__)"`)
- Python version and OS
- A minimal reproducible example
- The full traceback

---

## License

By contributing you agree that your contributions will be licensed under the
MIT License.
