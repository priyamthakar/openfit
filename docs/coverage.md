# Test Coverage

[![codecov](https://codecov.io/gh/priyamthakar/openfit/branch/master/graph/badge.svg)](https://codecov.io/gh/priyamthakar/openfit)

Coverage is reported automatically on every push to master via GitHub Actions + Codecov. Only the `ubuntu-latest` + Python 3.12 matrix entry reports coverage (to avoid noise from the matrix).

## What's covered

- Core fitting engine: `fit.py`, `results.py`, `spec.py`, `weighting.py`
- All model families
- Statistical inference: `uncertainty.py`, `compare.py`, `diagnostics.py`
- Global fit, ROUT, reports, and I/O

## What's excluded

The following paths are listed in `codecov.yml` and are not counted toward coverage:

- `tests/`
- `docs/`
- `paper/`
- `recipe/`

## Notes on coverage quality

NIST validation tests and synthetic certified-dataset tests contribute to coverage of the core engine paths that are hardest to reach with unit tests alone.

## Running coverage locally

```bash
pytest --cov=src/openfit --cov-report=term-missing --cov-report=html
open htmlcov/index.html
```
