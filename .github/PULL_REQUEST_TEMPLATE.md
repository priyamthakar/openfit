## Summary

<!-- One or two sentences describing what this PR does. -->

## Type of change

- [ ] Bug fix (non-breaking, fixes an issue)
- [ ] New model (requires validation test — see CONTRIBUTING.md)
- [ ] New feature (non-breaking)
- [ ] Breaking change (existing behaviour changes)
- [ ] Documentation / tests only

## Related issue

Closes #

## Changes

<!-- Bullet list of the key changes. -->

## Validation (required for new models)

<!-- If this PR adds a model, describe the validation test and reference:
     - Reference: <paper / textbook / NIST dataset>
     - Test file: tests/validation/...
     - Recovery tolerance: rtol=... -->

N/A

## Checklist

- [ ] `pytest` passes locally (no new failures)
- [ ] `ruff check src tests` passes
- [ ] `ruff format --check src tests` passes
- [ ] `mypy src/openfit` passes
- [ ] New public API has docstrings
- [ ] New or changed behaviour has tests
- [ ] New model has a validation test (see CONTRIBUTING.md)
- [ ] CHANGELOG.md updated under `[Unreleased]`
