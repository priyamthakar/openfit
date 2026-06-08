# Benchmark: openfit vs scipy.curve_fit

Both openfit and `scipy.curve_fit` call the same scipy optimizer under the hood
(`scipy.optimize.least_squares`), so **parameter agreement is expected within
floating-point tolerance** when both are given the same problem and starting point.
The benchmark is not a speed race — it documents the overhead cost of openfit's
correctness infrastructure and highlights where openfit provides value that raw
scipy does not.

## Key findings

### 1. Parameter agreement on NIST StRD datasets

On the Chwirut2 dataset (NIST StRD, easy difficulty) openfit and
`scipy.curve_fit` recover parameters that agree to better than `1e-6` relative
difference.  Both match NIST certified values to within the expected numerical
tolerance.  This confirms that openfit does not alter the optimizer or introduce
numerical artifacts.

### 2. Overhead ratio

openfit's per-fit overhead relative to a bare `curve_fit` call is typically
**less than 2×** for small datasets (7–25 points).  The overhead comes from four
sources, none of which are in the optimizer:

| Source | What it does |
|--------|-------------|
| `initial_guess()` | Data-driven starting point computation |
| Weighting setup | Constructing and normalizing the weight array |
| `FitResult` construction | Packaging parameters, SEs, CIs, residuals |
| `FitSpec` SHA-256 | Computing the reproducibility manifest hash |

For maximum raw throughput on simple problems, use `scipy.optimize.curve_fit`
directly.  openfit's value is in correctness infrastructure, not speed.

### 3. Weighting: explicit vs silent

`scipy.curve_fit` defaults silently to uniform weighting unless the caller passes
a `sigma=` array.  On heteroscedastic data (e.g. ELISA assays with ~20% CV),
this produces biased EC50 estimates.  openfit requires an explicit `weights=`
argument — there is no silent default.  Passing `weights="1/y2"` is equivalent to
`sigma=y` in scipy but is enforced rather than optional.

## Feature comparison

| Feature | openfit | scipy.curve_fit |
|---------|---------|-----------------|
| Built-in domain models | Yes (29) | No |
| Smart `initial_guess()` | Yes | No (manual `p0`) |
| Explicit weighting | Yes (required) | Manual `sigma=` |
| Profile-likelihood CI | Yes | No |
| ROUT outlier detection | Yes | No |
| Global/shared fitting | Yes | No |
| Reproducibility manifest | Yes (`FitSpec`) | No |
| NIST validation published | Yes | Not published |
| Publication reports | HTML/PDF/DOCX | No |
| Optimizer core | scipy LM+TRF | scipy LM |
| Parameter agreement | Identical* | Reference |

\* When both converge; openfit adds TRF for bounded problems.

## Recommendation

- **For speed-critical loops over simple functions**: use `scipy.optimize.curve_fit`
  directly — it has lower per-call overhead and no dependencies beyond scipy.
- **For scientific workflows where reproducibility, weighting correctness, and
  statistical inference matter**: use openfit.  The overhead is modest and the
  correctness guarantees are substantial.

See [06_benchmark_vs_scipy.ipynb](../notebooks/06_benchmark_vs_scipy.ipynb) for
the full runnable comparison.
