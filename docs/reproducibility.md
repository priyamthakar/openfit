# Reproducibility and FitSpec

## The Problem

A figure in a paper shows a fitted curve. The methods section says "data were fitted with a four-parameter logistic." You want to reproduce that curve exactly. You cannot — unless you also know:

- Which software and version was used.
- Which weight scheme was applied (`"1/y2"`? `"uniform"`? something else?).
- What initial parameter guesses were supplied or auto-generated.
- What random seed was used to generate bootstrap confidence intervals.

Without this information, even with the same data and the same model, different fitting software — or the same software at a different version — will produce numerically different parameter estimates. openfit solves this with **FitSpec**.

---

## What Is FitSpec?

FitSpec is a JSON reproducibility manifest that openfit saves alongside every fit result. It contains all information needed to verify and reproduce the exact fit.

| Field | Description |
|---|---|
| `model_id` | The model name string passed to `Fit()`, e.g. `"hill4p"`. |
| `param_values` | The exact fitted parameter values, serialised using lossless `repr()` so no precision is lost in the JSON round-trip. |
| `weight_scheme` | The weight scheme string, e.g. `"1/y2"`. |
| `data_hash` | SHA-256 hash of the input `x` and `y` arrays, computed over their float64 little-endian byte representations. Used to verify that the FitSpec corresponds to the correct dataset. |
| `openfit_version` | The openfit version at the time the fit was run (e.g. `"0.1.2"`). |
| `scipy_version` | The scipy version at the time the fit was run. |
| `numpy_version` | The numpy version at the time the fit was run. |
| `random_seed` | The integer seed passed to the bootstrap CI sampler. |

---

## Saving and Loading a FitSpec

```python
from openfit import Fit, FitSpec

result = Fit("hill4p", x, y, weights="1/y2").run()

# Save the FitSpec to disk
result.spec.to_json("my_fit.spec.json")

# Load and inspect a saved FitSpec
spec = FitSpec.from_json(open("my_fit.spec.json").read())
print(spec.model_id)       # "hill4p"
print(spec.param_values)   # {"Bottom": 0.47, "Top": 99.8, "EC50": 3.21, "HillSlope": 1.14}
print(spec.data_hash)      # "a3f8c2..." — compare against your data file's hash
```

`result.spec` is a `FitSpec` instance available on any `FitResult` returned by `Fit.run()`. If you run bootstrap CI, the random seed is recorded automatically.

### Verifying a Data Hash

To confirm that a deposited FitSpec corresponds to a particular data file, compute the hash of your data arrays and compare:

```python
import hashlib
import numpy as np
from openfit import FitSpec

# Load your data
x = np.array([...], dtype=np.float64)
y = np.array([...], dtype=np.float64)

# Compute the hash the same way openfit does
raw = x.astype("<f8").tobytes() + y.astype("<f8").tobytes()
computed_hash = hashlib.sha256(raw).hexdigest()

# Load and compare
spec = FitSpec.from_json(open("my_fit.spec.json").read())
assert computed_hash == spec.data_hash, "Data does not match FitSpec"
```

If the hashes match, you have confirmed that this FitSpec was produced from exactly these data.

---

## Using FitSpec in a Manuscript

Include the `.spec.json` file as supplementary data alongside the manuscript. Readers and reviewers can then:

1. Download the deposited dataset and the `.spec.json`.
2. Verify the `data_hash` against the dataset to confirm the FitSpec belongs to the deposited data.
3. Confirm that the reported parameters in the paper match `param_values` in the FitSpec.
4. Install the recorded software versions (see `openfit_version`, `scipy_version`, `numpy_version`) and reproduce the fit from scratch to obtain numerically identical results.

This workflow closes the reproducibility gap for fitted-curve figures.

---

## Limitations

FitSpec guarantees **exact numerical reproducibility** only when the identical scipy and numpy versions are used. This is the reason version pins are recorded.

Minor numerical differences in parameter estimates can occur across scipy major versions because the optimizer implementation (Levenberg-Marquardt, Trust Region Reflective) may change in ways that affect convergence paths. The differences are typically negligible in practice (well below the confidence interval width), but they are not zero. The version pins document this so that future readers know which version to install for exact reproduction.

openfit does not attempt to paper over these differences by locking internal optimizer state — doing so would create a false sense of precision. The honest representation is: same versions → identical results; different versions → results within normal numerical tolerance.

---

> **Tip:** Deposit both the raw data CSV and the `.spec.json` file in your data repository (Zenodo, Figshare, OSF, or equivalent) alongside the manuscript. Many journals now require this; even when not required, it is good scientific practice and costs nothing to do.
