# Model Reference

openfit includes 29 built-in models across 8 families. Pass the model name string to `Fit()`, `GlobalFit()`, or `rout_outliers()`. Names are case-insensitive.

---

## Sigmoidal

Four-parameter and related sigmoidal equations for dose-response and activation data.

### `hill3p` — Three-Parameter Hill Equation

**Equation:** `Y = Top * X^n / (EC50^n + X^n)`

| Parameter | Description |
|---|---|
| `Top` | Upper asymptote (maximum response). |
| `EC50` | Concentration producing 50% of Top. |
| `HillSlope` | Hill coefficient (cooperativity / steepness). |

**Typical use:** Dose-response assays where the baseline is known to be zero and can be fixed rather than estimated. Simpler than 4PL when zero is anchored by experimental design.

---

### `hill4p` — Four-Parameter Logistic (4PL)

**Equation:** `Y = Bottom + (Top - Bottom) / (1 + (EC50/X)^HillSlope)`

| Parameter | Description |
|---|---|
| `Bottom` | Lower asymptote (baseline response). |
| `Top` | Upper asymptote (maximum response). |
| `EC50` | Concentration at the midpoint between Bottom and Top. |
| `HillSlope` | Hill slope. Positive for activation curves, negative for inhibition. |

**Typical use:** The standard model for most dose-response and inhibition assays. Use when neither the baseline nor the ceiling is known a priori.

---

### `hill5p` — Five-Parameter Logistic (5PL)

**Equation:** `Y = Bottom + (Top - Bottom) / (1 + (EC50/X)^HillSlope)^Asymm`

| Parameter | Description |
|---|---|
| `Bottom` | Lower asymptote. |
| `Top` | Upper asymptote. |
| `EC50` | Inflection point concentration. |
| `HillSlope` | Slope at the inflection point. |
| `HillAsym` | Asymmetry factor. `HillAsym = 1` recovers the symmetric 4PL. |

**Typical use:** Assays with visibly asymmetric sigmoidal responses — i.e. when the curve rises or falls more steeply on one side of the inflection than the other. Requires more data than 4PL to fit reliably.

---

### `boltzmann` — Boltzmann Sigmoid

**Equation:** `Y = Bottom + (Top - Bottom) / (1 + exp((V50 - X) / Slope))`

| Parameter | Description |
|---|---|
| `Bottom` | Lower asymptote. |
| `Top` | Upper asymptote. |
| `V50` | Midpoint value (e.g. voltage at half-maximal activation). |
| `Slope` | Slope factor (in X units). |

**Typical use:** Voltage-gated ion channel activation/inactivation curves; thermodynamic unfolding transitions where the X axis is temperature or denaturant concentration.

---

## Exponential

### `monoexp` — Monoexponential Decay with Offset

**Equation:** `Y = A * exp(-k * X) + C`

| Parameter | Description |
|---|---|
| `A` | Amplitude of the exponential component. |
| `k` | Rate constant (units: 1/X). |
| `C` | Plateau (offset as X → ∞). |

**Typical use:** Radioactive decay, fluorescence lifetime, single-phase pharmacokinetic elimination.

---

### `biexp` — Biexponential Decay

**Equation:** `Y = A1 * exp(-k1 * X) + A2 * exp(-k2 * X) + C`

| Parameter | Description |
|---|---|
| `A1` | Amplitude of the fast component. |
| `k1` | Rate constant of the fast component. |
| `A2` | Amplitude of the slow component. |
| `k2` | Rate constant of the slow component. |
| `C` | Baseline offset. |

**Typical use:** Two-compartment pharmacokinetic models; fluorescence decay with two lifetime components.

---

### `expgrowth` — Exponential Growth

**Equation:** `Y = A * exp(k * X)`

| Parameter | Description |
|---|---|
| `A` | Value at X = 0. |
| `k` | Growth rate constant. |

**Typical use:** Bacterial growth in log phase; compound interest; early-phase epidemic growth.

---

### `expplateau` — Exponential Approach to Plateau

**Equation:** `Y = A * (1 - exp(-k * X))`

| Parameter | Description |
|---|---|
| `A` | Plateau (maximum Y as X → ∞). |
| `k` | Rate constant governing the approach to plateau. |

**Typical use:** Receptor binding saturation over time; capacitor charging; enzyme inactivation assays.

---

### `expdecay` — Simple Exponential Decay

**Equation:** `Y = A * exp(-k * X)`

| Parameter | Description |
|---|---|
| `A` | Value at X = 0. |
| `k` | Decay rate constant. |

**Typical use:** Zero-baseline monoexponential decay. Equivalent to `monoexp` with C fixed at 0.

---

## Enzyme Kinetics

### `michaelismenten` — Michaelis-Menten

**Equation:** `V = Vmax * S / (Km + S)`

| Parameter | Description |
|---|---|
| `Vmax` | Maximum reaction velocity. |
| `Km` | Substrate concentration at half-maximal velocity (Michaelis constant). |

**Typical use:** Enzyme velocity vs. substrate concentration for enzymes following standard Michaelis-Menten kinetics.

---

### `substrateinhibition` — Substrate Inhibition

**Equation:** `V = Vmax * S / (Km + S * (1 + S/Ki))`

| Parameter | Description |
|---|---|
| `Vmax` | Maximum velocity (theoretical, in absence of inhibition). |
| `Km` | Michaelis constant. |
| `Ki` | Substrate inhibition constant. |

**Typical use:** Enzymes that are inhibited at high substrate concentrations — produces a curve that rises then falls as substrate increases.

---

### `allosteric` — Allosteric (Hill) Enzyme Kinetics

**Equation:** `V = Vmax * S^n / (K^n + S^n)`

| Parameter | Description |
|---|---|
| `Vmax` | Maximum reaction velocity. |
| `K` | Substrate concentration at half-maximal velocity. |
| `n` | Hill coefficient. `n > 1` indicates positive cooperativity; `n < 1` indicates negative cooperativity. |

**Typical use:** Cooperative enzymes such as haemoglobin oxygen binding; allosteric proteins where substrate binding at one site influences other sites.

---

## Growth

### `logistic3p` — Three-Parameter Logistic

**Equation:** `Y = Asym / (1 + exp(-k * (X - xmid)))`

| Parameter | Description |
|---|---|
| `Asym` | Upper asymptote (carrying capacity). |
| `k` | Growth rate constant. |
| `xmid` | X value at the inflection point (half-maximal growth). |

**Typical use:** Bacterial or cell population growth with a known zero lower baseline.

---

### `logistic4p` — Four-Parameter Logistic Growth

**Equation:** `Y = Bottom + (Top - Bottom) / (1 + exp(-k * (X - xmid)))`

| Parameter | Description |
|---|---|
| `Bottom` | Lower asymptote. |
| `Top` | Upper asymptote (carrying capacity). |
| `k` | Growth rate constant. |
| `xmid` | X value at the inflection point. |

**Typical use:** Population growth where the initial value is not zero; tumour growth curves.

---

### `gompertz` — Gompertz Growth

**Equation:** `Y = Asym * exp(-exp(-k * (X - xmid)))`

| Parameter | Description |
|---|---|
| `Asym` | Upper asymptote. |
| `k` | Growth rate constant. |
| `xmid` | X value at the inflection point. |

**Typical use:** Tumour growth; actuarial mortality curves; product adoption curves. Asymmetric by construction — rises more steeply initially than it approaches the asymptote.

### `gompertz_asym` — Asymmetric Gompertz

**Equation:**  
`Y = K * exp(-exp(-r_left * (X - t_inf)))` for `X <= t_inf`  
`Y = K * exp(-exp(-r_right * (X - t_inf)))` for `X > t_inf`  

| Parameter | Description |
|---|---|
| `K` | Asymptotic maximum (carrying capacity). |
| `r_left` | Growth rate constant for the left side ($X \le t_{\text{inf}}$). |
| `r_right` | Growth rate constant for the right side ($X > t_{\text{inf}}$). |
| `t_inf` | Inflection point (time of maximum growth rate). |

**Typical use:** Growth processes with different growth rates before and after reaching the inflection point.

---

### `richards` — Richards / Generalised Logistic

**Equation:** `Y = Asym / (1 + exp(-k * (X - xmid)))^(1/d)`

| Parameter | Description |
|---|---|
| `Asym` | Upper asymptote. |
| `k` | Growth rate constant. |
| `xmid` | Inflection point. |
| `d` | Shape parameter. `d = 1` recovers the standard logistic. |

**Typical use:** The most flexible of the growth models — subsumes logistic and Gompertz as special cases. Use when neither standard model fits well.

---

## Gaussian

### `gaussian` — Gaussian (Normal) Peak

**Equation:** `Y = A * exp(-(X - mu)^2 / (2 * sigma^2))`

| Parameter | Description |
|---|---|
| `A` | Peak amplitude. |
| `mu` | Peak centre (mean). |
| `sigma` | Standard deviation (controls width). |

**Typical use:** Spectroscopic peaks; elution peaks in chromatography; normally distributed data.

---

### `bigaussian` — Sum of Two Gaussians

**Equation:** `Y = A1 * exp(-(X - mu1)^2 / (2 * sigma1^2)) + A2 * exp(-(X - mu2)^2 / (2 * sigma2^2))`

| Parameter | Description |
|---|---|
| `A1` | Amplitude of first peak. |
| `mu1` | Centre of first peak. |
| `sigma1` | Width of first peak. |
| `A2` | Amplitude of second peak. |
| `mu2` | Centre of second peak. |
| `sigma2` | Width of second peak. |

**Typical use:** Overlapping spectroscopic or chromatographic peaks; bimodal distributions.

---

### `lorentzian` — Lorentzian (Cauchy) Peak

**Equation:** `Y = A / (1 + ((X - x0) / gamma)^2)`

| Parameter | Description |
|---|---|
| `A` | Peak amplitude. |
| `x0` | Peak centre. |
| `gamma` | Half-width at half-maximum (HWHM). |

**Typical use:** Spectroscopic line shapes where natural linewidth dominates (NMR, Raman, Mössbauer spectroscopy); any system with a Cauchy distribution of frequencies.

---

## Polynomial

Six polynomial models with analytic Jacobians for fast, stable fitting.

| Model | Equation | Degree |
|---|---|---|
| `poly1` | `Y = a0 + a1*X` | 1 (linear) |
| `poly2` | `Y = a0 + a1*X + a2*X^2` | 2 (quadratic) |
| `poly3` | `Y = a0 + a1*X + a2*X^2 + a3*X^3` | 3 (cubic) |
| `poly4` | `Y = a0 + a1*X + ... + a4*X^4` | 4 |
| `poly5` | `Y = a0 + a1*X + ... + a5*X^5` | 5 |
| `poly6` | `Y = a0 + a1*X + ... + a6*X^6` | 6 |

Parameters are named `a0`, `a1`, ..., `an` (the intercept through the highest-degree coefficient).

**Typical use:** Empirical calibration curves where a mechanistic model is not needed; detector response linearisation.

---

## Binding

### `onesitebinding` — One-Site Binding (Langmuir Isotherm)

**Equation:** `B = Bmax * L / (Kd + L)`

| Parameter | Description |
|---|---|
| `Bmax` | Maximum specific binding (at saturation). |
| `Kd` | Equilibrium dissociation constant (ligand concentration at half-maximal binding). |

**Typical use:** Radioligand binding assays; surface plasmon resonance single-analyte experiments; adsorption isotherms.

---

### `twositebinding` — Two-Site Binding

**Equation:** `B = Bmax1 * L / (Kd1 + L) + Bmax2 * L / (Kd2 + L)`

| Parameter | Description |
|---|---|
| `Bmax1` | Maximum binding at site 1. |
| `Kd1` | Dissociation constant for site 1. |
| `Bmax2` | Maximum binding at site 2. |
| `Kd2` | Dissociation constant for site 2. |

**Typical use:** Receptors or proteins with two distinct binding sites of different affinity.

---

### `competitivebinding` — Competitive Binding (Cheng-Prusoff)

**Equation:** `B = Bmax * L / (Kd_app + L)` where `Kd_app = Kd * (1 + I/Ki)`

| Parameter | Description |
|---|---|
| `Bmax` | Maximum specific binding. |
| `Kd` | Dissociation constant of the labelled ligand in the absence of inhibitor. |
| `Ki` | Inhibition constant of the unlabelled competitor. |

**Typical use:** Competition binding assays. The Cheng-Prusoff relationship converts the apparent IC50 measured in a radioligand assay into the true Ki for the competitor.

---

## Custom Models

If no built-in model matches your data, wrap any Python callable with `CustomModel`:

```python
from openfit import Fit
from openfit.models import CustomModel

def my_equation(x, A, tau, offset):
    return A * (1 - np.exp(-x / tau)) + offset

model = CustomModel(my_equation)
result = Fit(model, x, y, weights="1/y2").run()
```

Parameter names are inferred directly from the function signature. All `Fit()` features — bootstrap CI, ROUT outlier detection, FitSpec, report generation — work with custom models.

---

## Model Registry Functions

```python
from openfit.models import list_models, get_model, register_model
```

### `list_models()`

Returns a list of all registered model names. Includes all 29 built-ins plus any custom models you have registered in the current session.

```python
names = list_models()
# ["hill3p", "hill4p", "hill5p", "boltzmann", "monoexp", ...]
```

### `get_model(name)`

Retrieves a model object by name. Lookup is case-insensitive.

```python
model = get_model("Hill4P")   # same as get_model("hill4p")
print(model.param_names)      # ["Bottom", "Top", "EC50", "HillSlope"]
```

### `register_model(name, model)`

Adds a `CustomModel` instance (or any compatible model object) to the registry under the given name. Once registered, the name can be used everywhere a built-in name is accepted.

```python
from openfit.models import CustomModel, register_model

def biphasic(x, A1, k1, A2, k2):
    return A1 * np.exp(-k1 * x) + A2 * np.exp(-k2 * x)

register_model("biphasic_decay", CustomModel(biphasic))

# Now usable by name
result = Fit("biphasic_decay", x, y).run()
```

Registered models persist for the lifetime of the Python session. To make a custom model permanently available, call `register_model` in your project's initialisation code or in a small helper module that you import at startup.
