---
title: "openfit: Reproducible nonlinear curve fitting with publication-quality reports"
tags:
  - Python
  - nonlinear regression
  - curve fitting
  - dose-response
  - reproducibility
  - pharmacology
  - biochemistry
  - NIST validation
authors:
  - name: Priyam Thakar
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 8 June 2026
bibliography: paper.bib
---

# Summary

`openfit` is a Python library for nonlinear curve fitting that combines a
validated, scipy-based engine with built-in domain models, explicit weighting,
statistical inference, and a reproducibility manifest. It provides 29 built-in
model equations across sigmoidal, exponential, enzyme-kinetic, growth,
Gaussian, polynomial, and binding families. Every fit emits a `FitSpec` — a
JSON record containing the model identifier, fitted parameters, weight scheme,
SHA-256 hash of the input data, and exact software versions — so that any
reader with the spec and the data can reproduce the identical numerical result.

# Statement of Need

Nonlinear curve fitting is a routine step in biochemistry, pharmacology, and
the life sciences, yet the current Python ecosystem leaves a significant gap
between low-level optimizers and the reproducibility standards expected in
peer-reviewed research.

`scipy.optimize.curve_fit` [@virtanen2020scipy] and `lmfit`
[@newville2014lmfit] are general-purpose tools: they expose the optimizer but
require the researcher to supply model equations, initial guesses, weight
schemes, and post-fit statistics individually. GraphPad Prism supplies these
conveniences but is closed-source, platform-locked, and produces results that
cannot be reproduced from a script alone. Neither category addresses the
reproducibility gap: there is no standard mechanism for a researcher to
communicate, alongside a figure, the exact model, weights, software version,
and random seed that produced it.

`openfit` is designed to close these gaps simultaneously:

1. **Domain models with data-driven initial guesses.** Four-parameter logistic
   (4PL/Hill4P), five-parameter logistic (5PL/Hill5P), Michaelis-Menten,
   one-site binding, competitive binding, Gompertz growth, and 23 further
   equations ship with per-model heuristics that produce reliable starting
   values from the data itself, eliminating the most common source of
   optimizer failure.

2. **Explicit, validated weighting.** The `weights=` argument is required;
   there is no silent uniform-weighting default. The five supported schemes
   (`uniform`, `1/y`, `1/y2`, `1/sd2`, `poisson`) cover the most common
   heteroscedastic patterns in assay data.

3. **ROUT outlier detection.** `openfit` provides the first Python
   implementation of the ROUT algorithm [@motulsky2006detecting], which
   combines a Lorentzian robust fit with Benjamini-Hochberg false discovery
   rate control to identify outliers without inflating the Type I error rate.
   Prior to `openfit`, ROUT was available only in GraphPad Prism and in
   unpublished R scripts.

4. **Global (shared-parameter) fitting.** `GlobalFit` accepts multiple
   datasets and a declarative list of shared and local parameters, performs a
   joint optimization, and reports an F-test for whether sharing is
   statistically justified — following the approach described in
   @motulsky2003fitting Chapter 25.

5. **Reproducibility manifest.** `FitSpec` records the data SHA-256 hash,
   fitted parameter values, weight scheme, random seed, and
   `openfit`/`scipy`/`numpy` version strings in a machine-readable JSON
   document. This is intended to accompany supplementary materials and to
   enable automated reproduction checks.

6. **Publication-ready reports.** `report_fit()` generates self-contained HTML,
   Markdown, PDF, or Word documents containing the fit plot, residual plot,
   Q-Q plot, parameter table, and full statistical output.

# Validation

`openfit` is validated against the NIST Statistical Reference Datasets (StRD)
for nonlinear regression [@nist_strd]. All 27 certified datasets are exercised
under both NIST Start I (far) and Start II (close) initial conditions; fitted
parameters agree with NIST 128-bit certified values to at least 6 significant
figures in all cases. RSS matches certified values to 6 significant figures for
26 of 27 datasets (Lanczos1 is excluded: the certified RSS of 1.43 × 10⁻²⁵ is
below the 64-bit floating-point floor, though its parameter tests pass).

Hill4P is additionally cross-validated against the R `drda` package
[@marasini2023drda] at the coefficient level: fitted parameters on the
`voropm2` dataset agree within a relative tolerance of 10⁻³ and RSS within
10⁻⁶.

Profile-likelihood confidence intervals are validated against the analytical
example in @motulsky2003fitting Table 22.1; the ROUT implementation is
validated against the reference dataset in @motulsky2006detecting Figure 2 and
Table 1; and the F-test model comparison is validated against the formula in
@motulsky2003fitting Chapter 22. The full validation suite comprises 379 tests
(374 passing, 5 skipped) covering the core engine, all model families, all
statistical inference methods, and all report formats.

# Comparison with Existing Tools

| Feature | openfit | scipy `curve_fit` | lmfit |
|---|---|---|---|
| 4PL/5PL built-in | Yes | No | No |
| Smart initial guesses | Yes (per model) | No | Partial |
| Explicit weighting | Yes (required) | Manual | Manual |
| Profile-likelihood CI | Yes | No | Yes |
| ROUT outlier detection | Yes | No | No |
| Global/shared fitting | Yes (declarative) | No | Manual |
| Reproducibility manifest | Yes (unique) | No | No |
| NIST StRD validation | Published | Not published | Not published |
| Publication reports | HTML/MD/PDF/DOCX | No | No |

# Acknowledgements

The NIST Statistical Reference Datasets were produced by the Statistical
Engineering Division of the National Institute of Standards and Technology and
are in the public domain. The ROUT algorithm is due to @motulsky2006detecting.
The `drda` R package cross-validation data are from @marasini2023drda.

# References
