"""GlobalFit: shared-parameter fitting across multiple datasets.

The key moat feature of openfit.  A single model is simultaneously fitted to
N datasets with some parameters forced to be equal across all datasets
(shared) and others estimated independently per dataset (local).

Reference
---------
Motulsky & Christopoulos, "Fitting Models to Biological Data using Linear
and Nonlinear Regression: A Practical Guide to Curve Fitting", Oxford
University Press, 2004.  Chapter 35: Fitting data from several experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.optimize
import scipy.stats

from openfit.models import get_model
from openfit.models.base import BaseModel
from openfit.spec import FitSpec, build_spec
from openfit.weighting import WeightScheme, apply_weights, parse_weight_scheme

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FTestSharing:
    """F-test comparing global (shared) vs. fully independent per-dataset fits.

    Parameters
    ----------
    f_statistic : float
        Computed F ratio.
    df_numerator : int
        Constraints imposed by sharing = n_shared * (n_datasets - 1).
    df_denominator : int
        Degrees of freedom for the full (independent) model
        = n_total - n_datasets * (n_shared + n_local).
    p_value : float
        Right-tail p-value from the F distribution.
    rss_shared : float
        Total weighted RSS from the global fit (constrained/restricted model).
    rss_independent : float
        Total weighted RSS when each dataset is fitted fully independently.
    sharing_justified : bool
        True when p_value > 0.05 (sharing does not significantly worsen fit).
    """

    f_statistic: float
    df_numerator: int
    df_denominator: int
    p_value: float
    rss_shared: float
    rss_independent: float
    sharing_justified: bool


@dataclass
class GlobalFitResult:
    """Result of a global shared-parameter fit.

    Parameters
    ----------
    shared_params : dict[str, float]
        Fitted values for parameters shared across all datasets.
    local_params : list[dict[str, float]]
        Per-dataset fitted values for local parameters only.
    all_params_per_dataset : list[dict[str, float]]
        Full parameter dictionaries (shared + local) for each dataset,
        ready to pass to model.equation().
    rss_total : float
        Sum of weighted residual sum-of-squares across all datasets.
    rss_per_dataset : list[float]
        Weighted RSS for each individual dataset.
    r_squared_per_dataset : list[float]
        Weighted R^2 for each individual dataset
        (1 - SS_res_weighted / SS_tot_weighted).
    n_obs_total : int
        Total number of observations across all datasets.
    n_obs_per_dataset : list[int]
        Number of observations in each dataset.
    n_shared : int
        Number of shared parameters.
    n_local : int
        Number of local parameters per dataset.
    n_datasets : int
        Number of datasets.
    model_id : str
        Identifier of the fitted model.
    shared_param_names : list[str]
        Names of shared parameters (in model.param_names order).
    local_param_names : list[str]
        Names of local parameters (in model.param_names order).
    weight_scheme : str
        Canonical weight-scheme string (e.g. "1/y2").
    f_test_sharing : FTestSharing | None
        F-test result; None if run_f_test=False or the test cannot be
        computed (e.g. zero shared params, non-positive df_denominator).
    specs : list[FitSpec]
        One reproducibility manifest per dataset.
    """

    shared_params: dict[str, float]
    local_params: list[dict[str, float]]
    all_params_per_dataset: list[dict[str, float]]
    rss_total: float
    rss_per_dataset: list[float]
    r_squared_per_dataset: list[float]
    n_obs_total: int
    n_obs_per_dataset: list[int]
    n_shared: int
    n_local: int
    n_datasets: int
    model_id: str
    shared_param_names: list[str]
    local_param_names: list[str]
    weight_scheme: str
    f_test_sharing: FTestSharing | None
    specs: list[FitSpec]
    
    # Private fields for report generation and plotting
    _datasets: list[tuple[np.ndarray, np.ndarray]] = field(repr=False, compare=False, default_factory=list)
    _model: BaseModel = field(repr=False, compare=False, default=None)
    _weights_per_ds: list[np.ndarray] = field(repr=False, compare=False, default_factory=list)

    # ------------------------------------------------------------------
    # report()
    # ------------------------------------------------------------------

    def report(self, path: str, fmt: str = "html") -> None:
        """Write a global fit report to a file.

        Parameters
        ----------
        path : str
            Output file path (e.g. "global_fit_result.html").
        fmt : str
            Output format: "html" or "markdown". Default "html".

        Raises
        ------
        ValueError
            If fmt is not "html" or "markdown".
        """
        _valid = {"html", "markdown"}
        if fmt not in _valid:
            raise ValueError(
                f"fmt must be one of {sorted(_valid)}, got {fmt!r}."
            )
        from openfit.report.global_fit import report_global_fit
        report_global_fit(self, path=path, fmt=fmt)

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return an ASCII summary of the global fit results.

        Returns
        -------
        str
            Multiline ASCII string suitable for print() or CLI output.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(f"GlobalFit  model={self.model_id!r}")
        lines.append(
            f"  datasets={self.n_datasets}  "
            f"n_obs_total={self.n_obs_total}  "
            f"weights={self.weight_scheme!r}"
        )
        lines.append("-" * 60)

        # Shared params
        lines.append("Shared parameters:")
        for name in self.shared_param_names:
            lines.append(f"  {name:<20s} = {self.shared_params[name]:.6g}")

        lines.append("-" * 60)

        # Per-dataset summary
        lines.append("Per-dataset local parameters and goodness-of-fit:")
        for i, (lp, rss, r2, n) in enumerate(
            zip(
                self.local_params,
                self.rss_per_dataset,
                self.r_squared_per_dataset,
                self.n_obs_per_dataset,
                strict=True,
            )
        ):
            lines.append(f"  Dataset {i}  (n={n}  RSS={rss:.4g}  R^2={r2:.4f})")
            for name in self.local_param_names:
                lines.append(f"    {name:<20s} = {lp[name]:.6g}")

        lines.append("-" * 60)
        lines.append(f"Total weighted RSS: {self.rss_total:.6g}")

        if self.f_test_sharing is not None:
            ft = self.f_test_sharing
            verdict = "sharing justified" if ft.sharing_justified else "sharing questionable"
            lines.append(
                f"F-test (sharing vs. independent): "
                f"F({ft.df_numerator},{ft.df_denominator})={ft.f_statistic:.4g}  "
                f"p={ft.p_value:.4g}  [{verdict}]"
            )
        else:
            lines.append("F-test: not computed")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_finite(x: np.ndarray, y: np.ndarray, index: int) -> None:
    """Raise ValueError if x or y contain NaN or Inf.

    Parameters
    ----------
    x : np.ndarray
        Independent-variable values for dataset *index*.
    y : np.ndarray
        Dependent-variable values for dataset *index*.
    index : int
        Dataset index used in the error message.

    Raises
    ------
    ValueError
        If x or y contain NaN or Inf values.
    """
    if not np.isfinite(x).all():
        raise ValueError(
            f"Dataset {index}: x contains NaN or Inf values. "
            "Clean the input data before fitting."
        )
    if not np.isfinite(y).all():
        raise ValueError(
            f"Dataset {index}: y contains NaN or Inf values. "
            "Clean the input data before fitting."
        )


def _weighted_r2(
    y: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute weighted R^2 for a single dataset.

    R^2 = 1 - SS_res_weighted / SS_tot_weighted

    SS_res_weighted = sum(w * (y - y_pred)^2)
    SS_tot_weighted = sum(w * (y - y_mean_weighted)^2)
    y_mean_weighted = sum(w * y) / sum(w)

    Per openfit correctness rule #9: for weighted fits, use weighted SS.

    Parameters
    ----------
    y : np.ndarray
        Observed values.
    y_pred : np.ndarray
        Model-predicted values.
    weights : np.ndarray
        Weight array (w_i = 1/sigma_i^2 or uniform).

    Returns
    -------
    float
        Weighted R^2 in (-inf, 1].
    """
    w_sum = float(np.sum(weights))
    y_mean = float(np.sum(weights * y) / w_sum)
    ss_res = float(np.sum(weights * (y - y_pred) ** 2))
    ss_tot = float(np.sum(weights * (y - y_mean) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _fit_single(
    model: BaseModel,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict[str, float], float]:
    """Fit *model* to a single dataset and return params and weighted RSS.

    This helper is intentionally self-contained (does not import openfit.fit)
    to avoid circular imports.  It mirrors the fitting logic used in the global
    residual function.

    Parameters
    ----------
    model : BaseModel
        Model instance to fit.
    x : np.ndarray
        Independent-variable values.
    y : np.ndarray
        Dependent-variable values.
    weights : np.ndarray
        Pre-computed weight array (same scheme used in the global fit).

    Returns
    -------
    tuple[dict[str, float], float]
        ``(fitted_params_dict, weighted_rss)`` where weighted_rss is the
        minimised sum(w * (y - y_pred)^2).

    Raises
    ------
    RuntimeError
        If scipy.optimize.least_squares fails to produce a finite result.
    """
    p0_dict = model.initial_guess(x, y)
    # Preserve model.param_names order.
    param_order = model.param_names
    p0 = np.array([p0_dict[n] for n in param_order], dtype=float)

    lb_list, ub_list = model.bounds()
    lb = np.array(lb_list, dtype=float)
    ub = np.array(ub_list, dtype=float)

    sqrt_w = np.sqrt(weights)

    def _resid(p: np.ndarray) -> np.ndarray:
        params = dict(zip(param_order, p, strict=True))
        y_pred = model.equation(x, **params)
        return np.asarray(sqrt_w * (y - y_pred))

    result = scipy.optimize.least_squares(
        _resid,
        p0,
        bounds=(lb, ub),
        method="trf",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=10_000 * len(p0),
    )

    fitted_params = dict(zip(param_order, result.x, strict=True))
    # Weighted RSS = sum of squared weighted residuals.
    weighted_rss = float(np.sum(result.fun ** 2))
    return fitted_params, weighted_rss


# ---------------------------------------------------------------------------
# GlobalFit
# ---------------------------------------------------------------------------


class GlobalFit:
    """Shared-parameter fitting across multiple datasets.

    A single model is fitted simultaneously to N datasets.  Some parameters
    are constrained to be equal across all datasets (shared); others are
    estimated independently per dataset (local).  The optimizer minimises the
    joint weighted residual sum-of-squares.

    Parameters
    ----------
    datasets : list[tuple[array-like, array-like]]
        List of (x, y) pairs.  Each element must be a pair of 1-D
        array-like objects of the same length.  At least 2 datasets required.
    model : str | BaseModel
        Model identifier string or BaseModel instance.  The same model is
        applied to every dataset.
    shared : list[str]
        Parameter names forced equal across all datasets.
        Must be a subset of ``model.param_names``.
    local : list[str]
        Parameter names estimated independently per dataset.
        Must be a subset of ``model.param_names``.
    weights : str | WeightScheme
        Weight scheme applied to all datasets.  Required.
        Accepted: "uniform", "1/y", "1/y2", "1/sd2", "poisson" and aliases.
    sd : list[np.ndarray | None] | None
        Per-dataset standard-deviation arrays.  Required only when
        ``weights="1/sd2"``.  Length must equal ``len(datasets)``; use
        ``None`` for individual datasets that have no SD data.
    run_f_test : bool
        If True (default), run the extra-sum-of-squares F-test comparing
        global (shared) to fully independent fits.  Adds compute time
        proportional to ``n_datasets`` single fits.
    random_seed : int
        Seed stored in FitSpec reproducibility manifests.  Default 0.

    Notes
    -----
    ``shared + local`` must together name every parameter in
    ``model.param_names`` exactly once.

    Examples
    --------
    >>> import numpy as np
    >>> from openfit import GlobalFit
    >>> datasets = [
    ...     (np.array([0.1, 1.0, 10.0, 100.0]), np.array([2.0, 30.0, 80.0, 95.0])),
    ...     (np.array([0.1, 1.0, 10.0, 100.0]), np.array([1.0, 15.0, 60.0, 90.0])),
    ... ]
    >>> result = GlobalFit(
    ...     datasets,
    ...     "hill4p",
    ...     shared=["HillSlope", "Bottom"],
    ...     local=["EC50", "Top"],
    ...     weights="1/y2",
    ... ).run()
    >>> print(result.summary())
    """

    def __init__(
        self,
        datasets: list[tuple[Any, Any]],
        model: str | BaseModel,
        shared: list[str],
        local: list[str],
        *,
        weights: str | WeightScheme,
        sd: list[np.ndarray | None] | None = None,
        run_f_test: bool = True,
        random_seed: int = 0,
    ) -> None:
        # Resolve model.
        if isinstance(model, str):
            self._model: BaseModel = get_model(model)
        else:
            self._model = model

        # Store raw inputs; validation happens in run().
        self._datasets_raw = datasets
        self._shared_raw = list(shared)
        self._local_raw = list(local)
        self._weights_raw = weights
        self._sd = sd if sd is not None else [None] * len(datasets)
        self._run_f_test = run_f_test
        self._random_seed = random_seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> GlobalFitResult:
        """Execute the global fit.

        Returns
        -------
        GlobalFitResult

        Raises
        ------
        ValueError
            If fewer than 2 datasets are provided.
            If shared + local does not exactly cover model.param_names.
            If shared and local lists overlap.
            If any dataset x or y array contains NaN or Inf.
        """
        model = self._model
        all_param_names: list[str] = model.param_names

        # ---- 1. Input validation ----------------------------------------

        n_datasets = len(self._datasets_raw)
        if n_datasets < 2:
            raise ValueError(
                f"GlobalFit requires at least 2 datasets; got {n_datasets}."
            )

        # Shared/local coverage check.
        shared_set = set(self._shared_raw)
        local_set = set(self._local_raw)
        overlap = shared_set & local_set
        if overlap:
            raise ValueError(
                f"Parameters appear in both shared and local: {sorted(overlap)}. "
                "Each parameter must appear in exactly one list."
            )
        all_set = set(all_param_names)
        covered = shared_set | local_set
        missing = all_set - covered
        extra = covered - all_set
        if missing:
            raise ValueError(
                f"Parameters not assigned to shared or local: {sorted(missing)}. "
                "Every model parameter must be listed in exactly one of "
                "shared or local."
            )
        if extra:
            raise ValueError(
                f"Unknown parameter names in shared/local: {sorted(extra)}. "
                f"Model '{model.model_id}' has params: {all_param_names}."
            )

        # Canonicalize order: follow model.param_names.
        shared_names: list[str] = [p for p in all_param_names if p in shared_set]
        local_names: list[str] = [p for p in all_param_names if p in local_set]
        n_shared = len(shared_names)
        n_local = len(local_names)

        # Pre-process datasets into numpy arrays and validate.
        datasets: list[tuple[np.ndarray, np.ndarray]] = []
        for i, (x_raw, y_raw) in enumerate(self._datasets_raw):
            x = np.asarray(x_raw, dtype=float)
            y = np.asarray(y_raw, dtype=float)
            _check_finite(x, y, i)
            datasets.append((x, y))

        # Weight arrays per dataset.
        weight_scheme = parse_weight_scheme(self._weights_raw)
        weight_label = weight_scheme.value

        if len(self._sd) != n_datasets:
            raise ValueError(
                f"sd list length ({len(self._sd)}) must equal number of "
                f"datasets ({n_datasets})."
            )

        weights_per_ds: list[np.ndarray] = []
        for i, (_x, y) in enumerate(datasets):
            sd_i = self._sd[i]
            if sd_i is not None:
                sd_i = np.asarray(sd_i, dtype=float)
            weights_per_ds.append(apply_weights(y, weight_scheme, sd=sd_i))

        n_obs_per_dataset = [len(y) for _, y in datasets]
        n_total = sum(n_obs_per_dataset)

        # ---- 2. Build flat parameter vector layout ----------------------
        # Layout: [shared_0, ..., shared_{ns-1},
        #          local_ds0_0, ..., local_ds0_{nl-1},
        #          local_ds1_0, ..., local_ds1_{nl-1},
        #          ...]
        # total length = n_shared + n_datasets * n_local

        # ---- 3. Build initial guess -------------------------------------
        # Shared: average per-dataset initial_guess across all datasets.
        # Local: per-dataset initial_guess.

        per_ds_guesses: list[dict[str, float]] = [
            model.initial_guess(x, y) for x, y in datasets
        ]

        p0_shared: list[float] = []
        for name in shared_names:
            avg = float(np.mean([g[name] for g in per_ds_guesses]))
            p0_shared.append(avg)

        p0_local_blocks: list[list[float]] = []
        for g in per_ds_guesses:
            block = [g[name] for name in local_names]
            p0_local_blocks.append(block)

        p0_blocks: list[list[float]] = [p0_shared] + p0_local_blocks
        p0: np.ndarray = np.concatenate(p0_blocks)

        # ---- 4. Build bounds --------------------------------------------
        lb_full_model, ub_full_model = model.bounds()
        # Map param name -> bound index.
        param_index = {name: i for i, name in enumerate(all_param_names)}

        lb_shared = [lb_full_model[param_index[n]] for n in shared_names]
        ub_shared = [ub_full_model[param_index[n]] for n in shared_names]

        lb_local_one = [lb_full_model[param_index[n]] for n in local_names]
        ub_local_one = [ub_full_model[param_index[n]] for n in local_names]

        lb_all: list[float] = lb_shared + lb_local_one * n_datasets
        ub_all: list[float] = ub_shared + ub_local_one * n_datasets

        # ---- 5. Build joint residual function ---------------------------
        sqrt_w_per_ds = [np.sqrt(w) for w in weights_per_ds]

        def _joint_residuals(p_flat: np.ndarray) -> np.ndarray:
            shared_vals = dict(zip(shared_names, p_flat[:n_shared], strict=True))
            chunks: list[np.ndarray] = []
            for i, (xi, yi) in enumerate(datasets):
                offset = n_shared + i * n_local
                local_vals = dict(
                    zip(local_names, p_flat[offset : offset + n_local], strict=True)
                )
                params = {**shared_vals, **local_vals}
                y_pred = model.equation(xi, **params)
                r = sqrt_w_per_ds[i] * (yi - y_pred)
                chunks.append(r)
            return np.concatenate(chunks)

        # ---- 6. Optimize ------------------------------------------------
        opt = scipy.optimize.least_squares(
            _joint_residuals,
            p0,
            bounds=(lb_all, ub_all),
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=10_000 * len(p0),
        )

        p_opt = opt.x

        # Unpack fitted values.
        fitted_shared: dict[str, float] = dict(
            zip(shared_names, p_opt[:n_shared].tolist(), strict=True)
        )
        fitted_local: list[dict[str, float]] = []
        all_params_per_ds: list[dict[str, float]] = []
        for i in range(n_datasets):
            offset = n_shared + i * n_local
            lp = dict(
                zip(local_names, p_opt[offset : offset + n_local].tolist(), strict=True)
            )
            fitted_local.append(lp)
            all_params_per_ds.append({**fitted_shared, **lp})

        # ---- 7. Per-dataset RSS and R^2 ---------------------------------
        rss_per_ds: list[float] = []
        r2_per_ds: list[float] = []
        for i, (xi, yi) in enumerate(datasets):
            y_pred_i = model.equation(xi, **all_params_per_ds[i])
            wi = weights_per_ds[i]
            rss_i = float(np.sum(wi * (yi - y_pred_i) ** 2))
            rss_per_ds.append(rss_i)
            r2_per_ds.append(_weighted_r2(yi, y_pred_i, wi))

        rss_shared_total = float(sum(rss_per_ds))

        # ---- 8. Optional F-test -----------------------------------------
        f_test: FTestSharing | None = None

        if self._run_f_test and n_shared > 0:
            # Fit each dataset independently using the same weight scheme.
            rss_independent_total = 0.0
            for i, (xi, yi) in enumerate(datasets):
                _, rss_i_indep = _fit_single(
                    model, xi, yi, weights_per_ds[i]
                )
                rss_independent_total += rss_i_indep

            # Extra-sum-of-squares F-test.
            # df_num = number of constraints imposed by sharing
            #        = n_shared * (n_datasets - 1)
            #        (each dataset loses n_shared free parameters except one)
            # df_den = df of the full (independent) model
            #        = n_total - n_datasets * (n_shared + n_local)
            # NOTE: df_den uses the FULL model's parameter count (n_datasets
            # independent fits, each with n_shared + n_local free params).
            # Using the restricted model's df in the denominator would be a
            # statistical error because the denominator RSS is rss_independent.
            df_num = n_shared * (n_datasets - 1)
            df_den = n_total - n_datasets * (n_shared + n_local)

            if df_den > 0 and df_num > 0:
                rss_diff = rss_shared_total - rss_independent_total
                # Clamp to zero: floating-point noise can give tiny negatives.
                rss_diff = max(rss_diff, 0.0)
                f_stat = (rss_diff / df_num) / (rss_independent_total / df_den)
                p_val = float(scipy.stats.f.sf(f_stat, df_num, df_den))
                f_test = FTestSharing(
                    f_statistic=float(f_stat),
                    df_numerator=df_num,
                    df_denominator=df_den,
                    p_value=p_val,
                    rss_shared=rss_shared_total,
                    rss_independent=rss_independent_total,
                    sharing_justified=p_val > 0.05,
                )
            # If df_den <= 0 or df_num <= 0, the model is over-parameterised
            # relative to the data; leave f_test=None.

        # ---- 9. Build FitSpec per dataset --------------------------------
        specs: list[FitSpec] = []
        for i, (xi, yi) in enumerate(datasets):
            specs.append(
                build_spec(
                    model_id=model.model_id,
                    param_values=all_params_per_ds[i],
                    weights=weight_label,
                    x=xi,
                    y=yi,
                    random_seed=self._random_seed,
                )
            )

        # ---- 10. Assemble result -----------------------------------------
        return GlobalFitResult(
            shared_params=fitted_shared,
            local_params=fitted_local,
            all_params_per_dataset=all_params_per_ds,
            rss_total=rss_shared_total,
            rss_per_dataset=rss_per_ds,
            r_squared_per_dataset=r2_per_ds,
            n_obs_total=n_total,
            n_obs_per_dataset=n_obs_per_dataset,
            n_shared=n_shared,
            n_local=n_local,
            n_datasets=n_datasets,
            model_id=model.model_id,
            shared_param_names=shared_names,
            local_param_names=local_names,
            weight_scheme=weight_label,
            f_test_sharing=f_test,
            specs=specs,
            _datasets=datasets,
            _model=model,
            _weights_per_ds=weights_per_ds,
        )
