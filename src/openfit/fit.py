# src/openfit/fit.py
"""Fit: main entry point for nonlinear least-squares fitting in openfit."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy.linalg
import scipy.optimize

from openfit.models import get_model
from openfit.models.base import BaseModel
from openfit.results import FitResult
from openfit.spec import build_spec
from openfit.weighting import WeightScheme, apply_weights, parse_weight_scheme


class Fit:
    """Configure and run a nonlinear least-squares fit.

    Parameters
    ----------
    model : str | BaseModel
        Model identifier (e.g. "hill4p") or a BaseModel instance.
    x : array-like
        Independent variable values.
    y : array-like
        Observed response values.
    weights : str | WeightScheme
        Weight scheme.  One of: "uniform", "1/y", "1/y2", "1/sd2", "poisson".
        This argument is REQUIRED -- there is no silent default (CLAUDE.md rule 1).
    sd : array-like | None
        Standard deviations per observation.  Required only when weights="1/sd2".
    method : str | None
        Optimization method: "lm" (Levenberg-Marquardt) or "trf"
        (Trust Region Reflective).  Default: "lm" when all bounds are infinite,
        "trf" automatically when any bound is finite.
    p0 : dict[str, float] | None
        Override initial parameter guesses.  None uses model.initial_guess(x, y).
    bounds : dict[str, tuple[float, float]] | None
        Per-parameter box bounds.  None uses model.bounds().
    random_seed : int
        Seed stored in the FitSpec for reproducible bootstrap CI.  Default 0.
    max_nfev : int | None
        Maximum number of function evaluations passed to scipy.  None = default.
    xtol : float | None
        Tolerance for change in the independent variables.  None uses scipy default (1e-8).
    ftol : float | None
        Tolerance for change in the cost function.  None uses scipy default (1e-8).
    gtol : float | None
        Tolerance for the norm of the gradient.  None uses scipy default (1e-8).
    x_scale : str | np.ndarray | None
        Characteristic scale of each variable.  ``"jac"`` enables Jacobian-based
        scaling which significantly improves convergence for ill-conditioned problems
        (e.g. datasets where parameters span many orders of magnitude).  None uses
        scipy default (uniform scaling, equivalent to 1.0).  Ignored when method="lm".
    diff_method : str | None
        Finite-difference scheme used when the model provides no analytic Jacobian.
        One of ``"2-point"`` (default), ``"3-point"``, or ``"cs"`` (complex-step).
        ``"cs"`` gives near-machine-precision gradients at the cost of complex
        function evaluations; valid for models implemented entirely with numpy ufuncs.
        Ignored when an analytic Jacobian is available.

    Examples
    --------
    >>> import numpy as np
    >>> from openfit import Fit
    >>> x = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    >>> y = np.array([2.0, 5.0, 20.0, 65.0, 90.0, 97.0, 99.0])
    >>> result = Fit("hill4p", x, y, weights="1/y2").run()
    >>> print(result.summary())
    """

    def __init__(
        self,
        model: str | BaseModel,
        x: Any,
        y: Any,
        *,
        weights: str | WeightScheme,
        sd: Any = None,
        method: str | None = None,
        p0: dict[str, float] | None = None,
        bounds: dict[str, tuple[float, float]] | None = None,
        random_seed: int = 0,
        max_nfev: int | None = None,
        xtol: float | None = None,
        ftol: float | None = None,
        gtol: float | None = None,
        x_scale: str | np.ndarray | None = None,
        diff_method: str | None = None,
    ) -> None:
        # Resolve model
        if isinstance(model, str):
            self._model: BaseModel = get_model(model)
        else:
            self._model = model

        # Stash raw arrays -- convert to float64 in run() after validation
        self._x_raw = x
        self._y_raw = y

        # Parse weight scheme eagerly so bad strings fail fast at construction time
        self._weight_scheme: WeightScheme = parse_weight_scheme(weights)
        self._sd = sd
        self._user_method = method
        self._p0 = p0
        self._user_bounds = bounds
        self._random_seed = random_seed
        self._max_nfev = max_nfev
        self._xtol = xtol
        self._ftol = ftol
        self._gtol = gtol
        self._x_scale = x_scale
        self._diff_method = diff_method

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(self) -> FitResult:
        """Execute the fit and return a FitResult.

        Returns
        -------
        FitResult
            Fitted parameters, standard errors, confidence intervals, GOF
            statistics, residuals, and a full reproducibility spec.

        Raises
        ------
        ValueError
            If x or y contain NaN or Inf (CLAUDE.md rule 8).
            If the number of observations is too small (n < n_params + 1).
            If scipy optimization fails to converge (status < 0).
        RuntimeError
            If the Jacobian is singular and covariance cannot be computed.
        """
        model = self._model

        # ----------------------------------------------------------------
        # 1. Validate and convert inputs
        # ----------------------------------------------------------------
        x = np.asarray(self._x_raw, dtype=np.float64).ravel()
        y = np.asarray(self._y_raw, dtype=np.float64).ravel()

        if not np.isfinite(x).all():
            raise ValueError("x contains NaN or Inf values. Clean the input data before fitting.")
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or Inf values. Clean the input data before fitting.")

        n_obs = len(x)
        n_params = len(model.param_names)

        if n_obs < n_params + 1:
            raise ValueError(
                f"Not enough observations: n_obs={n_obs} must be >= n_params + 1 = "
                f"{n_params + 1} to fit model '{model.model_id}'."
            )

        # ----------------------------------------------------------------
        # 2. Compute weights
        # ----------------------------------------------------------------
        sd_arr: np.ndarray | None = (
            np.asarray(self._sd, dtype=np.float64).ravel() if self._sd is not None else None
        )
        w = apply_weights(y, self._weight_scheme, sd=sd_arr)  # shape (n,), w_i > 0
        sqrt_w = np.sqrt(w)

        # ----------------------------------------------------------------
        # 3. Resolve initial guesses and bounds
        # ----------------------------------------------------------------
        if self._p0 is not None:
            p0_dict = {name: self._p0[name] for name in model.param_names}
        else:
            p0_dict = model.initial_guess(x, y)

        p0_arr = np.array([p0_dict[name] for name in model.param_names], dtype=np.float64)

        # Start from model bounds then apply any user overrides
        lb_list, ub_list = model.bounds()
        lb_arr = np.asarray(lb_list, dtype=np.float64)
        ub_arr = np.asarray(ub_list, dtype=np.float64)

        if self._user_bounds is not None:
            for i, name in enumerate(model.param_names):
                if name in self._user_bounds:
                    lo, hi = self._user_bounds[name]
                    lb_arr[i] = lo
                    ub_arr[i] = hi

        # ----------------------------------------------------------------
        # 4. Choose optimization method
        # ----------------------------------------------------------------
        has_finite_bounds = np.any(np.isfinite(lb_arr) & (lb_arr > -np.inf)) or np.any(
            np.isfinite(ub_arr) & (ub_arr < np.inf)
        )

        if self._user_method is not None:
            method = self._user_method
            # Guard: lm does not support bounds in scipy
            if method == "lm" and has_finite_bounds:
                method = "trf"
        else:
            method = "trf" if has_finite_bounds else "lm"

        # lm ignores bounds entirely in scipy; pass (-inf, inf) for safety
        if method == "lm":
            bounds_arg: tuple[Any, Any] = (-np.inf, np.inf)
        else:
            bounds_arg = (lb_arr, ub_arr)

        # Warn if user provided parameters that lm doesn't support
        if method == "lm":
            if self._x_scale is not None:
                warnings.warn(
                    'Parameter x_scale is not supported with method="lm" and will be ignored. '
                    'Use method="trf" to enable this parameter.',
                    UserWarning,
                    stacklevel=2,
                )
            if self._diff_method is not None:
                warnings.warn(
                    'Parameter diff_method is not supported with method="lm" and will be ignored. '
                    'Use method="trf" to enable this parameter.',
                    UserWarning,
                    stacklevel=2,
                )

        # ----------------------------------------------------------------
        # 5. Build residual function
        # ----------------------------------------------------------------
        def _residuals(p: np.ndarray) -> np.ndarray:
            params = dict(zip(model.param_names, p, strict=False))
            y_pred = model.equation(x, **params)
            return sqrt_w * (y - y_pred)

        # ----------------------------------------------------------------
        # 6. Build analytic Jacobian if the model provides one
        #    d(residual_i)/d(p_j) = d/dp_j [sqrt(w_i) * (y_i - f(x_i))]
        #                         = -sqrt(w_i) * d(f)/d(p_j)
        # ----------------------------------------------------------------
        jac_fn = None
        _test_jac = model.jacobian(x, **p0_dict)
        if _test_jac is not None:

            def jac_fn(p: np.ndarray) -> np.ndarray:  # type: ignore[misc]
                params = dict(zip(model.param_names, p, strict=False))
                J = model.jacobian(x, **params)  # shape (n_obs, n_params)
                if J is None:
                    return np.empty((n_obs, n_params))  # fallback, should not happen
                # Negative sign: d(sqrt(w)*(y-f))/dp = -sqrt(w) * dF/dp
                return -sqrt_w[:, np.newaxis] * np.asarray(J, dtype=np.float64)

        # ----------------------------------------------------------------
        # 7. Run optimization
        # ----------------------------------------------------------------
        # scipy does not accept jac=None; fall back to numerical diff.
        # "cs" (complex-step) gives near-machine-precision derivatives for
        # functions composed of standard numpy ufuncs and is the recommended
        # choice when no analytic Jacobian is provided.
        _fallback_diff = self._diff_method if self._diff_method is not None else "2-point"
        jac_arg: Any = jac_fn if jac_fn is not None else _fallback_diff

        opt_kwargs: dict[str, Any] = {
            "fun": _residuals,
            "x0": p0_arr,
            "jac": jac_arg,
            "bounds": bounds_arg,
            "method": method,
        }
        if self._max_nfev is not None:
            opt_kwargs["max_nfev"] = self._max_nfev
        if self._xtol is not None:
            opt_kwargs["xtol"] = self._xtol
        if self._ftol is not None:
            opt_kwargs["ftol"] = self._ftol
        if self._gtol is not None:
            opt_kwargs["gtol"] = self._gtol
        # x_scale only meaningful for trf/dogbox (not lm); pass through if set
        if self._x_scale is not None and method != "lm":
            opt_kwargs["x_scale"] = self._x_scale

        result_opt = scipy.optimize.least_squares(**opt_kwargs)

        if result_opt.status < 0:
            raise ValueError(
                f"Optimization failed for model '{model.model_id}' "
                f"(scipy status={result_opt.status}: {result_opt.message}). "
                "Try different initial guesses, a different weight scheme, or inspect "
                "the data for outliers."
            )

        # ----------------------------------------------------------------
        # 8. Extract fitted parameters
        # ----------------------------------------------------------------
        p_fit = result_opt.x
        params: dict[str, float] = {
            name: float(p_fit[i]) for i, name in enumerate(model.param_names)
        }

        # ----------------------------------------------------------------
        # 9. Compute asymptotic standard errors from the scaled Jacobian
        #    J (n x k) is already sqrt(w)-scaled (returned by scipy).
        #    Cov = (J^T J)^{-1} * (weighted_RSS / df)
        # ----------------------------------------------------------------
        J = result_opt.jac  # shape (n_obs, n_params), sqrt(w)-scaled
        rss_weighted = float(np.sum(result_opt.fun**2))
        df = n_obs - n_params

        try:
            jtj = J.T @ J
            cov = np.linalg.pinv(jtj) * (rss_weighted / max(df, 1))
            se_arr = np.sqrt(np.diag(cov))
            if not np.isfinite(se_arr).all():
                raise np.linalg.LinAlgError("Non-finite SE from covariance diagonal.")
            se: dict[str, float] = {
                name: float(se_arr[i]) for i, name in enumerate(model.param_names)
            }
        except np.linalg.LinAlgError:
            # Singular Jacobian: model is overparameterized or data is degenerate
            se = {name: float("inf") for name in model.param_names}
            cov = np.full((n_params, n_params), np.nan)

        # ----------------------------------------------------------------
        # 10. Asymptotic confidence intervals (lazy import for future-proofing)
        # ----------------------------------------------------------------
        try:
            from openfit.uncertainty import asymptotic_ci

            ci = asymptotic_ci(params, se, n_obs, n_params)
        except (ImportError, ValueError):
            # Fallback inline t-distribution CI if uncertainty module unavailable
            import scipy.stats as _stats

            df_ci = max(n_obs - n_params, 1)
            alpha = 0.05
            t_crit = float(_stats.t.ppf(1.0 - alpha / 2.0, df_ci))
            ci = {
                name: (
                    float(params[name] - t_crit * se[name]),
                    float(params[name] + t_crit * se[name]),
                )
                for name in model.param_names
            }

        # ----------------------------------------------------------------
        # 11. Compute GOF statistics
        # ----------------------------------------------------------------
        y_fitted = model.equation(x, **params)
        residuals = y - y_fitted
        rss_unweighted = float(np.sum(residuals**2))

        # R^2: for weighted fits use weighted SS (CLAUDE.md rule 9)
        # For uniform weighting w is all-ones, so weighted == unweighted
        is_uniform = self._weight_scheme is WeightScheme.UNIFORM
        if is_uniform:
            ss_res = float(np.sum(residuals**2))
            ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        else:
            y_wmean = float(np.average(y, weights=w))
            ss_res = float(np.sum(w * residuals**2))
            ss_tot = float(np.sum(w * (y - y_wmean) ** 2))

        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

        # Information criteria (unweighted RSS -- task spec)
        rss = rss_unweighted
        k = n_params
        n = n_obs
        aic = n * np.log(rss / n) + 2.0 * k
        bic = n * np.log(rss / n) + k * np.log(n)
        denom_aicc = n - k - 1
        aicc = float(aic + 2.0 * k * (k + 1) / denom_aicc) if denom_aicc > 0 else float("inf")

        # Residual diagnostics
        raw_resid = residuals  # y - y_fitted
        weighted_resid = sqrt_w * raw_resid
        std_resid_sd = float(np.std(raw_resid, ddof=1)) if n_obs > 1 else 1.0
        if std_resid_sd == 0.0:
            std_resid_sd = 1.0
        standardized_resid = raw_resid / std_resid_sd

        # ----------------------------------------------------------------
        # 12. Build FitSpec
        # ----------------------------------------------------------------
        spec = build_spec(
            model_id=model.model_id,
            param_values=params,
            weights=str(self._weight_scheme.value),
            x=x,
            y=y,
            random_seed=self._random_seed,
        )

        # ----------------------------------------------------------------
        # 13. Assemble and return FitResult
        # ----------------------------------------------------------------
        return FitResult(
            params=params,
            se=se,
            ci=ci,
            covariance=cov,
            r_squared=float(r_squared),
            aic=float(aic),
            bic=float(bic),
            aicc=float(aicc),
            rss=float(rss),
            x=x,
            y=y,
            y_fitted=np.asarray(y_fitted, dtype=np.float64),
            residuals=np.asarray(raw_resid, dtype=np.float64),
            weighted_residuals=np.asarray(weighted_resid, dtype=np.float64),
            standardized_residuals=np.asarray(standardized_resid, dtype=np.float64),
            n_obs=n_obs,
            n_params=n_params,
            model_id=model.model_id,
            weight_scheme=str(self._weight_scheme.value),
            spec=spec,
            _model=model,
            _weights=w,
        )
