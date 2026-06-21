# src/openfit/batch.py
"""BatchFit: running model fits across multiple datasets in batch."""

from __future__ import annotations

from typing import Any

import pandas as pd

from openfit.fit import Fit
from openfit.models.base import BaseModel
from openfit.results import FitResult


class BatchFit:
    """Run model fits across multiple datasets.

    Parameters
    ----------
    model : str | BaseModel
        Model identifier string or BaseModel instance.
    datasets : list[dict[str, Any]]
        List of datasets. Each dataset must contain 'x' and 'y' arrays.
        Can optionally contain: 'weights', 'sd', 'p0', 'bounds', 'fixed',
        'constraints', 'penalties' to override batch-level parameters.
    weights : str
        Default weight scheme applied to all datasets (unless overridden).
    """

    def __init__(
        self,
        model: str | BaseModel,
        datasets: list[dict[str, Any]],
        *,
        weights: str = "uniform",
    ) -> None:
        self.model = model
        self.datasets = datasets
        self.weights = weights

    def run(self) -> list[FitResult]:
        """Execute fitting sequentially for all datasets.

        Returns
        -------
        list[FitResult]
            FitResult objects corresponding to each dataset in input order.
        """
        results: list[FitResult] = []
        for i, ds in enumerate(self.datasets):
            if "x" not in ds or "y" not in ds:
                raise ValueError(f"Dataset at index {i} is missing required 'x' or 'y' keys.")

            # Override or fallback
            x = ds["x"]
            y = ds["y"]
            w = ds.get("weights", self.weights)
            sd = ds.get("sd", None)
            p0 = ds.get("p0", None)
            bounds = ds.get("bounds", None)
            fixed = ds.get("fixed", None)
            constraints = ds.get("constraints", None)
            penalties = ds.get("penalties", None)

            fit_obj = Fit(
                model=self.model,
                x=x,
                y=y,
                weights=w,
                sd=sd,
                p0=p0,
                bounds=bounds,
                fixed=fixed,
                constraints=constraints,
                penalties=penalties,
            )
            results.append(fit_obj.run())

        return results

    @staticmethod
    def summary_df(results: list[FitResult]) -> pd.DataFrame:
        """Construct a summary pandas DataFrame from a list of FitResult objects.

        Parameters
        ----------
        results : list[FitResult]
            Completed FitResult instances.

        Returns
        -------
        pd.DataFrame
            DataFrame where rows represent datasets and columns represent
            parameter estimates, standard errors, R^2, and information criteria.
        """
        records: list[dict[str, Any]] = []
        for i, res in enumerate(results):
            rec: dict[str, Any] = {
                "dataset_index": i,
                "r_squared": res.r_squared,
                "rss": res.rss,
                "aic": res.aic,
                "bic": res.bic,
                "aicc": res.aicc,
                "n_obs": res.n_obs,
                "n_params": res.n_params,
            }

            for name, val in res.params.items():
                rec[name] = val
                rec[f"{name}_se"] = res.se[name]

            records.append(rec)

        return pd.DataFrame(records)
