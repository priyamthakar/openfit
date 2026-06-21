"""openfit -- Reproducible, open-source nonlinear curve fitting."""

from importlib.metadata import PackageNotFoundError, version

from openfit.bands import BandResult, confidence_band, prediction_band
from openfit.batch import BatchFit
from openfit.compare import compare_models
from openfit.diagnostics import (
    DurbinWatsonResult,
    LackOfFitResult,
    durbin_watson,
    lack_of_fit_test,
)
from openfit.fit import Fit
from openfit.global_fit import GlobalFit
from openfit.leverage import LeverageResult, leverage_diagnostics
from openfit.models import get_model, list_models
from openfit.outliers import ROUTResult, rout_outliers
from openfit.results import FitResult
from openfit.spec import FitSpec
from openfit.weighting import WeightScheme

try:
    __version__ = version("openfit")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "BandResult",
    "BatchFit",
    "DurbinWatsonResult",
    "Fit",
    "FitResult",
    "FitSpec",
    "GlobalFit",
    "LackOfFitResult",
    "LeverageResult",
    "ROUTResult",
    "WeightScheme",
    "compare_models",
    "confidence_band",
    "durbin_watson",
    "get_model",
    "lack_of_fit_test",
    "leverage_diagnostics",
    "list_models",
    "prediction_band",
    "rout_outliers",
    "__version__",
]
