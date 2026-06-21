"""openfit -- Reproducible, open-source nonlinear curve fitting."""

from importlib.metadata import PackageNotFoundError, version

from openfit.batch import BatchFit
from openfit.compare import compare_models
from openfit.fit import Fit
from openfit.global_fit import GlobalFit
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
    "Fit",
    "FitResult",
    "FitSpec",
    "WeightScheme",
    "compare_models",
    "GlobalFit",
    "rout_outliers",
    "ROUTResult",
    "get_model",
    "list_models",
    "BatchFit",
    "__version__",
]
