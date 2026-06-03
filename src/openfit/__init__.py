"""openfit -- Reproducible, open-source nonlinear curve fitting."""
from openfit.fit import Fit
from openfit.results import FitResult
from openfit.spec import FitSpec
from openfit.weighting import WeightScheme
from openfit.compare import compare_models
from openfit.global_fit import GlobalFit
from openfit.outliers import rout_outliers, ROUTResult
from openfit.models import get_model, list_models
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("openfit")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Fit", "FitResult", "FitSpec", "WeightScheme",
    "compare_models", "GlobalFit", "rout_outliers", "ROUTResult",
    "get_model", "list_models",
    "__version__",
]
