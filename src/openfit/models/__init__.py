"""Model registry for openfit built-in models.

Built-in models are registered automatically on import.  Custom models can be
added at runtime with :func:`register_model`.

Functions
---------
get_model(model_id)      -- retrieve a model instance by ID
list_models()            -- sorted list of all registered model IDs
register_model(model)    -- add a model to the registry
"""

from __future__ import annotations

from openfit.models.base import BaseModel
from openfit.models.custom import CustomModel
from openfit.models.enzyme import Allosteric, MichaelisMenten, SubstrateInhibition
from openfit.models.exponential import BiExp, ExpDecay, ExpGrowth, ExpPlateau, MonoExp
from openfit.models.gaussian import BiGaussian, Gaussian, Lorentzian
from openfit.models.growth import Gompertz, Logistic3P, Logistic4P, Richards
from openfit.models.polynomial import Poly1, Poly2, Poly3, Poly4, Poly5, Poly6
from openfit.models.sigmoidal import Boltzmann, Hill3P, Hill4P, Hill5P

__all__ = [
    # Protocol
    "BaseModel",
    # Sigmoidal
    "Hill3P",
    "Hill4P",
    "Hill5P",
    "Boltzmann",
    # Exponential
    "MonoExp",
    "BiExp",
    "ExpGrowth",
    "ExpPlateau",
    "ExpDecay",
    # Enzyme
    "MichaelisMenten",
    "SubstrateInhibition",
    "Allosteric",
    # Growth
    "Logistic3P",
    "Logistic4P",
    "Gompertz",
    "Richards",
    # Gaussian / peak
    "Gaussian",
    "BiGaussian",
    "Lorentzian",
    # Polynomial
    "Poly1",
    "Poly2",
    "Poly3",
    "Poly4",
    "Poly5",
    "Poly6",
    # Custom
    "CustomModel",
    # Registry API
    "get_model",
    "list_models",
    "register_model",
]

# ---------------------------------------------------------------------------
# Internal registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, BaseModel] = {}


def _register_defaults() -> None:
    """Register all built-in model instances into the global registry."""
    built_ins: list[type[object]] = [
        Hill3P,
        Hill4P,
        Hill5P,
        Boltzmann,
        MonoExp,
        BiExp,
        ExpGrowth,
        ExpPlateau,
        ExpDecay,
        MichaelisMenten,
        SubstrateInhibition,
        Allosteric,
        Logistic3P,
        Logistic4P,
        Gompertz,
        Richards,
        Gaussian,
        BiGaussian,
        Lorentzian,
        Poly1,
        Poly2,
        Poly3,
        Poly4,
        Poly5,
        Poly6,
    ]
    for cls in built_ins:
        instance = cls()  # type: ignore[call-arg]
        _REGISTRY[instance.model_id] = instance  # type: ignore[union-attr]


_register_defaults()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_model(model_id: str) -> BaseModel:
    """Return a registered model instance by its ID.

    Parameters
    ----------
    model_id : str
        Unique model identifier (e.g. ``"hill4p"``).  Case-sensitive.

    Returns
    -------
    BaseModel
        Registered model instance.

    Raises
    ------
    KeyError
        If *model_id* is not found in the registry.
    """
    key = model_id.lower()
    try:
        return _REGISTRY[key]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Unknown model ID: {model_id!r}. "
            f"Available models: {available}"
        ) from None


def list_models() -> list[str]:
    """Return a sorted list of all registered model IDs.

    Returns
    -------
    list[str]
        Alphabetically sorted list of model identifiers.
    """
    return sorted(_REGISTRY.keys())


def register_model(model: BaseModel) -> None:
    """Register a model instance in the global registry.

    If a model with the same ``model_id`` already exists it is silently
    overwritten.

    Parameters
    ----------
    model : BaseModel
        Model instance to register.  Must satisfy the :class:`BaseModel`
        Protocol (i.e. have ``model_id``, ``param_names``, ``equation``,
        ``initial_guess``, ``bounds``, and ``jacobian``).

    Raises
    ------
    TypeError
        If *model* does not satisfy the BaseModel Protocol.
    """
    if not isinstance(model, BaseModel):
        raise TypeError(
            f"model must satisfy the BaseModel Protocol, got {type(model)!r}."
        )
    _REGISTRY[model.model_id] = model
