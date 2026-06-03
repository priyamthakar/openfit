"""FitSpec: reproducibility manifest for every openfit fit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import scipy  # type: ignore[import-untyped]


def compute_data_hash(x: np.ndarray, y: np.ndarray) -> str:
    """Compute a SHA-256 digest of the concatenated x and y arrays.

    Both arrays are cast to float64 (little-endian) before hashing so
    that the digest is independent of the caller's original dtype and
    platform byte order.

    Parameters
    ----------
    x : np.ndarray
        Independent-variable values.
    y : np.ndarray
        Dependent-variable (response) values.

    Returns
    -------
    str
        64-character lowercase hex digest.
    """
    x_bytes = np.asarray(x, dtype="<f8").tobytes()
    y_bytes = np.asarray(y, dtype="<f8").tobytes()
    return hashlib.sha256(x_bytes + y_bytes).hexdigest()


def build_spec(
    model_id: str,
    param_values: dict[str, float],
    weights: str,
    x: np.ndarray,
    y: np.ndarray,
    random_seed: int = 0,
) -> FitSpec:
    """Convenience factory that auto-populates version strings and timestamp.

    Parameters
    ----------
    model_id : str
        Registered model identifier (e.g. ``"hill4p"``).
    param_values : dict[str, float]
        Fitted parameter values keyed by parameter name.
    weights : str
        Weight scheme name (e.g. ``"1/y2"``, ``"uniform"``).
    x : np.ndarray
        Independent-variable values used in the fit.
    y : np.ndarray
        Dependent-variable values used in the fit.
    random_seed : int, optional
        Seed passed to bootstrap CI (default 0).

    Returns
    -------
    FitSpec
        Fully populated reproducibility manifest.
    """
    try:
        openfit_ver = version("openfit")
    except PackageNotFoundError:
        openfit_ver = "0.0.0"

    return FitSpec(
        model_id=model_id,
        param_values=dict(param_values),
        weights=weights,
        data_hash=compute_data_hash(x, y),
        openfit_version=openfit_ver,
        scipy_version=scipy.__version__,
        numpy_version=np.__version__,
        random_seed=random_seed,
    )


@dataclass
class FitSpec:
    """Reproducibility manifest for a nonlinear fit.

    Every ``FitResult`` embeds a ``FitSpec`` so that a fit can be reproduced
    exactly from the spec plus the original data.  ``param_values`` floats are
    stored as their ``repr()`` strings inside JSON to guarantee lossless
    round-tripping; they are converted back to ``float`` on deserialisation.

    Parameters
    ----------
    model_id : str
        Registered model identifier (e.g. ``"hill4p"``).
    param_values : dict[str, float]
        Fitted parameter values keyed by parameter name.
    weights : str
        Weight scheme name (e.g. ``"1/y2"``, ``"uniform"``).
    data_hash : str
        SHA-256 hex digest of the concatenated x and y arrays
        (see :func:`compute_data_hash`).
    openfit_version : str
        openfit package version that produced this fit.
    scipy_version : str
        scipy version active at fit time.
    numpy_version : str
        numpy version active at fit time.
    random_seed : int
        Seed used for bootstrap CI (0 when bootstrap was not requested).
    timestamp : str
        ISO 8601 UTC timestamp of when the fit completed.
    """

    model_id: str
    param_values: dict[str, float]
    weights: str
    data_hash: str
    openfit_version: str
    scipy_version: str
    numpy_version: str
    random_seed: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise to a JSON string with lossless float precision.

        ``param_values`` floats are stored as their ``repr()`` strings
        (e.g. ``"0.1"`` not ``0.1``) to avoid any precision loss through
        JSON's native float formatting.  All other fields are already
        strings or integers and serialise without transformation.

        Returns
        -------
        str
            Indented JSON string that can be written to a file or
            transmitted over a network.
        """
        raw: dict[str, Any] = asdict(self)
        raw["param_values"] = {k: repr(v) for k, v in self.param_values.items()}
        return json.dumps(raw, indent=2)

    @classmethod
    def from_json(cls, s: str) -> FitSpec:
        """Deserialise from a JSON string produced by :meth:`to_json`.

        ``param_values`` string representations are converted back to
        ``float`` via ``float()``, which is the exact inverse of
        ``repr()`` for IEEE 754 double-precision values.

        Parameters
        ----------
        s : str
            JSON string previously returned by :meth:`to_json`.

        Returns
        -------
        FitSpec
            Reconstructed manifest with identical field values.
        """
        raw: dict[str, Any] = json.loads(s)
        raw["param_values"] = {k: float(v) for k, v in raw["param_values"].items()}
        return cls(**raw)

    # ------------------------------------------------------------------
    # Equality
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Return True if all fields of both specs match exactly.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            ``True`` when *other* is a ``FitSpec`` and every field is
            equal, ``False`` otherwise.
        """
        if not isinstance(other, FitSpec):
            return NotImplemented  # type: ignore[return-value]
        return asdict(self) == asdict(other)
