"""NIST StRD nonlinear regression certified values and data loader.

Reads raw .dat files downloaded from:
  https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/<Name>.dat

All 27 nonlinear datasets are parsed at import time from the local cache in
tests/validation/nist_data/.  The data files are public domain (NIST).

Certified parameter values are given by NIST to 11 significant digits,
computed in 128-bit extended precision and confirmed by at least two
independent algorithms using analytic derivatives.

Each entry in NIST_DATASETS has the form::

    {
        "name":             str,                  # dataset name
        "difficulty":       str,                  # "Lower" | "Average" | "Higher"
        "n_params":         int,
        "n_obs":            int,
        "param_names":      list[str],            # ["b1", "b2", ...]
        "certified_params": dict[str, float],     # 11-digit certified values
        "certified_sd":     dict[str, float],     # certified standard deviations
        "certified_rss":    float,                # certified residual sum of squares
        "start1":           dict[str, float],     # NIST Start I  (far from solution)
        "start2":           dict[str, float],     # NIST Start II (closer to solution)
        "x":                list[float] | None,   # predictor values (None for 2-pred.)
        "x2":               list[float] | None,   # second predictor (Nelson only)
        "y":                list[float] | None,   # response values
        "model_eq":         str,                  # equation string from .dat file
        "n_predictors":     int,                  # 1 for almost all; 2 for Nelson
    }

Reference
---------
NIST/ITL Statistical Reference Datasets (StRD):
https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Directory containing the raw .dat files.
_NIST_DATA_DIR = Path(__file__).parent / "nist_data"

# All 27 dataset names in a canonical order.
ALL_DATASET_NAMES: list[str] = [
    # Lower difficulty (8 datasets)
    "Misra1a",
    "Misra1b",
    "Chwirut1",
    "Chwirut2",
    "Lanczos3",
    "Gauss1",
    "Gauss2",
    "DanWood",
    # Average difficulty (11 datasets)
    "Kirby2",
    "Hahn1",
    "Nelson",
    "MGH17",
    "Lanczos1",
    "Lanczos2",
    "Gauss3",
    "Misra1c",
    "Misra1d",
    "Roszman1",
    "ENSO",
    # Higher difficulty (8 datasets)
    "MGH09",
    "MGH10",
    "Thurber",
    "BoxBOD",
    "Rat42",
    "Rat43",
    "Eckerle4",
    "Bennett5",
]

# Datasets in the "must-pass" subset per CLAUDE.md.
MUST_PASS_NAMES: list[str] = [
    # Lower
    "Misra1a",
    "Chwirut2",
    "DanWood",
    "Gauss1",
    # Average
    "Hahn1",
    "MGH17",
    "Lanczos1",
    "Nelson",
    "ENSO",
    # Higher (ALL)
    "BoxBOD",
    "Eckerle4",
    "Rat42",
    "Rat43",
    "MGH09",
    "MGH10",
    "Thurber",
    "Bennett5",
]


# ---------------------------------------------------------------------------
# Internal .dat file parser
# ---------------------------------------------------------------------------

def _parse_nist_dat(filepath: Path) -> dict[str, Any]:
    """Parse one NIST StRD nonlinear .dat file.

    Parameters
    ----------
    filepath : Path
        Absolute path to the .dat file.

    Returns
    -------
    dict[str, Any]
        Parsed record with keys described in the module docstring.
    """
    with open(filepath, errors="replace") as fh:
        lines = fh.readlines()

    name = filepath.stem  # filename without extension

    param_names: list[str] = []
    start1: dict[str, float] = {}
    start2: dict[str, float] = {}
    cert_params: dict[str, float] = {}
    cert_sd: dict[str, float] = {}
    cert_rss: float | None = None
    n_obs: int | None = None
    difficulty: str | None = None
    model_eq: str | None = None
    n_predictors: int = 1

    data_x: list[float] = []
    data_x2: list[float] = []  # second predictor (Nelson only)
    data_y: list[float] = []

    # Locate the LAST "Data:   y  [x1  [x2]]" line -- that marks the data header.
    data_header_idx: int | None = None
    for i, line in enumerate(lines):
        if re.match(r"\s*Data:\s+y\s+x", line, re.IGNORECASE):
            data_header_idx = i
            # Check for two predictors (Nelson: "y  x1  x2")
            if re.search(r"x2", line, re.IGNORECASE):
                n_predictors = 2

    # Header parsing (everything before the data header).
    header_end = data_header_idx if data_header_idx is not None else len(lines)
    for line in lines[:header_end]:
        stripped = line.strip()

        # Difficulty level
        m = re.search(r"(Lower|Average|Higher)\s+Level", stripped)
        if m:
            difficulty = m.group(1)

        # Number of observations
        m = re.match(r"(\d+)\s+Observations", stripped)
        if m:
            n_obs = int(m.group(1))

        # Model equation (first 'y = ...' line before data section)
        m = re.match(r"y\s*=\s*(.+)", stripped)
        if m and model_eq is None:
            model_eq = stripped

        # Multi-line model equation (log[y] = ... for Nelson)
        m = re.match(r"log\[y\]\s*=\s*(.+)", stripped)
        if m and model_eq is None:
            model_eq = stripped

        # Parameter lines:
        #   "  b1 =   500         250           2.3894212918E+02  2.7070075241E+00"
        m = re.match(
            r"\s*(b\d+)\s*=\s+(\S+)\s+(\S+)\s+([\d.E+\-]+)\s+([\d.E+\-]+)", line
        )
        if m:
            pname = m.group(1)
            param_names.append(pname)
            start1[pname] = float(m.group(2))
            start2[pname] = float(m.group(3))
            cert_params[pname] = float(m.group(4))
            cert_sd[pname] = float(m.group(5))

        # Residual Sum of Squares
        m = re.search(r"Residual Sum of Squares:\s+([\d.E+\-]+)", stripped)
        if m:
            cert_rss = float(m.group(1))

    # Data parsing (everything after the data header).
    if data_header_idx is not None:
        for line in lines[data_header_idx + 1 :]:
            stripped = line.strip()
            if not stripped:
                continue
            # Extract all numeric tokens.
            # Pattern handles numbers starting with a decimal point (e.g. ".500E0").
            nums = re.findall(
                r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+\-]?\d+)?", stripped
            )
            if n_predictors == 1 and len(nums) >= 2:
                try:
                    data_y.append(float(nums[0]))
                    data_x.append(float(nums[1]))
                except ValueError:
                    pass
            elif n_predictors == 2 and len(nums) >= 3:
                try:
                    data_y.append(float(nums[0]))
                    data_x.append(float(nums[1]))
                    data_x2.append(float(nums[2]))
                except ValueError:
                    pass

    return {
        "name": name,
        "difficulty": difficulty,
        "n_params": len(param_names),
        "n_obs": n_obs,
        "n_predictors": n_predictors,
        "param_names": param_names,
        "certified_params": cert_params,
        "certified_sd": cert_sd,
        "certified_rss": cert_rss,
        "start1": start1,
        "start2": start2,
        "x": data_x if data_x else None,
        "x2": data_x2 if data_x2 else None,
        "y": data_y if data_y else None,
        "model_eq": model_eq,
    }


def _load_all() -> dict[str, dict[str, Any]]:
    """Load all available NIST .dat files from the nist_data directory."""
    datasets: dict[str, dict[str, Any]] = {}
    if not _NIST_DATA_DIR.exists():
        return datasets
    for name in ALL_DATASET_NAMES:
        dat_path = _NIST_DATA_DIR / f"{name}.dat"
        if dat_path.exists():
            datasets[name] = _parse_nist_dat(dat_path)
    return datasets


# Populated at import time -- tests iterate over this dict.
NIST_DATASETS: dict[str, dict[str, Any]] = _load_all()
