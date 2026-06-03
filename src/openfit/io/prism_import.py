"""Read-only Prism .pzfx file parser for data migration."""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PrismTable:
    """A single XY data table extracted from a .pzfx file."""

    title: str
    x: np.ndarray
    y: np.ndarray
    y_replicates: list[np.ndarray]
    y_sd: np.ndarray | None
    table_id: str


def _parse_subcolumn(subcolumn: ET.Element) -> np.ndarray:
    """Extract float values from a Subcolumn element, NaN for missing entries.

    Parameters
    ----------
    subcolumn : ET.Element
        A Subcolumn XML element containing zero or more <d> children.

    Returns
    -------
    np.ndarray
        Float64 array with NaN for empty or absent <d> text.
    """
    values: list[float] = []
    for d in subcolumn.findall("{*}d"):
        text = d.text
        if text is None or text.strip() == "":
            values.append(float("nan"))
        else:
            try:
                values.append(float(text.strip()))
            except ValueError:
                values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def _parse_table(table: ET.Element) -> PrismTable | None:
    """Parse a single Table element into a PrismTable.

    Returns None if the table is not of type XY or has no usable x/y data.

    Parameters
    ----------
    table : ET.Element
        A Table XML element.

    Returns
    -------
    PrismTable | None
        Parsed table, or None for non-XY or unparseable tables.
    """
    table_type = table.get("TableType", "")
    if table_type != "XY":
        return None

    table_id = table.get("ID", "")

    title_el = table.find("{*}Title")
    title = title_el.text.strip() if (title_el is not None and title_el.text) else ""

    # --- X column ---
    x_col_el = table.find("{*}XColumn")
    if x_col_el is None:
        return None

    x_subcols = x_col_el.findall("{*}Subcolumn")
    if not x_subcols:
        return None
    x_raw = _parse_subcolumn(x_subcols[0])

    # --- Y columns (each Subcolumn is one replicate) ---
    y_col_el = table.find("{*}YColumn")
    if y_col_el is None:
        return None

    y_subcols = y_col_el.findall("{*}Subcolumn")
    if not y_subcols:
        return None

    # Pad ragged subcolumns to max length with NaN before stacking
    max_len = max(len(x_raw), max(len(_parse_subcolumn(sc)) for sc in y_subcols))

    def _pad(arr: np.ndarray, length: int) -> np.ndarray:
        if len(arr) >= length:
            return arr[:length]
        padded = np.full(length, float("nan"), dtype=np.float64)
        padded[: len(arr)] = arr
        return padded

    x_raw = _pad(x_raw, max_len)
    rep_arrays = [_pad(_parse_subcolumn(sc), max_len) for sc in y_subcols]

    # Stack replicates and compute mean y and sd.
    # Suppress RuntimeWarnings that fire on all-NaN rows (e.g., "Mean of empty slice",
    # "Degrees of freedom <= 0 for slice"). The valid_mask below discards those rows.
    rep_matrix = np.stack(rep_arrays, axis=1)  # shape (max_len, n_reps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y_mean = np.nanmean(rep_matrix, axis=1)
        if len(y_subcols) >= 2:
            y_sd: np.ndarray | None = np.nanstd(rep_matrix, axis=1, ddof=1)
        else:
            y_sd = None

    # Drop rows where x is NaN or mean-y is NaN
    valid_mask = np.isfinite(x_raw) & np.isfinite(y_mean)
    x_out = x_raw[valid_mask]
    y_out = y_mean[valid_mask]
    reps_out = [arr[valid_mask] for arr in rep_arrays]
    y_sd_out = y_sd[valid_mask] if y_sd is not None else None

    return PrismTable(
        title=title,
        x=x_out,
        y=y_out,
        y_replicates=reps_out,
        y_sd=y_sd_out,
        table_id=table_id,
    )


def load_pzfx(path: str | Path) -> list[PrismTable]:
    """Parse a Prism .pzfx file and return all XY tables.

    .pzfx files are ZIP archives containing an XML document. Non-XY tables
    (e.g., Column, Grouped) are silently skipped.

    Parameters
    ----------
    path : str | Path
        Path to the .pzfx file.

    Returns
    -------
    list[PrismTable]
        One entry per XY table found. Returns an empty list if none exist.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValueError
        If the file is not a valid .pzfx ZIP archive or its XML cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Open ZIP archive
    try:
        zf = zipfile.ZipFile(path, "r")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{path.name} is not a valid .pzfx file (bad ZIP archive).") from exc

    with zf:
        names = zf.namelist()
        xml_names = [n for n in names if n.lower().endswith(".xml")]
        if not xml_names:
            # Fall back: try the first member regardless of extension
            if not names:
                raise ValueError(f"{path.name} is an empty archive; no XML found.")
            xml_names = [names[0]]

        xml_name = xml_names[0]
        try:
            xml_bytes = zf.read(xml_name)
        except KeyError as exc:
            raise ValueError(f"Could not read '{xml_name}' from {path.name}.") from exc

    # Parse XML
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise ValueError(f"{path.name} contains invalid XML: {exc}") from exc

    # Find all Table elements regardless of namespace
    tables: list[PrismTable] = []
    for table_el in root.findall(".//{*}Table"):
        result = _parse_table(table_el)
        if result is not None:
            tables.append(result)

    return tables
