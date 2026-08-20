"""Shared IMOS dataset utilities for mooring_proc parsers.

Provides helpers for MATLAB datenum conversions, IMOS-style xr.Dataset
scaffolding, and range-tag generation.  All parsers in this package use
these helpers so that the IMOS structural conventions (TIME as
datetime64, scaffold scalars, quality_control flags) are applied
consistently.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Time conversions
# ---------------------------------------------------------------------------

def matlab_datenum_to_datetime64(datenum: float | np.ndarray) -> np.ndarray:
    """Convert MATLAB datenum(s) to numpy datetime64[ns].

    MATLAB datenum is days since 0000-01-01 in the proleptic Gregorian
    calendar.  Python's ``datetime.toordinal`` is days since 0001-01-01,
    so the offset is 366 days.
    """
    dn = np.atleast_1d(np.asarray(datenum, dtype=float))
    # Epoch: 1970-01-01 as MATLAB datenum
    matlab_epoch_1970 = 719529.0  # datetime(1970,1,1).toordinal() + 366
    seconds_since_epoch = (dn - matlab_epoch_1970) * 86400.0
    ns = (seconds_since_epoch * 1e9).astype("int64")
    return ns.astype("datetime64[ns]")


def datetime64_to_matlab_datenum(dt64: np.ndarray) -> np.ndarray:
    """Convert numpy datetime64[ns] array to MATLAB datenum array."""
    matlab_epoch_1970 = 719529.0
    epoch = np.datetime64("1970-01-01T00:00:00", "ns")
    ns = (dt64.astype("datetime64[ns]") - epoch).astype("int64")
    return ns / 1e9 / 86400.0 + matlab_epoch_1970


def python_datetime_to_matlab_datenum(dt: datetime) -> float:
    """Convert a Python datetime to a MATLAB datenum (float)."""
    ordinal = dt.toordinal()
    frac = (dt - datetime(dt.year, dt.month, dt.day)).total_seconds() / 86400.0
    return ordinal + 366.0 + frac


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def is_blank(value: Any) -> bool:
    """Return True if *value* is None, NaN, or an empty/null string."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none"}


def coerce_row(config: Any) -> dict[str, Any]:
    """Normalise a *config* argument into a plain ``dict``.

    Accepts ``None``, a plain ``dict``, a ``dict`` with a
    ``'metadata_row'`` key (as produced by ``run_proc1``), or any
    object that exposes a ``to_dict()`` method (e.g. a
    ``pandas.Series``).
    """
    if config is None:
        return {}
    if isinstance(config, dict):
        if "metadata_row" in config and config["metadata_row"] is not None:
            return coerce_row(config["metadata_row"])
        return dict(config)
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return {}


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def range_tag_from_times(time_values: np.ndarray | pd.DatetimeIndex) -> str:
    """Return an ISO range string ``YYYYmmddTHHMMSS_YYYYmmddTHHMMSS``."""
    times = pd.to_datetime(time_values)
    return f"{times.min():%Y%m%dT%H%M%S}_{times.max():%Y%m%dT%H%M%S}"


def build_imos_dataset(
    time_values: np.ndarray,
    data_vars: dict[str, np.ndarray],
    row: dict[str, Any],
    source_file: Path | str,
    qc_variables: list[str] | None = None,
    extra_var_attrs: dict[str, dict[str, Any]] | None = None,
    global_attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Build a minimal IMOS-style ``xr.Dataset``.

    Parameters
    ----------
    time_values:
        1-D numpy datetime64 array for the TIME dimension.
    data_vars:
        Mapping of IMOS variable name → 1-D numpy array (same length as
        *time_values*).
    row:
        Metadata row dict; used for LATITUDE, LONGITUDE, NOMINAL_DEPTH
        and global attribute defaults.
    source_file:
        Path of the raw input file (stored as a global attribute).
    qc_variables:
        Variables for which a ``<name>_quality_control`` flag (int8,
        initialised to 1) is automatically created.
    extra_var_attrs:
        Per-variable extra attributes that override the defaults.
    global_attrs:
        Extra global attributes merged over the defaults.

    Returns
    -------
    xr.Dataset
        An IMOS-shaped dataset ready for downstream QC and export.
    """
    n_time = len(time_values)
    coordinates = "TIME LATITUDE LONGITUDE NOMINAL_DEPTH"
    extra_var_attrs = extra_var_attrs or {}

    xr_vars: dict[str, xr.DataArray] = {
        "TIMESERIES": xr.DataArray(np.int32(1), attrs={"cf_role": "timeseries_id"}),
        "LATITUDE": xr.DataArray(
            np.float64(row.get("latitude", np.nan)),
            attrs={"units": "degrees_north"},
        ),
        "LONGITUDE": xr.DataArray(
            np.float64(row.get("longitude", np.nan)),
            attrs={"units": "degrees_east"},
        ),
        "NOMINAL_DEPTH": xr.DataArray(
            np.float32(row.get("nominal_depth", np.nan)),
            attrs={"units": "m", "positive": "down"},
        ),
    }

    for name, data in data_vars.items():
        attrs: dict[str, Any] = {"coordinates": coordinates}
        attrs.update(extra_var_attrs.get(name, {}))
        xr_vars[name] = xr.DataArray(data, dims=["TIME"], attrs=attrs)

    if qc_variables:
        for var in qc_variables:
            xr_vars[f"{var}_quality_control"] = xr.DataArray(
                np.ones(n_time, dtype=np.int8), dims=["TIME"]
            )

    range_tag = range_tag_from_times(time_values)

    default_attrs: dict[str, Any] = {
        "source_file": str(source_file),
        "instrument": str(row.get("inst_type", "")),
        "serial": str(row.get("inst_id", "")),
        "location": str(row.get("location", "")),
        "deployment_id": str(row.get("deployment_id", "")),
        "range_tag": range_tag,
    }
    if global_attrs:
        default_attrs.update(global_attrs)

    ds = xr.Dataset(
        data_vars=xr_vars,
        coords={"TIME": time_values},
        attrs=default_attrs,
    )

    return ds
