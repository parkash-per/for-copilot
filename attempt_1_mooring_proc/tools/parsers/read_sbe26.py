"""SBE26 parser: ``.tid`` text file support.

Adapted from khannakarishma/imos-toolbox
``python/src/imos_toolbox/parsers/sbe26.py``.

File format (space-separated)::

    <meas_no> <MM/DD/YYYY> <HH:MM:SS> <pressure_psia> <temperature_C>

Conversions applied
-------------------
- Pressure: psia → dbar (multiply by 0.6894757)
- Time: centre of the 4-minute measurement window is used (+2 minutes).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .imos_utils import (
    coerce_row,
    is_blank,
    python_datetime_to_matlab_datenum,
    matlab_datenum_to_datetime64,
    range_tag_from_times,
)


def read_sbe26(input_path, config=None):
    """Read SBE26 ``.tid`` file and return dataframe, dataset, and metadata.

    Parameters
    ----------
    input_path:
        Path to the ``.tid`` file, or ``None`` to resolve from *config*.
    config:
        Metadata dict or ``{'metadata_row': <row>}``.

    Returns
    -------
    dict with keys ``dataframe``, ``dataset``, ``input_path``, ``range_tag``.
    """
    row = coerce_row(config)
    resolved = _resolve_input_path(input_path, row)

    if resolved.suffix.lower() != ".tid":
        raise ValueError(
            f"SBE26 parser: expected a .tid file, got '{resolved.suffix}'."
        )

    time_datenums, pressures_dbar, temps = _read_tid(resolved)

    if not temps:
        raise ValueError(f"No valid SBE26 samples found in {resolved}")

    time64 = matlab_datenum_to_datetime64(np.asarray(time_datenums))
    pressures = np.asarray(pressures_dbar, dtype=np.float32)
    temperatures = np.asarray(temps, dtype=np.float32)

    # Sample interval (median, in seconds)
    if len(time_datenums) > 1:
        sample_interval = float(np.median(np.diff(np.asarray(time_datenums))) * 86400.0)
    else:
        sample_interval = float("nan")

    dataset = _build_dataset(time64, pressures, temperatures, row, resolved, sample_interval)
    df = pd.DataFrame(
        {"PRES_REL": pressures, "TEMP": temperatures},
        index=pd.DatetimeIndex(time64, name="TIME"),
    )
    range_tag = dataset.attrs["range_tag"]
    return {
        "dataframe": df,
        "dataset": dataset,
        "input_path": str(resolved),
        "range_tag": range_tag,
    }


# ---------------------------------------------------------------------------
# Internal parser
# ---------------------------------------------------------------------------

def _read_tid(source_file: Path) -> tuple[list[float], list[float], list[float]]:
    """Parse a SBE26 .tid file into lists of time, pressure, temperature."""
    time_values: list[float] = []
    pressures: list[float] = []
    temps: list[float] = []

    content = source_file.read_text(encoding="utf-8", errors="ignore")
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Normalise delimiters: replace / and : with spaces
        normalised = line.replace("/", " ").replace(":", " ")
        tokens = normalised.split()
        # Expected: meas_no M D Y H MN S pressure_psia temp_c
        if len(tokens) < 9:
            continue
        try:
            month, day, year = int(tokens[1]), int(tokens[2]), int(tokens[3])
            hour, minute = int(tokens[4]), int(tokens[5])
            second_f = float(tokens[6])
            pressure_psia = float(tokens[7])
            temp_c = float(tokens[8])

            whole_sec = int(second_f)
            microsec = int(round((second_f - whole_sec) * 1_000_000))
            dt = datetime(year, month, day, hour, minute, whole_sec, microsec)
        except (ValueError, IndexError):
            continue

        # Shift +2 min to centre of the 4-minute measurement window
        dt_centre = dt + timedelta(minutes=2)
        time_values.append(python_datetime_to_matlab_datenum(dt_centre))
        pressures.append(pressure_psia * 0.6894757)
        temps.append(temp_c)

    return time_values, pressures, temps


def _build_dataset(
    time64: np.ndarray,
    pressures: np.ndarray,
    temperatures: np.ndarray,
    row: dict[str, Any],
    source_file: Path,
    sample_interval: float,
) -> xr.Dataset:
    coordinates = "TIME LATITUDE LONGITUDE NOMINAL_DEPTH"
    n = len(time64)
    range_tag = range_tag_from_times(time64)

    ds = xr.Dataset(
        data_vars={
            "TIMESERIES": xr.DataArray(np.int32(1), attrs={"cf_role": "timeseries_id"}),
            "LATITUDE": xr.DataArray(
                np.float64(row.get("latitude", np.nan)), attrs={"units": "degrees_north"}
            ),
            "LONGITUDE": xr.DataArray(
                np.float64(row.get("longitude", np.nan)), attrs={"units": "degrees_east"}
            ),
            "NOMINAL_DEPTH": xr.DataArray(
                np.float32(row.get("nominal_depth", np.nan)),
                attrs={"units": "m", "positive": "down"},
            ),
            "PRES_REL": xr.DataArray(
                pressures,
                dims=["TIME"],
                attrs={
                    "coordinates": coordinates,
                    "applied_offset": np.float32(-14.7 * 0.689476),
                    "comment": (
                        "Relative pressure with atmospheric offset "
                        "of ~-10.13 dbar applied by SeaBird software."
                    ),
                },
            ),
            "PRES_REL_quality_control": xr.DataArray(
                np.ones(n, dtype=np.int8), dims=["TIME"]
            ),
            "TEMP": xr.DataArray(
                temperatures, dims=["TIME"], attrs={"coordinates": coordinates}
            ),
            "TEMP_quality_control": xr.DataArray(
                np.ones(n, dtype=np.int8), dims=["TIME"]
            ),
        },
        coords={
            "TIME": time64,
        },
        attrs={
            "source_file": str(source_file),
            "instrument": str(row.get("inst_type", "SBE26")),
            "instrument_make": "Seabird",
            "instrument_model": "SBE26",
            "instrument_serial_no": str(row.get("inst_id", "")),
            "instrument_sample_interval": sample_interval,
            "serial": str(row.get("inst_id", "")),
            "location": str(row.get("location", "")),
            "deployment_id": str(row.get("deployment_id", "")),
            "range_tag": range_tag,
        },
    )

    ds["TIME"].attrs["comment"] = (
        "Time stamp corresponds to the centre of the measurement "
        "which lasts 4 minutes."
    )
    return ds


# ---------------------------------------------------------------------------
# Path resolver
# ---------------------------------------------------------------------------

def _resolve_input_path(input_path: Any, row: dict[str, Any]) -> Path:
    if not is_blank(input_path):
        p = Path(str(input_path)).expanduser()
        return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
    data_in_path = row.get("data_in_path")
    data_in_file = row.get("data_in_file")
    if is_blank(data_in_path) or is_blank(data_in_file):
        raise ValueError(
            "SBE26 input path missing; provide input_path or "
            "metadata data_in_path/data_in_file."
        )
    base = Path(str(data_in_path)).expanduser()
    base = base.resolve() if base.is_absolute() else (Path.cwd() / base).resolve()
    return base / str(data_in_file).strip()

