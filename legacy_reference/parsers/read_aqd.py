"""Parser for Nortek AQD .aqd files used in the AQD notebook workflow.

Integration:
- Use in AQD notebooks/scripts to parse AQD text and optionally build export dataset.
- File resolution can be explicit (path) or metadata-row based.

Copy-paste:
    from tools.parsers.read_aqd import (
        read_aqd_file,
        read_aqd_from_metadata,
        build_aqd_dataset,
        build_aqd_dataset_from_metadata,
    )

    # 1) Dataframe only (explicit file path)
    df = read_aqd_file("/path/to/input.aqd", verbose=True)

    # 2) Dataframe + resolved file path (metadata row)
    df, dat_file = read_aqd_from_metadata(_row, cwd=Path.cwd(), verbose=True)

    # 3) Build xarray dataset from existing dataframe
    ds, range_tag = build_aqd_dataset(df, _row, dat_file)

    # 4) One-step: metadata row -> dataframe + dataset + file path
    df, ds, dat_file, range_tag = build_aqd_dataset_from_metadata(_row, cwd=Path.cwd(), verbose=True)
"""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def _parse_aquad_line(line: str) -> Optional[Tuple[datetime, list[float]]]:
    # Expected date layout in first 19 chars: MM DD YYYY HH MM SS
    if len(line) < 21:
        return None

    date_tokens = line[:19].split()
    if len(date_tokens) != 6:
        return None

    try:
        mo_i, day_i, yy, hh, mi, ss = [int(v) for v in date_tokens]
        dt = datetime(yy, mo_i, day_i, hh, mi, ss)
    except (ValueError, TypeError):
        return None

    fields = line[20:].split()
    if len(fields) < 19:
        return None

    try:
        x = [float(v) for v in fields[:19]]
    except ValueError:
        return None

    return dt, x


def read_aqd_file(dat_file: str | Path, verbose: bool = True) -> pd.DataFrame:
    """Read AQD text file and return dataframe with datetime, Temperature, Depth, UCUR, VCUR."""
    dat_file = Path(dat_file)
    if not dat_file.exists():
        raise FileNotFoundError(f"AQD input file not found: {dat_file}")

    rows = []
    with open(dat_file, "rt", encoding="utf-8", errors="ignore") as fid:
        for line in fid:
            parsed = _parse_aquad_line(line)
            if parsed is None:
                continue
            dt, x = parsed
            spd = x[17]
            direction = x[18]
            direction_rad = math.radians(direction)
            rows.append(
                {
                    "datetime": dt,
                    "Temperature": x[14],
                    "Depth": x[13],
                    "UCUR": spd * math.sin(direction_rad) * 100.0,
                    "VCUR": spd * math.cos(direction_rad) * 100.0,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No valid AQD records parsed from {dat_file}")

    if verbose:
        print(f"Loaded AQD rows: {len(df)}")
        print(f"Input file: {dat_file}")
        print(df.head())

    return df


def read_aqd_from_metadata(row: Any, cwd: str | Path | None = None, verbose: bool = True) -> tuple[pd.DataFrame, Path]:
    """Resolve AQD input file from metadata row and return parsed dataframe and file path."""
    data_in_path = row.get("data_in_path", None)
    data_in_file = row.get("data_in_file", None)
    if _is_missing(data_in_path) or _is_missing(data_in_file):
        raise ValueError("Database fields data_in_path/data_in_file are required for AQD processing.")

    dat_path = Path(str(data_in_path))
    if not dat_path.is_absolute():
        base = Path(cwd) if cwd is not None else Path.cwd()
        dat_path = base / dat_path
    dat_file = dat_path / str(data_in_file)

    df = read_aqd_file(dat_file, verbose=verbose)
    return df, dat_file


def _time_to_days_since_1950(time_index: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    epoch_days_1950 = pd.Timestamp("1950-01-01").value / 86400e9
    return (pd.to_datetime(time_index).astype("int64") / 86400e9 - epoch_days_1950).astype(np.float64)


def build_aqd_dataset(df: pd.DataFrame, row: Any, dat_file: str | Path) -> tuple[xr.Dataset, str]:
    """Build AQD xarray dataset and return (dataset, range_tag)."""
    export_df = df.copy()
    if export_df.empty:
        raise ValueError("No data available for NetCDF export.")

    # Ensure expected columns exist
    required_cols = {"datetime", "Temperature", "Depth", "UCUR", "VCUR"}
    missing = required_cols - set(export_df.columns)
    if missing:
        raise ValueError(f"AQD dataframe missing required columns: {sorted(missing)}")

    time_start = pd.to_datetime(export_df["datetime"].min())
    time_end = pd.to_datetime(export_df["datetime"].max())
    range_tag = f"{time_start:%Y%m%dT%H%M%S}_{time_end:%Y%m%dT%H%M%S}"

    time_days = _time_to_days_since_1950(export_df["datetime"])
    n_time = len(export_df)

    lat_val = float(row.get("latitude", np.nan))
    lon_val = float(row.get("longitude", np.nan))
    nom_depth_val = float(row.get("nominal_depth", np.nan))

    ds = xr.Dataset(
        data_vars={
            "TEMP": ("TIME", export_df["Temperature"].to_numpy(dtype=np.float32)),
            "DEPTH": ("TIME", export_df["Depth"].to_numpy(dtype=np.float32)),
            "UCUR": ("TIME", export_df["UCUR"].to_numpy(dtype=np.float32)),
            "VCUR": ("TIME", export_df["VCUR"].to_numpy(dtype=np.float32)),
            "TEMP_quality_control": ("TIME", np.ones(n_time, dtype=np.int8)),
            "DEPTH_quality_control": ("TIME", np.ones(n_time, dtype=np.int8)),
            "UCUR_quality_control": ("TIME", np.ones(n_time, dtype=np.int8)),
            "VCUR_quality_control": ("TIME", np.ones(n_time, dtype=np.int8)),
        },
        coords={"TIME": time_days},
        attrs={
            "source_file": str(dat_file),
            "instrument": str(row.get("inst_type", "AQD")),
            "serial": str(row.get("inst_id", "")),
        },
    )

    ds["LATITUDE"] = xr.DataArray(lat_val)
    ds["LONGITUDE"] = xr.DataArray(lon_val)
    ds["NOMINAL_DEPTH"] = xr.DataArray(np.float32(nom_depth_val))
    ds["TIME"].attrs["units"] = "days since 1950-01-01T00:00:00 UTC"
    ds["TIME"].attrs["calendar"] = "gregorian"

    return ds, range_tag


def build_aqd_dataset_from_metadata(
    row: Any,
    cwd: str | Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, xr.Dataset, Path, str]:
    """Resolve input from metadata, parse dataframe, and build dataset.

    Returns:
        (df, ds, dat_file, range_tag)
    """
    df, dat_file = read_aqd_from_metadata(row, cwd=cwd, verbose=verbose)
    ds, range_tag = build_aqd_dataset(df=df, row=row, dat_file=dat_file)

    if verbose:
        time_vals = pd.to_datetime(df["datetime"])
        print(f"Dataset rows: {ds.sizes['TIME']}")
        print(f"Time range: {time_vals.min()} -> {time_vals.max()}")

    return df, ds, dat_file, range_tag


__all__ = [
    "read_aqd_file",
    "read_aqd_from_metadata",
    "build_aqd_dataset",
    "build_aqd_dataset_from_metadata",
]