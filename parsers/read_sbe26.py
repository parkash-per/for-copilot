"""Parser utilities for SBE26 processing workflows.

Integration:
- Use in SBE26 notebooks/scripts to parse .tid exports and build
  proc-ready xarray datasets.
- Supports metadata-row based file resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


REFERENCE_DATE = pd.Timestamp("1950-01-01")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def _resolve_input_file(row: Any, cwd: str | Path | None = None) -> Path:
    data_in_path = row.get("data_in_path", None)
    data_in_file = row.get("data_in_file", None)
    if _is_missing(data_in_path):
        raise ValueError("Database field data_in_path is required for SBE26 processing.")

    root = Path(str(data_in_path)).expanduser()
    if not root.is_absolute():
        base = Path(cwd) if cwd is not None else Path.cwd()
        root = (base / root).resolve()

    candidates: list[Path] = []

    if not _is_missing(data_in_file):
        direct = root / str(data_in_file)
        if direct.exists() and direct.is_file():
            candidates.append(direct)

    candidates.extend(sorted(root.rglob("*.tid")))
    candidates.extend(sorted(root.rglob("*.txt")))

    seen = set()
    unique_candidates = []
    for c in candidates:
        rc = c.resolve()
        if rc in seen:
            continue
        seen.add(rc)
        unique_candidates.append(rc)

    if not unique_candidates:
        raise FileNotFoundError(f"No readable SBE26 file found under {root}")

    return unique_candidates[0]


def read_sbe26_file(input_file: str | Path, verbose: bool = True) -> pd.DataFrame:
    """Read SBE26 .tid text file and normalize expected columns."""
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"SBE26 input file not found: {input_file}")

    df = pd.read_csv(
        input_file,
        sep=r"\s+",
        header=None,
        names=["ID", "Date", "time", "PRES", "TEMP"],
        engine="python",
    )

    if df.empty:
        raise ValueError(f"No rows parsed from {input_file}")

    df["Instrument Time"] = pd.to_datetime(df["Date"] + " " + df["time"])
    df["Pressure [db]"] = pd.to_numeric(df["PRES"], errors="coerce") * 0.6894757
    df["TEMP"] = pd.to_numeric(df["TEMP"], errors="coerce")

    out = (
        df[["Instrument Time", "Pressure [db]", "TEMP"]]
        .dropna(subset=["Instrument Time", "Pressure [db]", "TEMP"])
        .set_index("Instrument Time")
        .sort_index()
    )

    if verbose:
        print(f"Loaded SBE26 rows: {len(out)}")
        print(f"Input file: {input_file}")
        print(out.head())

    return out


def centered_rolling_avg(
    df: pd.DataFrame,
    window_seconds: int,
    time_vector: pd.DatetimeIndex,
    column_name: str,
) -> pd.DataFrame:
    result = []
    half = pd.Timedelta(seconds=window_seconds / 2)
    for t in time_vector:
        seg = df.loc[t - half : t + half]
        result.append({"Time": t, "average": seg[column_name].mean()})
    return pd.DataFrame(result)


def _to_days_since_1950(dt_index: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    return ((pd.to_datetime(dt_index) - REFERENCE_DATE) / pd.Timedelta(days=1)).to_numpy(dtype=np.float64)


def build_sbe26_dataset(
    df: pd.DataFrame,
    row: Any,
    pressure_threshold: float = 37.0,
    window_seconds: int = 4 * 60,
    output_freq: str = "5min",
    edge_trim_minutes: int = 5,
    center_offset_minutes: int = 2,
) -> tuple[pd.DataFrame, xr.Dataset, dict[str, Any]]:
    """Filter, average, and build a proc-ready xarray dataset for SBE26."""
    in_water = df[df["Pressure [db]"] > float(pressure_threshold)].sort_index().copy()
    if in_water.empty:
        raise ValueError("No in-water data found above threshold.")

    start_time = pd.to_datetime(in_water.index[0])
    end_time = pd.to_datetime(in_water.index[-1])

    r_start = start_time.ceil(output_freq) + pd.Timedelta(minutes=edge_trim_minutes)
    r_end = end_time.floor(output_freq) - pd.Timedelta(minutes=edge_trim_minutes)
    if r_end <= r_start:
        raise ValueError(
            f"Averaging window is empty after edge trimming. start={r_start}, end={r_end}"
        )

    time_series = pd.date_range(start=r_start, end=r_end, freq=output_freq) + pd.Timedelta(
        minutes=center_offset_minutes
    )

    averaged_dfp = centered_rolling_avg(in_water, window_seconds, time_series, "Pressure [db]").rename(
        columns={"average": "PRES"}
    )
    averaged_dft = centered_rolling_avg(in_water, window_seconds, time_series, "TEMP").rename(
        columns={"average": "TEMP"}
    )
    averaged = pd.concat([averaged_dfp, averaged_dft.drop(columns=["Time"])], axis=1)

    time_days = _to_days_since_1950(averaged["Time"])
    n_time = len(averaged)

    lat_val = float(row.get("latitude", np.nan))
    lon_val = float(row.get("longitude", np.nan))
    nom_depth = float(row.get("nominal_depth", np.nan))

    ds = xr.Dataset(
        data_vars={
            "PRES": ("TIME", averaged["PRES"].to_numpy(dtype=np.float32)),
            "TEMP": ("TIME", averaged["TEMP"].to_numpy(dtype=np.float32)),
            "PRES_quality_control": ("TIME", np.ones(n_time, dtype=np.int8)),
            "TEMP_quality_control": ("TIME", np.ones(n_time, dtype=np.int8)),
        },
        coords={"TIME": time_days},
        attrs={
            "instrument": str(row.get("inst_type", "SBE26")),
            "serial": str(row.get("inst_id", "")),
            "mooring_channels": str(row.get("mooring_channels", "PT")),
            "time_coverage_start": str(pd.to_datetime(r_start)),
            "time_coverage_end": str(pd.to_datetime(r_end)),
        },
    )

    ds["LATITUDE"] = xr.DataArray(lat_val)
    ds["LONGITUDE"] = xr.DataArray(lon_val)
    ds["NOMINAL_DEPTH"] = xr.DataArray(np.float32(nom_depth))
    ds["TIME"].attrs["units"] = "days since 1950-01-01T00:00:00 UTC"
    ds["TIME"].attrs["calendar"] = "gregorian"

    info = {
        "raw_start": start_time,
        "raw_end": end_time,
        "time_coverage_start": pd.to_datetime(r_start),
        "time_coverage_end": pd.to_datetime(r_end),
        "averaged_samples": int(n_time),
        "mean_pressure": float(in_water["Pressure [db]"].mean()),
        "mean_temperature": float(in_water["TEMP"].mean()),
    }

    return averaged, ds, info


def build_sbe26_dataset_from_metadata(
    row: Any,
    cwd: str | Path | None = None,
    pressure_threshold: float = 37.0,
    window_seconds: int = 4 * 60,
    output_freq: str = "5min",
    edge_trim_minutes: int = 5,
    center_offset_minutes: int = 2,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, xr.Dataset, Path, dict[str, Any]]:
    """Resolve input from metadata, parse, average, and build xarray dataset.

    Returns:
        (raw_df, averaged_df, ds, input_file, info)
    """
    input_file = _resolve_input_file(row=row, cwd=cwd)
    raw_df = read_sbe26_file(input_file=input_file, verbose=verbose)

    averaged_df, ds, info = build_sbe26_dataset(
        df=raw_df,
        row=row,
        pressure_threshold=pressure_threshold,
        window_seconds=window_seconds,
        output_freq=output_freq,
        edge_trim_minutes=edge_trim_minutes,
        center_offset_minutes=center_offset_minutes,
    )

    if verbose:
        print(f"Averaged samples: {info['averaged_samples']}")
        print(f"Coverage: {info['time_coverage_start']} -> {info['time_coverage_end']}")

    return raw_df, averaged_df, ds, input_file, info


__all__ = [
    "read_sbe26_file",
    "centered_rolling_avg",
    "build_sbe26_dataset",
    "build_sbe26_dataset_from_metadata",
]
