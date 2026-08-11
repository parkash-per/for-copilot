"""Writers for IMOS-style output products."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


TIME_UNITS = "days since 1950-01-01T00:00:00 UTC"

VARIABLE_ATTRS = {
    "TEMP": {"units": "degrees_Celsius", "long_name": "sea_water_temperature"},
    "DEPTH": {"units": "m", "long_name": "instrument_depth"},
    "UCUR": {"units": "m/s", "long_name": "eastward_sea_water_velocity"},
    "VCUR": {"units": "m/s", "long_name": "northward_sea_water_velocity"},
    "LATITUDE": {"units": "degrees_north"},
    "LONGITUDE": {"units": "degrees_east"},
    "NOMINAL_DEPTH": {"units": "m"},
}


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none"}


def _normalize_processing_version(version: Any) -> str:
    if _is_blank(version):
        return ""
    text = str(version).strip()
    try:
        numeric = int(float(text))
        if 0 <= numeric < 100:
            return f"{numeric:02d}"
    except (TypeError, ValueError):
        pass
    return text


def _parse_datetime(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, format="mixed", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, dayfirst=True, format="mixed", errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Unable to parse datetime value: {value}")
    return parsed


def _time_values_to_datetime(time_values) -> pd.DatetimeIndex:
    values = np.asarray(time_values)
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values)
    return pd.to_datetime("1950-01-01") + pd.to_timedelta(values, unit="D")


def _to_days_since_1950(time_values) -> np.ndarray:
    timestamps = pd.to_datetime(time_values)
    reference = pd.Timestamp("1950-01-01")
    return ((timestamps - reference) / pd.Timedelta(days=1)).to_numpy(dtype=np.float64)


def _site_token(value: Any) -> str:
    return str(value or "").strip().replace(" ", "")


def _instrument_token(value: Any) -> str:
    return str(value or "").strip().upper().replace("_", "").replace(" ", "")


def _depth_token(value: Any) -> str:
    try:
        return f"{int(round(float(value)))}m"
    except (TypeError, ValueError):
        return f"{str(value).strip()}m"


def build_output_filename(metadata=None):
    """Build an internal or delivery filename from metadata."""
    metadata = metadata or {}
    output_name_mode = str(metadata.get("output_name_mode", "internal")).strip().lower()
    start_time = _parse_datetime(
        metadata.get("start_of_good_data")
        or metadata.get("time_coverage_start")
        or metadata.get("deploy_date")
    )
    if output_name_mode == "imos":
        version = _normalize_processing_version(metadata.get("version", "1"))
        return (
            f"IMOS_SRSALT_{metadata.get('inst_channels', metadata.get('mooring_channels', ''))}_"
            f"{start_time.strftime('%Y%m%dT%H%M%SZ')}_{_site_token(metadata.get('location'))}_"
            f"FV{version}_{_instrument_token(metadata.get('instrument', metadata.get('inst_type', 'AQD')))}"
            f"d{int(round(float(metadata.get('depth', metadata.get('nominal_depth', 0)))))}m.nc"
        )

    return (
        f"{_site_token(metadata.get('location'))}_{start_time.strftime('%Y%m')}_"
        f"{_instrument_token(metadata.get('instrument', metadata.get('inst_type', 'AQD')))}_"
        f"{str(metadata.get('inst_id', metadata.get('serial', ''))).strip()}_"
        f"{_depth_token(metadata.get('depth', metadata.get('nominal_depth', '')))}.nc"
    )


def _resolve_output_path(output_path, metadata=None) -> Path:
    metadata = metadata or {}
    if _is_blank(output_path):
        directory = metadata.get("output_dir")
        if _is_blank(directory):
            raise ValueError("An output path or output_dir metadata value is required.")
        base_path = Path(str(directory)).expanduser()
    else:
        base_path = Path(str(output_path)).expanduser()

    if base_path.suffix.lower() == ".nc":
        resolved_path = base_path
    else:
        resolved_path = base_path / build_output_filename(metadata)

    if not resolved_path.is_absolute():
        resolved_path = (Path.cwd() / resolved_path).resolve()
    else:
        resolved_path = resolved_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path


def _apply_variable_attrs(dataset: xr.Dataset) -> xr.Dataset:
    prepared = dataset.copy(deep=True)
    for variable_name, attrs in VARIABLE_ATTRS.items():
        if variable_name in prepared.variables:
            prepared[variable_name].attrs.update(attrs)
    for variable_name in prepared.data_vars:
        if variable_name.endswith("_quality_control"):
            prepared[variable_name].attrs.update(
                {
                    "long_name": f"{variable_name.removesuffix('_quality_control')} quality control flag",
                    "flag_values": np.array([1, 3, 4, 5], dtype=np.int8),
                    "flag_meanings": "good_data probably_bad_data bad_data missing_data",
                }
            )
    return prepared


def _apply_global_attrs(dataset: xr.Dataset, metadata: dict[str, Any]) -> xr.Dataset:
    prepared = dataset.copy(deep=True)
    time_values = _time_values_to_datetime(prepared["TIME"].values)
    if len(time_values) == 0:
        raise ValueError("Cannot write an empty dataset.")
    derived_start = pd.to_datetime(time_values.min())
    derived_end = pd.to_datetime(time_values.max())
    derived_start_text = "" if pd.isna(derived_start) else derived_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    derived_end_text = "" if pd.isna(derived_end) else derived_end.strftime("%Y-%m-%dT%H:%M:%SZ")
    attrs = dict(prepared.attrs)
    attrs.update(
        {
            "instrument": str(metadata.get("inst_type", metadata.get("instrument", attrs.get("instrument", "AQD")))),
            "serial": str(metadata.get("inst_id", metadata.get("serial", attrs.get("serial", "")))),
            "location": str(metadata.get("location", attrs.get("location", ""))),
            "deployment_id": str(metadata.get("deployment_id", attrs.get("deployment_id", ""))),
            "mooring_channels": str(metadata.get("mooring_channels", metadata.get("inst_channels", attrs.get("mooring_channels", "")))),
            "processing_version": _normalize_processing_version(metadata.get("version", attrs.get("processing_version", ""))),
            "time_coverage_start": str(metadata.get("time_coverage_start") or derived_start_text),
            "time_coverage_end": str(metadata.get("time_coverage_end") or derived_end_text),
            "output_stage": str(metadata.get("output_stage", attrs.get("output_stage", ""))),
        }
    )
    prepared.attrs = attrs
    return prepared


def _prepare_dataset_for_write(dataset: xr.Dataset, metadata: dict[str, Any]) -> tuple[xr.Dataset, dict[str, Any]]:
    prepared = _apply_variable_attrs(dataset)
    prepared = _apply_global_attrs(prepared, metadata)
    prepared = prepared.copy(deep=True)

    time_values = _time_values_to_datetime(prepared["TIME"].values)
    prepared = prepared.assign_coords(TIME=_to_days_since_1950(time_values))
    prepared["TIME"].attrs.update({"units": TIME_UNITS, "calendar": "gregorian"})

    encoding = {"TIME": {"dtype": "float64"}}
    for variable_name in ("TEMP", "DEPTH", "UCUR", "VCUR"):
        if variable_name in prepared.variables:
            encoding[variable_name] = {"dtype": "float32"}
    for variable_name in ("TEMP_quality_control", "DEPTH_quality_control", "UCUR_quality_control", "VCUR_quality_control"):
        if variable_name in prepared.variables:
            prepared[variable_name] = prepared[variable_name].astype(np.int8)
            encoding[variable_name] = {"dtype": "int8"}
    return prepared, encoding


def write_imos_file(dataset, output_path, metadata=None):
    """Write an AQD proc or IMOS delivery NetCDF file."""
    if "TIME" not in dataset:
        raise KeyError("TIME not found in dataset")

    metadata = dict(metadata or {})
    resolved_output_path = _resolve_output_path(output_path, metadata)
    prepared_dataset, encoding = _prepare_dataset_for_write(dataset, metadata)
    if resolved_output_path.exists():
        resolved_output_path.unlink()
    prepared_dataset.to_netcdf(resolved_output_path, encoding=encoding)
    return str(resolved_output_path)
