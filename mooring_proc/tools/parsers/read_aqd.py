"""AQD parser for raw Nortek text files."""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none"}


def _coerce_row(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        if "metadata_row" in config and config["metadata_row"] is not None:
            return _coerce_row(config["metadata_row"])
        return dict(config)
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return {}


def _resolve_input_path(input_path: Any, row: dict[str, Any]) -> Path:
    if not _is_blank(input_path):
        resolved = Path(str(input_path)).expanduser()
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
        else:
            resolved = resolved.resolve()
        return resolved

    data_in_path = row.get("data_in_path")
    data_in_file = row.get("data_in_file")
    if _is_blank(data_in_path) or _is_blank(data_in_file):
        raise ValueError("AQD input path is missing; provide input_path or metadata data_in_path/data_in_file.")

    base_dir = Path(str(data_in_path)).expanduser()
    if not base_dir.is_absolute():
        base_dir = (Path.cwd() / base_dir).resolve()
    else:
        base_dir = base_dir.resolve()
    return base_dir / str(data_in_file).strip()


def _parse_aqd_line(line: str) -> tuple[datetime, list[float]] | None:
    if len(line) < 21:
        return None

    date_tokens = line[:19].split()
    if len(date_tokens) != 6:
        return None

    try:
        month, day, year, hour, minute, second = [int(token) for token in date_tokens]
        timestamp = datetime(year, month, day, hour, minute, second)
    except (TypeError, ValueError):
        return None

    fields = line[20:].split()
    if len(fields) < 19:
        return None

    try:
        numeric_fields = [float(value) for value in fields[:19]]
    except ValueError:
        return None

    return timestamp, numeric_fields


def _build_dataframe(input_file: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with open(input_file, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parsed = _parse_aqd_line(line)
            if parsed is None:
                continue
            timestamp, numeric_fields = parsed
            speed = numeric_fields[17]
            direction_radians = math.radians(numeric_fields[18])
            rows.append(
                {
                    "datetime": timestamp,
                    "Temperature": numeric_fields[14],
                    "Depth": numeric_fields[13],
                    "UCUR": speed * math.sin(direction_radians) * 100.0,
                    "VCUR": speed * math.cos(direction_radians) * 100.0,
                }
            )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise ValueError(f"No valid AQD records parsed from {input_file}")
    return dataframe


def _range_tag(time_values: pd.Series) -> str:
    time_start = pd.to_datetime(time_values.min())
    time_end = pd.to_datetime(time_values.max())
    return f"{time_start:%Y%m%dT%H%M%S}_{time_end:%Y%m%dT%H%M%S}"


def _build_dataset(dataframe: pd.DataFrame, row: dict[str, Any], input_file: Path) -> xr.Dataset:
    n_time = len(dataframe)
    dataset = xr.Dataset(
        data_vars={
            "TEMP": ("TIME", dataframe["Temperature"].to_numpy(dtype=np.float32)),
            "DEPTH": ("TIME", dataframe["Depth"].to_numpy(dtype=np.float32)),
            "UCUR": ("TIME", dataframe["UCUR"].to_numpy(dtype=np.float32)),
            "VCUR": ("TIME", dataframe["VCUR"].to_numpy(dtype=np.float32)),
            "TEMP_quality_control": ("TIME", np.ones(n_time, dtype=np.int8)),
            "DEPTH_quality_control": ("TIME", np.ones(n_time, dtype=np.int8)),
            "UCUR_quality_control": ("TIME", np.ones(n_time, dtype=np.int8)),
            "VCUR_quality_control": ("TIME", np.ones(n_time, dtype=np.int8)),
        },
        coords={"TIME": pd.to_datetime(dataframe["datetime"]).to_numpy()},
        attrs={
            "source_file": str(input_file),
            "instrument": str(row.get("inst_type", "AQD")),
            "serial": str(row.get("inst_id", "")),
            "location": str(row.get("location", "")),
            "deployment_id": str(row.get("deployment_id", "")),
            "range_tag": _range_tag(dataframe["datetime"]),
        },
    )

    dataset["LATITUDE"] = xr.DataArray(np.float64(row.get("latitude", np.nan)))
    dataset["LONGITUDE"] = xr.DataArray(np.float64(row.get("longitude", np.nan)))
    dataset["NOMINAL_DEPTH"] = xr.DataArray(np.float32(row.get("nominal_depth", np.nan)))
    return dataset


def read_aqd(input_path, config=None):
    """Read AQD input and return dataframe, dataset, and file metadata."""
    row = _coerce_row(config)
    resolved_input_path = _resolve_input_path(input_path, row)
    dataframe = _build_dataframe(resolved_input_path)
    dataset = _build_dataset(dataframe, row, resolved_input_path)
    return {
        "dataframe": dataframe,
        "dataset": dataset,
        "input_path": str(resolved_input_path),
        "range_tag": dataset.attrs["range_tag"],
    }
