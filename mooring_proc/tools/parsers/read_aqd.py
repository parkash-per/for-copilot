"""AQD parser for raw Nortek text files.

Supports two text export formats:

``.aqd`` — standard Nortek Aquadopp text export
    ``MM DD YYYY HH MM SS <19 numeric fields>``
    Fields 13 = Depth (m), 14 = Temperature (°C),
    17 = Speed (m/s), 18 = Direction (deg).

``.dat`` — historic Nortek export (adapter)
    Two common layouts are tried automatically:

    1. Same layout as ``.aqd`` — detected when the first six tokens form
       a valid date (no leading burst counter).
    2. Burst-counter-prefixed layout —
       ``BurstNo MM DD YYYY HH MM SS SoundSpeed Heading Pitch Roll
         Pressure(dbar) Temp(°C) East(m/s) North(m/s) Up(m/s) A1 A2 A3``

    **Format ambiguity note:** no canonical sample ``.dat`` files were
    available during development.  If your ``.dat`` files use a different
    column ordering, pass ``dat_layout='aqd'`` or ``dat_layout='burst'``
    via the ``config`` dict to force a specific path, or raise an issue
    with a sample file.
"""

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


# ---------------------------------------------------------------------------
# .dat historic format adapter
# ---------------------------------------------------------------------------

def _parse_dat_line_burst_prefix(
    tokens: list[str],
) -> dict[str, Any] | None:
    """Parse a Nortek .dat line with a leading burst counter.

    Expected layout (0-based token indices)::

        0         1   2    3     4    5   6       7          8
        BurstNo   MM  DD  YYYY   HH  MM  SS  SoundSpeed  Heading
            9      10    11       12    13     14    15   16  17  18
        Pitch   Roll  Pressure  Temp  East  North  Up   A1  A2  A3
    """
    if len(tokens) < 16:
        return None
    try:
        # Token 0 is the burst counter (integer — skip it)
        int(tokens[0])
        month, day, year = int(tokens[1]), int(tokens[2]), int(tokens[3])
        hour, minute, second = int(tokens[4]), int(tokens[5]), int(tokens[6])
        timestamp = datetime(year, month, day, hour, minute, second)
        # Pressure (dbar) at index 11, temperature at 12
        depth = float(tokens[11])
        temp = float(tokens[12])
        # ENU velocities (m/s) → store in cm/s to match .aqd convention
        ucur = float(tokens[13]) * 100.0
        vcur = float(tokens[14]) * 100.0
    except (ValueError, IndexError):
        return None
    return {"datetime": timestamp, "Temperature": temp, "Depth": depth,
            "UCUR": ucur, "VCUR": vcur}


def _build_dataframe_from_dat(input_file: Path, layout: str = "auto") -> pd.DataFrame:
    """Read a historic Nortek AQD ``.dat`` file.

    Parameters
    ----------
    input_file:
        Path to the ``.dat`` file.
    layout:
        ``'auto'`` — try the standard ``.aqd`` text layout first, then
        the burst-counter-prefixed layout.
        ``'aqd'`` — force the standard ``.aqd`` text layout.
        ``'burst'`` — force the burst-counter-prefixed layout.
    """
    rows: list[dict[str, Any]] = []
    with open(input_file, "rt", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(("%", "#", "/")):
                continue
            tokens = line.split()

            # --- try standard .aqd layout ---------------------------------
            if layout in ("auto", "aqd"):
                parsed = _parse_aqd_line(line)
                if parsed is not None:
                    timestamp, numeric_fields = parsed
                    if len(numeric_fields) >= 19:
                        speed = numeric_fields[17]
                        dir_rad = math.radians(numeric_fields[18])
                        rows.append({
                            "datetime": timestamp,
                            "Temperature": numeric_fields[14],
                            "Depth": numeric_fields[13],
                            "UCUR": speed * math.sin(dir_rad) * 100.0,
                            "VCUR": speed * math.cos(dir_rad) * 100.0,
                        })
                        continue
                    if layout == "aqd":
                        continue

            # --- try burst-counter-prefixed layout -----------------------
            if layout in ("auto", "burst"):
                row_dict = _parse_dat_line_burst_prefix(tokens)
                if row_dict is not None:
                    rows.append(row_dict)

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise ValueError(
            f"No valid records parsed from .dat file: {input_file}.  "
            "If the column layout differs from the two supported formats "
            "('aqd' and 'burst'), set dat_layout explicitly in config."
        )
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
    """Read AQD input and return dataframe, dataset, and file metadata.

    Supports ``.aqd`` (standard Nortek text export) and ``.dat`` (historic
    text export).  The appropriate parser is selected automatically based
    on the file suffix.

    Parameters
    ----------
    input_path:
        Path to the input file, or ``None`` to resolve from *config*.
    config:
        Metadata dict (or ``{'metadata_row': <row>}``).  Recognised keys:

        - Standard metadata keys (``latitude``, ``longitude``, etc.)
        - ``dat_layout``: ``'auto'`` *(default)*, ``'aqd'``, or
          ``'burst'`` — controls which column layout is tried for
          ``.dat`` files.
    """
    row = _coerce_row(config)
    resolved_input_path = _resolve_input_path(input_path, row)
    suffix = resolved_input_path.suffix.lower()

    if suffix == ".dat":
        dat_layout = str(row.get("dat_layout", "auto")).strip().lower()
        dataframe = _build_dataframe_from_dat(resolved_input_path, layout=dat_layout)
    else:
        dataframe = _build_dataframe(resolved_input_path)

    dataset = _build_dataset(dataframe, row, resolved_input_path)
    return {
        "dataframe": dataframe,
        "dataset": dataset,
        "input_path": str(resolved_input_path),
        "range_tag": dataset.attrs["range_tag"],
    }
