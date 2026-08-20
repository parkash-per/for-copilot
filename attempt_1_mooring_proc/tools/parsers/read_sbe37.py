"""SBE37 parser: CNV, ASC, and HEX/XML/XMLCON (metadata-only) support.

Parser paths
------------
``.cnv``
    SeaBird CNV processed output.  Variable names are mapped to IMOS
    standard names via ``convert_sbe_var``.  Time is reconstructed from
    the elapsed-time column in the file (``timeS``, ``timeM``, etc.) or
    from the ``start_time`` header line.

``.asc``
    SeaBird ASC ASCII output for SBE37
    (``TEMP CNDC PRES_REL PSAL date time`` columns).

``.hex``, ``.XML``, ``.xmlcon``, ``.hdr``
    Treated as metadata-only sources.  No data parsing is performed.
    Callers can combine them with a matching ``.cnv`` or ``.asc`` file.

Reuses
------
``convert_sbe_var`` from this package for IMOS variable-name mapping.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .convert_sbe_var import convert_sbe_var, _matlab_datenum
from .imos_utils import (
    coerce_row,
    is_blank,
    matlab_datenum_to_datetime64,
    range_tag_from_times,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def read_sbe37(input_path, config=None):
    """Read SBE37 data and return dataframe, dataset, and file metadata.

    Parameters
    ----------
    input_path:
        Path to a ``.cnv`` or ``.asc`` file, or ``None`` to resolve from
        *config* (keys ``data_in_path`` + ``data_in_file``).
    config:
        Metadata dict or ``{'metadata_row': <row>}``.

    Returns
    -------
    dict with keys ``dataframe``, ``dataset``, ``input_path``, ``range_tag``.
    """
    row = coerce_row(config)
    resolved = _resolve_input_path(input_path, row)
    suffix = resolved.suffix.lower()

    metadata_only_suffixes = {".hex", ".xml", ".xmlcon", ".hdr"}
    if suffix in metadata_only_suffixes:
        companion = _resolve_companion_data_file(resolved)
        if companion is None:
            raise ValueError(
                f"SBE37 parser: {suffix} files are metadata-only and no matching .cnv/.asc file was found."
            )
        resolved = companion
        suffix = resolved.suffix.lower()

    if suffix == ".cnv":
        dataframe, dataset = _parse_cnv(resolved, row)
    elif suffix == ".asc":
        dataframe, dataset = _parse_asc(resolved, row)
    else:
        raise ValueError(
            f"SBE37 parser: unsupported suffix '{suffix}'.  "
            "Supported: .cnv, .asc"
        )

    range_tag = dataset.attrs["range_tag"]
    return {
        "dataframe": dataframe,
        "dataset": dataset,
        "input_path": str(resolved),
        "range_tag": range_tag,
    }


# ---------------------------------------------------------------------------
# CNV parser
# ---------------------------------------------------------------------------

def _parse_cnv(source_file: Path, row: dict[str, Any]):
    content = source_file.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()

    inst_header_lines: list[str] = []
    proc_header_lines: list[str] = []
    data_start_idx = 0

    for idx, line in enumerate(lines):
        s = line.strip()
        if s.startswith("*") and s != "*END*":
            inst_header_lines.append(s)
        elif s.startswith("#"):
            proc_header_lines.append(s)
        elif s == "*END*":
            data_start_idx = idx + 1
            break

    inst_header = _parse_inst_header(inst_header_lines)
    proc_header = _parse_proc_header(proc_header_lines)

    column_names: list[str] = proc_header.get("columns", [])
    bad_flag: float = proc_header.get("badFlag", -9.99e-29)
    start_time_datenum: float = proc_header.get("startTime", 0.0)

    if not column_names:
        raise ValueError(f"No variable names found in CNV header: {source_file}")

    data_rows: list[list[float]] = []
    for line in lines[data_start_idx:]:
        s = line.strip()
        if not s or s.startswith(("*", "#")):
            continue
        try:
            values = [float(x) for x in s.split()]
            if len(values) == len(column_names):
                values = [np.nan if v == bad_flag else v for v in values]
                data_rows.append(values)
        except ValueError:
            continue

    if not data_rows:
        raise ValueError(f"No valid data rows found in {source_file}")

    arr = np.array(data_rows)
    data_dict: dict[str, np.ndarray] = {}
    comment_dict: dict[str, str] = {}

    for col_idx, raw_name in enumerate(column_names):
        imos_name, converted, comment = convert_sbe_var(
            name=raw_name,
            data=arr[:, col_idx],
            time_offset=start_time_datenum,
            mode="timeSeries",
            inst_header=inst_header,
            proc_header=proc_header,
        )
        if not imos_name:
            continue
        final_name = imos_name
        count = 1
        while final_name in data_dict:
            final_name = f"{imos_name}_{count}"
            count += 1
        data_dict[final_name] = converted
        comment_dict[final_name] = comment

    # Generate TIME if not present
    if "TIME" not in data_dict:
        data_dict["TIME"] = _generate_timestamps(inst_header, len(next(iter(data_dict.values()))))
        comment_dict["TIME"] = "Generated from header information"

    return _build_result(data_dict, comment_dict, inst_header, source_file, row)


# ---------------------------------------------------------------------------
# ASC parser  (SBE37 ASCII export)
# ---------------------------------------------------------------------------

_ASC_PATTERNS = [
    # temperature conductivity pressure salinity date time
    re.compile(
        r"^\s*(?P<temp>[-\d.]+)\s+(?P<cndc>[-\d.]+)\s+(?P<pres>[-\d.]+)"
        r"\s+(?P<psal>[-\d.]+)\s+"
        r"(?P<month>\d+)/(?P<day>\d+)/(?P<year>\d+)\s+"
        r"(?P<hour>\d+):(?P<minute>\d+):(?P<second>\d+)"
    ),
]


def _parse_asc(source_file: Path, row: dict[str, Any]):
    rows: list[dict[str, Any]] = []

    with open(source_file, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith(("%", "#", "*")):
                continue
            for pat in _ASC_PATTERNS:
                m = pat.match(stripped)
                if m:
                    try:
                        dt = datetime(
                            int(m.group("year")), int(m.group("month")), int(m.group("day")),
                            int(m.group("hour")), int(m.group("minute")), int(m.group("second")),
                        )
                    except ValueError:
                        continue
                    rows.append({
                        "datetime": dt,
                        "TEMP": float(m.group("temp")),
                        "CNDC": float(m.group("cndc")),
                        "PRES_REL": float(m.group("pres")),
                        "PSAL": float(m.group("psal")),
                    })
                    break

    if not rows:
        raise ValueError(f"No valid SBE37 ASC records found in {source_file}")

    df = pd.DataFrame(rows)
    time64 = pd.to_datetime(df["datetime"]).to_numpy()
    data_dict = {
        "TEMP": df["TEMP"].to_numpy(dtype=np.float32),
        "CNDC": df["CNDC"].to_numpy(dtype=np.float32),
        "PRES_REL": df["PRES_REL"].to_numpy(dtype=np.float32),
        "PSAL": df["PSAL"].to_numpy(dtype=np.float32),
    }
    comment_dict = {k: "" for k in data_dict}
    inst_header: dict[str, Any] = {
        "instrument_model": "SBE37",
        "instrument_make": "Seabird",
        "instrument_serial_no": str(row.get("inst_id", "")),
    }
    return _build_result(
        {"TIME": time64, **data_dict}, comment_dict, inst_header, source_file, row,
        time_already_datetime64=True,
    )


# ---------------------------------------------------------------------------
# Header parsers
# ---------------------------------------------------------------------------

def _parse_inst_header(lines: list[str]) -> dict[str, Any]:
    header: dict[str, Any] = {}
    for line in lines:
        if m := re.search(r"\*\s*(?:Sea-Bird\s+)?(SBE[\s\-]?\S+|SeacatPlus)\s+V\s+(\S+)\s+SERIAL NO\.\s+(\d+)", line, re.IGNORECASE):
            header.setdefault("instrument_model", m.group(1).replace(" ", ""))
            header["instrument_firmware"] = m.group(2)
            header["instrument_serial_no"] = m.group(3)
        elif m := re.search(r"sample interval = (\d+) (\w+)", line):
            val = int(m.group(1))
            unit = m.group(2).lower()
            header["sampleInterval"] = val if "second" in unit else val * 60
        elif m := re.search(r"cast\s+\d+\s+(\d+ \w+ \d+ \d+:\d+:\d+)", line):
            header["castDate"] = _parse_sbe_datetime(m.group(1))
        elif m := re.search(r"Cast Time = (\w+ \d+ \d+ \d+:\d+:\d+)", line):
            header.setdefault("castDate", _parse_sbe_datetime(m.group(1)))
    return header


def _parse_proc_header(lines: list[str]) -> dict[str, Any]:
    header: dict[str, Any] = {"columns": []}
    for line in lines:
        if m := re.search(r"# name \d+ = ([^:]+):", line):
            header["columns"].append(m.group(1).strip())
        elif m := re.search(r"# bad_flag = (.+)$", line):
            try:
                header["badFlag"] = float(m.group(1).strip())
            except ValueError:
                pass
        elif m := re.search(r"# start_time = (\w+ \d+ \d+ \d+:\d+:\d+)", line):
            header["startTime"] = _parse_sbe_datetime(m.group(1).strip())
    return header


def _parse_sbe_datetime(text: str) -> float:
    """Parse a SeaBird header datetime string to MATLAB datenum."""
    for fmt in ("%b %d %Y %H:%M:%S", "%d %b %Y %H:%M:%S"):
        try:
            dt = datetime.strptime(text.strip(), fmt)
            return _matlab_datenum(dt)
        except ValueError:
            continue
    return 0.0


def _generate_timestamps(inst_header: dict[str, Any], n_samples: int) -> np.ndarray:
    """Generate MATLAB datenum timestamps from header info."""
    interval_s = inst_header.get("sampleInterval", 60)
    start = inst_header.get("castDate", 0.0)
    if start == 0.0:
        start = _matlab_datenum(datetime(2000, 1, 1))
    return start + np.arange(n_samples) * interval_s / 86400.0


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

_PRES_REL_ATTRS: dict[str, Any] = {
    "applied_offset": np.float32(-14.7 * 0.689476),
    "comment": "Relative pressure with atmospheric offset (~-10.13 dbar) applied.",
}


def _build_result(
    data_dict: dict[str, np.ndarray],
    comment_dict: dict[str, str],
    inst_header: dict[str, Any],
    source_file: Path,
    row: dict[str, Any],
    time_already_datetime64: bool = False,
) -> tuple[pd.DataFrame, xr.Dataset]:
    coordinates = "TIME LATITUDE LONGITUDE NOMINAL_DEPTH"

    if time_already_datetime64:
        time64 = data_dict["TIME"]
    else:
        time64 = matlab_datenum_to_datetime64(data_dict["TIME"])

    primary_vars = [k for k in data_dict if k != "TIME"]

    xr_vars: dict[str, xr.DataArray] = {
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
    }

    for vname in primary_vars:
        attrs: dict[str, Any] = {"coordinates": coordinates}
        comment = comment_dict.get(vname, "")
        if comment:
            attrs["comment"] = comment
        if vname.startswith("PRES_REL"):
            attrs.update(_PRES_REL_ATTRS)
        xr_vars[vname] = xr.DataArray(data_dict[vname], dims=["TIME"], attrs=attrs)
        xr_vars[f"{vname}_quality_control"] = xr.DataArray(
            np.ones(len(time64), dtype=np.int8), dims=["TIME"]
        )

    range_tag = range_tag_from_times(time64)

    sample_interval = inst_header.get(
        "instrument_sample_interval",
        inst_header.get("sampleInterval", float("nan")),
    )

    ds = xr.Dataset(
        data_vars=xr_vars,
        coords={"TIME": time64},
        attrs={
            "source_file": str(source_file),
            "instrument": str(inst_header.get("instrument_model", row.get("inst_type", "SBE37"))),
            "instrument_make": str(inst_header.get("instrument_make", "Seabird")),
            "instrument_serial_no": str(
                inst_header.get("instrument_serial_no", row.get("inst_id", ""))
            ),
            "instrument_firmware": str(inst_header.get("instrument_firmware", "")),
            "instrument_sample_interval": float(sample_interval)
            if not isinstance(sample_interval, float)
            else sample_interval,
            "serial": str(row.get("inst_id", "")),
            "location": str(row.get("location", "")),
            "deployment_id": str(row.get("deployment_id", "")),
            "range_tag": range_tag,
        },
    )

    # Build dataframe (TIME as datetime index)
    df_data = {vname: data_dict[vname] for vname in primary_vars}
    df = pd.DataFrame(df_data, index=pd.DatetimeIndex(time64, name="TIME"))

    return df, ds


# ---------------------------------------------------------------------------
# Path resolver (mirrors read_aqd convention)
# ---------------------------------------------------------------------------

def _resolve_input_path(input_path: Any, row: dict[str, Any]) -> Path:
    if not is_blank(input_path):
        p = Path(str(input_path)).expanduser()
        return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
    data_in_path = row.get("data_in_path")
    data_in_file = row.get("data_in_file")
    if is_blank(data_in_path) or is_blank(data_in_file):
        raise ValueError(
            "SBE37 input path missing; provide input_path or "
            "metadata data_in_path/data_in_file."
        )
    base = Path(str(data_in_path)).expanduser()
    base = base.resolve() if base.is_absolute() else (Path.cwd() / base).resolve()
    return base / str(data_in_file).strip()


def _resolve_companion_data_file(metadata_file: Path) -> Path | None:
    preferred = [metadata_file.with_suffix(".cnv"), metadata_file.with_suffix(".asc")]
    for candidate in preferred:
        if candidate.exists():
            return candidate
    for suffix in ("*.cnv", "*.asc"):
        matches = sorted(metadata_file.parent.glob(suffix))
        if matches:
            return matches[0]
    return None
