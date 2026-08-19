"""RBRQ / RBR parser: Ruskin text exports, zip archives, and .rsk SQLite.

Supported input formats
-----------------------
``.rsk``
    RBR SQLite database (requires ``sqlite3``, part of the Python
    standard library).  Channels, sample timestamps, and data are read
    directly from the ``rawdata`` and ``channels`` tables.

Ruskin ZIP export
    A ``.zip`` produced by Ruskin software containing text files
    ``*_data.txt``, ``*_events.txt``, ``*_metadata.txt``.  The data
    file is a tab- or comma-separated table; the metadata file is used
    to extract serial number and sample interval.

``*_data.txt`` (plain)
    Same CSV/TSV text file as produced by a Ruskin zip, but extracted
    directly.

``*.txt`` / ``*.csv``
    Legacy or ad-hoc text exports with a datetime index (first column)
    and named columns ``pressure`` / ``temperature`` or
    ``BPR pressure`` / ``BPR temperature``.

Parser resolution order
-----------------------
1. If *input_path* ends in ``.rsk`` → SQLite path.
2. If *input_path* ends in ``.zip`` → extract and parse Ruskin zip.
3. Otherwise resolve via ``data_in_path`` / ``data_in_file`` from
   *config*, preferring ``*_data.txt`` files.
"""

from __future__ import annotations

import io
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .imos_utils import coerce_row, is_blank, range_tag_from_times


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def read_rbrq(input_path, config=None):
    """Read RBRQ / RBR data and return dataframe, dataset, and metadata.

    Parameters
    ----------
    input_path:
        Path to a ``.rsk``, ``.zip``, or ``*_data.txt`` file, or
        ``None`` to resolve from *config*.
    config:
        Metadata dict or ``{'metadata_row': <row>}``.

    Returns
    -------
    dict with keys ``dataframe``, ``dataset``, ``input_path``, ``range_tag``.
    """
    row = coerce_row(config)
    resolved = _resolve_input_path(input_path, row)
    suffix = resolved.suffix.lower()

    meta: dict[str, Any] = {}

    if suffix == ".rsk":
        df = _read_rsk(resolved, meta)
    elif suffix == ".zip":
        df, meta = _read_ruskin_zip(resolved)
    else:
        df = _read_text_file(resolved)

    if df.empty:
        raise ValueError(f"No RBRQ data parsed from {resolved}")

    dataset = _build_dataset(df, row, resolved, meta)
    range_tag = dataset.attrs["range_tag"]
    return {
        "dataframe": df,
        "dataset": dataset,
        "input_path": str(resolved),
        "range_tag": range_tag,
    }


# ---------------------------------------------------------------------------
# .rsk (SQLite) reader
# ---------------------------------------------------------------------------

def _read_rsk(rsk_file: Path, meta: dict[str, Any]) -> pd.DataFrame:
    """Read an RBR .rsk SQLite database."""
    con = sqlite3.connect(str(rsk_file))
    try:
        # Channels table: gives column names and units
        channels = pd.read_sql("SELECT * FROM channels", con)
        channel_map: dict[int, str] = {}
        for _, ch in channels.iterrows():
            ch_type = str(ch.get("shortName", ch.get("channelType", "unknown"))).lower()
            ch_id = int(ch.get("channelID", ch.get("id", -1)))
            channel_map[ch_id] = ch_type

        # Try rawdata table
        try:
            raw = pd.read_sql("SELECT * FROM rawdata ORDER BY tstamp", con)
        except Exception:
            raw = pd.read_sql("SELECT * FROM data ORDER BY tstamp", con)

        # tstamp is Unix time in milliseconds
        raw.index = pd.to_datetime(raw["tstamp"], unit="ms", utc=True).dt.tz_localize(None)
        raw.index.name = "TIME"

        # Map channel columns (usually named channel01, channel02, …)
        col_rename: dict[str, str] = {}
        for col in raw.columns:
            if col.lower().startswith("channel"):
                try:
                    ch_num = int(col[7:])
                    if ch_num in channel_map:
                        col_rename[col] = channel_map[ch_num]
                except ValueError:
                    pass
        raw = raw.rename(columns=col_rename)

        # Try to read instrument metadata
        try:
            deploy = pd.read_sql("SELECT * FROM deployments LIMIT 1", con)
            if not deploy.empty:
                meta["instrument_serial_no"] = str(deploy.iloc[0].get("serialID", ""))
                meta["instrument_sample_interval"] = float(
                    deploy.iloc[0].get("samplePeriod", float("nan")) or float("nan")
                )
        except Exception:
            pass

    finally:
        con.close()

    # Normalise column names to IMOS
    rename_map = {
        "pres": "BPR pressure",
        "pressure": "BPR pressure",
        "pres_dbar": "BPR pressure",
        "temp": "BPR temperature",
        "temperature": "BPR temperature",
        "temp_c": "BPR temperature",
    }
    raw = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})
    keep = [c for c in ("BPR pressure", "BPR temperature") if c in raw.columns]
    if not keep:
        # Return all numeric columns if mapping failed
        keep = list(raw.select_dtypes(include=np.number).columns)
    return raw[keep].dropna(how="all")


# ---------------------------------------------------------------------------
# Ruskin ZIP reader
# ---------------------------------------------------------------------------

def _read_ruskin_zip(zip_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract and parse a Ruskin zip export."""
    meta: dict[str, Any] = {}
    data_bytes: bytes | None = None
    metadata_bytes: bytes | None = None

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        names = zf.namelist()
        for name in names:
            lower = name.lower()
            if lower.endswith("_data.txt"):
                data_bytes = zf.read(name)
            elif lower.endswith("_metadata.txt"):
                metadata_bytes = zf.read(name)

    if data_bytes is None:
        # Fall back to first txt file
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith(".txt"):
                    data_bytes = zf.read(name)
                    break

    if data_bytes is None:
        raise ValueError(f"No data text file found in zip: {zip_path}")

    if metadata_bytes is not None:
        meta = _parse_ruskin_metadata(metadata_bytes.decode("utf-8", errors="ignore"))

    df = _parse_ruskin_data_text(io.StringIO(data_bytes.decode("utf-8", errors="ignore")))
    return df, meta


def _parse_ruskin_metadata(text: str) -> dict[str, Any]:
    """Extract key fields from a Ruskin *_metadata.txt file."""
    meta: dict[str, Any] = {}
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("serial"):
            parts = s.split(":", 1)
            if len(parts) == 2:
                meta["instrument_serial_no"] = parts[1].strip()
        elif "sample" in s.lower() and "period" in s.lower():
            parts = s.split(":", 1)
            if len(parts) == 2:
                try:
                    meta["instrument_sample_interval"] = float(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
    return meta


def _parse_ruskin_data_text(stream: Any) -> pd.DataFrame:
    """Parse a Ruskin *_data.txt (tab- or comma-separated)."""
    text = stream.read() if hasattr(stream, "read") else stream
    # Try tab-separated first, then comma
    for sep in ("\t", ","):
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, index_col=0, parse_dates=True)
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                df.index.name = "TIME"
                break
        except Exception:
            continue
    else:
        raise ValueError("Could not parse Ruskin data text file")

    # Normalise column names
    df = df.rename(
        columns={
            "pressure": "BPR pressure",
            "Pressure": "BPR pressure",
            "temperature": "BPR temperature",
            "Temperature": "BPR temperature",
        },
        errors="ignore",
    )
    keep = [c for c in ("BPR pressure", "BPR temperature") if c in df.columns]
    return df[keep] if keep else df


# ---------------------------------------------------------------------------
# Plain text file reader
# ---------------------------------------------------------------------------

def _read_text_file(path: Path) -> pd.DataFrame:
    """Read a plain CSV/TSV text file with a datetime index."""
    for sep in ("\t", ",", r"\s+"):
        try:
            df = pd.read_csv(path, sep=sep, index_col=0, parse_dates=True, engine="python")
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                df.index.name = "TIME"
                df = df.rename(
                    columns={
                        "pressure": "BPR pressure",
                        "Pressure": "BPR pressure",
                        "temperature": "BPR temperature",
                        "Temperature": "BPR temperature",
                    },
                    errors="ignore",
                )
                return df
        except Exception:
            continue
    raise ValueError(f"Could not parse RBRQ text file: {path}")


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _build_dataset(
    df: pd.DataFrame,
    row: dict[str, Any],
    source_file: Path,
    meta: dict[str, Any],
) -> xr.Dataset:
    time64 = df.index.to_numpy(dtype="datetime64[ns]")
    n = len(time64)
    coordinates = "TIME LATITUDE LONGITUDE NOMINAL_DEPTH"
    range_tag = range_tag_from_times(time64)

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

    # Map dataframe columns to IMOS variable names
    col_to_imos = {
        "BPR pressure": "PRES_REL",
        "BPR temperature": "TEMP",
    }
    for col, imos_name in col_to_imos.items():
        if col in df.columns:
            data = df[col].to_numpy(dtype=np.float32)
            attrs: dict[str, Any] = {"coordinates": coordinates}
            if imos_name == "PRES_REL":
                attrs["applied_offset"] = np.float32(-14.7 * 0.689476)
            xr_vars[imos_name] = xr.DataArray(data, dims=["TIME"], attrs=attrs)
            xr_vars[f"{imos_name}_quality_control"] = xr.DataArray(
                np.ones(n, dtype=np.int8), dims=["TIME"]
            )

    # Additional numeric columns get passed through under their original names
    for col in df.columns:
        if col not in col_to_imos:
            clean = col.replace(" ", "_").upper()
            if clean not in xr_vars:
                try:
                    data = df[col].to_numpy(dtype=np.float32)
                    xr_vars[clean] = xr.DataArray(
                        data, dims=["TIME"], attrs={"coordinates": coordinates}
                    )
                    xr_vars[f"{clean}_quality_control"] = xr.DataArray(
                        np.ones(n, dtype=np.int8), dims=["TIME"]
                    )
                except Exception:
                    pass

    # Sample interval
    if len(time64) > 1:
        diff_s = float(np.median(np.diff(time64.astype("int64"))) / 1e9)
    else:
        diff_s = float("nan")

    sample_interval = meta.get("instrument_sample_interval", diff_s)

    ds = xr.Dataset(
        data_vars=xr_vars,
        coords={"TIME": time64},
        attrs={
            "source_file": str(source_file),
            "instrument": str(row.get("inst_type", "RBRQ")),
            "instrument_make": "RBR",
            "instrument_model": str(row.get("inst_type", "RBRQ")),
            "instrument_serial_no": str(
                meta.get("instrument_serial_no", row.get("inst_id", ""))
            ),
            "instrument_sample_interval": float(sample_interval)
            if not isinstance(sample_interval, float)
            else sample_interval,
            "serial": str(row.get("inst_id", "")),
            "location": str(row.get("location", "")),
            "deployment_id": str(row.get("deployment_id", "")),
            "range_tag": range_tag,
        },
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

    if is_blank(data_in_path):
        raise ValueError(
            "RBRQ input path missing; provide input_path or "
            "metadata data_in_path/data_in_file."
        )

    base = Path(str(data_in_path)).expanduser()
    base = base.resolve() if base.is_absolute() else (Path.cwd() / base).resolve()

    if not is_blank(data_in_file):
        direct = base / str(data_in_file).strip()
        if direct.exists():
            return direct

    # Auto-discover: prefer *_data.txt, then .rsk, then .zip, then any .txt
    for pattern in ("*_data.txt", "*.rsk", "*.zip", "*.txt", "*.csv"):
        candidates = sorted(base.rglob(pattern))
        if candidates:
            return candidates[0]

    raise FileNotFoundError(f"No RBRQ data file found under {base}")

