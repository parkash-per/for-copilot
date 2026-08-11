"""Window-based QC helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _time_values_to_datetime(time_values):
    values = np.asarray(time_values)
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values)
    return pd.to_datetime("1950-01-01") + pd.to_timedelta(values, unit="D")


def _coerce_row(window_spec: Any) -> dict[str, Any]:
    if isinstance(window_spec, dict):
        row = window_spec.get("row", window_spec.get("metadata_row"))
        if row is None:
            return window_spec
        if hasattr(row, "to_dict"):
            return row.to_dict()
        if isinstance(row, dict):
            return row
    if hasattr(window_spec, "to_dict"):
        return window_spec.to_dict()
    return {}


def _normalize_window(window: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "start": window.get("start", ""),
        "end": window.get("end", ""),
        "flag": int(window.get("flag", 4)),
    }
    if "qc_vars" in window:
        normalized["qc_vars"] = list(window["qc_vars"]) if isinstance(window["qc_vars"], (list, tuple, set)) else window["qc_vars"]
    if "comment" in window:
        normalized["comment"] = window["comment"]
    return normalized


def _resolve_bounds(dataset, window_spec: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    row = _coerce_row(window_spec)
    time_values = _time_values_to_datetime(dataset["TIME"].values)
    file_start = pd.to_datetime(time_values.min())
    file_end = pd.to_datetime(time_values.max())

    start_raw = (
        window_spec.get("time_coverage_start")
        or row.get("time_coverage_start")
        or row.get("deploy_date")
        or file_start
    )
    end_raw = (
        window_spec.get("time_coverage_end")
        or row.get("time_coverage_end")
        or row.get("recovery_date")
        or file_end
    )

    start_time = pd.to_datetime(start_raw, dayfirst=True, format="mixed", errors="coerce")
    end_time = pd.to_datetime(end_raw, dayfirst=True, format="mixed", errors="coerce")
    if pd.isna(start_time):
        start_time = file_start
    if pd.isna(end_time):
        end_time = file_end
    return start_time, end_time


def build_qc_windows(dataset, window_spec):
    """Build deployment-window QC windows for AQD workflows."""
    if "TIME" not in dataset:
        raise KeyError("TIME not found in dataset")

    if isinstance(window_spec, (list, tuple)):
        return [_normalize_window(window) for window in window_spec]

    if isinstance(window_spec, dict) and isinstance(window_spec.get("windows"), (list, tuple)):
        return [_normalize_window(window) for window in window_spec["windows"]]

    spec = dict(window_spec or {})
    start_time, end_time = _resolve_bounds(dataset, spec)
    flag = int(spec.get("flag", 4))
    qc_vars = spec.get("qc_vars")
    comment = spec.get("comment", "outside deployment window")

    windows = [
        {"start": "", "end": start_time.strftime("%Y-%m-%d %H:%M:%S"), "flag": flag, "comment": comment},
        {"start": end_time.strftime("%Y-%m-%d %H:%M:%S"), "end": "", "flag": flag, "comment": comment},
    ]
    if qc_vars is not None:
        for window in windows:
            window["qc_vars"] = qc_vars
    return windows
