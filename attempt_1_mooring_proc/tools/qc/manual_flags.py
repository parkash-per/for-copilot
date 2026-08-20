"""Manual QC flagging helpers."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd


def _time_values_to_datetime(time_values):
    values = np.asarray(time_values)
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values)
    return pd.to_datetime("1950-01-01") + pd.to_timedelta(values, unit="D")


def _expand_qc_names(dataset, variable_names):
    if variable_names is None:
        return []
    expanded = []
    for name in variable_names:
        if name in dataset.variables:
            expanded.append(name)
            continue
        qc_name = f"{name}_quality_control"
        if qc_name in dataset.variables:
            expanded.append(qc_name)
    return expanded


def apply_qc_flag_windows(dataset, flag_windows, variable_names=None):
    """Apply QC flags over inclusive time windows."""
    if "TIME" not in dataset:
        raise KeyError("TIME not found in dataset")

    updated = dataset.copy(deep=True)
    time_values = _time_values_to_datetime(updated["TIME"].values)
    file_start = pd.to_datetime(time_values.min())
    file_end = pd.to_datetime(time_values.max())

    qc_variables = [name for name in updated.data_vars if name.endswith("_quality_control")]
    explicit_qc_variables = _expand_qc_names(updated, variable_names)
    if explicit_qc_variables:
        qc_variables = explicit_qc_variables

    for window in flag_windows or []:
        start_value = file_start if window.get("start") in {None, ""} else pd.to_datetime(window["start"])
        end_value = file_end if window.get("end") in {None, ""} else pd.to_datetime(window["end"])
        if end_value < start_value:
            raise ValueError("QC window end precedes start.")

        mask = (time_values >= start_value) & (time_values <= end_value)
        requested_qc_vars = window.get("qc_vars")
        if requested_qc_vars is None:
            window_qc_variables = qc_variables
        else:
            window_qc_variables = _expand_qc_names(updated, requested_qc_vars)

        for qc_name in window_qc_variables:
            values = updated[qc_name].values.copy()
            values[mask] = np.int8(window["flag"])
            updated[qc_name].values = values

    return updated


def write_manual_qc_flags_txt(flag_windows, output_path, metadata=None):
    """Write manual QC flag windows beside an output NetCDF file."""
    metadata = metadata or {}
    suffix = metadata.get("filename_suffix", "_manual_qc_flags.txt")
    output_file = Path(str(output_path)).expanduser()
    if not output_file.is_absolute():
        output_file = (Path.cwd() / output_file).resolve()
    else:
        output_file = output_file.resolve()
    text_path = output_file.with_name(f"{output_file.stem}{suffix}")

    rows = []
    for window in flag_windows or []:
        qc_vars = window.get("qc_vars")
        if qc_vars in (None, ""):
            qc_list = ["ALL"]
        elif isinstance(qc_vars, (list, tuple, set)):
            qc_list = [str(value) for value in qc_vars]
        else:
            qc_list = [str(qc_vars)]

        for qc_name in qc_list:
            rows.append(
                [
                    qc_name,
                    str(window.get("flag", "")),
                    str(window.get("start", "")),
                    str(window.get("end", "")),
                    str(window.get("comment", "")),
                ]
            )

    text_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        text_path.write_text("no data flagged\n", encoding="utf-8")
        return text_path

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["qc_var", "flag", "start", "end", "comment"])
    writer.writerows(rows)
    text_path.write_text(buffer.getvalue(), encoding="utf-8")
    return text_path
