"""Manual QA/QC flagging helpers shared across notebooks.

Integration:
- Apply after dataset creation and before plotting/export.
- Updates ``*_quality_control`` variables in-place based on time windows.

Copy-paste:
    from tools.helpers import apply_qc_flag_windows

    qc_windows = [
        {"start": "", "end": "2025-02-07 01:55:00", "flag": 4},
        {"start": "2026-03-04 00:40:00", "end": "", "flag": 4},
    ]
    ds = apply_qc_flag_windows(ds, qc_windows, time_name="TIME")
"""

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


def apply_qc_flag_windows(ds, windows, time_name="TIME", verbose=True):
    """
    Apply QC flags over multiple time windows.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing TIME and *_quality_control variables.
    windows : list of dict
        Each dict supports:
          - start: datetime-like string (inclusive). Use "" or None for file start.
          - end: datetime-like string (inclusive). Use "" or None for file end.
          - flag: int (e.g. 3, 4, 5)
          - qc_vars: optional list of QC variable names.
                     If omitted/None, applies to ALL *_quality_control variables.
    time_name : str
        Name of time coordinate (default "TIME").
    verbose : bool
        If True, print per-window/per-variable updates. If False, run silently.
    """
    if time_name not in ds:
        raise KeyError(f"{time_name} not found in dataset")

    t = _time_values_to_datetime(ds[time_name].values)
    file_start, file_end = t.min(), t.max()

    all_qc = [v for v in ds.data_vars if "_quality_control" in v]
    if not all_qc:
        if verbose:
            print("No *_quality_control variables found.")
        return ds

    for i, w in enumerate(windows, start=1):
        raw_start = w.get("start", None)
        raw_end = w.get("end", None)

        t0 = file_start if raw_start in (None, "") else pd.to_datetime(raw_start)
        t1 = file_end if raw_end in (None, "") else pd.to_datetime(raw_end)
        flag = np.int8(w["flag"])

        if t1 < t0:
            raise ValueError(f"Window {i}: end before start")

        mask = (t >= t0) & (t <= t1)
        n = int(mask.sum())
        if n == 0:
            if verbose:
                print(f"Window {i}: no samples in range ({t0} to {t1}); skipped")
            continue

        requested = w.get("qc_vars", None)
        if requested is None:
            qc_list = all_qc
            missing = []
        else:
            qc_list = [q for q in requested if q in ds.variables]
            missing = [q for q in requested if q not in ds.variables]

        if missing:
            if verbose:
                print(f"Window {i}: requested QC vars not found (ignored): {missing}")

        if not qc_list:
            if verbose:
                print(f"Window {i}: no matching QC vars; skipped")
            continue

        for qv in qc_list:
            arr = ds[qv].values.copy()
            arr[mask] = flag
            ds[qv].values = arr
            if verbose:
                print(f"Window {i}: set {qv}={int(flag)} for {n} samples ({t0} to {t1})")

    return ds


def write_manual_qc_flags_txt(output_nc_path, windows, filename_suffix="_manual_qc_flags.txt"):
    """
    Write manual QC flag windows to a text file beside a generated NetCDF output.

    Parameters
    ----------
    output_nc_path : str or pathlib.Path
        Path to the written NetCDF output file.
    windows : list of dict
        Manual QC windows (e.g., items passed to apply_qc_flag_windows).
        A ``comment`` field is supported per item and will be written if present.
    filename_suffix : str
        Suffix used for the output text file name.

    Returns
    -------
    pathlib.Path
        Path to the written text file.
    """
    nc_path = Path(output_nc_path).expanduser().resolve()
    txt_path = nc_path.with_name(f"{nc_path.stem}{filename_suffix}")

    lines = []
    windows = windows or []
    for item in windows:
        qc_vars = item.get("qc_vars", None)
        if qc_vars in (None, ""):
            qc_var_list = ["ALL"]
        elif isinstance(qc_vars, (list, tuple, set)):
            qc_var_list = list(qc_vars)
        else:
            qc_var_list = [str(qc_vars)]

        for qc_var in qc_var_list:
            lines.append(
                [
                    str(qc_var),
                    str(item.get("flag", "")),
                    str(item.get("start", "")),
                    str(item.get("end", "")),
                    str(item.get("comment", "")),
                ]
            )

    if not lines:
        lines = ["no data flagged"]

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    if lines == ["no data flagged"]:
        txt_path.write_text("no data flagged\n", encoding="utf-8")
    else:
        output_buffer = io.StringIO()
        writer = csv.writer(output_buffer)
        writer.writerow(["qc_var", "flag", "start", "end", "comment"])
        writer.writerows(lines)
        txt_path.write_text(output_buffer.getvalue(), encoding="utf-8")
    return txt_path

def build_deployment_good_data_windows(
    ds,
    row=None,
    time_coverage_start=None,
    time_coverage_end=None,
    flag=4,
    time_name="TIME",
    dayfirst=True,
):
    """
    Build QC windows that flag outside deployment good-data interval.

    Priority:
            explicit time_coverage_start/time_coverage_end
            -> row['time_coverage_start']/row['time_coverage_end']
      -> row['deploy_date']/row['recovery_date']
      -> dataset bounds
    """
    if row is None:
        row_dict = {}
    elif hasattr(row, "to_dict"):
        row_dict = row.to_dict()
    else:
        row_dict = dict(row)
    
    def _is_blank(x):
        return x is None or (isinstance(x, str) and x.strip() == "")

    def _parse(x):
        if _is_blank(x):
            return pd.NaT
        return pd.to_datetime(x, dayfirst=dayfirst, format="mixed", errors="coerce")

    t_ds = _time_values_to_datetime(ds[time_name].values)
    ds_start = pd.to_datetime(t_ds.min())
    ds_end = pd.to_datetime(t_ds.max())

    resolved_start_good_data = _parse(time_coverage_start)
    resolved_end_good_data = _parse(time_coverage_end)

    if pd.isna(resolved_start_good_data):
        resolved_start_good_data = _parse(row_dict.get("time_coverage_start", None))
    if pd.isna(resolved_end_good_data):
        resolved_end_good_data = _parse(row_dict.get("time_coverage_end", None))

    if pd.isna(resolved_start_good_data):
        resolved_start_good_data = _parse(row_dict.get("deploy_date", None))
    if pd.isna(resolved_end_good_data):
        resolved_end_good_data = _parse(row_dict.get("recovery_date", None))

    if pd.isna(resolved_start_good_data):
        resolved_start_good_data = ds_start
    if pd.isna(resolved_end_good_data):
        resolved_end_good_data = ds_end

    deployment_good_data = {
        "time_coverage_start": resolved_start_good_data.strftime("%Y-%m-%d %H:%M:%S"),
        "time_coverage_end": resolved_end_good_data.strftime("%Y-%m-%d %H:%M:%S"),
        "flag": int(flag),
    }

    windows = [
        {"start": "", "end": deployment_good_data["time_coverage_start"], "flag": int(flag)},
        {"start": deployment_good_data["time_coverage_end"], "end": "", "flag": int(flag)},
    ]

    return windows, deployment_good_data

__all__ = [
    "apply_qc_flag_windows",
    "write_manual_qc_flags_txt",
    "build_deployment_good_data_windows",
]