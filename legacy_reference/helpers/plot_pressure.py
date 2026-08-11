"""Reusable pressure comparison helpers for mooring workflows.

Integration:
- Use after loading source dataframe with ``datetime`` and ``Pressure [db]``.
- Supports comparison source via deploy ID metadata or explicit file path.

Copy-paste:
    from tools.helpers import plot_pressure_comparison

    fig, label_used, file_used = plot_pressure_comparison(
        df=df,
        database=database,
        pressure_inst_deploy_id=269,
        x_start=None,
        x_end=None,
    )
    fig.show()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from tools.database_lookup import resolve_pressure_source_by_id
from .plot_qa_qc import _to_py_dt


def _load_pressure(filepath):
    """Load pressure series (datetime index, pressure in dbar) from supported files."""
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f"Pressure file not found: {filepath}")

    ext = fp.suffix.lower()

    def _finalize_series(s):
        s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
        return s[~s.index.duplicated(keep="first")]

    if ext == ".tid":
        raw = pd.read_csv(
            fp,
            sep=r"\s+",
            header=None,
            names=["ID", "Date", "time", "PRES", "TEMP"],
            engine="python",
            encoding="utf-8",
            encoding_errors="replace",
        )
        dt = pd.to_datetime(
            raw["Date"].astype(str) + " " + raw["time"].astype(str),
            dayfirst=True,
            format="mixed",
            errors="coerce",
        )
        pres_dbar = pd.to_numeric(raw["PRES"], errors="coerce") * 0.6894757
        return _finalize_series(pd.Series(pres_dbar.values, index=dt, name="pressure_dbar"))

    if ext in (".csv", ".txt", ".aqd"):
        last_err = None
        for enc in ("utf-8", "cp1252", "latin1"):
            try:
                try:
                    raw = pd.read_csv(fp, encoding=enc)
                except pd.errors.ParserError:
                    raw = pd.read_csv(
                        fp,
                        sep=None,
                        engine="python",
                        comment="#",
                        on_bad_lines="skip",
                        encoding=enc,
                    )

                dt_col = None
                for c in raw.columns:
                    cl = str(c).strip().lower()
                    if cl in ("datetime", "date_time", "timestamp", "instrument time", "time"):
                        dt_col = c
                        break

                if dt_col is not None:
                    idx = pd.to_datetime(raw[dt_col], errors="coerce", dayfirst=True, format="mixed")
                    raw = raw.drop(columns=[dt_col])
                    raw.index = idx
                else:
                    raw.index = pd.to_datetime(raw.index, errors="coerce", dayfirst=True, format="mixed")

                pcol = None
                for c in raw.columns:
                    cl = str(c).lower()
                    if cl in ("pres", "pressure", "pressure [db]", "bpr pressure") or ("press" in cl):
                        pcol = c
                        break
                if pcol is None:
                    raise KeyError(f"No pressure-like column found. Columns={list(raw.columns)}")

                return _finalize_series(pd.Series(raw[pcol].values, index=raw.index, name="pressure_dbar"))
            except Exception as e:
                last_err = e
                continue

        raise ValueError(f"Could not parse {ext} file: {filepath} | last error: {last_err}")

    raise ValueError(f"Unsupported pressure file extension: {ext}")


def _resolve_pressure_file_by_id_data_in_first(database, pressure_inst_deploy_id, use_proc1=True):
    """Resolve comparison pressure file from metadata with data_in-path preference."""
    id_col = None
    for c in database.columns:
        if str(c).strip().lower() in ("inst_deploy_id", "inst_deployid", "inst_deploy id"):
            id_col = c
            break
    if id_col is None:
        raise KeyError("Could not find inst_deploy_ID column in metadata dataframe.")

    rows = database[pd.to_numeric(database[id_col], errors="coerce") == int(pressure_inst_deploy_id)]
    if rows.empty:
        raise KeyError(f"No row found for pressure_inst_deploy_id={pressure_inst_deploy_id}")

    prow = rows.iloc[0]

    def _ok(v):
        return v is not None and str(v).strip() != "" and str(v).strip().lower() != "nan"

    allowed = {".tid", ".csv", ".txt", ".aqd"}
    candidates = []

    if _ok(prow.get("data_in_path")) and _ok(prow.get("data_in_file")):
        candidates.append(Path(str(prow["data_in_path"])) / str(prow["data_in_file"]))

    if use_proc1 and _ok(prow.get("proc_1_path")) and _ok(prow.get("proc_1_file")):
        candidates.append(Path(str(prow["proc_1_path"])) / str(prow["proc_1_file"]))

    try:
        _prow2, pfile2, _fallback = resolve_pressure_source_by_id(database, pressure_inst_deploy_id, use_proc1=use_proc1)
        if pfile2:
            candidates.append(Path(str(pfile2)))
    except Exception:
        pass

    checked = []
    for p in candidates:
        p = Path(str(p))
        checked.append(str(p))
        if p.suffix.lower() in allowed and p.exists():
            inst_type = str(prow.get("inst_type", "")).strip()
            inst_id = str(prow.get("inst_id", "")).strip()
            label = f"{inst_type}_{inst_id}" if inst_type and inst_id and inst_id.lower() != "nan" else "Comparison instrument"
            return str(p), label, prow

    raise FileNotFoundError(
        "No valid pressure comparison file found for deploy ID "
        f"{pressure_inst_deploy_id}. Checked: {checked}. Allowed: {sorted(allowed)}"
    )


def plot_pressure_comparison(
    df,
    database=None,
    pressure_inst_deploy_id=None,
    pressure_file: Optional[str] = None,
    comparison_label: Optional[str] = None,
    x_start=None,
    x_end=None,
    y_zoom_to_good=False,
):
    """Build pressure-comparison figure and residuals against a second pressure source."""
    _ = y_zoom_to_good

    if "datetime" not in df or "Pressure [db]" not in df:
        raise KeyError("df must contain 'datetime' and 'Pressure [db]' columns")

    t = pd.to_datetime(df["datetime"])
    p_ctd = pd.to_numeric(df["Pressure [db]"], errors="coerce").to_numpy()

    resolved_label = comparison_label if comparison_label else "Comparison instrument"
    pressure_file_used = None

    if pressure_file is not None:
        pressure_file_used = str(pressure_file)
    elif pressure_inst_deploy_id is not None:
        if database is None:
            raise ValueError("database is required when using pressure_inst_deploy_id")
        pressure_file_used, auto_label, _prow = _resolve_pressure_file_by_id_data_in_first(
            database,
            pressure_inst_deploy_id,
            use_proc1=True,
        )
        if comparison_label is None:
            resolved_label = auto_label
    else:
        raise ValueError("Provide either pressure_file or pressure_inst_deploy_id")

    pf = Path(pressure_file_used)
    if pf.suffix.lower() == ".nc":
        raise ValueError(
            f"Resolved pressure file is NetCDF ({pressure_file_used}). "
            "Expected .tid/.csv/.aqd/.txt"
        )

    ext = _load_pressure(pressure_file_used)
    if ext.empty:
        raise ValueError(f"Comparison pressure series is empty: {pressure_file_used}")

    p_ext = np.interp(
        t.astype("int64"),
        ext.index.astype("int64"),
        ext.values.astype(float),
        left=np.nan,
        right=np.nan,
    )
    diff = p_ctd - p_ext

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=["Pressure vs time (SBE37 and comparison)", "Residuals: SBE37 - comparison"],
    )

    fig.add_trace(
        go.Scattergl(x=t, y=p_ctd, mode="markers", marker=dict(size=3, color="#1F77B4"), name="SBE37"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(x=t, y=p_ext, mode="markers", marker=dict(size=3, color="#FF7F0E"), name=resolved_label),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=diff,
            mode="markers",
            marker=dict(size=3, color="#9467BD"),
            name="Residual (SBE37 - comparison)",
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Pressure (dbar)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (dbar)", row=2, col=1)

    if x_start is not None or x_end is not None:
        x0 = _to_py_dt(x_start if x_start is not None else t.min())
        x1 = _to_py_dt(x_end if x_end is not None else t.max())
        fig.update_xaxes(range=[x0, x1])

    fig.update_xaxes(title_text="Date-Time", row=2, col=1)
    fig.update_layout(
        title="Pressure comparison",
        template="plotly_white",
        width=1450,
        height=620,
    )

    print(f"Comparison file: {pressure_file_used}")
    print(f"Comparison label: {resolved_label}")
    print(f"Median residual: {np.nanmedian(diff):.3f} dbar")
    print(f"Std residual:    {np.nanstd(diff):.3f} dbar")

    return fig, resolved_label, pressure_file_used


__all__ = ["plot_pressure_comparison"]