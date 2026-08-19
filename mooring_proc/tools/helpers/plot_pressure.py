"""Reusable pressure comparison helpers for mooring workflows.

Provides a two-panel Plotly figure comparing an instrument's pressure
timeseries against a second reference source (e.g. a tide gauge or a
co-deployed CTD), together with the residual series.

Typical usage in a notebook proc_1 review cell::

    from tools.helpers import plot_pressure_comparison

    # Option A – resolve the reference from the metadata table by deploy ID
    fig, label, file_used = plot_pressure_comparison(
        df=proc1_result["dataframe"],
        database=metadata_csv,
        pressure_inst_deploy_id=269,
    )
    fig.show()

    # Option B – supply the reference file path directly
    fig, label, file_used = plot_pressure_comparison(
        df=proc1_result["dataframe"],
        pressure_file="/data/ref_pressure.tid",
        comparison_label="BPR",
    )
    fig.show()

The ``df`` argument must contain ``"datetime"`` and ``"Pressure [db]"``
columns.  These are the column names produced by the raw-source parsers
(read_sbe26, read_rbrq, etc.) before the data are converted to IMOS
NetCDF variables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..database_lookup import load_metadata_table
from .plot_qa_qc import _to_py_dt


# ---------------------------------------------------------------------------
# Internal file loaders
# ---------------------------------------------------------------------------

def _load_pressure(filepath: str) -> pd.Series:
    """Load a pressure timeseries from a supported reference file.

    Returns a pandas Series with a datetime index and pressure values in
    dbar, sorted and deduplicated.  Supports ``.tid`` (tide-gauge binary-
    text), ``.csv``, ``.txt``, and ``.aqd`` formats.
    """
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f"Pressure reference file not found: {filepath}")

    ext = fp.suffix.lower()

    def _finalize(s: pd.Series) -> pd.Series:
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
        return _finalize(pd.Series(pres_dbar.values, index=dt, name="pressure_dbar"))

    if ext in (".csv", ".txt", ".aqd"):
        last_err: Exception | None = None
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
                    if str(c).strip().lower() in ("datetime", "date_time", "timestamp", "instrument time", "time"):
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
                    if cl in ("pres", "pressure", "pressure [db]", "bpr pressure") or "press" in cl:
                        pcol = c
                        break
                if pcol is None:
                    raise KeyError(f"No pressure-like column found. Columns={list(raw.columns)}")

                return _finalize(pd.Series(raw[pcol].values, index=raw.index, name="pressure_dbar"))
            except Exception as exc:
                last_err = exc
                continue

        raise ValueError(f"Could not parse {ext} file: {filepath} | last error: {last_err}")

    raise ValueError(f"Unsupported pressure file extension: {ext}")


def _resolve_pressure_file(database, pressure_inst_deploy_id: int) -> tuple[str, str]:
    """Return ``(filepath, label)`` for the given deploy ID from the metadata table.

    Prefers ``data_in_path/data_in_file`` over ``proc_1_path/proc_1_file``.
    The label is ``<inst_type>_<inst_id>`` when available, otherwise
    ``"Comparison instrument"``.
    """
    df = load_metadata_table(database) if not isinstance(database, pd.DataFrame) else database

    id_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("inst_deploy_id", "inst_deployid", "inst_deploy id"):
            id_col = c
            break
    if id_col is None:
        raise KeyError("Could not find inst_deploy_ID column in metadata table.")

    rows = df[pd.to_numeric(df[id_col], errors="coerce") == int(pressure_inst_deploy_id)]
    if rows.empty:
        raise KeyError(f"No metadata row for pressure_inst_deploy_id={pressure_inst_deploy_id}")

    row = rows.iloc[0]

    def _ok(v: object) -> bool:
        return v is not None and str(v).strip() not in ("", "nan")

    allowed = {".tid", ".csv", ".txt", ".aqd"}
    candidates: list[Path] = []
    if _ok(row.get("data_in_path")) and _ok(row.get("data_in_file")):
        candidates.append(Path(str(row["data_in_path"])) / str(row["data_in_file"]))
    if _ok(row.get("proc_1_path")) and _ok(row.get("proc_1_file")):
        candidates.append(Path(str(row["proc_1_path"])) / str(row["proc_1_file"]))

    checked: list[str] = []
    for p in candidates:
        checked.append(str(p))
        if p.suffix.lower() in allowed and p.exists():
            inst_type = str(row.get("inst_type", "")).strip()
            inst_id = str(row.get("inst_id", "")).strip()
            label = (
                f"{inst_type}_{inst_id}"
                if inst_type and inst_id and inst_id.lower() != "nan"
                else "Comparison instrument"
            )
            return str(p), label

    raise FileNotFoundError(
        f"No valid pressure reference file found for deploy ID {pressure_inst_deploy_id}. "
        f"Checked: {checked}. Allowed extensions: {sorted(allowed)}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_pressure_comparison(
    df: pd.DataFrame,
    database=None,
    pressure_inst_deploy_id: Optional[int] = None,
    pressure_file: Optional[str] = None,
    comparison_label: Optional[str] = None,
    primary_label: str = "Instrument",
    x_start=None,
    x_end=None,
) -> tuple:
    """Build a two-panel pressure-comparison figure.

    Panel 1: overlaid pressure timeseries (instrument vs reference).
    Panel 2: residuals (instrument − reference).

    Parameters
    ----------
    df:
        DataFrame with ``"datetime"`` and ``"Pressure [db]"`` columns, as
        produced by the instrument parsers.
    database:
        Path to the metadata CSV, or a loaded ``pd.DataFrame``.  Required
        when ``pressure_inst_deploy_id`` is provided.
    pressure_inst_deploy_id:
        ``inst_deploy_ID`` value of the reference pressure instrument in the
        metadata table.  Mutually exclusive with ``pressure_file``.
    pressure_file:
        Explicit path to the reference pressure file.  Mutually exclusive
        with ``pressure_inst_deploy_id``.
    comparison_label:
        Legend label for the reference series.  Auto-resolved from metadata
        when ``pressure_inst_deploy_id`` is used and this is ``None``.
    primary_label:
        Legend label for the instrument series (default ``"Instrument"``).
        Override with e.g. ``"SBE37"`` or ``"SBE26"``.
    x_start, x_end:
        Optional time range for the x-axis.

    Returns
    -------
    tuple of ``(fig, label_used, file_used)`` where *fig* is the Plotly
    figure, *label_used* is the resolved comparison label, and *file_used*
    is the resolved reference file path.
    """
    if "datetime" not in df.columns or "Pressure [db]" not in df.columns:
        raise KeyError("df must contain 'datetime' and 'Pressure [db]' columns.")

    t = pd.to_datetime(df["datetime"])
    p_instrument = pd.to_numeric(df["Pressure [db]"], errors="coerce").to_numpy()

    resolved_label = comparison_label or "Comparison instrument"
    pressure_file_used: str

    if pressure_file is not None:
        pressure_file_used = str(pressure_file)
    elif pressure_inst_deploy_id is not None:
        if database is None:
            raise ValueError("database is required when using pressure_inst_deploy_id.")
        pressure_file_used, auto_label = _resolve_pressure_file(database, pressure_inst_deploy_id)
        if comparison_label is None:
            resolved_label = auto_label
    else:
        raise ValueError("Provide either pressure_file or pressure_inst_deploy_id.")

    if Path(pressure_file_used).suffix.lower() == ".nc":
        raise ValueError(
            f"Reference pressure file is a NetCDF ({pressure_file_used}). "
            "Expected .tid / .csv / .aqd / .txt."
        )

    ref_series = _load_pressure(pressure_file_used)
    if ref_series.empty:
        raise ValueError(f"Reference pressure series is empty: {pressure_file_used}")

    p_ref = np.interp(
        t.astype("int64"),
        ref_series.index.astype("int64"),
        ref_series.values.astype(float),
        left=np.nan,
        right=np.nan,
    )
    residuals = p_instrument - p_ref

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=[
            f"Pressure vs time ({primary_label} and {resolved_label})",
            f"Residuals: {primary_label} \u2212 {resolved_label}",
        ],
    )

    fig.add_trace(
        go.Scattergl(
            x=t,
            y=p_instrument,
            mode="markers",
            marker=dict(size=3, color="#1F77B4"),
            name=primary_label,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=p_ref,
            mode="markers",
            marker=dict(size=3, color="#FF7F0E"),
            name=resolved_label,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=residuals,
            mode="markers",
            marker=dict(size=3, color="#9467BD"),
            name=f"Residual ({primary_label} \u2212 {resolved_label})",
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Pressure (dbar)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (dbar)", row=2, col=1)
    fig.update_xaxes(title_text="Date-Time", row=2, col=1)

    if x_start is not None or x_end is not None:
        x0 = _to_py_dt(x_start if x_start is not None else t.min())
        x1 = _to_py_dt(x_end if x_end is not None else t.max())
        fig.update_xaxes(range=[x0, x1])

    fig.update_layout(
        title="Pressure comparison",
        template="plotly_white",
        width=1450,
        height=620,
    )

    print(f"Reference file:   {pressure_file_used}")
    print(f"Reference label:  {resolved_label}")
    print(f"Median residual:  {np.nanmedian(residuals):.3f} dbar")
    print(f"Std residual:     {np.nanstd(residuals):.3f} dbar")

    return fig, resolved_label, pressure_file_used


__all__ = ["plot_pressure_comparison"]
