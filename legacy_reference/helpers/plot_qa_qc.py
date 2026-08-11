# Reusable Plotly helpers for QC-flagged datasets.

# Integration:
# - Use for quick QA/QC visual checks and static/HTML exports.
# - Expects IMOS-style TIME and ``*_quality_control`` companion variables.

# Copy-paste:
#     from tools.helpers import plot_data_by_qc, save_plotly_figure

#     fig = plot_data_by_qc(ds, flags_to_plot=[1], y_zoom_to_good=True)
#     fig = plot_data_by_qc(ds)  # all mapped vars
#     fig = plot_data_by_qc(ds, variables=["TEMP", "PSAL"])  # selected vars
#     save_plotly_figure(fig, out_dir="./plots", filename="post_qc.png")


from __future__ import annotations

import os
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _to_py_dt(x):
    """Convert datetime-like values to python datetime for safe Plotly export."""
    if x is None:
        return None
    try:
        return pd.to_datetime(x).to_pydatetime()
    except Exception:
        return x


def _time_values_to_datetime(time_values):
    values = np.asarray(time_values)
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values)
    return pd.to_datetime("1950-01-01") + pd.to_timedelta(values, unit="D")


def save_plotly_figure(fig, out_dir: str, filename: str, width: int = 1450, height: int = 900, scale: int = 2):
    """Save a Plotly figure as static image or HTML."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    ext = os.path.splitext(filename)[1].lower()

    fig.update_layout(width=width, height=height)

    # Sanitize x-axis ranges for kaleido JSON serialization.
    for k in fig.layout:
        if str(k).startswith("xaxis"):
            ax = fig.layout[k]
            if hasattr(ax, "range") and ax.range is not None and len(ax.range) == 2:
                fig.layout[k].range = [_to_py_dt(ax.range[0]), _to_py_dt(ax.range[1])]

    if ext in [".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf"]:
        fig.write_image(out_path, scale=scale)
    elif ext == ".html":
        fig.write_html(out_path, include_plotlyjs="cdn")
    else:
        raise ValueError(f"Unsupported extension '{ext}'")

    print(f"Saved plot: {out_path}")
    return out_path


def _compute_good_data_ylims(
    y,
    qc,
    t=None,
    x_start=None,
    x_end=None,
    pad_frac: float = 0.08,
    min_span: float = 1e-6,
):
    """Compute y-limits from QC==1 points, optionally within a time window."""
    y = np.asarray(y, dtype=float)
    qc = np.asarray(qc)

    m_good = (qc == 1) & np.isfinite(y)

    if t is not None and (x_start is not None or x_end is not None):
        tt = pd.to_datetime(t)
        if x_start is not None:
            m_good &= tt >= pd.to_datetime(x_start)
        if x_end is not None:
            m_good &= tt <= pd.to_datetime(x_end)

    if not np.any(m_good):
        return (None, None)

    yg = y[m_good]
    y0 = np.nanmin(yg)
    y1 = np.nanmax(yg)
    span = max(y1 - y0, min_span)
    pad = span * pad_frac
    return (y0 - pad, y1 + pad)


def plot_data_by_qc(
    ds,
    mappings: Optional[Sequence[Tuple[str, str, str]]] = None,
    time_name: str = "TIME",
    title: Optional[str] = None,
    x_start=None,
    x_end=None,
    y_zoom_to_good: bool = False,
    y_pad_frac: float = 0.08,
    flags_to_plot: Optional[Union[Sequence[int], Mapping[str, Iterable[int]]]] = None,
    legend: Optional[bool] = True,
    variables: Optional[Sequence[str]] = None,
    style: str = "clean",
):
    """Plot variables grouped by QC flag using one subplot per variable.

    flags_to_plot options:
      - None: plot all flags present
      - list/set/tuple of ints: same flags for all variables
      - dict: per-variable flags, keys are variable names

    variables options:
      - None: plot all variables from mappings that exist in ds
      - list/set/tuple of variable names: only plot those variables
    """
    if mappings is None:
        mappings = [
            ("TEMP", "TEMP_quality_control", "Temperature (°C)"),
            ("CNDC", "CNDC_quality_control", "Conductivity (S/m)"),
            ("PSAL", "PSAL_quality_control", "Salinity (PSU)"),
            ("PRES", "PRES_quality_control", "Pressure (dbar)"),
            ("DEPTH", "DEPTH_quality_control", "Depth (m)"),
            ("UCUR", "UCUR_quality_control", "U current (m/s)"),
            ("VCUR", "VCUR_quality_control", "V current (m/s)"),
        ]

    mapping_by_var = {v: (v, q, yl) for (v, q, yl) in mappings}

    if variables is None:
        candidate = list(mapping_by_var.keys())
    else:
        candidate = [str(v) for v in variables]

    selected_mappings = []
    missing = []
    for v in candidate:
        if v in mapping_by_var:
            vv, qq, yl = mapping_by_var[v]
        else:
            vv, qq, yl = v, f"{v}_quality_control", v  # auto-map

        if vv in ds and qq in ds:
            selected_mappings.append((vv, qq, yl))
        else:
            missing.append(v)

    if not selected_mappings:
        raise ValueError(
            f"No plottable variable/QC pairs found. Requested={candidate}. "
            f"Missing vars or QC companions={missing}"
        )




    # Optional variable filter
    selected_vars = None if variables is None else set(variables)

    t = _time_values_to_datetime(ds[time_name].values)
    valid = [
        (v, q, yl)
        for (v, q, yl) in mappings
        if v in ds and q in ds and (selected_vars is None or v in selected_vars)
    ]
    if not valid:
        if selected_vars is None:
            raise ValueError("No variable/QC pairs found in dataset.")
        raise ValueError(
            f"No requested variables found with QC pairs. Requested={sorted(selected_vars)}"
        )

    qc_colors = {
        0: "#6B7C93",
        1: "#1F77B4",
        2: "#2CA02C",
        3: "#FF7F0E",
        4: "#D62728",
        5: "#9467BD",
        6: "#BDBDBD",
        7: "#BDBDBD",
        8: "#BDBDBD",
        9: "#FFFFFF",
    }

    var_colors = {
        "PRES": "#636EFA",
        "TEMP": "#FFA500",
        "CNDC": "#00CC96",
        "PSAL": "#AB63FA",
        "DEPTH": "#19D3F3",
        "UCUR": "#EF553B",
        "VCUR": "#FF6692",
    }

    fig = make_subplots(
        rows=len(valid),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[yl for (_, _, yl) in valid],
    )

    global_flags = None
    per_var_flags = None
    if flags_to_plot is None:
        pass
    elif isinstance(flags_to_plot, dict):
        per_var_flags = {k: set(int(x) for x in v) for k, v in flags_to_plot.items()}
    else:
        global_flags = set(int(x) for x in flags_to_plot)

    show_legend = bool(legend) if legend is not None else False
    style_key = str(style).strip().lower()
    if style_key not in {"clean", "classic"}:
        style_key = "clean"
    variable_legend_mode = style_key == "clean" and global_flags == {1}

    for r, (var, qc_name, ylab) in enumerate(valid, start=1):
        y = np.asarray(ds[var].values, dtype=float)
        qv = np.asarray(ds[qc_name].values).astype(np.int16)
        flags_present = np.unique(qv)

        if per_var_flags is not None:
            wanted = per_var_flags.get(var, set(flags_present.tolist()))
        elif global_flags is not None:
            wanted = global_flags
        else:
            wanted = set(flags_present.tolist())

        flags = [f for f in flags_present if int(f) in wanted]

        for f in flags:
            m = qv == f
            if not np.any(m):
                continue

            is_good = int(f) == 1
            trace_color = qc_colors.get(int(f), "#333333")
            trace_name = f"QC = {int(f)}"
            trace_group = f"QC={int(f)}"
            trace_showlegend = show_legend and r == 1

            if variable_legend_mode and is_good:
                trace_color = var_colors.get(var, trace_color)
                trace_name = var
                trace_group = var
                trace_showlegend = show_legend

            if style_key == "clean":
                mode = "lines" if is_good else "markers"
                marker_size = 3 if is_good else 4
                marker_opacity = 0.85 if is_good else 0.65
                line_width = 1.2 if is_good else 0.0
            else:
                mode = "markers"
                marker_size = 3
                marker_opacity = 1.0
                line_width = 0.0

            fig.add_trace(
                go.Scatter(
                    x=t[m],
                    y=y[m],
                    mode=mode,
                    marker=dict(
                        size=marker_size,
                        opacity=marker_opacity,
                        color=trace_color,
                        line=dict(width=0),
                    ),
                    line=dict(
                        width=line_width,
                        color=trace_color,
                    ),
                    name=trace_name,
                    legendgroup=trace_group,
                    showlegend=trace_showlegend,
                ),
                row=r,
                col=1,
            )

        fig.update_yaxes(title_text=ylab, row=r, col=1)

        if y_zoom_to_good:
            ymin, ymax = _compute_good_data_ylims(
                y,
                qv,
                t=t,
                x_start=x_start,
                x_end=x_end,
                pad_frac=y_pad_frac,
            )
            if ymin is not None and ymax is not None:
                fig.update_yaxes(range=[ymin, ymax], row=r, col=1)

    if x_start is not None or x_end is not None:
        x0 = _to_py_dt(x_start if x_start is not None else t.min())
        x1 = _to_py_dt(x_end if x_end is not None else t.max())
        fig.update_xaxes(range=[x0, x1])

    fig.update_xaxes(title_text="Time", row=len(valid), col=1)
    layout_kwargs = dict(
        template="plotly",
        width=1150,
        height=max(340 * len(valid), 650),
        showlegend=show_legend,
        margin=dict(l=70, r=40, t=70, b=60),
    )
    if show_legend:
        layout_kwargs["legend"] = dict(
            title=("" if variable_legend_mode else "Quality Control Flag"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        )
    if title is not None:
        layout_kwargs["title"] = title
    fig.update_layout(**layout_kwargs)

    fig.update_xaxes(showgrid=True, gridcolor="#e9eef6")
    fig.update_yaxes(showgrid=True, gridcolor="#e9eef6")
    return fig


__all__ = ["plot_data_by_qc", "save_plotly_figure"]
