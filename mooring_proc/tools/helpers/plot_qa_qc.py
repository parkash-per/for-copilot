"""Reusable Plotly helpers for QC-flagged datasets."""

from __future__ import annotations

import os
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _to_py_dt(value):
    """Convert datetime-like values to Python datetimes for Plotly range handling."""
    if value is None:
        return None
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return value


def _time_values_to_datetime(time_values):
    values = np.asarray(time_values)
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values)
    return pd.to_datetime("1950-01-01") + pd.to_timedelta(values, unit="D")


def save_plotly_figure(
    fig,
    out_dir: str,
    filename: str,
    width: int = 1450,
    height: int = 900,
    scale: int = 2,
):
    """Save a Plotly figure as a static image or HTML."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    ext = os.path.splitext(filename)[1].lower()

    fig.update_layout(width=width, height=height)

    for key in fig.layout:
        if str(key).startswith("xaxis"):
            axis = fig.layout[key]
            if hasattr(axis, "range") and axis.range is not None and len(axis.range) == 2:
                fig.layout[key].range = [_to_py_dt(axis.range[0]), _to_py_dt(axis.range[1])]

    if ext in [".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf"]:
        fig.write_image(out_path, scale=scale)
    elif ext == ".html":
        fig.write_html(out_path, include_plotlyjs="cdn")
    else:
        raise ValueError(f"Unsupported extension '{ext}'")

    print(f"Saved plot: {out_path}")
    return out_path


def _compute_good_data_ylims(
    values,
    qc_values,
    time_values=None,
    x_start=None,
    x_end=None,
    pad_frac: float = 0.08,
    min_span: float = 1e-6,
):
    """Compute y-limits from QC=1 points, optionally within a time window."""
    y = np.asarray(values, dtype=float)
    qc = np.asarray(qc_values)

    mask = (qc == 1) & np.isfinite(y)
    if time_values is not None and (x_start is not None or x_end is not None):
        time_index = pd.to_datetime(time_values)
        if x_start is not None:
            mask &= time_index >= pd.to_datetime(x_start)
        if x_end is not None:
            mask &= time_index <= pd.to_datetime(x_end)

    if not np.any(mask):
        return (None, None)

    good = y[mask]
    y_min = np.nanmin(good)
    y_max = np.nanmax(good)
    span = max(y_max - y_min, min_span)
    pad = span * pad_frac
    return (y_min - pad, y_max + pad)


def plot_data_by_qc(
    dataset,
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
    """Plot variables grouped by QC flag using one subplot per variable."""
    if mappings is None:
        mappings = [
            ("TEMP", "TEMP_quality_control", "Temperature (°C)"),
            ("CNDC", "CNDC_quality_control", "Conductivity (S/m)"),
            ("PSAL", "PSAL_quality_control", "Salinity (PSU)"),
            ("PRES", "PRES_quality_control", "Pressure (dbar)"),
            ("DEPTH", "DEPTH_quality_control", "Depth (m)"),
            ("UCUR", "UCUR_quality_control", "U current (cm/s)"),
            ("VCUR", "VCUR_quality_control", "V current (cm/s)"),
        ]

    mapping_by_var = {variable_name: (variable_name, qc_name, y_label) for (variable_name, qc_name, y_label) in mappings}

    if variables is None:
        requested_variables = list(mapping_by_var.keys())
    else:
        requested_variables = [str(name) for name in variables]

    selected_mappings = []
    missing = []
    for variable_name in requested_variables:
        if variable_name in mapping_by_var:
            mapped_var, mapped_qc, y_label = mapping_by_var[variable_name]
        else:
            mapped_var, mapped_qc, y_label = variable_name, f"{variable_name}_quality_control", variable_name

        if mapped_var in dataset and mapped_qc in dataset:
            selected_mappings.append((mapped_var, mapped_qc, y_label))
        else:
            missing.append(variable_name)

    if not selected_mappings:
        raise ValueError(
            f"No plottable variable/QC pairs found. Requested={requested_variables}. "
            f"Missing vars or QC companions={missing}"
        )

    time_values = _time_values_to_datetime(dataset[time_name].values)

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

    figure = make_subplots(
        rows=len(selected_mappings),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[y_label for (_, _, y_label) in selected_mappings],
    )

    global_flags = None
    per_var_flags = None
    if flags_to_plot is None:
        pass
    elif isinstance(flags_to_plot, dict):
        per_var_flags = {key: set(int(item) for item in value) for key, value in flags_to_plot.items()}
    else:
        global_flags = set(int(item) for item in flags_to_plot)

    show_legend = bool(legend) if legend is not None else False
    style_key = str(style).strip().lower()
    if style_key not in {"clean", "classic"}:
        style_key = "clean"
    variable_legend_mode = style_key == "clean" and global_flags == {1}

    for row_index, (variable_name, qc_name, y_label) in enumerate(selected_mappings, start=1):
        y_values = np.asarray(dataset[variable_name].values, dtype=float)
        qc_values = np.asarray(dataset[qc_name].values).astype(np.int16)
        flags_present = np.unique(qc_values)

        if per_var_flags is not None:
            wanted_flags = per_var_flags.get(variable_name, set(flags_present.tolist()))
        elif global_flags is not None:
            wanted_flags = global_flags
        else:
            wanted_flags = set(flags_present.tolist())

        for flag in [value for value in flags_present if int(value) in wanted_flags]:
            mask = qc_values == flag
            if not np.any(mask):
                continue

            is_good = int(flag) == 1
            trace_color = qc_colors.get(int(flag), "#333333")
            trace_name = f"QC = {int(flag)}"
            legend_group = f"QC={int(flag)}"
            trace_showlegend = show_legend and row_index == 1

            if variable_legend_mode and is_good:
                trace_color = var_colors.get(variable_name, trace_color)
                trace_name = variable_name
                legend_group = variable_name
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

            figure.add_trace(
                go.Scatter(
                    x=time_values[mask],
                    y=y_values[mask],
                    mode=mode,
                    marker=dict(
                        size=marker_size,
                        opacity=marker_opacity,
                        color=trace_color,
                        line=dict(width=0),
                    ),
                    line=dict(width=line_width, color=trace_color),
                    name=trace_name,
                    legendgroup=legend_group,
                    showlegend=trace_showlegend,
                ),
                row=row_index,
                col=1,
            )

        figure.update_yaxes(title_text=y_label, row=row_index, col=1)

        if y_zoom_to_good:
            y_min, y_max = _compute_good_data_ylims(
                y_values,
                qc_values,
                time_values=time_values,
                x_start=x_start,
                x_end=x_end,
                pad_frac=y_pad_frac,
            )
            if y_min is not None and y_max is not None:
                figure.update_yaxes(range=[y_min, y_max], row=row_index, col=1)

    if x_start is not None or x_end is not None:
        x0 = _to_py_dt(x_start if x_start is not None else time_values.min())
        x1 = _to_py_dt(x_end if x_end is not None else time_values.max())
        figure.update_xaxes(range=[x0, x1])

    figure.update_xaxes(title_text="Time", row=len(selected_mappings), col=1)

    layout_kwargs = dict(
        template="plotly",
        width=1150,
        height=max(340 * len(selected_mappings), 650),
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
    figure.update_layout(**layout_kwargs)
    figure.update_xaxes(showgrid=True, gridcolor="#e9eef6")
    figure.update_yaxes(showgrid=True, gridcolor="#e9eef6")
    return figure


__all__ = ["plot_data_by_qc", "save_plotly_figure"]
