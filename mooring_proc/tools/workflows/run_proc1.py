"""Instrument proc_1 workflow (parse + trim)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..database_lookup import get_instrument_context, update_metadata_file_fields
from ..imos.writer import build_output_filename, write_imos_file
from ..parsers.read_aqd import read_aqd
from ..parsers.read_rbrq import read_rbrq
from ..parsers.read_sbe26 import read_sbe26
from ..parsers.read_sbe37 import read_sbe37
from ..parsers.read_sig500 import read_sig500
from ..qc.manual_flags import apply_qc_flag_windows
from ..qc.windows import build_qc_windows


def _metadata_source(config: dict[str, Any]):
    for key in ("metadata_table", "metadata_csv", "metadata_source"):
        if config.get(key) is not None:
            return config[key]
    raise ValueError("config must provide metadata_table, metadata_csv, or metadata_source.")


def _instrument_key(config: dict[str, Any], instrument_id: Any):
    if instrument_id is not None:
        return instrument_id
    for key in ("inst_deploy_ID", "instrument_id"):
        if config.get(key) is not None:
            return config[key]
    raise ValueError("An inst_deploy_ID or instrument_id is required.")


_PARSERS = {
    "AQD": read_aqd,
    "SBE26": read_sbe26,
    "SBE37": read_sbe37,
    "RBRQ": read_rbrq,
    "SIG500": read_sig500,
}


def _instrument_type(row: Any) -> str:
    inst = str(row.get("inst_type", "")).strip().upper()
    if inst not in _PARSERS:
        raise NotImplementedError(f"Unsupported instrument '{inst}'.")
    return inst


def _resolve_time_bounds(row, overrides: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_raw = overrides.get("time_coverage_start") or row.get("time_coverage_start") or row.get("deploy_date")
    end_raw = overrides.get("time_coverage_end") or row.get("time_coverage_end") or row.get("recovery_date")
    start_time = pd.to_datetime(start_raw, format="mixed", errors="coerce")
    end_time = pd.to_datetime(end_raw, format="mixed", errors="coerce")
    if pd.isna(start_time):
        start_time = pd.to_datetime(start_raw, dayfirst=True, format="mixed", errors="coerce")
    if pd.isna(end_time):
        end_time = pd.to_datetime(end_raw, dayfirst=True, format="mixed", errors="coerce")
    if pd.isna(start_time) or pd.isna(end_time):
        raise ValueError("Deployment time_coverage_start/time_coverage_end must be available for proc_1.")
    return start_time, end_time


def _stage_dir(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plot_review(dataset, output_path: Path, title: str, start_time: pd.Timestamp, end_time: pd.Timestamp):
    import matplotlib.pyplot as plt

    required = ["TEMP", "DEPTH", "UCUR", "VCUR"]
    if not all(name in dataset.variables for name in required):
        return
    frame = dataset[required].to_dataframe().reset_index()
    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for axis, variable_name in zip(axes, ("TEMP", "DEPTH", "UCUR", "VCUR")):
        axis.plot(frame["TIME"], frame[variable_name], linewidth=0.8)
        axis.axvline(start_time, color="tab:red", linestyle="--", linewidth=1)
        axis.axvline(end_time, color="tab:red", linestyle="--", linewidth=1)
        axis.set_ylabel(variable_name)
    axes[-1].set_xlabel("TIME")
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _select_dataset(parsed: dict[str, Any], config: dict[str, Any]):
    if "datasets" in parsed and isinstance(parsed["datasets"], list) and parsed["datasets"]:
        index = int(config.get("sig500_dataset_index", 0))
        return parsed["datasets"][index]
    return parsed["dataset"]


def run_proc1(config, instrument_id=None, source_path=None):
    """Run proc_1 for AQD, SBE26, SBE37, RBRQ, or SIG500."""
    metadata_source = _metadata_source(config)
    inst_deploy_id = _instrument_key(config, instrument_id)
    _, row, cfg, _ = get_instrument_context(
        metadata_source,
        inst_deploy_id,
        deployment_id=config.get("deployment_id"),
    )
    inst_type = _instrument_type(row)
    parser = _PARSERS[inst_type]

    parsed = parser(source_path, config={"metadata_row": row.to_dict()})
    dataset = _select_dataset(parsed, config)
    start_time, end_time = _resolve_time_bounds(row, config)

    stage_metadata = {**cfg, **row.to_dict(), **config}
    stage_metadata.update(
        {
            "output_stage": "proc_1",
            "output_name_mode": "internal",
            "location": cfg.get("location", row.get("location", "")),
            "instrument": row.get("inst_type", inst_type),
            "inst_type": row.get("inst_type", inst_type),
            "inst_id": row.get("inst_id", ""),
            "depth": row.get("nominal_depth", cfg.get("nominal_depth", 0)),
            "start_of_good_data": start_time,
            "time_coverage_start": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "time_coverage_end": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "output_dir": row.get("proc_1_path"),
        }
    )

    output_name = build_output_filename(stage_metadata)
    proc_1_dir = _stage_dir(row.get("proc_1_path"))
    output_path = proc_1_dir / output_name
    pre_plot_path = proc_1_dir / f"{output_path.stem}_pre_trim_review.png"
    post_plot_path = proc_1_dir / f"{output_path.stem}_post_trim_review.png"

    if inst_type == "AQD":
        _plot_review(dataset, pre_plot_path, "AQD proc_1 pre-trim review", start_time, end_time)

    qc_vars = [name for name in dataset.data_vars if name.endswith("_quality_control")]
    deployment_windows = build_qc_windows(
        dataset,
        {
            "row": row.to_dict(),
            "time_coverage_start": start_time,
            "time_coverage_end": end_time,
            "flag": 4,
            "qc_vars": qc_vars or None,
            "comment": "outside deployment window",
        },
    )
    dataset_with_qc = apply_qc_flag_windows(dataset, deployment_windows)
    trimmed_dataset = dataset_with_qc.sel(TIME=slice(start_time, end_time))
    if inst_type == "AQD":
        _plot_review(trimmed_dataset, post_plot_path, "AQD proc_1 post-trim review", start_time, end_time)

    proc_1_output = write_imos_file(trimmed_dataset, output_path, metadata=stage_metadata)
    update_metadata_file_fields(metadata_source, inst_deploy_id, {"proc_1_file": Path(proc_1_output).name})
    return {
        "metadata_row": row,
        "dataset": trimmed_dataset,
        "output_path": proc_1_output,
        "pre_trim_plot": str(pre_plot_path) if inst_type == "AQD" else None,
        "post_trim_plot": str(post_plot_path) if inst_type == "AQD" else None,
    }
