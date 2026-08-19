"""Instrument proc_1 workflow (parse + trim)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

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
    def _normalize_timestamp(value: pd.Timestamp) -> pd.Timestamp:
        if getattr(value, "tzinfo", None) is not None:
            return value.tz_convert("UTC").tz_localize(None)
        return value

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
    return _normalize_timestamp(start_time), _normalize_timestamp(end_time)


def _stage_dir(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _select_dataset(parsed: dict[str, Any], config: dict[str, Any]):
    if "datasets" in parsed and isinstance(parsed["datasets"], list) and parsed["datasets"]:
        index = int(config.get("sig500_dataset_index", 0))
        return parsed["datasets"][index]
    return parsed["dataset"]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_stage_file(stage_dir: Path, configured_name: Any) -> Path:
    if configured_name is not None and str(configured_name).strip():
        candidate = stage_dir / str(configured_name).strip()
        if candidate.exists():
            return candidate
    candidates = sorted(stage_dir.glob("*.nc"))
    if not candidates:
        raise FileNotFoundError(f"No NetCDF files found in {stage_dir}")
    return candidates[-1]


def _resolve_existing_proc1_path(row, config: dict[str, Any]) -> Path:
    explicit_path = (
        config.get("proc_1_input_dataset")
        or config.get("existing_proc1_path")
        or config.get("existing_proc_1_path")
    )
    if explicit_path is not None and str(explicit_path).strip():
        path = Path(str(explicit_path)).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Configured proc_1_input_dataset does not exist: {path}")
        return path

    proc_1_dir = _stage_dir(row.get("proc_1_path"))
    return _resolve_stage_file(proc_1_dir, row.get("proc_1_file"))


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

    if _as_bool(config.get("reuse_existing_proc1", False)):
        proc_1_path = _resolve_existing_proc1_path(row, config)
        with xr.open_dataset(proc_1_path) as existing:
            existing_dataset = existing.load()
        update_metadata_file_fields(metadata_source, inst_deploy_id, {"proc_1_file": proc_1_path.name})
        return {
            "metadata_row": row,
            "dataset": existing_dataset,
            "review_dataset": existing_dataset,
            "deployment_windows": [],
            "output_path": str(proc_1_path),
            "reused_existing": True,
        }

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

    proc_1_output = write_imos_file(trimmed_dataset, output_path, metadata=stage_metadata)
    update_metadata_file_fields(metadata_source, inst_deploy_id, {"proc_1_file": Path(proc_1_output).name})
    return {
        "metadata_row": row,
        "dataset": trimmed_dataset,
        "review_dataset": dataset_with_qc,
        "deployment_windows": deployment_windows,
        "output_path": proc_1_output,
        "reused_existing": False,
    }
