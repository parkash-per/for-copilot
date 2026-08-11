"""AQD proc_2 workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from ..database_lookup import get_instrument_context, update_metadata_file_fields
from ..imos.writer import build_output_filename, write_imos_file
from ..qc.manual_flags import apply_qc_flag_windows, write_manual_qc_flags_txt


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


def _require_aqd(row):
    if str(row.get("inst_type", "")).strip().upper() != "AQD":
        raise NotImplementedError("Only AQD workflows are implemented.")


def _stage_dir(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_stage_file(stage_dir: Path, configured_name: Any) -> Path:
    if configured_name is not None and str(configured_name).strip():
        candidate = stage_dir / str(configured_name).strip()
        if candidate.exists():
            return candidate
    candidates = sorted(stage_dir.glob("*.nc"))
    if not candidates:
        raise FileNotFoundError(f"No NetCDF files found in {stage_dir}")
    return candidates[-1]


def _load_dataset(input_dataset, row) -> xr.Dataset:
    if isinstance(input_dataset, xr.Dataset):
        return input_dataset.copy(deep=True)

    if input_dataset is not None:
        dataset_path = Path(str(input_dataset)).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = (Path.cwd() / dataset_path).resolve()
        else:
            dataset_path = dataset_path.resolve()
    else:
        stage_dir = _stage_dir(row.get("proc_1_path"))
        dataset_path = _resolve_stage_file(stage_dir, row.get("proc_1_file"))

    with xr.open_dataset(dataset_path) as opened_dataset:
        return opened_dataset.load()


def run_proc2(config, instrument_id=None, input_dataset=None):
    """Run the AQD proc_2 workflow without re-trimming proc_1 data."""
    metadata_source = _metadata_source(config)
    inst_deploy_id = _instrument_key(config, instrument_id)
    _, row, cfg, _ = get_instrument_context(
        metadata_source,
        inst_deploy_id,
        deployment_id=config.get("deployment_id"),
    )
    _require_aqd(row)

    proc_1_dataset = _load_dataset(input_dataset, row)
    manual_qc_flags = list(config.get("manual_qc_flags", config.get("flag_windows", [])) or [])
    proc_2_dataset = apply_qc_flag_windows(proc_1_dataset, manual_qc_flags)

    stage_metadata = {**cfg, **row.to_dict(), **config, **proc_2_dataset.attrs}
    stage_metadata.update(
        {
            "output_stage": "proc_2",
            "output_name_mode": "internal",
            "location": cfg.get("location", row.get("location", "")),
            "instrument": row.get("inst_type", "AQD"),
            "inst_type": row.get("inst_type", "AQD"),
            "inst_id": row.get("inst_id", ""),
            "depth": row.get("nominal_depth", cfg.get("nominal_depth", 0)),
            "start_of_good_data": proc_2_dataset.attrs.get(
                "time_coverage_start",
                row.get("time_coverage_start", row.get("deploy_date")),
            ),
            "output_dir": row.get("proc_2_path"),
        }
    )

    proc_2_dir = _stage_dir(row.get("proc_2_path"))
    output_path = proc_2_dir / build_output_filename(stage_metadata)
    proc_2_output = write_imos_file(proc_2_dataset, output_path, metadata=stage_metadata)
    manual_qc_log = write_manual_qc_flags_txt(manual_qc_flags, proc_2_output)
    update_metadata_file_fields(metadata_source, inst_deploy_id, {"proc_2_file": Path(proc_2_output).name})
    return {
        "metadata_row": row,
        "dataset": proc_2_dataset,
        "output_path": proc_2_output,
        "manual_qc_log": str(manual_qc_log),
    }
