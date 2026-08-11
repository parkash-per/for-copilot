"""AQD IMOS delivery workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from ..database_lookup import get_instrument_context, update_metadata_file_fields
from ..imos.publisher import publish_delivery


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


def _delivery_metadata(row, cfg, version: str, input_path: Path) -> dict[str, Any]:
    with xr.open_dataset(input_path) as opened_dataset:
        dataset = opened_dataset.load()

    attrs = dict(dataset.attrs)
    depth_value = attrs.get("NOMINAL_DEPTH", row.get("nominal_depth", cfg.get("nominal_depth", 0)))
    if "NOMINAL_DEPTH" in dataset.variables:
        depth_value = float(dataset["NOMINAL_DEPTH"].values)
    return {
        **cfg,
        **row.to_dict(),
        **attrs,
        "output_name_mode": "imos",
        "output_stage": "imos_delivery",
        "version": version,
        "location": cfg.get("location", row.get("location", "")),
        "instrument": row.get("inst_type", "AQD"),
        "inst_type": row.get("inst_type", "AQD"),
        "inst_id": row.get("inst_id", ""),
        "depth": depth_value,
        "start_of_good_data": attrs.get("time_coverage_start", row.get("time_coverage_start", row.get("deploy_date"))),
        "time_coverage_start": attrs.get("time_coverage_start", row.get("time_coverage_start", row.get("deploy_date"))),
        "time_coverage_end": attrs.get("time_coverage_end", row.get("time_coverage_end", row.get("recovery_date"))),
        "inst_channels": attrs.get("mooring_channels", cfg.get("inst_channels", row.get("mooring_channels", ""))),
        "mooring_channels": attrs.get("mooring_channels", cfg.get("mooring_channels", row.get("mooring_channels", ""))),
    }


def run_imos_delivery(config, instrument_id=None, input_dataset=None):
    """Publish proc_1 as FV00 and proc_2 as FV01 for AQD."""
    metadata_source = _metadata_source(config)
    inst_deploy_id = _instrument_key(config, instrument_id)
    _, row, cfg, _ = get_instrument_context(
        metadata_source,
        inst_deploy_id,
        deployment_id=config.get("deployment_id"),
    )
    _require_aqd(row)

    proc_1_path = _resolve_stage_file(_stage_dir(row.get("proc_1_path")), row.get("proc_1_file"))
    proc_2_path = _resolve_stage_file(_stage_dir(row.get("proc_2_path")), row.get("proc_2_file"))
    delivery_dir = _stage_dir(row.get("imos_deliverables_path", row.get("imos_path", "")))

    fv00_output = publish_delivery(proc_1_path, delivery_dir, metadata=_delivery_metadata(row, cfg, "0", proc_1_path))
    fv01_output = publish_delivery(proc_2_path, delivery_dir, metadata=_delivery_metadata(row, cfg, "1", proc_2_path))

    deliverable_names = ";".join([Path(fv00_output).name, Path(fv01_output).name])
    update_metadata_file_fields(
        metadata_source,
        inst_deploy_id,
        {"imos_deliverables_file": deliverable_names},
    )
    return {
        "metadata_row": row,
        "proc_1_delivery": fv00_output,
        "proc_2_delivery": fv01_output,
        "imos_deliverables_file": deliverable_names,
    }
