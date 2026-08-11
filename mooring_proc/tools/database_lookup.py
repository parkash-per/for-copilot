"""Metadata lookup helpers for mooring_proc."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none"}


def _coerce_table(source: Any, table_name: str | None = None) -> tuple[pd.DataFrame, Path | None]:
    if isinstance(source, pd.DataFrame):
        table = source.copy()
        table.columns = table.columns.str.strip()
        return table, None

    if isinstance(source, dict) and table_name and table_name in source:
        return _coerce_table(source[table_name], None)

    if _is_blank(source):
        raise ValueError("A metadata table source is required.")

    source_path = Path(str(source)).expanduser()
    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).resolve()
    else:
        source_path = source_path.resolve()

    separator = "\t" if source_path.suffix.lower() in {".tsv", ".tab"} else ","
    table = pd.read_csv(source_path, sep=separator)
    table.columns = table.columns.str.strip()
    return table, source_path


def _format_yyyymm(value: Any) -> str:
    if _is_blank(value):
        return ""
    parsed = pd.to_datetime(value, format="mixed", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, dayfirst=True, format="mixed", errors="coerce")
    if pd.isna(parsed):
        return str(value).strip()
    return parsed.strftime("%Y%m")


def _coerce_int_text(value: Any) -> str:
    if _is_blank(value):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def _find_row_index(metadata_table: pd.DataFrame, instrument_id: Any, deployment_id: Any = None) -> Any:
    if "inst_deploy_ID" in metadata_table.columns:
        mask = metadata_table["inst_deploy_ID"].astype(str).str.strip() == str(instrument_id).strip()
        matches = metadata_table.index[mask]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1 and deployment_id is None:
            return matches[0]
        if len(matches) > 1 and "deployment_id" in metadata_table.columns:
            deployment_mask = (
                metadata_table.loc[matches, "deployment_id"].astype(str).str.strip()
                == str(deployment_id).strip()
            )
            deployment_matches = metadata_table.loc[matches].index[deployment_mask]
            if len(deployment_matches) == 1:
                return deployment_matches[0]

    if {"inst_id", "deployment_id"}.issubset(metadata_table.columns) and deployment_id is not None:
        mask = (
            metadata_table["inst_id"].astype(str).str.strip() == str(instrument_id).strip()
        ) & (
            metadata_table["deployment_id"].astype(str).str.strip() == str(deployment_id).strip()
        )
        matches = metadata_table.index[mask]
        if len(matches) == 1:
            return matches[0]

    raise ValueError(f"Instrument deployment '{instrument_id}' was not found in metadata.")


def _build_cfg(row: pd.Series) -> dict[str, Any]:
    time_start = row.get("time_coverage_start", row.get("deploy_date", ""))
    time_end = row.get("time_coverage_end", row.get("recovery_date", ""))
    return {
        "inst_deploy_ID": row.get("inst_deploy_ID", ""),
        "input_file": f"{row.get('data_in_path', '')}/{row.get('data_in_file', '')}".strip("/"),
        "data_in_path": row.get("data_in_path", ""),
        "data_in_file": row.get("data_in_file", ""),
        "proc_1_path": row.get("proc_1_path", ""),
        "proc_1_file": row.get("proc_1_file", ""),
        "proc_2_path": row.get("proc_2_path", ""),
        "proc_2_file": row.get("proc_2_file", ""),
        "imos_deliverables_path": row.get("imos_deliverables_path", row.get("imos_path", "")),
        "imos_deliverables_file": row.get("imos_deliverables_file", row.get("imos_file", "")),
        "location": row.get("location", ""),
        "deployment_id": row.get("deployment_id", ""),
        "longitude": row.get("longitude", ""),
        "latitude": row.get("latitude", ""),
        "time_coverage_start": time_start,
        "time_coverage_end": time_end,
        "time_coverage_start_yyyymm": _format_yyyymm(time_start),
        "time_coverage_end_yyyymm": _format_yyyymm(time_end),
        "inst_type": row.get("inst_type", ""),
        "instrument": str(row.get("inst_type", "")).lower(),
        "inst_id": row.get("inst_id", ""),
        "mooring_channels": str(row.get("mooring_channels", "")),
        "inst_channels": str(row.get("mooring_channels", row.get("inst_channels", ""))),
        "nominal_depth": row.get("nominal_depth", ""),
        "nominal_inst_depth": row.get("nominal_inst_depth", ""),
        "version": row.get("version", "1"),
        "deploy_date": row.get("deploy_date", ""),
        "recovery_date": row.get("recovery_date", ""),
        "serial": _coerce_int_text(row.get("inst_id", "")),
        "depth": f"{_coerce_int_text(row.get('nominal_depth', ''))}m".strip(),
    }


def _format_metadata_lines(row: pd.Series) -> list[str]:
    return [
        f"inst_deploy_ID : {row.get('inst_deploy_ID', '')}",
        f"site           : {row.get('location', '')} | {row.get('deployment_id', '')} | {row.get('mooring_channels', '')}",
        f"position       : {row.get('longitude', '')}, {row.get('latitude', '')}",
        f"nominal_depth  : {row.get('nominal_depth', '')}",
        f"instrument     : {row.get('inst_type', '')} | {row.get('inst_id', '')} | {row.get('nominal_inst_depth', '')}",
        f"time_coverage  : {row.get('time_coverage_start', row.get('deploy_date', ''))} to {row.get('time_coverage_end', row.get('recovery_date', ''))}",
        f"data_in_path   : {row.get('data_in_path', '')}",
        f"data_in_file   : {row.get('data_in_file', '')}",
        f"proc_1_path    : {row.get('proc_1_path', '')}",
        f"proc_1_file    : {row.get('proc_1_file', '')}",
        f"proc_2_path    : {row.get('proc_2_path', '')}",
        f"proc_2_file    : {row.get('proc_2_file', '')}",
        f"imos_path      : {row.get('imos_deliverables_path', row.get('imos_path', ''))}",
        f"imos_file      : {row.get('imos_deliverables_file', row.get('imos_file', ''))}",
    ]


def _resolve_identifier(file_record: Any) -> tuple[Any, Any]:
    if isinstance(file_record, pd.Series):
        return file_record.get("inst_deploy_ID"), file_record.get("deployment_id")
    if isinstance(file_record, dict):
        return file_record.get("inst_deploy_ID", file_record.get("instrument_id")), file_record.get("deployment_id")
    return file_record, None


def load_metadata_table(source, table_name=None):
    """Load a metadata table from a dataframe or delimited file."""
    metadata_table, _ = _coerce_table(source, table_name=table_name)
    return metadata_table


def get_instrument_context(metadata_table, instrument_id, deployment_id=None):
    """Return ``(metadata_table, row, cfg, metadata_lines)`` for one deployment."""
    table, _ = _coerce_table(metadata_table)
    row_index = _find_row_index(table, instrument_id, deployment_id=deployment_id)
    row = table.loc[row_index].copy()
    cfg = _build_cfg(row)
    metadata_lines = _format_metadata_lines(row)
    return table, row, cfg, metadata_lines


def update_metadata_file_fields(metadata_table, file_record, updates):
    """Update output filename fields in a metadata table."""
    if not isinstance(updates, dict) or not updates:
        raise ValueError("updates must be a non-empty mapping.")

    table, source_path = _coerce_table(metadata_table)
    inst_deploy_id, deployment_id = _resolve_identifier(file_record)
    row_index = _find_row_index(table, inst_deploy_id, deployment_id=deployment_id)

    alias_map = {
        "proc_1_file": "proc_1_file",
        "proc_2_file": "proc_2_file",
        "imos_file": "imos_deliverables_file",
        "imos_deliverables_file": "imos_deliverables_file",
    }

    for key, value in updates.items():
        if key not in alias_map:
            valid = ", ".join(sorted(alias_map))
            raise ValueError(f"Unsupported metadata field '{key}'. Use one of: {valid}")
        target_column = alias_map[key]
        if target_column not in table.columns:
            table[target_column] = ""
        if table[target_column].dtype != object:
            table[target_column] = table[target_column].astype(object)
        table.at[row_index, target_column] = "" if value is None else str(value)

    if source_path is not None:
        table.to_csv(source_path, index=False)

    return table.loc[row_index].copy()


def list_instruments(
    metadata_table,
    year=None,
    location=None,
    instrument=None,
    mooring_channel=None,
    deployment_id=None,
    include_paths=False,
):
    """List instruments using lightweight metadata filters."""
    table, _ = _coerce_table(metadata_table)
    filtered = table.copy()

    if year is not None:
        parsed = pd.to_datetime(
            filtered.get("time_coverage_start", filtered.get("deploy_date")),
            format="mixed",
            errors="coerce",
        )
        year_mask = parsed.dt.year.eq(int(year))
        if "deploy_date" in filtered.columns:
            deploy_years = pd.to_datetime(
                filtered["deploy_date"],
                format="mixed",
                errors="coerce",
            ).dt.year.eq(int(year))
            year_mask = year_mask | deploy_years
        filtered = filtered.loc[year_mask.fillna(False)]

    if location is not None and "location" in filtered.columns:
        filtered = filtered.loc[
            filtered["location"].astype(str).str.strip().str.casefold() == str(location).strip().casefold()
        ]

    if instrument is not None and "inst_type" in filtered.columns:
        filtered = filtered.loc[
            filtered["inst_type"].astype(str).str.strip().str.casefold() == str(instrument).strip().casefold()
        ]

    if mooring_channel is not None and "mooring_channels" in filtered.columns:
        filtered = filtered.loc[
            filtered["mooring_channels"]
            .astype(str)
            .str.casefold()
            .str.contains(str(mooring_channel).strip().casefold(), na=False)
        ]

    if deployment_id is not None and "deployment_id" in filtered.columns:
        filtered = filtered.loc[
            filtered["deployment_id"].astype(str).str.strip() == str(deployment_id).strip()
        ]

    if not include_paths:
        drop_columns = [column for column in filtered.columns if column.endswith("_path")]
        filtered = filtered.drop(columns=drop_columns)

    return filtered.reset_index(drop=True)
