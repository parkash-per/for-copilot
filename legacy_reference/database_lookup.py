import pandas as pd
from pathlib import Path


DEFAULT_METADATA_CSV = "/datasets/work/oa-srsalt/work/preqa/SWOT/cal_val/jason_calval/all_mooring_data/reference_mooring_proc_info/satellite_altimetry_moorings_metadata.csv"


def load_metadata_table(metadata_csv=DEFAULT_METADATA_CSV):
    database = pd.read_csv(metadata_csv)
    database.columns = database.columns.str.strip()
    return database


def get_deployment_row(database, inst_deploy_id):
    matches = database.loc[database["inst_deploy_ID"] == inst_deploy_id]
    if matches.empty:
        raise ValueError(f"inst_deploy_ID {inst_deploy_id} not found in metadata table")
    return matches.iloc[0]

def _format_yyyymm(value):
    s = str(value).strip()
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y"):
        try:
            return pd.to_datetime(s, format=fmt, dayfirst=True).strftime("%Y%m")
        except ValueError:
            pass
    # final fallback
    return pd.to_datetime(s, format="mixed", dayfirst=True, errors="raise").strftime("%Y%m")

def build_cfg(row):
    time_coverage_start = row.get("time_coverage_start", row.get("deploy_date"))
    time_coverage_end = row.get("time_coverage_end", row.get("recovery_date"))
    return {
        "input_file": f"{row['data_in_path']}/{row['data_in_file']}",
        "data_in_path": row.get("data_in_path", ""),
        "data_in_file": row.get("data_in_file", ""),
        "output_dir": row["proc_1_path"],
        "proc_1_path": row.get("proc_1_path", ""),
        "proc_1_file": row.get("proc_1_file", ""),
        "proc_2_path": row.get("proc_2_path", ""),
        "proc_2_file": row.get("proc_2_file", ""),
        "imos_deliverables_path": row.get("imos_deliverables_path", ""),
        "imos_deliverables_file": row.get("imos_deliverables_file", ""),
        "location": row["location"],
        "longitude": row.get("longitude"),
        "latitude": row.get("latitude"),
        "time_coverage_start": _format_yyyymm(time_coverage_start),
        "time_coverage_end": _format_yyyymm(time_coverage_end),
        "time_coverage_start_raw": row.get("time_coverage_start", ""),
        "time_coverage_end_raw": row.get("time_coverage_end", ""),
        "dep_date": _format_yyyymm(time_coverage_start),
        "rec_date": _format_yyyymm(time_coverage_end),
        "instrument": str(row["inst_type"]).lower(),
        "inst_type": str(row.get("inst_type", "")),
        "inst_id": row.get("inst_id"),
        "inst_channels": str(row.get("mooring_channels", "")),
        "mooring_channels": str(row.get("mooring_channels", "")),
        "nominal_depth": row.get("nominal_depth"),
        "nominal_inst_depth": row.get("nominal_inst_depth", ""),
        "version": row.get("version", "1"),
        "deployment_id": row.get("deployment_id", ""),
        "deploy_date": row.get("deploy_date", ""),
        "recovery_date": row.get("recovery_date", ""),
        "serial": str(int(row["inst_id"])),
        "depth": f"{int(row['nominal_depth'])}m",
    }


def format_metadata_lines(row):
    return [
        f"inst_deploy_ID : {row['inst_deploy_ID']}",
        f"site           : {row['location']} | {row['deployment_id']} | {row['mooring_channels']}",
        f"position       : {row['longitude']}, {row['latitude']}",
        f"nominal_depth  : {row['nominal_depth']}",
        f"instrument     : {row['inst_type']} | {row['inst_id']} | {row['nominal_inst_depth']}",
        f"time_coverage  : {row.get('time_coverage_start', row['deploy_date'])} to {row.get('time_coverage_end', row['recovery_date'])}",
        f"data_in_path   : {row['data_in_path']}",
        f"data_in_file   : {row['data_in_file']}",
        f"proc_1_path    : {row['proc_1_path']}",
        f"proc_1_file    : {row['proc_1_file']}",
        f"proc_2_path    : {row['proc_2_path']}",
        f"proc_2_file    : {row['proc_2_file']}",
        f"imos_path      : {row['imos_deliverables_path']}",
        f"imos_file      : {row['imos_deliverables_file']}",
    ]


def get_instrument_context(inst_deploy_id, metadata_csv=DEFAULT_METADATA_CSV, print_details=True):
    database = load_metadata_table(metadata_csv)
    row = get_deployment_row(database, inst_deploy_id)
    cfg = build_cfg(row)
    metadata_lines = format_metadata_lines(row)

    if print_details:
        print("\n".join(metadata_lines))

    return database, row, cfg, metadata_lines


def resolve_pressure_source_by_id(database, pressure_inst_deploy_id, use_proc1=False):
    row = get_deployment_row(database, pressure_inst_deploy_id)
    if use_proc1:
        filepath = f"{row['proc_1_path']}/{row['proc_1_file']}"
    else:
        filepath = f"{row['data_in_path']}/{row['data_in_file']}"
    label = f"{row['inst_type']}_{int(row['inst_id'])} pressure"
    return row, filepath, label


def _path_matches_expected_dir(output_path, expected_dir_value, working_dir=None):
    if output_path is None:
        return True

    if expected_dir_value is None:
        return True

    if isinstance(expected_dir_value, float) and pd.isna(expected_dir_value):
        return True

    expected_dir_text = str(expected_dir_value).strip()
    if expected_dir_text == "" or expected_dir_text.lower() in ("nan", "none"):
        return True

    actual_parent = Path(str(output_path)).expanduser().resolve().parent
    expected_dir = Path(expected_dir_text).expanduser()

    if expected_dir.is_absolute():
        return actual_parent == expected_dir.resolve()

    if working_dir is None:
        return False

    resolved_working_dir = Path(str(working_dir)).expanduser().resolve()
    resolved_expected_dir = (resolved_working_dir / expected_dir).resolve()
    return actual_parent == resolved_expected_dir


def update_metadata_file_fields(
    inst_deploy_id,
    updates,
    metadata_csv=DEFAULT_METADATA_CSV,
    output_paths=None,
    warn_on_path_mismatch=True,
    working_dir=None,
):
    alias_map = {
        "proc_1_file": "proc_1_file",
        "proc_2_file": "proc_2_file",
        "imos_file": "imos_deliverables_file",
        "imos_deliverables_file": "imos_deliverables_file",
    }
    expected_dir_map = {
        "proc_1_file": "proc_1_path",
        "proc_2_file": "proc_2_path",
        "imos_deliverables_file": "imos_deliverables_path",
    }

    database = load_metadata_table(metadata_csv)
    matches = database.index[database["inst_deploy_ID"] == inst_deploy_id]
    if len(matches) == 0:
        raise ValueError(f"inst_deploy_ID {inst_deploy_id} not found in metadata table")

    row_idx = matches[0]
    row = database.loc[row_idx]
    normalized_updates = {}
    for key, value in updates.items():
        if key not in alias_map:
            valid_keys = ", ".join(sorted(alias_map))
            raise ValueError(f"Unsupported metadata field '{key}'. Use one of: {valid_keys}")
        normalized_updates[alias_map[key]] = "" if value is None else str(value)

    normalized_output_paths = {}
    if output_paths:
        for key, value in output_paths.items():
            if key not in alias_map:
                valid_keys = ", ".join(sorted(alias_map))
                raise ValueError(f"Unsupported output_paths field '{key}'. Use one of: {valid_keys}")
            normalized_output_paths[alias_map[key]] = value

    if warn_on_path_mismatch:
        for key, output_path in normalized_output_paths.items():
            expected_dir_key = expected_dir_map.get(key)
            expected_dir_value = row.get(expected_dir_key) if expected_dir_key else None
            if not _path_matches_expected_dir(output_path, expected_dir_value, working_dir=working_dir):
                print(
                    f"WARNING: {key} is being recorded in metadata, but the file path does not match "
                    f"{expected_dir_key}: {output_path}"
                )

    for key, value in normalized_updates.items():
        database.at[row_idx, key] = value

    try:
        database.to_csv(metadata_csv, index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"Metadata file is not writable: {metadata_csv}. "
            "Provide a writable metadata_csv path to update_metadata_file_fields(...) or request write access."
        ) from exc
    return database.loc[row_idx]
