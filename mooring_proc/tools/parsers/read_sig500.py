"""SIG500 / Nortek Signature parser: ``.ad2cp`` primary, OceanContour `.nc` fallback.

Parser paths
------------
``.ad2cp`` (primary)
    Direct binary parsing via ``read_ad2cp_binary`` (ported from
    khannakarishma/imos-toolbox).  Returns one dataset per acquisition
    mode (Burst / Average) as separate xr.Dataset objects.  The result
    dict contains a ``datasets`` key with the list; ``dataset`` holds
    the first (or only) one for backward-compatibility.

OceanContour ``.nc`` (fallback)
    Reads a NetCDF file exported by Nortek OceanContour software.
    Adapted from khannakarishma/imos-toolbox ``ocean_contour.py``.
    The ``/Config`` group supplies metadata; each subgroup under
    ``/Data`` becomes a separate dataset.

IMOS variable mapping (ENU coordinate system)
    =====================  ================
    OceanContour / AD2CP   IMOS name
    =====================  ================
    Vel_East / East        UCUR (or UCUR_MAG)
    Vel_North / North      VCUR (or VCUR_MAG)
    Vel_Up1 / Up           WCUR
    Amp_Beam1–4            ABSI1–4
    Cor_Beam1–4            CMAG1–4
    WaterTemperature       TEMP
    Pressure               PRES_REL
    SpeedOfSound           SSPD
    Battery                BAT_VOLT
    Pitch / Roll / Heading PITCH / ROLL / HEADING(_MAG)
    =====================  ================
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .imos_utils import (
    coerce_row,
    is_blank,
    matlab_datenum_to_datetime64,
    range_tag_from_times,
)
from .read_ad2cp_binary import read_ad2cp_binary

# Beam angle by instrument model (from OceanContour parser)
_BEAM_ANGLES = {
    "Signature250": 20.0,
    "Signature500": 25.0,
    "Signature1000": 25.0,
}

# OceanContour NetCDF variable → IMOS name (ENU, no mag declination)
_NC_VARMAP_2D = {
    "Vel_East": "UCUR",
    "Vel_North": "VCUR",
    "Vel_Up1": "WCUR",
    "Vel_Up2": "WCUR_2",
    "Amp_Beam1": "ABSI1",
    "Amp_Beam2": "ABSI2",
    "Amp_Beam3": "ABSI3",
    "Amp_Beam4": "ABSI4",
    "Cor_Beam1": "CMAG1",
    "Cor_Beam2": "CMAG2",
    "Cor_Beam3": "CMAG3",
    "Cor_Beam4": "CMAG4",
}
_NC_VARMAP_2D_MAG = {**_NC_VARMAP_2D, "Vel_East": "UCUR_MAG", "Vel_North": "VCUR_MAG"}

_NC_VARMAP_1D = {
    "WaterTemperature": "TEMP",
    "Pressure": "PRES_REL",
    "SpeedOfSound": "SSPD",
    "Battery": "BAT_VOLT",
    "Pitch": "PITCH",
    "Roll": "ROLL",
    "Heading": "HEADING",
}
_NC_VARMAP_1D_MAG = {**_NC_VARMAP_1D, "Heading": "HEADING_MAG"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def read_sig500(input_path, config=None):
    """Read SIG500 / Nortek Signature data.

    Parameters
    ----------
    input_path:
        Path to a ``.ad2cp`` or OceanContour ``.nc`` file, or ``None``
        to resolve from *config*.
    config:
        Metadata dict or ``{'metadata_row': <row>}``.

    Returns
    -------
    dict with keys:

    - ``dataset`` — first (or only) ``xr.Dataset``
    - ``datasets`` — list of all datasets (one per acquisition mode)
    - ``input_path`` — resolved path string
    - ``range_tag`` — ISO range string
    - ``dataframe`` — ``None`` (profile/ADCP data does not reduce to a
      single 2-D frame; use the dataset directly)
    """
    row = coerce_row(config)
    resolved = _resolve_input_path(input_path, row)
    suffix = resolved.suffix.lower()

    if suffix == ".ad2cp":
        datasets = _parse_ad2cp(resolved, row)
    elif suffix == ".nc":
        datasets = _parse_ocean_contour_nc(resolved, row)
    else:
        raise ValueError(
            f"SIG500 parser: unsupported suffix '{suffix}'.  "
            "Supported: .ad2cp, .nc"
        )

    if not datasets:
        raise ValueError(f"No valid datasets produced from {resolved}")

    primary = datasets[0]
    range_tag = primary.attrs.get("range_tag", "")
    return {
        "dataframe": None,
        "dataset": primary,
        "datasets": datasets,
        "input_path": str(resolved),
        "range_tag": range_tag,
    }


# ---------------------------------------------------------------------------
# AD2CP binary parser
# ---------------------------------------------------------------------------

def _parse_ad2cp(source_file: Path, row: dict[str, Any]) -> list[xr.Dataset]:
    structures = read_ad2cp_binary(source_file)

    # Extract instrument info from string records (IdA0)
    instrument_model = ""
    mag_dec = 0.0
    serial_number = ""

    for key in list(structures.keys()):
        if key.startswith("IdA0"):
            for rec in structures[key]["Data"]:
                string_data = rec.get("String", "")
                if string_data:
                    model = _extract_header_key(string_data, "ID", "STR")
                    if model:
                        instrument_model = model
                    decl = _extract_header_key(string_data, "GETUSER", "DECL")
                    if decl:
                        try:
                            mag_dec = float(decl)
                        except ValueError:
                            pass
            del structures[key]
            break

    # Remove unsupported record types
    for key in [k for k in structures if "Id1A" in k or "Id1C" in k or "Id1D" in k]:
        del structures[key]

    # Prefer Version 3 records
    acq_modes = [k for k in structures if "Version3" in k and ("Id15" in k or "Id16" in k)]
    if not acq_modes:
        acq_modes = [k for k in structures if "Id15" in k or "Id16" in k]

    if not acq_modes:
        raise ValueError(f"No burst/average records found in {source_file}")

    # Infer serial number from first data record
    first_data = structures[acq_modes[0]]["Data"]
    if first_data and "SerialNumber" in first_data[0]:
        serial_number = str(first_data[0]["SerialNumber"])

    datasets: list[xr.Dataset] = []
    for mode_key in acq_modes:
        ds = _build_ad2cp_dataset(
            source_file=source_file,
            mode_key=mode_key,
            records=structures[mode_key]["Data"],
            instrument_model=instrument_model,
            serial_number=serial_number,
            mag_dec=mag_dec,
            row=row,
        )
        if ds is not None:
            datasets.append(ds)

    return datasets


def _build_ad2cp_dataset(
    source_file: Path,
    mode_key: str,
    records: list[dict],
    instrument_model: str,
    serial_number: str,
    mag_dec: float,
    row: dict[str, Any],
) -> xr.Dataset | None:
    if not records:
        return None

    first = records[0]
    n_cells = first.get("nCells", 0)
    n_beams = first.get("nBeams", 0)
    coord_sys = first.get("coordSys", 0)
    cell_size = first.get("CellSize", 0) * 0.001   # mm → m
    blank_dist = first.get("Blanking", 0) * 0.001  # mm → m

    if n_cells == 0 or n_beams == 0:
        return None

    n_samples = len(records)

    time_dn = np.array([r.get("Time", np.nan) for r in records])
    temperature = np.array([r.get("Temperature", 0) for r in records]) * 0.01
    pressure = np.array([r.get("Pressure", 0) for r in records]) * 0.001
    heading = np.array([r.get("Heading", 0) for r in records]) * 0.01
    pitch = np.array([r.get("Pitch", 0) for r in records]) * 0.01
    roll = np.array([r.get("Roll", 0) for r in records]) * 0.01
    speed_of_sound = np.array([r.get("SpeedOfSound", 0) for r in records]) * 0.1
    battery = np.array([r.get("BatteryVoltage", 0) for r in records]) * 0.1

    distance = blank_dist + cell_size / 2.0 + np.arange(n_cells) * cell_size
    vel_scale = 10.0 ** first.get("VelocityScaling", -3)

    velocity: np.ndarray | None = None
    amplitude: np.ndarray | None = None
    correlation: np.ndarray | None = None

    if "VelocityData" in first:
        velocity = np.zeros((n_samples, n_beams, n_cells), dtype=np.float32)
        for i, r in enumerate(records):
            vd = r.get("VelocityData")
            if vd is not None and vd.shape == (n_beams, n_cells):
                velocity[i] = vd * vel_scale

    if "AmplitudeData" in first:
        amplitude = np.zeros((n_samples, n_beams, n_cells), dtype=np.float32)
        for i, r in enumerate(records):
            ad = r.get("AmplitudeData")
            if ad is not None and ad.shape == (n_beams, n_cells):
                amplitude[i] = ad

    if "CorrelationData" in first:
        correlation = np.zeros((n_samples, n_beams, n_cells), dtype=np.float32)
        for i, r in enumerate(records):
            cd = r.get("CorrelationData")
            if cd is not None and cd.shape == (n_beams, n_cells):
                correlation[i] = cd

    time64 = matlab_datenum_to_datetime64(time_dn)
    range_tag = range_tag_from_times(time64)

    dist_dim = "HEIGHT_ABOVE_SENSOR" if coord_sys == 0 else "DIST_ALONG_BEAMS"
    coords_2d = f"TIME LATITUDE LONGITUDE {dist_dim}"
    coords_1d = "TIME LATITUDE LONGITUDE NOMINAL_DEPTH"

    xr_vars: dict[str, xr.DataArray] = {
        "TIMESERIES": xr.DataArray(np.int32(1), attrs={"cf_role": "timeseries_id"}),
        "LATITUDE": xr.DataArray(
            np.float64(row.get("latitude", np.nan)), attrs={"units": "degrees_north"}
        ),
        "LONGITUDE": xr.DataArray(
            np.float64(row.get("longitude", np.nan)), attrs={"units": "degrees_east"}
        ),
        "NOMINAL_DEPTH": xr.DataArray(
            np.float32(row.get("nominal_depth", np.nan)),
            attrs={"units": "m", "positive": "down"},
        ),
    }

    if velocity is not None:
        if coord_sys == 0:  # ENU
            mag_ext = "_MAG" if mag_dec == 0 else ""
            vel_names = [f"UCUR{mag_ext}", f"VCUR{mag_ext}", "WCUR"]
            if n_beams > 3:
                vel_names.append("WCUR_2")
        else:
            vel_names = [f"VEL{i + 1}" for i in range(n_beams)]

        for bi, vname in enumerate(vel_names[:n_beams]):
            xr_vars[vname] = xr.DataArray(
                velocity[:, bi, :],
                dims=["TIME", dist_dim],
                attrs={"coordinates": coords_2d},
            )

    if amplitude is not None:
        for bi in range(n_beams):
            xr_vars[f"ABSI{bi + 1}"] = xr.DataArray(
                amplitude[:, bi, :],
                dims=["TIME", dist_dim],
                attrs={"coordinates": coords_2d},
            )

    if correlation is not None:
        for bi in range(n_beams):
            xr_vars[f"CMAG{bi + 1}"] = xr.DataArray(
                correlation[:, bi, :],
                dims=["TIME", dist_dim],
                attrs={"coordinates": coords_2d},
            )

    heading_name = "HEADING_MAG" if mag_dec == 0 else "HEADING"
    for name, data in [
        ("TEMP", temperature), ("PRES_REL", pressure),
        ("SSPD", speed_of_sound), ("BAT_VOLT", battery),
        ("PITCH", pitch), ("ROLL", roll), (heading_name, heading),
    ]:
        xr_vars[name] = xr.DataArray(data, dims=["TIME"], attrs={"coordinates": coords_1d})

    sample_interval = (
        float(np.median(np.diff(time_dn) * 86400.0)) if len(time_dn) > 1 else float("nan")
    )

    ds = xr.Dataset(
        data_vars=xr_vars,
        coords={
            "TIME": time64,
            dist_dim: distance.astype(np.float32),
        },
        attrs={
            "source_file": str(source_file),
            "instrument": str(instrument_model or row.get("inst_type", "Signature")),
            "instrument_make": "Nortek",
            "instrument_model": str(instrument_model or "Signature"),
            "instrument_serial_no": str(serial_number or row.get("inst_id", "")),
            "instrument_sample_interval": sample_interval,
            "beam_angle": float(_BEAM_ANGLES.get(str(instrument_model), 25.0)),
            "nBeams": n_beams,
            "nCells": n_cells,
            "cellSize": cell_size,
            "blankDist": blank_dist,
            "coordinate_system": {0: "ENU", 1: "XYZ", 2: "Beam"}.get(coord_sys, "unknown"),
            "acquisition_mode": mode_key,
            "serial": str(row.get("inst_id", "")),
            "location": str(row.get("location", "")),
            "deployment_id": str(row.get("deployment_id", "")),
            "range_tag": range_tag,
        },
    )
    return ds


def _extract_header_key(header_string: str, section: str, key: str) -> str | None:
    pattern = rf"{section}.*?{key}\s*=\s*\"?([^\"\\n,]+)\"?"
    match = re.search(pattern, header_string, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


# ---------------------------------------------------------------------------
# OceanContour NetCDF parser
# ---------------------------------------------------------------------------

def _parse_ocean_contour_nc(source_file: Path, row: dict[str, Any]) -> list[xr.Dataset]:
    try:
        import netCDF4  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "OceanContour .nc support requires netCDF4.  "
            "Install with: pip install netCDF4"
        ) from exc

    root = netCDF4.Dataset(str(source_file), "r")
    try:
        config_group = root.groups.get("Config")
        if config_group is None:
            raise ValueError("OceanContour file missing 'Config' group")
        file_meta = {attr: config_group.getncattr(attr) for attr in config_group.ncattrs()}

        data_group = root.groups.get("Data")
        if data_group is None:
            raise ValueError("OceanContour file missing 'Data' group")

        group_names = list(data_group.groups.keys())
        if not group_names:
            raise ValueError("OceanContour 'Data' group has no subgroups")

        datasets: list[xr.Dataset] = []
        for gname in group_names:
            ds = _parse_nc_group(
                source_file, data_group.groups[gname], gname, file_meta, row
            )
            if ds is not None:
                datasets.append(ds)
    finally:
        root.close()

    return datasets


def _parse_nc_group(
    source_file: Path,
    grp: Any,
    group_name: str,
    file_meta: dict[str, Any],
    row: dict[str, Any],
) -> xr.Dataset | None:
    meta_mid = group_name[0].lower() + group_name[1:]

    def get_meta(key: str) -> Any:
        if key in file_meta:
            return file_meta[key]
        return file_meta.get(f"Instrument_{meta_mid}_{key}")

    mag_dec = float(get_meta("Instrument_user_decl") or 0.0)
    custom_mag = bool(mag_dec)

    # Only ENU is supported
    coord_key = f"Instrument_{meta_mid}_coordSystem"
    coord_system = file_meta.get(coord_key, "ENU")
    if coord_system != "ENU":
        return None

    def read_nc(nc_name: str) -> np.ndarray | None:
        if nc_name in grp.variables:
            return np.asarray(grp.variables[nc_name][:])
        return None

    time = read_nc("MatlabTimeStamp")
    if time is None and "time" in grp.variables:
        time = np.asarray(grp.variables["time"][:])
    if time is None:
        return None

    # HEIGHT_ABOVE_SENSOR (Range)
    height = None
    for vn in grp.variables:
        if "VelocityENU_Range" in vn or ("Range" in vn and "Velocity" in vn):
            height = np.asarray(grp.variables[vn][:])
            break
    if height is None:
        for vn in grp.variables:
            if "Range" in vn:
                height = np.asarray(grp.variables[vn][:])
                break
    if height is None:
        return None

    time64 = matlab_datenum_to_datetime64(time.ravel())
    range_tag = range_tag_from_times(time64)

    varmap_2d = _NC_VARMAP_2D_MAG if custom_mag else _NC_VARMAP_2D
    varmap_1d = _NC_VARMAP_1D_MAG if custom_mag else _NC_VARMAP_1D

    coords_2d = "TIME LATITUDE LONGITUDE HEIGHT_ABOVE_SENSOR"
    coords_1d = "TIME LATITUDE LONGITUDE NOMINAL_DEPTH"

    n_beams_val = get_meta("nBeams") or 4
    n_beams = int(n_beams_val)

    xr_vars: dict[str, xr.DataArray] = {
        "TIMESERIES": xr.DataArray(np.int32(1), attrs={"cf_role": "timeseries_id"}),
        "LATITUDE": xr.DataArray(
            np.float64(row.get("latitude", np.nan)), attrs={"units": "degrees_north"}
        ),
        "LONGITUDE": xr.DataArray(
            np.float64(row.get("longitude", np.nan)), attrs={"units": "degrees_east"}
        ),
        "NOMINAL_DEPTH": xr.DataArray(
            np.float32(row.get("nominal_depth", np.nan)),
            attrs={"units": "m", "positive": "down"},
        ),
    }

    for nc_name, imos_name in varmap_2d.items():
        if n_beams <= 3 and nc_name in ("Vel_Up2", "Amp_Beam4", "Cor_Beam4"):
            continue
        data = read_nc(nc_name)
        if data is not None:
            if data.ndim == 2:
                nh, nt = len(height.ravel()), len(time.ravel())
                if data.shape == (nh, nt):
                    data = data.T
            xr_vars[imos_name] = xr.DataArray(
                data, dims=["TIME", "HEIGHT_ABOVE_SENSOR"],
                attrs={"coordinates": coords_2d},
            )

    for nc_name, imos_name in varmap_1d.items():
        data = read_nc(nc_name)
        if data is not None:
            xr_vars[imos_name] = xr.DataArray(
                data.ravel(), dims=["TIME"], attrs={"coordinates": coords_1d}
            )

    instrument_model = str(file_meta.get("Instrument_instrumentName", "Signature"))
    beam_angle = float(_BEAM_ANGLES.get(instrument_model, 25.0))

    # Serial number
    serial_data = read_nc("SerialNumber")
    serial_no = (
        str(int(serial_data[0]))
        if serial_data is not None and serial_data.size > 0
        else str(row.get("inst_id", ""))
    )

    sample_interval_key = f"Instrument_{meta_mid}_measurementInterval"
    sample_interval = file_meta.get(sample_interval_key, float("nan"))

    ds = xr.Dataset(
        data_vars=xr_vars,
        coords={
            "TIME": time64,
            "HEIGHT_ABOVE_SENSOR": height.ravel().astype(np.float32),
        },
        attrs={
            "source_file": str(source_file),
            "instrument": instrument_model,
            "instrument_make": "Nortek",
            "instrument_model": instrument_model,
            "instrument_serial_no": serial_no,
            "instrument_sample_interval": float(sample_interval)
            if not isinstance(sample_interval, float)
            else sample_interval,
            "beam_angle": beam_angle,
            "nBeams": n_beams,
            "coordinate_system": coord_system,
            "netcdf_group_name": group_name,
            "serial": str(row.get("inst_id", "")),
            "location": str(row.get("location", "")),
            "deployment_id": str(row.get("deployment_id", "")),
            "range_tag": range_tag,
        },
    )
    return ds


# ---------------------------------------------------------------------------
# Path resolver
# ---------------------------------------------------------------------------

def _resolve_input_path(input_path: Any, row: dict[str, Any]) -> Path:
    if not is_blank(input_path):
        p = Path(str(input_path)).expanduser()
        return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
    data_in_path = row.get("data_in_path")
    data_in_file = row.get("data_in_file")
    if is_blank(data_in_path) or is_blank(data_in_file):
        raise ValueError(
            "SIG500 input path missing; provide input_path or "
            "metadata data_in_path/data_in_file."
        )
    base = Path(str(data_in_path)).expanduser()
    base = base.resolve() if base.is_absolute() else (Path.cwd() / base).resolve()
    return base / str(data_in_file).strip()

