"""SeaBird CNV variable name → IMOS name and unit conversion.

Adapted from khannakarishma/imos-toolbox
``python/src/imos_toolbox/parsers/convert_sbe_var.py``.

The function ``convert_sbe_var`` maps the short variable identifiers used
in Seabird CNV files (e.g. ``tv290C``, ``pr``, ``sal00``) to their IMOS
standard counterparts (e.g. ``TEMP``, ``PRES_REL``, ``PSAL``) and applies
any required unit conversions.

Only the names that appear in practice for the instruments in scope
(SBE26, SBE37, SBE37SM) are included.  Unknown names return an empty
string and should be silently skipped by callers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np


def convert_sbe_var(
    name: str,
    data: np.ndarray,
    time_offset: float = 0.0,
    mode: str = "timeSeries",
    inst_header: dict[str, Any] | None = None,
    proc_header: dict[str, Any] | None = None,
) -> tuple[str, np.ndarray, str]:
    """Map a SeaBird CNV variable to its IMOS equivalent.

    Parameters
    ----------
    name:
        Short SeaBird variable name extracted from the CNV ``# name``
        header line (the part before the colon).
    data:
        1-D numpy array of raw values.
    time_offset:
        MATLAB datenum reference used for elapsed-time conversions
        (e.g. start_time from the processed header).
    mode:
        ``'timeSeries'`` or ``'profile'``.
    inst_header:
        Parsed instrument header dict (used for voltage channel mapping).
    proc_header:
        Parsed processed header dict (used for voltage channel comments).

    Returns
    -------
    tuple[str, np.ndarray, str]
        ``(imos_name, converted_data, comment)``.  Returns
        ``('', empty_array, '')`` when the variable should be skipped.
    """
    # ---- TIME variables (elapsed in various units) -------------------------
    if name == "timeS":           # seconds since start
        return "TIME", data / 86400.0 + time_offset, ""
    if name == "timeM":           # minutes since start
        return "TIME", data / 1440.0 + time_offset, ""
    if name == "timeH":           # hours since start
        return "TIME", data / 24.0 + time_offset, ""
    if name == "timeJ":           # Julian days since start of year
        if time_offset == 0:
            time_offset = 2010
        start_year = datetime.fromordinal(int(time_offset) - 366).year
        converted = data + _matlab_datenum(datetime(start_year - 1, 12, 31))
        return "TIME", converted, ""
    if name == "timeK":           # seconds since 2000-01-01
        epoch_2000 = _matlab_datenum(datetime(2000, 1, 1))
        return "TIME", data / 86400.0 + epoch_2000, ""
    if name == "timeY":           # seconds since 1970-01-01 (Unix)
        epoch_1970 = _matlab_datenum(datetime(1970, 1, 1))
        return "TIME", data / 86400.0 + epoch_1970, ""

    # ---- Pressure ---------------------------------------------------------
    if name in ("pr", "prM", "prdM", "prDM", "prSM"):
        return "PRES_REL", data, ""
    if name == "prdE":            # psi → dbar
        return "PRES_REL", data * 0.68948, ""

    # ---- Temperature ------------------------------------------------------
    if name in ("t090C", "tv290C", "t090"):
        return "TEMP", data, ""
    if name in ("t190C", "tv190C"):
        return "TEMP_2", data, ""

    # ---- Conductivity -----------------------------------------------------
    if name in ("c0S0x2Fm", "cond0S0x2Fm"):
        return "CNDC", data, ""
    if name in ("c0ms0x2Fcm", "cond0ms0x2Fcm", "c0mS0x2Fcm", "cond0mS0x2Fcm"):
        return "CNDC", data / 10.0, ""       # mS/cm → S/m
    if name in ("c0us0x2Fcm", "cond0us0x2Fcm", "c0uS0x2Fcm", "cond0uS0x2Fcm"):
        return "CNDC", data / 10000.0, ""    # µS/cm → S/m

    # ---- Salinity ---------------------------------------------------------
    if name == "sal00":
        return "PSAL", data, ""
    if name == "sal11":
        return "PSAL_2", data, ""

    # ---- Fluorescence / Chlorophyll ----------------------------------------
    if name == "flC":
        return "CPHL", data, "WET Labs ECO-AFL/FL (Ex: 430 nm, Em: 685 nm)"
    if name == "flCUVA":
        return "CPHL", data, "Aquatracka III fluorescence (Ex: 430 nm, Em: 685 nm)"
    if name == "flECO0x2DAFL":
        return "CPHL", data, "WET Labs ECO-AFL/FL (Ex: 470 nm, Em: 695 nm)"

    # ---- Oxygen -----------------------------------------------------------
    if name == "sbeox0Mg0x2FL":
        return "DOXY", data, ""         # mg/l
    if name == "sbeox0ML0x2FL":
        return "DOX", data, ""          # ml/l
    if name == "sbeox0Mm0x2FL":
        return "DOX1", data, ""         # µmol/L
    if name in ("sbeox0Mm0x2FKg", "sbeopoxMm0x2FKg"):
        return "DOX2", data, ""         # µmol/kg
    if name in ("sbeopoxPS", "sbeox0PS"):
        return "DOXS", data, ""         # % saturation
    if name == "sbeoxTC":
        return "DOXY_TEMP", data, ""

    # ---- PAR --------------------------------------------------------------
    if name in ("par0x2Fsat0x2Flog", "par/sat/log", "par0x2Flog"):
        return "PAR", data, "PAR/Logarithmic/Satlantic"
    if name == "par":
        return "PAR", data, ""
    if name == "cpar":
        return "CPAR", data, ""

    # ---- Turbidity --------------------------------------------------------
    if name in ("obs", "obs30x2B", "turbWETntu0", "upoly0"):
        return "TURB", data, ""

    # ---- Beam attenuation / transmission ----------------------------------
    if name in ("bat", "CStarAt0"):
        return "BAT", data, ""
    if name == "CStarTr0":
        return "BAT_PERCENT", data, "Beam Transmission, WET Labs C-Star [%]"

    # ---- Depth / Altimeter ------------------------------------------------
    if name in ("depSM", "depFM"):
        return "DEPTH", data, ""
    if name == "altM":
        return "ALTIMETER", data, ""

    # ---- Density ----------------------------------------------------------
    if name == "density00":
        return "DENS", data, ""

    # ---- Descent rate -----------------------------------------------------
    if name == "dz0x2FdtM":
        return "DESC", data, ""

    # ---- Position ---------------------------------------------------------
    if name == "latitude":
        return "LATITUDE_CAST", data, ""
    if name == "longitude":
        return "LONGITUDE_CAST", data, ""

    # ---- Profile-only auxiliary variables ---------------------------------
    if mode == "profile":
        if name == "f1":
            return "CNDC_FREQ", data, "Conductivity frequency [Hz]"
        if name == "flag":
            return "SBE_FLAG", data, "SBE processing flag (0=good)"
        if name == "scan":
            return "ETIME", data / 4.0, "Elapsed time [s]"

    # ---- Voltage channels (v0–v7) ----------------------------------------
    if len(name) == 2 and name[0] == "v" and name[1].isdigit():
        return _convert_voltage(name, data, inst_header, proc_header)

    # Unknown variable — caller should skip it
    return "", np.array([]), ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert_voltage(
    name: str,
    data: np.ndarray,
    inst_header: dict[str, Any] | None,
    proc_header: dict[str, Any] | None,
) -> tuple[str, np.ndarray, str]:
    if inst_header is None or proc_header is None:
        return "", np.array([]), ""
    sensor_ids = inst_header.get("sensorIds", [])
    sensor_types = inst_header.get("sensorTypes", [])
    channel_num = name[1]
    volt_id = f"volt {channel_num}"
    sensor_type = "not_assigned"
    for i, sid in enumerate(sensor_ids):
        if sid.lower() == volt_id.lower() and i < len(sensor_types):
            sensor_type = sensor_types[i]
            break
    if sensor_type == "not_assigned":
        return "", np.array([]), ""
    clean_type = "".join(c if c.isalnum() else "_" for c in sensor_type)
    imos_name = f"volt_{clean_type}"
    comment = proc_header.get(f"volt{channel_num}Expr", "")
    return imos_name, data, comment


def _matlab_datenum(dt: datetime) -> float:
    ordinal = dt.toordinal()
    frac = (dt - datetime(dt.year, dt.month, dt.day)).total_seconds() / 86400.0
    return ordinal + 366.0 + frac
