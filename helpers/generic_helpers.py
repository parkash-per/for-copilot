"""Reusable generic helpers shared across notebooks and scripts.

Integration:
- Use for metadata/date safety checks and filename-safe tokens.
- Intended to replace repeated notebook utility functions.

Copy-paste:
    from tools.helpers import clean_label, is_missing, resolve_good_data_window, safe_tag

    token = safe_tag("BASS3A 2026-03")
    missing = is_missing(row.get("proc_1_path"))
    x_start, x_end = resolve_good_data_window(row)
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


def is_missing(v: object) -> bool:
    """Return True when value is None/NaN/empty-like text."""
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    s = str(v).strip().lower()
    return s in ("", "nan", "nat", "none")


def safe_tag(s: object) -> str:
    """Create a filename-safe text token."""
    if s is None:
        return "unknown"
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "", s)
    return s if s else "unknown"


def clean_label(var_code: str, desc: str) -> str:
    """Convert raw variable metadata labels into cleaner display labels."""
    v = var_code.lower().strip()
    d = desc.strip()

    unit_match = re.search(r"\[[^\]]+\]", d)
    unit = unit_match.group(0) if unit_match else ""
    main = d.split(",", 1)[0].strip()

    if v.startswith("time") or "julian" in d.lower():
        return "Instrument Time"
    if v.startswith("tv") or "temperature" in d.lower():
        return "Temperature"
    if v.startswith("cond") or "conductivity" in d.lower():
        return f"Conductivity {unit}".strip()
    if v.startswith("sal") or "salinity" in d.lower():
        return f"Salinity {unit}".strip()
    if v.startswith("prd") or "pressure" in d.lower():
        return f"Pressure {unit}".strip()
    if v.startswith("flag") or main.lower() == "flag":
        return "flag"

    return f"{main} {unit}".strip()


def resolve_good_data_window(
    row: Mapping[str, Any],
    time_coverage_start: Optional[Any] = None,
    time_coverage_end: Optional[Any] = None,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Resolve preferred start/end window from manual values then metadata row."""
    if not is_missing(time_coverage_start):
        x_start = pd.to_datetime(time_coverage_start, dayfirst=True, format="mixed", errors="coerce")
    else:
        x_start = pd.to_datetime(row.get("time_coverage_start", None), dayfirst=True, format="mixed", errors="coerce")
        if pd.isna(x_start):
            x_start = pd.to_datetime(row.get("deploy_date", None), dayfirst=True, format="mixed", errors="coerce")

    if not is_missing(time_coverage_end):
        x_end = pd.to_datetime(time_coverage_end, dayfirst=True, format="mixed", errors="coerce")
    else:
        x_end = pd.to_datetime(row.get("time_coverage_end", None), dayfirst=True, format="mixed", errors="coerce")
        if pd.isna(x_end):
            x_end = pd.to_datetime(row.get("recovery_date", None), dayfirst=True, format="mixed", errors="coerce")

    if pd.isna(x_start):
        x_start = None
    if pd.isna(x_end):
        x_end = None

    return x_start, x_end


__all__ = ["is_missing", "safe_tag", "clean_label", "resolve_good_data_window"]