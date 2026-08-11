"""Helpers for applying atmospheric pressure offsets.

Integration:
- Use in notebooks after building dataset variable ``ds``.
- Designed for pressure variable names like ``PRES``.

Copy-paste:
    from tools.helpers import apply_atmospheric_pressure_offset

    ds = apply_atmospheric_pressure_offset(
        ds,
        offset_dbar=10.1325,
        pres_var="PRES",
        mode="add",
        in_place=True,
    )
"""

from __future__ import annotations

import xarray as xr


def apply_atmospheric_pressure_offset(
    ds: xr.Dataset,
    offset_dbar: float | None,
    pres_var: str = "PRES",
    mode: str = "add",  # "add" converts relative->absolute if relative = absolute - atm
    in_place: bool = True,
):
    """
    Apply atmospheric pressure offset to pressure variable.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing pressure variable (default 'PRES').
    offset_dbar : float or None
        Atmospheric pressure offset in dbar. If None, no change is made.
    pres_var : str
        Pressure variable name in ds.
    mode : str
        "add"      : PRES_new = PRES_old + offset_dbar
        "subtract" : PRES_new = PRES_old - offset_dbar
    in_place : bool
        If True, modify ds directly. If False, returns modified copy.

    Returns
    -------
    xr.Dataset
    """
    if pres_var not in ds:
        raise KeyError(f"{pres_var} not found in dataset")

    if offset_dbar is None:
        print("No atmospheric pressure offset provided; PRES unchanged.")
        return ds

    target = ds if in_place else ds.copy(deep=True)
    offset = float(offset_dbar)

    print(offset)

    if mode == "add":
        target[pres_var].values = target[pres_var].values + offset
        op_txt = f"+{offset}"
    elif mode == "subtract":
        target[pres_var].values = target[pres_var].values - offset
        op_txt = f"-{offset}"
    else:
        raise ValueError("mode must be 'add' or 'subtract'")

    # # Record processing history
    prev = str(target.attrs.get("processing", "")).strip()
    step = f"{pres_var} adjusted by {op_txt} dbar atmospheric offset (mode={mode})"
    # target.attrs["processing"] = f"{prev}; {step}" if prev else step
    # target.attrs["atmospheric_pressure_offset_dbar"] = offset
    # target.attrs["atmospheric_pressure_offset_mode"] = mode

    print(f"Applied atmospheric pressure offset: {step}")
    return target


__all__ = ["apply_atmospheric_pressure_offset"]