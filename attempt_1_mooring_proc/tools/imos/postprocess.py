"""Post-processing hook for IMOS-style datasets before final export.

Call ``apply_postprocess`` after parsing and QC to override global
attributes, rename/drop variables, adjust coordinates, or append any
workflow-specific metadata before the dataset is written to NetCDF.

Usage example
-------------
::

    from tools.parsers.read_sbe37 import read_sbe37
    from tools.imos.postprocess import apply_postprocess

    parsed = read_sbe37("mydata.cnv", config=row)
    ds = apply_postprocess(
        parsed["dataset"],
        global_attr_overrides={
            "institution": "IMOS / AODN",
            "Conventions": "CF-1.8, IMOS-1.4",
            "acknowledgement": "This data was collected under IMOS.",
        },
        qc_overrides={"TEMP": 1},       # force all TEMP flags to 1
        rename_variables={"SSPD": "SOUND_SPEED"},
        drop_variables=["TIMESERIES"],
    )
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


def apply_postprocess(
    dataset: xr.Dataset,
    *,
    global_attr_overrides: dict[str, Any] | None = None,
    qc_overrides: dict[str, int] | None = None,
    rename_variables: dict[str, str] | None = None,
    drop_variables: list[str] | None = None,
    coordinate_overrides: dict[str, float] | None = None,
) -> xr.Dataset:
    """Apply workflow-level overrides to an IMOS-style dataset.

    This is a lightweight hook that does not alter the dataset structure;
    all changes are surgical and reversible.

    Parameters
    ----------
    dataset:
        The xr.Dataset produced by any parser in ``tools.parsers``.
    global_attr_overrides:
        Mapping of global attribute name → value.  Merged over the
        existing dataset attributes (existing values are preserved for
        keys not listed here).
    qc_overrides:
        Mapping of variable name → integer flag value (e.g. ``1`` =
        good, ``4`` = bad).  Applies to the corresponding
        ``<name>_quality_control`` variable if it exists.
    rename_variables:
        Mapping of old name → new name.  Variables that do not exist in
        the dataset are silently skipped.
    drop_variables:
        List of variable names to remove from the dataset.  Variables
        that do not exist are silently skipped.
    coordinate_overrides:
        Mapping of scalar coordinate/variable name → float value.
        Useful for setting LATITUDE, LONGITUDE, NOMINAL_DEPTH when they
        were not available at parse time.

    Returns
    -------
    xr.Dataset
        A new dataset with the requested modifications applied.  The
        original dataset is not mutated.
    """
    ds = dataset.copy(deep=False)

    # --- drop variables -------------------------------------------------------
    if drop_variables:
        to_drop = [v for v in drop_variables if v in ds]
        if to_drop:
            ds = ds.drop_vars(to_drop)

    # --- rename variables -----------------------------------------------------
    if rename_variables:
        valid_renames = {k: v for k, v in rename_variables.items() if k in ds}
        if valid_renames:
            ds = ds.rename(valid_renames)

    # --- coordinate overrides -------------------------------------------------
    if coordinate_overrides:
        updates: dict[str, xr.DataArray] = {}
        for name, value in coordinate_overrides.items():
            if name in ds:
                dtype = ds[name].dtype
                if np.issubdtype(dtype, np.floating):
                    updates[name] = xr.DataArray(dtype.type(value))
                else:
                    updates[name] = xr.DataArray(value)
        if updates:
            ds = ds.assign(updates)

    # --- QC flag overrides ----------------------------------------------------
    if qc_overrides:
        updates_qc: dict[str, xr.DataArray] = {}
        for var_name, flag_value in qc_overrides.items():
            qc_name = f"{var_name}_quality_control"
            if qc_name in ds:
                flag_array = np.full_like(ds[qc_name].values, fill_value=flag_value, dtype=np.int8)
                updates_qc[qc_name] = xr.DataArray(flag_array, dims=ds[qc_name].dims)
        if updates_qc:
            ds = ds.assign(updates_qc)

    # --- global attribute overrides -------------------------------------------
    new_attrs = dict(ds.attrs)
    if global_attr_overrides:
        new_attrs.update(global_attr_overrides)
    ds.attrs = new_attrs

    return ds


def set_deployment_coordinates(
    dataset: xr.Dataset,
    latitude: float,
    longitude: float,
    nominal_depth: float,
) -> xr.Dataset:
    """Convenience wrapper to set deployment coordinates after parsing.

    Parsers leave LATITUDE, LONGITUDE, and NOMINAL_DEPTH as NaN when
    those values are not in the raw file.  Call this helper after
    resolving coordinates from the metadata table.

    Parameters
    ----------
    dataset:
        Dataset returned by any parser in ``tools.parsers``.
    latitude, longitude, nominal_depth:
        Deployment coordinates (WGS-84 decimal degrees and metres).

    Returns
    -------
    xr.Dataset
        Updated dataset (original is not mutated).
    """
    return apply_postprocess(
        dataset,
        coordinate_overrides={
            "LATITUDE": float(latitude),
            "LONGITUDE": float(longitude),
            "NOMINAL_DEPTH": float(nominal_depth),
        },
    )
