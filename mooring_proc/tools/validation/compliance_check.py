"""Lightweight schema-driven compliance checks for delivery files."""

from __future__ import annotations

from typing import Any

from ..config_manager import load_global_attributes, load_instrument_schema


def _required_variables_from_schema(schema: dict[str, Any]) -> list[str]:
    if isinstance(schema.get("required_variables"), list):
        return [str(name) for name in schema["required_variables"]]
    output_variables = schema.get("output_variables", {}) or {}
    required: list[str] = ["TIME"]
    for name, meta in output_variables.items():
        if isinstance(meta, dict) and bool(meta.get("required")):
            required.append(str(name))
    return sorted(set(required))


def _required_global_attrs(global_schema: dict[str, Any], *, enforce: bool = False) -> list[str]:
    if not enforce:
        return []
    mandatory = global_schema.get("mandatory_attributes", {}) or {}
    return [str(name) for name in mandatory.keys()]


def run_compliance_check(dataset, schema=None):
    """Run schema-driven checks and raise ``ValueError`` when invalid."""
    if "TIME" not in dataset:
        raise ValueError("Compliance check failed: TIME variable is missing.")

    schema = schema or {}
    instrument = str(schema.get("instrument", dataset.attrs.get("instrument", ""))).strip().upper()
    schema_dir = schema.get("schema_dir")
    if instrument:
        inst_schema = load_instrument_schema(instrument, schema_dir=schema_dir)
    else:
        inst_schema = {}
    global_schema = load_global_attributes(schema_dir=schema_dir)

    missing_variables = [name for name in _required_variables_from_schema(inst_schema) if name not in dataset.variables]
    enforce_attrs = bool(schema.get("enforce_global_attrs", False))
    missing_attrs = [
        name for name in _required_global_attrs(global_schema, enforce=enforce_attrs) if name not in dataset.attrs
    ]
    if missing_variables or missing_attrs:
        detail = []
        if missing_variables:
            detail.append(f"missing variables: {', '.join(missing_variables)}")
        if missing_attrs:
            detail.append(f"missing global attributes: {', '.join(missing_attrs)}")
        raise ValueError("Compliance check failed - " + "; ".join(detail))

    return {"ok": True, "instrument": instrument, "checked_variables": len(dataset.variables)}
