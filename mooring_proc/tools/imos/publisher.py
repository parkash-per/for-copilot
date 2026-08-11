"""Publishing helpers for IMOS deliveries."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from .writer import build_output_filename, write_imos_file


def publish_delivery(delivery_path, destination, metadata=None):
    """Publish an existing NetCDF file with IMOS delivery naming."""
    metadata = dict(metadata or {})
    input_path = Path(str(delivery_path)).expanduser()
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()
    else:
        input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Delivery input file not found: {input_path}")

    destination_path = Path(str(destination)).expanduser()
    if destination_path.suffix.lower() != ".nc":
        destination_path = destination_path / build_output_filename({**metadata, "output_name_mode": "imos"})
    if not destination_path.is_absolute():
        destination_path = (Path.cwd() / destination_path).resolve()
    else:
        destination_path = destination_path.resolve()

    with xr.open_dataset(input_path) as opened_dataset:
        dataset = opened_dataset.load()

    combined_metadata = dict(dataset.attrs)
    combined_metadata.update(metadata)
    combined_metadata["output_name_mode"] = "imos"
    combined_metadata.setdefault("source_file", str(input_path))
    return write_imos_file(dataset, destination_path, metadata=combined_metadata)
