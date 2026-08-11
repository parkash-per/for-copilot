#!/usr/bin/env python3

# IMOS NetCDF Converter - Config-Driven Schema Edition
# Refactored for maintainability: all variable schemas defined in YAML configs
# Instrument classes are now minimal and focused on data population logic

import os
import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from netCDF4 import Dataset, num2date
from datetime import datetime, timezone

# Import config manager
from .config_manager import get_schema_manager

# AUTO-SETUP: Add this directory to Python path when imported
_current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


def _build_output_path(output_dir: str, file_name: str) -> str:
    return os.path.join(output_dir, file_name)


def _parse_start_of_good_data(value, robust: bool = False):
    if not isinstance(value, str):
        return value
    if not robust:
        return pd.to_datetime(value)
    try:
        return pd.to_datetime(value, format="%Y%m%d")
    except Exception:
        try:
            return pd.to_datetime(value, format="mixed", dayfirst=True)
        except Exception:
            return pd.to_datetime(value)


def _normalize_iso_z(value) -> str | None:
    """Return strict ISO8601 UTC string with trailing Z."""
    if value is None:
        return None
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return str(value)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _site_identity(location: str) -> str:
    """Single source of truth for site identity."""
    return str(location).strip()


def _is_missing_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    txt = str(value).strip()
    return txt == "" or txt.lower() in ("none", "nan")


def _coerce_metadata_row(metadata_row) -> dict | None:
    if metadata_row is None:
        return None
    if isinstance(metadata_row, dict):
        return metadata_row
    if hasattr(metadata_row, "to_dict"):
        return metadata_row.to_dict()
    return None


def _build_internal_stage_filename(location: str, start_of_good_data, instrument: str, inst_id, inst_depth) -> str:
    """
    Internal filename format for proc_1/proc_2 style outputs:
    [location]_[yyyymm]_[inst]_[inst_serial]_[instdepth]m.nc
    Example: BASS3A_202603_SBE26_1340_48m.nc
    """
    dt = _parse_start_of_good_data(start_of_good_data, robust=True)
    if pd.isna(dt):
        dt = pd.Timestamp.utcnow()

    location_token = str(location).strip().replace(" ", "")
    inst_token = str(instrument).strip().upper().replace(" ", "").replace("_", "")
    serial_token = str(inst_id).strip()

    try:
        depth_token = f"{int(round(float(inst_depth)))}m"
    except Exception:
        depth_token = f"{str(inst_depth).strip()}m"

    return f"{location_token}_{dt.strftime('%Y%m')}_{inst_token}_{serial_token}_{depth_token}.nc"


def _normalize_processing_version(version: str) -> str:
    version_str = str(version).strip()
    try:
        vnum = float(version_str)
        if vnum.is_integer() and 0 <= int(vnum) < 100:
            return f"{int(vnum):02d}"
    except Exception:
        pass
    if version_str.isdigit() and len(version_str) == 1:
        return f"0{version_str}"
    return version_str


def _build_imos_delivery_filename(inst_channels: str, start_of_good_data, location: str, version: str, instrument: str, depth) -> str:
    version_str = _normalize_processing_version(version)
    start_dt = _parse_start_of_good_data(start_of_good_data, robust=True)
    if pd.isna(start_dt):
        raise ValueError("start_of_good_data is missing/invalid; provide a valid datetime.")
    site_token = _site_identity(location)
    instrument_token = str(instrument).strip().upper().replace("_", "")
    return (
        f"IMOS_SRSALT_{inst_channels}_{start_dt.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{site_token}_FV{version_str}_{instrument_token}d{int(depth)}m.nc"
    )


def _build_output_filename(
    *,
    inst_channels: str,
    start_of_good_data,
    location: str,
    version: str,
    instrument: str,
    depth,
    inst_id,
    output_name_mode: str,
) -> str:
    if str(output_name_mode).lower() == "internal":
        return _build_internal_stage_filename(
            location=location,
            start_of_good_data=start_of_good_data,
            instrument=instrument,
            inst_id=inst_id,
            inst_depth=depth,
        )

    return _build_imos_delivery_filename(
        inst_channels=inst_channels,
        start_of_good_data=start_of_good_data,
        location=location,
        version=version,
        instrument=instrument,
        depth=depth,
    )


def _resolve_stage_output_dir(output_stage: str | None = None, output_dir: str | None = None, metadata_row: dict | None = None, default_output_dir: str | None = None) -> str:
    if output_dir and str(output_dir).strip():
        out = Path(str(output_dir)).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        return str(out)

    row = _coerce_metadata_row(metadata_row)
    stage = str(output_stage).strip().lower() if output_stage is not None else ""

    if row is not None:
        stage_key_map = {
            "proc_1": "proc_1_path",
            "proc1": "proc_1_path",
            "proc_2": "proc_2_path",
            "proc2": "proc_2_path",
            "imos_delivery": "imos_deliverables_path",
            "imos": "imos_deliverables_path",
            "delivery": "imos_deliverables_path",
        }
        path_key = stage_key_map.get(stage)
        if path_key:
            candidate = row.get(path_key, None)
            if _is_missing_value(candidate) and path_key == "imos_deliverables_path":
                candidate = row.get("imos_path", None)
            if _is_missing_value(candidate):
                raise ValueError(
                    f"Stage output path is missing in metadata: '{path_key}' for stage '{stage}'. "
                    "Set the stage path in metadata before running conversion."
                )
            out = Path(str(candidate)).expanduser()
            if not out.is_absolute():
                out = (Path.cwd() / out).resolve()
            else:
                out = out.resolve()
            out.mkdir(parents=True, exist_ok=True)
            return str(out)

    if default_output_dir and str(default_output_dir).strip():
        out = Path(str(default_output_dir)).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        return str(out)

    out = Path.cwd().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def _apply_common_global_attrs(
    nc: Dataset,
    *,
    latitude: float,
    longitude: float,
    depth: float,
    time_coverage_start=None,
    time_coverage_end=None,
    processing_version: str,
    site_code: str,
    instrument: str,
    instrument_serial_number,
    nominal_inst_depth=None,
    title: str,
    source: str,
    keywords: str,
    references: str | None = None,
    site: str | None = None,
    author_email: str | None = None,
    author: str | None = None,
    principal_investigator: str | None = None,
    set_attr=None,
) -> None:
    """Apply mandatory + common IMOS global attributes from schema."""
    if set_attr is None:
        def set_attr(nc_obj, attr_name, attr_value):
            setattr(nc_obj, attr_name, attr_value)

    # Load global attributes from YAML
    schema_manager = get_schema_manager()
    global_attrs = schema_manager.load_global_attributes()
    
    mandatory_attrs = global_attrs.get("mandatory_attributes", {})
    defaults = global_attrs.get("defaults", {})
    
    # Set all mandatory attributes
    for attr_name, attr_value in mandatory_attrs.items():
        setattr(nc, attr_name, attr_value)
    
    # Set date_created (always current)
    nc.date_created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Set configurable attributes with fallbacks to defaults
    set_attr(nc, "author", author or defaults.get("author"))
    set_attr(nc, "author_email", author_email or defaults.get("author_email"))
    set_attr(nc, "principal_investigator", principal_investigator or defaults.get("principal_investigator"))
    set_attr(nc, "references", references or defaults.get("references"))
    set_attr(nc, "site", site or defaults.get("site"))
    
    # Set geospatial attributes
    nc.geospatial_lat_max = np.float32(latitude)
    nc.geospatial_lat_min = np.float32(latitude)
    nc.geospatial_lon_max = np.float32(longitude)
    nc.geospatial_lon_min = np.float32(longitude)
    nc.geospatial_vertical_max = np.float32(depth)
    nc.geospatial_vertical_min = np.float32(depth)
    nc.geospatial_vertical_positive = "down"

    # Set instrument-related attributes
    set_attr(nc, "instrument", instrument)
    set_attr(nc, "instrument_serial_number", str(instrument_serial_number))
    if not _is_missing_value(nominal_inst_depth):
        set_attr(nc, "instrument_nominal_depth", nominal_inst_depth)
    
    # Set keywords
    nc.keywords = keywords
    
    # Set site code and source
    set_attr(nc, "site_code", str(site_code))
    set_attr(nc, "source", source)
    
    # Set processing version
    set_attr(nc, "processing_version", str(processing_version))
    
    # Set title
    set_attr(nc, "title", title)

    # Set time coverage
    if time_coverage_start is not None:
        nc.time_coverage_start = _normalize_iso_z(time_coverage_start)
    else:
        nc.time_coverage_start = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if time_coverage_end is not None:
        nc.time_coverage_end = _normalize_iso_z(time_coverage_end)
    else:
        nc.time_coverage_end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================================================
# BASE CLASS: Common logic for all instrument converters
# ============================================================================

class IMOSNetCDFConverterBase:
    """Base class with common methods for all IMOS NetCDF converters."""
    
    def __init__(self, input_folder: str, input_file: str, output_dir: str, config_dir: str = "configs"):
        self.input_folder = input_folder
        self.input_file = input_file
        self.output_dir = output_dir
        self.schema_manager = get_schema_manager(config_dir)
        self.instrument_name = self._get_instrument_name()
        self.schema = self.schema_manager.load_schema(self.instrument_name)

    def _get_instrument_name(self) -> str:
        """Extract instrument name from class name. e.g., IMOSNetCDFConverter_AQD -> AQD"""
        return self.__class__.__name__.split('_')[-1]

    def determine_channel_from_filename(self, input_filename: str) -> str | None:
        """Override in subclasses if needed. Default: None."""
        return None

    def generate_file_name(self, inst_channels: str, start_of_good_data, location: str, version: str, instrument: str, depth) -> str:
        """Default uses IMOS delivery filename. Override in subclasses if needed."""
        return _build_imos_delivery_filename(
            inst_channels=inst_channels,
            start_of_good_data=start_of_good_data,
            location=location,
            version=version,
            instrument=instrument,
            depth=depth,
        )

    def _default_channel_code(self) -> str:
        """Get from schema instead of hardcoding."""
        return self.schema_manager.get_default_channel_code(self.instrument_name)

    def _extract_arrays_from_input_nc(self, input_nc_file: str, depth_fallback: float) -> dict:
        """Generic extraction using schema definitions."""
        with Dataset(input_nc_file, "r") as src:
            if "TIME" not in src.variables:
                raise ValueError("Input NetCDF must contain TIME variable.")

            t_in = src.variables["TIME"]
            time_vals = np.asarray(t_in[:], dtype=np.float64)
            time_units = getattr(t_in, "units", "days since 1950-01-01T00:00:00 UTC")
            time_calendar = getattr(t_in, "calendar", "gregorian")
            n_time = len(time_vals)

            def _read_or_default(var_name: str, default_value, dtype_str: str):
                if var_name in src.variables:
                    return np.asarray(src.variables[var_name][:], dtype=np.dtype(dtype_str))
                if default_value is None:
                    return None
                return np.full(n_time, default_value, dtype=np.dtype(dtype_str))

            result = {"time_data": time_vals}
            
            # Read all input variables from schema
            for var_name, var_config in self.schema.get("input_variables", {}).items():
                default_val = var_config.get("default")
                dtype_str = var_config.get("dtype", "f4")
                result[var_name] = _read_or_default(var_name, default_val, dtype_str)

        # Parse time coverage
        t0 = datetime.now(timezone.utc)
        t1 = datetime.now(timezone.utc)
        try:
            if isinstance(time_units, str) and "since" in time_units.lower():
                dt = num2date(time_vals, units=time_units, calendar=time_calendar)
                t0 = pd.to_datetime(str(dt[0]), utc=True)
                t1 = pd.to_datetime(str(dt[-1]), utc=True)
            else:
                t0 = pd.to_datetime(time_vals[0], unit="D", origin="1950-01-01", utc=True)
                t1 = pd.to_datetime(time_vals[-1], unit="D", origin="1950-01-01", utc=True)
        except Exception:
            pass

        result["time_coverage_start_default"] = t0
        result["time_coverage_end_default"] = t1
        result["depth"] = depth_fallback
        return result

    def _extract_time_coverage(self, input_nc_file: str) -> tuple[str | None, str | None]:
        """
        Extract time_coverage_start and time_coverage_end from the actual TIME variable
        in the NetCDF file. Falls back to file attributes if TIME variable doesn't exist.
        
        Args:
            input_nc_file: Path to input NetCDF file
        
        Returns:
            tuple: (time_coverage_start, time_coverage_end) as ISO8601 strings, or (None, None) if extraction fails
        """
        with Dataset(input_nc_file, "r") as src:
            # Try to extract from TIME variable
            if "TIME" in src.variables:
                time_vals = np.asarray(src.variables["TIME"][:], dtype=np.float64)
                if len(time_vals) > 0:
                    time_units = getattr(src.variables["TIME"], "units", "days since 1950-01-01T00:00:00 UTC")
                    time_calendar = getattr(src.variables["TIME"], "calendar", "gregorian")
                    
                    try:
                        if isinstance(time_units, str) and "since" in time_units.lower():
                            dt = num2date(time_vals, units=time_units, calendar=time_calendar)
                            t_start = pd.to_datetime(str(dt[0]), utc=True)
                            t_end = pd.to_datetime(str(dt[-1]), utc=True)
                        else:
                            t_start = pd.to_datetime(time_vals[0], unit="D", origin="1950-01-01", utc=True)
                            t_end = pd.to_datetime(time_vals[-1], unit="D", origin="1950-01-01", utc=True)
                        
                        return (
                            t_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            t_end.strftime("%Y-%m-%dT%H:%M:%SZ")
                        )
                    except Exception:
                        pass
            
            # Fallback to file attributes
            time_start_attr = getattr(src, "time_coverage_start", None)
            time_end_attr = getattr(src, "time_coverage_end", None)
            
            if time_start_attr and time_end_attr:
                return (str(time_start_attr), str(time_end_attr))
        
        # Last resort: return None tuple
        return (None, None)

    def _create_output_variables(self, nc_dataset: Dataset, n_time: int) -> dict:
        """Create all output variables from schema."""
        output_vars = self.schema.get("output_variables", {})
        created_vars = {}
        
        for var_name, var_config in output_vars.items():
            var_type = var_config.get("type", "f4")
            dimensions = var_config.get("dimensions", [])
            fill_value = var_config.get("fill_value")
            
            # Convert dimensions: keep "TIME" as string (it's a dimension name)
            dims = tuple(d for d in dimensions)
            
            # If dimensions are empty (scalar), pass empty tuple
            if not dims:
                if fill_value is not None:
                    var = nc_dataset.createVariable(var_name, var_type, fill_value=fill_value)
                else:
                    var = nc_dataset.createVariable(var_name, var_type)
            else:
                # For dimensioned variables, dims must be tuple of dimension names (strings)
                if fill_value is not None:
                    var = nc_dataset.createVariable(var_name, var_type, dims, fill_value=fill_value)
                else:
                    var = nc_dataset.createVariable(var_name, var_type, dims)
            
            # Set attributes from schema
            for attr_name, attr_value in var_config.get("attributes", {}).items():
                setattr(var, attr_name, attr_value)
            
            created_vars[var_name] = var
        
        return created_vars

    def _resolve_channel_code(self, inst_channels, metadata: dict) -> str:
        """Resolve channel code with fallbacks."""
        channel_code = inst_channels
        if _is_missing_value(channel_code):
            channel_code = metadata.get("mooring_channels")
        if _is_missing_value(channel_code):
            channel_code = metadata.get("inst_channels")
        if _is_missing_value(channel_code):
            channel_code = self._default_channel_code()
        return str(channel_code).replace(" ", "").replace(",", "").upper()

    def _setup_attr_setter(self, metadata_mode: str):
        """Create the _set_attr callable based on metadata_mode."""
        if metadata_mode not in ("fill_missing", "overwrite"):
            metadata_mode = "fill_missing"

        def _set_attr(nc_obj, attr_name: str, attr_value):
            if _is_missing_value(attr_value):
                return
            if metadata_mode == "overwrite":
                setattr(nc_obj, attr_name, attr_value)
                return
            existing = getattr(nc_obj, attr_name, None)
            if _is_missing_value(existing):
                setattr(nc_obj, attr_name, attr_value)

        return _set_attr

    def publish_netcdf(
        self,
        input_nc_file: str,
        inst_channels: str,
        start_of_good_data,
        location: str,
        version: str,
        instrument: str,
        depth: float,
        inst_id,
        output_name_mode: str = "imos",
        output_dir: str | None = None,
    ) -> str:
        """
        MODE 2: PUBLISH - Rename existing IMOS NetCDF to delivery format.
        
        This is used in imos_delivery to take a proc_1 or proc_2 NetCDF
        (which is already IMOS-compliant) and:
        1. Copy it to the output directory
        2. Rename it to IMOS delivery naming convention
        3. Update the processing_version attribute
        
        Args:
            input_nc_file: Path to existing IMOS-compliant NetCDF (from proc_1 or proc_2)
            inst_channels: Channel code (e.g., "PTSUV")
            start_of_good_data: Used in filename generation
            location: Site code/location name
            version: Processing version (becomes FV00, FV01, etc.)
            instrument: Instrument name
            depth: Instrument depth
            inst_id: Instrument serial number
            output_name_mode: "imos" for delivery format
            output_dir: Output directory path
        
        Returns:
            Path to renamed NetCDF file in delivery format
        """
        if not input_nc_file or not os.path.exists(input_nc_file):
            raise FileNotFoundError(f"{self.__class__.__name__} delivery publish requires an existing input NetCDF file.")

        processing_version = _normalize_processing_version(version)

        # Generate IMOS delivery filename
        file_name = _build_output_filename(
            inst_channels=inst_channels,
            start_of_good_data=start_of_good_data,
            location=location,
            version=version,
            instrument=instrument,
            depth=depth,
            inst_id=inst_id,
            output_name_mode=output_name_mode,  # "imos"
        )
        output_path = _build_output_path(output_dir or self.output_dir, file_name)

        # Check if input and output are the same (already in correct location)
        if os.path.abspath(input_nc_file) == os.path.abspath(output_path):
            print(f"{self.__class__.__name__} delivery file already available at: {output_path}")
            return output_path

        # Remove existing output file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)

        # Copy the existing IMOS NetCDF to new location with delivery name
        shutil.copy2(input_nc_file, output_path)

        # Update processing_version in the copied file
        with Dataset(output_path, "r+") as nc:
            nc.processing_version = processing_version

        print(f"Copied NetCDF deliverable to: {file_name}")
        return output_path

    def _resolve_process_mode(self, output_stage: str | None, output_name_mode: str, process_mode: str | None) -> str:
        """Auto-detect process mode if not provided."""
        if process_mode is None:
            stage_token = str(output_stage).strip().lower() if output_stage is not None else ""
            name_mode_token = str(output_name_mode).strip().lower()
            if stage_token in ("imos_delivery", "imos", "delivery") and name_mode_token == "imos":
                process_mode = "publish"
            else:
                process_mode = "convert"

        process_mode = str(process_mode).strip().lower()
        if process_mode not in ("convert", "publish"):
            raise ValueError("process_mode must be 'convert' or 'publish'.")
        
        return process_mode

    def process(self, *args, **kwargs) -> str:
        """Common process() entry point."""
        input_nc_path = kwargs.get("input_nc_path", None)
        
        if input_nc_path is None:
            output_path = self.create_netcdf(*args, **kwargs)
            print(f"IMOS NetCDF file created: {os.path.basename(output_path)}")
            return output_path

        # Extract common parameters
        longitude = kwargs.get("longitude")
        latitude = kwargs.get("latitude")
        depth = kwargs.get("depth")
        inst_channels = kwargs.get("inst_channels")
        start_of_good_data = kwargs.get("start_of_good_data")
        site_code = kwargs.get("site_code", "")
        version = kwargs.get("version", "1")
        instrument = kwargs.get("instrument", self.instrument_name)
        inst_id = kwargs.get("inst_id", "")
        location = kwargs.get("location", site_code)
        output_name_mode = kwargs.get("output_name_mode", "imos")
        output_stage = kwargs.get("output_stage")
        output_dir = kwargs.get("output_dir")
        metadata_row = kwargs.get("metadata_row")
        process_mode = kwargs.get("process_mode")

        reserved = {
            "input_nc_path", "longitude", "latitude", "depth", "inst_channels",
            "start_of_good_data", "site_code", "version", "instrument", "inst_id",
            "location", "output_name_mode", "output_stage", "output_dir", "metadata_row",
            "process_mode",
        }
        metadata = {k: v for k, v in kwargs.items() if k not in reserved}

        channel_code = self._resolve_channel_code(inst_channels, metadata)

        resolved_output_dir = _resolve_stage_output_dir(
            output_stage=output_stage,
            output_dir=output_dir,
            metadata_row=metadata_row,
            default_output_dir=self.output_dir,
        )

        process_mode = self._resolve_process_mode(output_stage, output_name_mode, process_mode)

        print(f"Processing {os.path.basename(input_nc_path)}...")
        print(f"  Output directory: {resolved_output_dir}")

        if process_mode == "publish":
            print(f"  Publishing {channel_code} channel file with delivery naming")
            output_file = self.publish_netcdf(
                input_nc_file=input_nc_path,
                inst_channels=channel_code,
                start_of_good_data=start_of_good_data,
                location=location,
                version=version,
                instrument=instrument,
                depth=depth,
                inst_id=inst_id,
                output_name_mode=output_name_mode,
                output_dir=resolved_output_dir,
            )
        else:
            print(f"  Applying IMOS attributes to {channel_code} channel file")
            output_file = self.create_netcdf(
                time_data=None,
                longitude=longitude,
                latitude=latitude,
                depth=depth,
                pres_data=None,
                temp_data=None,
                pres_qc_data=None,
                temp_qc_data=None,
                inst_channels=channel_code,
                start_of_good_data=start_of_good_data,
                site_code=site_code,
                version=version,
                instrument=instrument,
                inst_id=inst_id,
                location=location,
                input_nc_file=input_nc_path,
                output_name_mode=output_name_mode,
                output_dir=resolved_output_dir,
                **metadata,
            )

        print(f"Successfully processed {os.path.basename(input_nc_path)}")
        return output_file

    def create_netcdf(self, *args, **kwargs) -> str:
        """Override in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__}.create_netcdf() not implemented")


# ============================================================================
# INSTRUMENT-SPECIFIC CONVERTERS (Simplified with schema-driven approach)
# ============================================================================

class IMOSNetCDFConverter_AQD(IMOSNetCDFConverterBase):
    """
    Aquadopp current meter converter with two modes:
    
    MODE 1 - CREATE (proc_1/proc_2):
        Input: xarray Dataset or arrays from build_aqd_dataset_from_metadata()
        Output: IMOS-compliant NetCDF with INTERNAL naming
        
    MODE 2 - PUBLISH (IMOS delivery):
        Input: Existing IMOS-compliant NetCDF from proc_1/proc_2
        Output: Same NetCDF with IMOS DELIVERY naming + updated version
    """
    
    def create_netcdf(
        self,
        time_data,
        longitude: float,
        latitude: float,
        depth: float,
        pres_data=None,
        temp_data=None,
        pres_qc_data=None,
        temp_qc_data=None,
        inst_channels: str = "",
        start_of_good_data=None,
        site_code: str = "",
        version: str = "1",
        instrument: str = "",
        inst_id="",
        location: str = "",
        input_nc_file: str | None = None,
        output_name_mode: str = "internal",
        output_dir: str | None = None,
        output_stage: str | None = None, 
        metadata_row: dict | None = None,  
        **metadata,
    ) -> str:
        """
        MODE 1: CREATE - Build IMOS-compliant NetCDF from scratch.
        
        This is used in proc_1/proc_2 to create the NetCDF with proper
        IMOS schema and attributes from the beginning.
        
        Args:
            time_data: TIME array from processed data
            longitude, latitude, depth: Coordinates
            temp_data, temp_qc_data: Temperature data and QC flags
            ucur_data, ucur_qc_data: U velocity data and QC flags (passed in metadata)
            vcur_data, vcur_qc_data: V velocity data and QC flags (passed in metadata)
            depth_data, depth_qc_data: Depth data and QC flags (passed in metadata)
            inst_channels: Channel code (e.g., "UVTD")
            start_of_good_data: Start of deployment/good data
            output_name_mode: "internal" for proc_1/proc_2, "imos" for delivery
        
        Returns:
            Path to created NetCDF file
        """
        # Resolve output directory using stage mapping
        resolved_output_dir = _resolve_stage_output_dir(
            output_stage=output_stage,
            output_dir=output_dir,
            metadata_row=metadata_row,
            default_output_dir=self.output_dir,
        )
        
        file_name = _build_output_filename(
            inst_channels=inst_channels,
            start_of_good_data=start_of_good_data,
            location=location,
            version=version,
            instrument=instrument,
            depth=depth,
            inst_id=inst_id,
            output_name_mode=output_name_mode,  # "internal" or "imos"
        )

        output_path = _build_output_path(resolved_output_dir, file_name)
        
        if time_data is None:
            raise ValueError("time_data is required for create_netcdf mode.")

        n_time = len(time_data)

        if os.path.exists(output_path):
            os.remove(output_path)

        processing_version = _normalize_processing_version(version)
        _set_attr = self._setup_attr_setter(metadata.get("metadata_mode", "fill_missing"))

        # Create new NetCDF from scratch with IMOS schema
        with Dataset(output_path, "w", format="NETCDF4") as nc:
            # Create TIME dimension
            time_dim = nc.createDimension("TIME", n_time)

            # Create all variables from schema
            created_vars = self._create_output_variables(nc, n_time)

            # Populate coordinate variables
            created_vars["TIME"][:] = time_data
            created_vars["LATITUDE"].assignValue(float(latitude))
            created_vars["LONGITUDE"].assignValue(float(longitude))
            created_vars["NOMINAL_DEPTH"].assignValue(np.float32(depth))

            # Populate data variables (from metadata passed in)
            if temp_data is not None and "TEMP" in created_vars:
                created_vars["TEMP"][:] = temp_data
            if temp_qc_data is not None and "TEMP_quality_control" in created_vars:
                created_vars["TEMP_quality_control"][:] = temp_qc_data

            # Handle AQD-specific velocity data (passed in metadata)
            ucur_data = metadata.get("ucur_data")
            ucur_qc_data = metadata.get("ucur_qc_data")
            vcur_data = metadata.get("vcur_data")
            vcur_qc_data = metadata.get("vcur_qc_data")
            depth_data = metadata.get("depth_data")
            depth_qc_data = metadata.get("depth_qc_data")

            if ucur_data is not None and "UCUR" in created_vars:
                created_vars["UCUR"][:] = ucur_data
            if ucur_qc_data is not None and "UCUR_quality_control" in created_vars:
                created_vars["UCUR_quality_control"][:] = ucur_qc_data

            if vcur_data is not None and "VCUR" in created_vars:
                created_vars["VCUR"][:] = vcur_data
            if vcur_qc_data is not None and "VCUR_quality_control" in created_vars:
                created_vars["VCUR_quality_control"][:] = vcur_qc_data

            if depth_data is not None and "DEPTH" in created_vars:
                created_vars["DEPTH"][:] = depth_data
            if depth_qc_data is not None and "DEPTH_quality_control" in created_vars:
                created_vars["DEPTH_quality_control"][:] = depth_qc_data

            # Apply IMOS global attributes
            start_date = _parse_start_of_good_data(start_of_good_data, robust=True)
            _apply_common_global_attrs(
                nc,
                latitude=latitude,
                longitude=longitude,
                depth=depth,
                time_coverage_start=metadata.get("time_coverage_start", metadata.get("time_deployment_start")),
                time_coverage_end=metadata.get("time_coverage_end", metadata.get("time_deployment_end")),
                processing_version=processing_version,
                site_code=_site_identity(location),
                instrument=metadata.get("inst_type", str(instrument).upper()),
                instrument_serial_number=metadata.get("inst_id", inst_id),
                nominal_inst_depth=metadata.get("nominal_inst_depth"),
                title=self.schema.get("default_title", "").format(location=location),
                source=f"CSIRO coastal mooring observations {start_date.strftime('%Y-%m-%d')}",
                keywords=self.schema.get("default_keywords", ""),
                references=metadata.get("references"),
                site=metadata.get("site"),
                author_email=metadata.get("author_email"),
                author=metadata.get("author"),
                principal_investigator=metadata.get("principal_investigator"),
                set_attr=_set_attr,
            )

            # Add deployment-specific attributes
            _set_attr(nc, "deployment_id", metadata.get("deployment_id"))
            _set_attr(nc, "mooring_channels", metadata.get("mooring_channels"))

        print(f"IMOS NetCDF created with internal naming: {file_name}")
        return output_path

class IMOSNetCDFConverter_RBRQ(IMOSNetCDFConverterBase):
    """RBR Quartz Pressure converter."""
    
    def create_netcdf(
        self,
        time_data,
        longitude: float,
        latitude: float,
        depth: float,
        pres_data,
        temp_data,
        pres_qc_data,
        temp_qc_data,
        inst_channels: str,
        start_of_good_data,
        site_code: str,
        version: str,
        instrument: str,
        inst_id,
        location: str,
        input_nc_file: str | None = None,
        output_name_mode: str = "imos",
        output_dir: str | None = None,
        **metadata,
    ) -> str:
        file_name = _build_output_filename(
            inst_channels=inst_channels,
            start_of_good_data=start_of_good_data,
            location=location,
            version=version,
            instrument=instrument,
            depth=depth,
            inst_id=inst_id,
            output_name_mode=output_name_mode,
        )

        output_path = _build_output_path(output_dir or self.output_dir, file_name)

        if input_nc_file:
            if not os.path.exists(input_nc_file):
                raise FileNotFoundError(f"Input NetCDF file not found: {input_nc_file}")
            extracted = self._extract_arrays_from_input_nc(input_nc_file, depth)
            time_data = extracted["time_data"]
            t0 = extracted["time_coverage_start_default"]
            t1 = extracted["time_coverage_end_default"]
        else:
            if time_data is None:
                raise ValueError("time_data is required when input_nc_file is not provided.")
            t0 = pd.to_datetime(time_data[0], unit="D", origin="1950-01-01")
            t1 = pd.to_datetime(time_data[-1], unit="D", origin="1950-01-01")
            extracted = {}

        n_time = len(time_data)
        start_date = _parse_start_of_good_data(start_of_good_data, robust=True)

        processing_version = _normalize_processing_version(version)
        _set_attr = self._setup_attr_setter(metadata.get("metadata_mode", "fill_missing"))

        if os.path.exists(output_path):
            os.remove(output_path)

        with Dataset(output_path, "w", format="NETCDF4") as nc:
            nc.createDimension("TIME", n_time)

            created_vars = self._create_output_variables(nc, n_time)

            created_vars["TIME"][:] = time_data
            created_vars["LATITUDE"].assignValue(float(latitude))
            created_vars["LONGITUDE"].assignValue(float(longitude))
            created_vars["NOMINAL_DEPTH"].assignValue(np.float32(depth))

            # Populate PRES/TEMP from extracted or passed data
            for var_name in ["PRES", "PRES_quality_control", "TEMP", "TEMP_quality_control"]:
                if var_name in extracted and extracted[var_name] is not None and var_name in created_vars:
                    created_vars[var_name][:] = extracted[var_name]
                elif var_name == "PRES" and pres_data is not None:
                    created_vars[var_name][:] = pres_data
                elif var_name == "TEMP" and temp_data is not None:
                    created_vars[var_name][:] = temp_data
                elif var_name == "PRES_quality_control" and pres_qc_data is not None:
                    created_vars[var_name][:] = pres_qc_data
                elif var_name == "TEMP_quality_control" and temp_qc_data is not None:
                    created_vars[var_name][:] = temp_qc_data

            _apply_common_global_attrs(
                nc,
                latitude=latitude,
                longitude=longitude,
                depth=depth,
                time_coverage_start=metadata.get("time_coverage_start", metadata.get("time_deployment_start", t0)),
                time_coverage_end=metadata.get("time_coverage_end", metadata.get("time_deployment_end", t1)),
                processing_version=processing_version,
                site_code=_site_identity(location),
                instrument=metadata.get("inst_type", str(instrument).upper()),
                instrument_serial_number=metadata.get("inst_id", inst_id),
                nominal_inst_depth=metadata.get("nominal_inst_depth"),
                title=self.schema.get("default_title", "").format(location=location),
                source=f"CSIRO coastal mooring observations {start_date.strftime('%Y-%m-%d')}",
                keywords=self.schema.get("default_keywords", ""),
                references=metadata.get("references", "http://imos.org.au/facilities/srs/srscalval/"),
                site=metadata.get("site", "Bass Strait, TAS"),
                author_email=metadata.get("author_email", "benoit.legresy@csiro.au"),
                author=metadata.get("author", "Dr Benoit LEGRESY, CSIRO"),
                principal_investigator=metadata.get("principal_investigator", "Dr Benoit LEGRESY, CSIRO"),
                set_attr=_set_attr,
            )

        return output_path


class IMOSNetCDFConverter_SBE26(IMOSNetCDFConverterBase):
    """SBE26 Pressure converter."""
    
    def create_netcdf(
        self,
        time_data,
        longitude: float,
        latitude: float,
        depth: float,
        pres_data,
        temp_data,
        pres_qc_data,
        temp_qc_data,
        inst_channels: str,
        start_of_good_data,
        site_code: str,
        version: str,
        instrument: str,
        inst_id,
        location: str,
        input_nc_file: str | None = None,
        output_name_mode: str = "imos",
        output_dir: str | None = None,
        **metadata,
    ) -> str:
        file_name = _build_output_filename(
            inst_channels=inst_channels,
            start_of_good_data=start_of_good_data,
            location=location,
            version=version,
            instrument=instrument,
            depth=depth,
            inst_id=inst_id,
            output_name_mode=output_name_mode,
        )

        output_path = _build_output_path(output_dir or self.output_dir, file_name)

        if input_nc_file:
            if not os.path.exists(input_nc_file):
                raise FileNotFoundError(f"Input NetCDF file not found: {input_nc_file}")
            extracted = self._extract_arrays_from_input_nc(input_nc_file, depth)
            time_data = extracted["time_data"]
            t0 = extracted["time_coverage_start_default"]
            t1 = extracted["time_coverage_end_default"]
        else:
            if time_data is None:
                raise ValueError("time_data is required when input_nc_file is not provided.")
            t0 = pd.to_datetime(time_data[0], unit="D", origin="1950-01-01")
            t1 = pd.to_datetime(time_data[-1], unit="D", origin="1950-01-01")
            extracted = {}

        n_time = len(time_data)
        start_date = _parse_start_of_good_data(start_of_good_data, robust=True)

        processing_version = _normalize_processing_version(version)
        _set_attr = self._setup_attr_setter(metadata.get("metadata_mode", "fill_missing"))

        if os.path.exists(output_path):
            os.remove(output_path)

        with Dataset(output_path, "w", format="NETCDF4") as nc:
            nc.createDimension("TIME", n_time)

            created_vars = self._create_output_variables(nc, n_time)

            created_vars["TIME"][:] = time_data
            created_vars["LATITUDE"].assignValue(float(latitude))
            created_vars["LONGITUDE"].assignValue(float(longitude))
            created_vars["NOMINAL_DEPTH"].assignValue(np.float32(depth))

            for var_name in ["PRES", "PRES_quality_control", "TEMP", "TEMP_quality_control"]:
                if var_name in extracted and extracted[var_name] is not None and var_name in created_vars:
                    created_vars[var_name][:] = extracted[var_name]
                elif var_name == "PRES" and pres_data is not None:
                    created_vars[var_name][:] = pres_data
                elif var_name == "TEMP" and temp_data is not None:
                    created_vars[var_name][:] = temp_data
                elif var_name == "PRES_quality_control" and pres_qc_data is not None:
                    created_vars[var_name][:] = pres_qc_data
                elif var_name == "TEMP_quality_control" and temp_qc_data is not None:
                    created_vars[var_name][:] = temp_qc_data

            _apply_common_global_attrs(
                nc,
                latitude=latitude,
                longitude=longitude,
                depth=depth,
                time_coverage_start=metadata.get("time_coverage_start", metadata.get("time_deployment_start", t0)),
                time_coverage_end=metadata.get("time_coverage_end", metadata.get("time_deployment_end", t1)),
                processing_version=processing_version,
                site_code=_site_identity(location),
                instrument=metadata.get("inst_type", str(instrument).upper()),
                instrument_serial_number=metadata.get("inst_id", inst_id),
                nominal_inst_depth=metadata.get("nominal_inst_depth"),
                title=self.schema.get("default_title", "").format(location=location),
                source=f"CSIRO coastal mooring observations {start_date.strftime('%Y-%m-%d')}",
                keywords=self.schema.get("default_keywords", ""),
                references=metadata.get("references", "http://imos.org.au/facilities/srs/srscalval/"),
                site=metadata.get("site", "Bass Strait, TAS"),
                author_email=metadata.get("author_email", "benoit.legresy@csiro.au"),
                author=metadata.get("author", "Dr Benoit LEGRESY, CSIRO"),
                principal_investigator=metadata.get("principal_investigator", "Dr Benoit LEGRESY, CSIRO"),
                set_attr=_set_attr,
            )

        return output_path


class IMOSNetCDFConverter_SBE37(IMOSNetCDFConverterBase):
    """SBE37 CTD converter."""
    
    def create_netcdf(
        self,
        time_data,
        longitude: float,
        latitude: float,
        depth: float,
        pres_data,
        temp_data,
        pres_qc_data,
        temp_qc_data,
        inst_channels: str,
        start_of_good_data,
        site_code: str,
        version: str,
        instrument: str,
        inst_id,
        location: str,
        input_nc_file: str | None = None,
        output_name_mode: str = "imos",
        output_dir: str | None = None,
        **metadata,
    ) -> str:
        if str(output_name_mode).lower() == "internal":
            file_name = _build_internal_stage_filename(
                location=location,
                start_of_good_data=start_of_good_data,
                instrument=instrument,
                inst_id=inst_id,
                inst_depth=depth,
            )
        else:
            file_name = _build_imos_delivery_filename(
                inst_channels=inst_channels,
                start_of_good_data=start_of_good_data,
                location=location,
                version=version,
                instrument=instrument,
                depth=depth,
            )

        output_path = _build_output_path(self.output_dir, file_name)

        if not input_nc_file or not os.path.exists(input_nc_file):
            raise FileNotFoundError("SBE37 conversion requires an existing input NetCDF file.")

        extracted = self._extract_arrays_from_input_nc(input_nc_file, depth)
        time_data = extracted["time_data"]
        n_time = len(time_data)
        time_coverage_start = extracted["time_coverage_start_default"]
        time_coverage_end = extracted["time_coverage_end_default"]

        if os.path.exists(output_path):
            os.remove(output_path)

        processing_version = _normalize_processing_version(version)
        _set_attr = self._setup_attr_setter(metadata.get("metadata_mode", "fill_missing"))

        with Dataset(output_path, "w", format="NETCDF4") as nc:
            nc.createDimension("TIME", n_time)

            created_vars = self._create_output_variables(nc, n_time)

            created_vars["TIME"][:] = time_data
            created_vars["TIMESERIES"].assignValue(1)
            created_vars["LATITUDE"].assignValue(float(latitude))
            created_vars["LONGITUDE"].assignValue(float(longitude))
            created_vars["NOMINAL_DEPTH"].assignValue(np.float32(depth))

            # Populate all SBE37 variables
            for var_name in ["TEMP", "TEMP_quality_control", "CNDC", "CNDC_quality_control",
                           "PSAL", "PSAL_quality_control", "PRES", "PRES_quality_control",
                           "DEPTH", "DEPTH_quality_control"]:
                if var_name in extracted and extracted[var_name] is not None and var_name in created_vars:
                    created_vars[var_name][:] = extracted[var_name]

            start_date = _parse_start_of_good_data(start_of_good_data, robust=True)
            _apply_common_global_attrs(
                nc,
                latitude=latitude,
                longitude=longitude,
                depth=depth,
                time_coverage_start=metadata.get("time_coverage_start", metadata.get("time_deployment_start", time_coverage_start)),
                time_coverage_end=metadata.get("time_coverage_end", metadata.get("time_deployment_end", time_coverage_end)),
                processing_version=processing_version,
                site_code=_site_identity(location),
                instrument=metadata.get("inst_type", str(instrument).upper()),
                instrument_serial_number=metadata.get("inst_id", inst_id),
                nominal_inst_depth=metadata.get("nominal_inst_depth"),
                title=self.schema.get("default_title", "").format(location=location),
                source=f"CSIRO coastal mooring observations {start_date.strftime('%Y-%m-%d')}",
                keywords=self.schema.get("default_keywords", ""),
                references=metadata.get("references", "http://imos.org.au/facilities/srs/srscalval/"),
                site=metadata.get("site", "Bass Strait, TAS"),
                author_email=metadata.get("author_email", "benoit.legresy@csiro.au"),
                author=metadata.get("author", "Dr Benoit LEGRESY, CSIRO"),
                principal_investigator=metadata.get("principal_investigator", "Dr Benoit LEGRESY, CSIRO"),
                set_attr=_set_attr,
            )

            _set_attr(nc, "deployment_id", metadata.get("deployment_id"))
            _set_attr(nc, "mooring_channels", metadata.get("mooring_channels"))
            _set_attr(nc, "deploy_date", metadata.get("deploy_date"))
            _set_attr(nc, "recovery_date", metadata.get("recovery_date"))

        print(f"IMOS attributes added and file renamed to: {file_name}")
        return output_path


class IMOSNetCDFConverter_SIG500(IMOSNetCDFConverterBase):
    """Nortek Signature 500 converter."""
    
    def determine_channel_from_filename(self, input_filename: str) -> str | None:
        """Extract channel from filename."""
        filename_lower = input_filename.lower()
        if "z" in filename_lower:
            return "Z"
        if "v" in filename_lower:
            return "V"
        if "w" in filename_lower:
            return "W"
        return None

    def create_netcdf(
        self,
        time_data,
        longitude: float,
        latitude: float,
        depth: float,
        pres_data,
        temp_data,
        pres_qc_data,
        temp_qc_data,
        inst_channels: str,
        start_of_good_data,
        site_code: str,
        version: str,
        instrument: str,
        inst_id,
        location: str,
        input_nc_file: str | None = None,
        output_name_mode: str = "imos",
        output_dir: str | None = None,
        **metadata,
    ) -> str:
        if str(output_name_mode).lower() == "internal":
            file_name = _build_internal_stage_filename(
                location=location,
                start_of_good_data=start_of_good_data,
                instrument=instrument,
                inst_id=inst_id,
                inst_depth=depth,
            )
        else:
            file_name = _build_imos_delivery_filename(
                inst_channels=inst_channels,
                start_of_good_data=start_of_good_data,
                location=location,
                version=version,
                instrument=instrument,
                depth=depth,
            )

        output_path = _build_output_path(self.output_dir, file_name)

        if not input_nc_file or not os.path.exists(input_nc_file):
            raise FileNotFoundError("SIG500 conversion requires an existing input NetCDF file.")

        shutil.copy2(input_nc_file, output_path)

        with Dataset(output_path, "r") as nc_read:
            if "TIME" in nc_read.variables:
                tvals = nc_read.variables["TIME"][:]
                t0 = pd.to_datetime(tvals[0], unit="D", origin="1950-01-01")
                t1 = pd.to_datetime(tvals[-1], unit="D", origin="1950-01-01")
            else:
                t0 = datetime.now(timezone.utc)
                t1 = datetime.now(timezone.utc)

        _set_attr = self._setup_attr_setter(metadata.get("metadata_mode", "fill_missing"))

        with Dataset(output_path, "a") as nc:
            start_date = _parse_start_of_good_data(start_of_good_data, robust=True)
            _apply_common_global_attrs(
                nc,
                latitude=latitude,
                longitude=longitude,
                depth=depth,
                time_coverage_start=metadata.get("time_coverage_start", metadata.get("time_deployment_start", t0)),
                time_coverage_end=metadata.get("time_coverage_end", metadata.get("time_deployment_end", t1)),
                processing_version=version,
                site_code=_site_identity(location),
                instrument=metadata.get("inst_type", instrument),
                instrument_serial_number=metadata.get("inst_id", inst_id),
                nominal_inst_depth=metadata.get("nominal_inst_depth"),
                title=self.schema.get("default_title", "").format(location=location),
                source=f"CSIRO coastal mooring observations {start_date.strftime('%Y-%m-%d')}",
                keywords=self.schema.get("default_keywords", ""),
                references=metadata.get("references", "http://imos.org.au/facilities/srs/srscalval/"),
                site=metadata.get("site", "Bass Strait, TAS"),
                author_email=metadata.get("author_email", "benoit.legresy@csiro.au"),
                author=metadata.get("author", "Dr Benoit LEGRESY, CSIRO"),
                principal_investigator=metadata.get("principal_investigator", "Dr Benoit LEGRESY, CSIRO"),
                set_attr=_set_attr,
            )

            _set_attr(nc, "magnetic_declination", metadata.get("magnetic_declination_text", "13.5 degrees"))
            _set_attr(nc, "magnetic_declination_disclaimer", metadata.get("magnetic_declination_disclaimer"))
            _set_attr(nc, "deployment_id", metadata.get("deployment_id"))
            _set_attr(nc, "mooring_channels", metadata.get("mooring_channels"))
            _set_attr(nc, "deploy_date", metadata.get("deploy_date"))
            _set_attr(nc, "recovery_date", metadata.get("recovery_date"))

        print(f"IMOS attributes added and file renamed to: {file_name}")
        return output_path