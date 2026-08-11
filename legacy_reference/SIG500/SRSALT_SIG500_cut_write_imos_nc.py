# ============================================================
# ### SIG500 processing (Simplified Filepath Input)
# ============================================================
# 
# Modify Signature 500 NetCDF files to comply with IMOS global attribute requirements
# Create 3 individual files for Altimeter, current, and wave data 

# Inputs required:
#   Data file path (*_data.txt)         
#   Metadata CSV path (.csv)
#   Processed output path (.nc)
#   IMOS output directory (.nc)
# ============================================================

# Imports
import numpy as np
import os
import sys
import glob
import gsw
import datetime
import pandas as pd
from netCDF4 import Dataset
import re
import shutil


def print_existing_global_metadata(nc_path):
    """Print a compact set of global attributes from an input NetCDF file."""
    attrs_to_show = [
        'site_code',
        'site',
        'instrument',
        'instrument_serial_number',
        'processing_version',
        'title',
        'source',
    ]

    print(f"\nMetadata preview from file: {os.path.basename(nc_path)}")
    try:
        with Dataset(nc_path, 'r') as ds:
            for attr_name in attrs_to_show:
                value = getattr(ds, attr_name, '<missing>')
                print(f"  {attr_name}: {value}")
    except Exception as e:
        print(f"  Could not read global attributes: {e}")


def prompt_metadata_mode():
    """Choose how metadata from table should be applied to output NetCDF files."""
    prompt = (
        "\nProceed with file metadata (fill missing from database) [P], "
        "or overwrite from database [O]? (P/O, default P): "
    )

    while True:
        choice = input(prompt).strip().lower()
        if choice in ('', 'p', 'proceed'):
            return 'fill_missing'
        if choice in ('o', 'overwrite'):
            return 'overwrite'
        print("Please enter P to proceed/fill missing, or O to overwrite.")

converter_path = '/datasets/work/oa-srsalt/work/scripts/ash/mooring_proc/tools/imos_nc_converter'
sys.path.insert(0, converter_path)
from imos_converter import IMOSNetCDFConverter_SIG500

sys.path.insert(0, '/datasets/work/oa-srsalt/work/scripts/ash/mooring_proc/tools/imos_to_csiro_fn_converter')
from imos_to_csiro_nc_fn import convert_imos_to_csiro_filename

# ============================================================
## Configuration ##           << To be updated with 'selector'
# ============================================================

input_filepath = "/datasets/work/oa-srsalt/work/preqa/SWOT/cal_val/jason_calval/all_mooring_data/proc_2/rec_202508/BASS3B_PTSUVW_202508/SIG500_104424/"
metadata_file = "/datasets/work/oa-srsalt/work/preqa/SWOT/cal_val/jason_calval/all_mooring_data/bass_strait_metadata.csv"
# ^^^^^ Change this to match your mooring_proc path!
processed_output_dir_input = "/datasets/work/oa-srsalt/work/preqa/SWOT/cal_val/jason_calval/all_mooring_data/proc_2/rec_202508/BASS3B_PTSUVW_202508/SIG500_104424/"
imos_output_dir = "/datasets/work/oa-srsalt/work/preqa/SWOT/cal_val/jason_calval/all_mooring_data/imos_delivery/rec_202508/2024d/"

# ============================================================
## Metadata ##
# ============================================================

# Extract directory from filepath
data_folder = os.path.dirname(input_filepath)

# Extract serial number from directory path (assumes: .../SIG500_SERIALNUM/filename)
# Look for SIG500_XXXXXX pattern in the path
dir_match = re.search(r'SIG500_(\d+)', input_filepath)

if dir_match:
    selected_serial_number = int(dir_match.group(1))
else:
    print(f"WARNING: Could not auto-detect serial number from path: {input_filepath}")
    print("Please enter the serial number manually:")
    selected_serial_number = int(input("Serial number: ").strip())

# Load metadata
metadata_df = pd.read_csv(metadata_file, dtype={'start_YYYYMM': str, 'end_YYYYMM': str})

# Try to match by serial number first
matching_rows = metadata_df[metadata_df['inst_id'] == selected_serial_number]

# If no match by serial number, try matching by data_in_file column
if len(matching_rows) == 0 and 'data_in_file' in metadata_df.columns:
    print(f"\nNo direct serial match found. Attempting to match by file path...")
    # Normalize paths for comparison
    for idx, row in metadata_df.iterrows():
        if 'data_in_file' in row and pd.notna(row['data_in_file']):
            if str(row['data_in_file']) in input_filepath or input_filepath.endswith(str(row['data_in_file']).split('/')[-1]):
                matching_rows = pd.DataFrame([row])
                print(f"Matched by file path")
                break

if len(matching_rows) == 0:
    raise ValueError(f"No metadata found for serial number: {selected_serial_number}")
elif len(matching_rows) > 1:
    # Remove duplicate rows based on key fields
    matching_rows = matching_rows.drop_duplicates(
        subset=['location', 'deployment_id', 'start_YYYYMM', 'end_YYYYMM', 'inst_id'],
        keep='first'
    )
    
    # Check again after removing duplicates
    if len(matching_rows) > 1:
        print(f"\nMultiple deployments found for serial {selected_serial_number}:")
        for i, (idx, row) in enumerate(matching_rows.iterrows(), 1):
            print(f"  {i}. Location: {row['location']}, Deployment: {row.get('deployment_id', 'N/A')}, Date: {row.get('start_YYYYMM', 'N/A')}")
        
        while True:
            choice = input(f"Select deployment (1-{len(matching_rows)}): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(matching_rows):
                    selected_row = matching_rows.iloc[idx]
                    break
            print(f"Please enter a number between 1 and {len(matching_rows)}")
    else:
        selected_row = matching_rows.iloc[0]
        print(f"Using deployment: {selected_row['location']}, {selected_row.get('deployment_id', 'N/A')}, {selected_row.get('start_YYYYMM', 'N/A')}")
else:
    selected_row = matching_rows.iloc[0]

# Extract all needed parameters from metadata
selected_location = selected_row['location']
selected_instrument = selected_row['inst_type']
selected_inst_id = selected_row['inst_id']
start_date_code = selected_row['start_YYYYMM']
end_date_code = selected_row['end_YYYYMM']

site_metadata = selected_row

# Generate output NC filename from input filename (remove _data.txt, add .nc)
input_filename = os.path.basename(input_filepath)
output_filename = input_filename.replace('_data.txt', '.nc')
processed_nc_path = os.path.join(processed_output_dir_input, output_filename)

# ============================================================
## Configuration and Path Setup ##
# ============================================================

# Extract site and instrument info from metadata
site_code = str(site_metadata['location'])
longitude = float(site_metadata['longitude'])
latitude = float(site_metadata['latitude'])
depth = float(site_metadata['nominal_depth'])
channels = str(site_metadata['mooring_channels'])
version = str(site_metadata['version'])
instrument = f"Signature500"
inst_id = str(selected_inst_id)

print(f"\nSite information:")
print(f"  Site code: {site_code}")
print(f"  Coordinates: {latitude:.6f}, {longitude:.6f}")
print(f"  Depth: {depth} m")
print(f"  Channels: {channels}")
print(f"  Instrument: {instrument} (ID: {inst_id})")
print(f"  Version: {version}")
print(f"  Data period: {start_date_code} → {end_date_code}")

# ============================================================
## Directory setup and validation ##
# ============================================================

# Create output directories
os.makedirs(processed_output_dir_input, exist_ok=True)
os.makedirs(imos_output_dir, exist_ok=True)

# Validate that the input directory exists
if not os.path.exists(input_filepath):
    print(f"ERROR: Input directory does not exist: {input_filepath}")
    raise FileNotFoundError(f"Input directory not found: {input_filepath}")

# ============================================================
## Find the three separate files (V, W, Z) ##
# ============================================================

print("\nSearching for V, W, Z files...")

# Look for the three files
v_files = glob.glob(os.path.join(input_filepath, '*_V*.nc'))
w_files = glob.glob(os.path.join(input_filepath, '*_W*.nc'))
z_files = glob.glob(os.path.join(input_filepath, '*_Z*.nc'))

if not v_files:
    raise FileNotFoundError(f"No *_V*.nc file found in {input_filepath}")
if not w_files:
    raise FileNotFoundError(f"No *_W*.nc file found in {input_filepath}")
if not z_files:
    raise FileNotFoundError(f"No *_Z*.nc file found in {input_filepath}")

# Use the first match for each
v_file = v_files[0]
w_file = w_files[0]
z_file = z_files[0]

print(f"  V file: {os.path.basename(v_file)}")
print(f"  W file: {os.path.basename(w_file)}")
print(f"  Z file: {os.path.basename(z_file)}")

nc_files = [v_file, w_file, z_file]

# Preview existing global metadata from each input file before choosing mode
for preview_file in nc_files:
    print_existing_global_metadata(preview_file)

metadata_mode = prompt_metadata_mode()
print(f"Selected metadata mode: {metadata_mode}")

# ============================================================
## Process each file with IMOS converter ##
# ============================================================

# Extract site and instrument info from metadata
site_code = f"{selected_location}"
longitude = float(site_metadata['longitude'])
latitude = float(site_metadata['latitude'])
depth = float(site_metadata['nominal_depth'])
version = str(site_metadata['version'])
instrument = selected_instrument
inst_id = str(selected_inst_id)

print(f"\nSite information:")
print(f"  Site code: {site_code}")
print(f"  Coordinates: {latitude:.6f}, {longitude:.6f}")
print(f"  Depth: {depth} m")
print(f"  Instrument: {instrument} (ID: {inst_id})")
print(f"  Version: {version}")
print(f"  Data period: {start_date_code} → {end_date_code}")

print("\nProcessing NetCDF files with IMOS converter...")
print(f"Site: {site_code}")
print(f"Instrument: {instrument} (Serial: {inst_id})")
print(f"Location: {latitude:.6f}°, {longitude:.6f}°")
print(f"Depth: {depth} m")
print("=" * 60)

# Initialize converter
converter = IMOSNetCDFConverter_SIG500(
    input_folder=input_filepath,
    input_file="",  # Will be set for each file
    output_dir=imos_output_dir
)

# Process each NetCDF file
successful_files = []
failed_files = []

for i, nc_file_path in enumerate(nc_files, 1):
    filename = os.path.basename(nc_file_path)
    print(f"\n[{i}/{len(nc_files)}] Processing: {filename}")
    
    try:
        # Extract channel from filename (V, W, Z, etc.)
        # Expected pattern: LOCATION_V_YYYYMM.nc -> channel = 'V'
        filename_parts = filename.replace('.nc', '').split('_')
        # Look for V, W, or Z in the filename parts
        file_channel = None
        for part in filename_parts:
            if part in ['V', 'W', 'Z']:
                file_channel = part
                break
        
        if file_channel:
            print(f"  → Using channel '{file_channel}' from filename")
        else:
            # Try to extract from position if pattern matches
            if len(filename_parts) >= 2:
                file_channel = filename_parts[1]  # Assume second part is channel
                print(f"  → Using channel '{file_channel}' from filename position")
            else:
                file_channel = 'UNKNOWN'
                print(f"  → WARNING: Could not extract channel from filename")
                
        # Extract start_of_good_data from the current NetCDF file
        try:
            with Dataset(nc_file_path, 'r') as nc:
                if 'TIME' in nc.variables:
                    time_data = nc.variables['TIME'][:]
                    # Convert from days since 1950-01-01 to datetime
                    start_of_good_data = pd.to_datetime(time_data[0], unit='D', origin='1950-01-01')
                else:
                    # Fallback to metadata start date
                    start_of_good_data = pd.to_datetime(f"{start_date_code}01", format='%Y%m%d')
        except Exception as e:
            print(f"  ⚠ WARNING: Could not read TIME from NetCDF file: {e}")
            print("  Using metadata start date as fallback")
            start_of_good_data = pd.to_datetime(f"{start_date_code}01", format='%Y%m%d')
        
        # Process this file with the specific channel
        result_path = converter.process(
            input_nc_path=nc_file_path,
            longitude=longitude,
            latitude=latitude,
            depth=depth,
            inst_channels=file_channel,  
            start_of_good_data=start_of_good_data,
            site_code=site_code,
            version=version,
            instrument=instrument,
            inst_id=inst_id,
            location=selected_location,
            metadata_mode=metadata_mode,
            deployment_id=site_metadata.get('deployment_id', ''),
            mooring_channels=site_metadata.get('mooring_channels', ''),
            deploy_date=site_metadata.get('deploy_date', ''),
            recovery_date=site_metadata.get('recovery_date', ''),
            nominal_inst_depth=site_metadata.get('nominal_inst_depth', ''),
        )
        
        successful_files.append((filename, os.path.basename(result_path)))
        print(f"  ✓ SUCCESS: {os.path.basename(result_path)}")
        
    except Exception as e:
        failed_files.append((filename, str(e)))
        print(f"  ✗ ERROR: Failed to process {filename}: {e}")

# ============================================================
## Find and process generated IMOS files ##
# ============================================================

# # Find all generated IMOS files
# imos_files = [f for f in os.listdir(imos_output_dir) if f.endswith('.nc') and f.startswith('IMOS_')]

# if not imos_files:
#     raise FileNotFoundError("No IMOS NetCDF files were created")

# print(f"\nFound {len(imos_files)} IMOS file(s)")

# # Process each IMOS file to CSIRO format
# csiro_files = []
# for imos_filename in sorted(imos_files):
#     imos_filepath = os.path.join(imos_output_dir, imos_filename)
    
#     # Convert IMOS filename to CSIRO format
#     csiro_filename = convert_imos_to_csiro_filename(
#         imos_filename=imos_filename,
#         location=selected_location,
#         instrument=instrument,
#         inst_id=inst_id,
#         depth_m=depth
#     )
    
#     # Copy IMOS file to processed directory with CSIRO name
#     processed_nc_path = os.path.join(processed_output_dir_input, csiro_filename)
#     os.makedirs(os.path.dirname(processed_nc_path), exist_ok=True)
#     shutil.copy2(imos_filepath, processed_nc_path)
    
#     csiro_files.append((imos_filename, csiro_filename))
#     print(f"Processed NetCDF file created: {processed_nc_path}")

# ============================================================
## Processing Summary ##
# ============================================================

print("\n" + "=" * 80)
print("PROCESSING SUMMARY")
print("=" * 80)

print(f"  Successfully processed SIG500 data from {selected_location}")
print(f"  Instrument: {selected_instrument} (Serial: {selected_inst_id})")
print(f"  Data period: {start_date_code} to {end_date_code}")
print(f"  Input files processed: {len(nc_files)}")
print(f"  IMOS files generated: {len(imos_files)}")
# print(f"  CSIRO files generated: {len(csiro_files)}")

# if csiro_files:
#     print(f"\n IMOS to CSIRO conversions:")
#     for imos_name, csiro_name in csiro_files:
#         print(f"  {imos_name} → {csiro_name}")

# print(f"\nIMOS output directory: {imos_output_dir}")
# print(f"Processed output directory: {processed_output_dir_input}")

# if successful_files:
#     print(f"\n Successfully processed files:")
#     for original, imos_name in successful_files:
#         print(f"  {original} → {imos_name}")

# if failed_files:
#     print(f"\n Failed files:")
#     for original, error in failed_files:
#         print(f"  {original}: {error}")

print(f"\nIMOS files saved to: {imos_output_dir}")
print("\n" + "=" * 80)
print("PROCESSING COMPLETE")
print("=" * 80)
