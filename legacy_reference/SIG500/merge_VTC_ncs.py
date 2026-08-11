# ### SIG500 post-processing

# #### Concatenating velocity files from OceanContour - Burst_*VTC.nc

# Imports
import numpy as np
import xarray as xr
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob
import datetime

#--------------------------------------------------------------------------
# hardcoded source
generic_folder="/datasets/work/oa-srsalt/work/preqa/SWOT/cal_val/jason_calval/all_mooring_data"

generic_input_folder="proc_1/rec_202603/BASJAS_PTSUVW_202603/SIG500_103017"
generic_ocprojectfolder="BASJAS-PTSUVW_202603/Exported_Data/S103017A002_BSJSN2502_0000/Burst"
input_folder=os.path.join(generic_folder,generic_input_folder,generic_ocprojectfolder) 

generic_output_folder="proc_2/rec_202603/BASJAS_PTSUVW_202603/SIG500_103017"
output_folder=os.path.join(generic_folder,generic_output_folder)

os.chdir(input_folder)
input_file_list = sorted(glob.glob("Burst_???.VTC.nc"))

output_file = "BASJAS_UV_202603_VTC.nc"

#------------------------------------------------------------------------
 # Main
os.chdir(input_folder)

# Open first file with decode_times=False to avoid timestamp overflow
file_in=input_file_list[0]
print(file_in)

ds = xr.open_dataset(file_in, group='Data/Burst', decode_times=False)

# sort out the time on the first file
epoch_1950 = pd.Timestamp('1950-01-01 00:00:00', tz='UTC')
seconds_offset_1950_to_1970 = (pd.Timestamp('1970-01-01 00:00:00', tz='UTC') - epoch_1950).total_seconds()
days_offset_1950_to_1970 = seconds_offset_1950_to_1970 / 86400.0
if 'TimeStamp' in ds.variables:
    ts_sec = ds['TimeStamp'].values.astype('float64')
    time_days = ts_sec / 86400.0 + days_offset_1950_to_1970
    ds = ds.assign_coords(time=time_days.astype('float64'))
    ds['time'].encoding.update({
        'units': 'days since 1950-01-01 00:00:00',
        'calendar': 'gregorian',
        'dtype': 'float64'
    })
# Concatenate all remaining files
list_ind = range(len(input_file_list)-1)
for ific in list_ind:
    file_in = input_file_list[ific+1]
    print(file_in)
    ds2 = xr.open_dataset(file_in, group='Data/Burst', decode_times=False)
    
    # Manually convert TimeStamp seconds to days since 1950
    if 'TimeStamp' in ds2.variables:
        ts_sec2 = ds2['TimeStamp'].values.astype('float64')
        time_days2 = ts_sec2 / 86400.0 + days_offset_1950_to_1970
        ds2 = ds2.assign_coords(time=time_days2.astype('float64'))
        ds2['time'].encoding.update({
            'units': 'days since 1950-01-01 00:00:00',
            'calendar': 'gregorian',
            'dtype': 'float64'
        })
    
    # Use concat for time series data
    ds = xr.concat([ds, ds2], dim='time')

os.chdir(output_folder)
ds.to_netcdf(output_file)
