# ### SIG500 processing

# #### Concatenating velocity files from OceanContour - Burst_*VTC.DSEL.AVER.nc

import numpy as np
import xarray as xr
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob
import datetime


folder="/datasets/work/oa-srsalt/work/preqa/SWOT/cal_val/jason_calval/all_mooring_data/proc_1/rec_202603/BASJAS_PTSUVW_202603/SIG500_103017/Exported_Data/S103017A002_BSJSN2502_0000/Burst"
os.chdir(folder)

fic1='Burst_001.VTC.DSEL.AVER.nc'
fic2='Burst_002.VTC.DSEL.AVER.nc'
filist = sorted(glob.glob("Burst_???.VTC.DSEL.AVER.nc"))


list_ind = range(len(filist)-1)
list_ind
for ific in list_ind :
    file_in = filist[ific+1]
    print(file_in)


# Open first file with decode_times=False to avoid timestamp overflow
file_in=filist[0]
ds = xr.open_dataset(file_in, group='Data/Burst', decode_times=False)

# Convert time using TimeStamp seconds since 1970 -> days since 1950 (gregorian)
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
list_ind = range(len(filist)-1)
for ific in list_ind:
    file_in = filist[ific+1]
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

ds

# Save with time encoded as days since 1950-01-01 00:00:00 UTC
output_folder = "/datasets/work/oa-srsalt/work/scripts/ash"
output_file = os.path.join(output_folder, "BASS3B_V_202508_swapdims.nc")
ds.to_netcdf(output_file)
# output_folder = "/datasets/work/oa-srsalt/work/preqa/SWOT/cal_val/jason_calval/all_mooring_data/proc_2/rec_202508/BASS3B_PTSUVW_202508/SIG500_104424"
# os.chdir(output_folder)

# Process BASS3B_Z_202508.nc
file_in = output_file
ds_z = xr.open_dataset(file_in)

# Convert TimeStamp to dimension coordinate (days since 1950), then rename to TIME
if 'TimeStamp' in ds_z.variables:
    ts_sec_z = ds_z['TimeStamp'].values.astype('float64')
    time_days_z = ts_sec_z / 86400.0 + days_offset_1950_to_1970
    # Replace time dimension with TimeStamp, then rename
    ds_z = ds_z.assign_coords(TimeStamp=('time', time_days_z.astype('float64')))
    ds_z = ds_z.swap_dims({'time': 'TimeStamp'})
    ds_z = ds_z.rename({'TimeStamp': 'TIME'})
    ds_z = ds_z.rename_vars({'time': 'time_instrument'})

# Preserve CF time encoding for TIME coordinate as float days since 1950
# Disable automatic CF time decoding by setting _FillValue and ensuring no datetime decoding
if 'TIME' in ds_z.coords:
    ds_z['TIME'].encoding.update({
        'units': 'days since 1950-01-01 00:00:00',
        'calendar': 'gregorian',
        'dtype': 'float64',
        '_FillValue': np.nan
    })
    # Also update any variables using TIME to have the same encoding
    for var in ds_z.data_vars:
        if ds_z[var].dims and ds_z[var].dims[0] == 'TIME':
            ds_z[var].encoding.update({'unlimited_dims': ['TIME']})

# Drop problematic timedelta variable to avoid overflow during CF encoding
if 'MatlabTimeStamp' in ds_z.variables:
    ds_z = ds_z.drop_vars('MatlabTimeStamp')

# Save the processed file without CF time encoding to avoid overflow
output_file_z = os.path.join(output_folder, "BASS3B_Z_202508_swapdims.nc")
ds_z.to_netcdf(output_file_z)
print(f"Saved: {output_file_z}")



# # Open dataset - use chunks if dask is available
# try:
#     ds = xr.open_dataset(file_in, chunks={'time': 1000})
# except ValueError:
#     # Dask not available, open without chunking
#     ds = xr.open_dataset(file_in)

# # start_plot = datetime.datetime(2023, 2, 14, 00)
# start_plot = datetime.datetime(2024, 7, 31, 00)
# # end_plot = datetime.datetime(2023, 8, 6, 00)
# end_plot = datetime.datetime(2025, 8, 23, 00)
# dsc=ds.sel(time=slice(start_plot, end_plot))

# # Downsample for faster plotting
# downsample = 10  # Adjust this value - higher = faster but less detail

# # Create output folder for plots
# plot_folder = output_folder
# os.makedirs(plot_folder, exist_ok=True)

# # Plot Altimeter data
# plt.figure(figsize=(15,5))
# plt.plot(dsc.time[::downsample], dsc.Altimeter_AST[::downsample])
# plt.xlabel('Time')
# plt.ylabel('Altimeter AST (m)')
# plt.title('Altimeter AST (Selected Period - Downsampled)')
# plt.grid(True)
# plot_file = os.path.join(plot_folder, 'BASS3B_V_202508_Altimeter.png')
# plt.savefig(plot_file, dpi=150, bbox_inches='tight')
# print(f"Saved plot: {plot_file}")
# plt.close()

# # Plot velocity data - East and North components
# if 'Vel_East' in dsc.variables and 'Vel_North' in dsc.variables:
#     plt.figure(figsize=(15,5))
#     # Plot mean across all range bins
#     vel_east_mean = dsc.Vel_East.mean(dim='BurstVelocityENU_Range')
#     vel_north_mean = dsc.Vel_North.mean(dim='BurstVelocityENU_Range')
#     plt.plot(dsc.time[::downsample], vel_east_mean[::downsample], label='East', linewidth=1)
#     plt.plot(dsc.time[::downsample], vel_north_mean[::downsample], label='North', linewidth=1)
#     plt.xlabel('Time')
#     plt.ylabel('Mean Velocity (m/s)')
#     plt.title('Burst Velocity - East & North (Mean across depth bins)')
#     plt.legend()
#     plt.grid(True)
#     plot_file = os.path.join(plot_folder, 'BASS3B_V_202508_Velocity_EastNorth.png')
#     plt.savefig(plot_file, dpi=150, bbox_inches='tight')
#     print(f"Saved plot: {plot_file}")
#     plt.close()
# else:
#     print("Warning: Vel_East or Vel_North not found in dataset")

# # Plot vertical velocity components if available
# if 'Vel_Up1' in dsc.variables:
#     plt.figure(figsize=(15,5))
#     vel_up1_mean = dsc.Vel_Up1.mean(dim='BurstVelocityENU_Range')
#     plt.plot(dsc.time[::downsample], vel_up1_mean[::downsample], label='Up1', linewidth=1)
#     if 'Vel_Up2' in dsc.variables:
#         vel_up2_mean = dsc.Vel_Up2.mean(dim='BurstVelocityENU_Range')
#         plt.plot(dsc.time[::downsample], vel_up2_mean[::downsample], label='Up2', linewidth=1)
#     plt.xlabel('Time')
#     plt.ylabel('Mean Vertical Velocity (m/s)')
#     plt.title('Burst Velocity - Vertical Components (Mean across depth bins)')
#     plt.legend()
#     plt.grid(True)
#     plot_file = os.path.join(plot_folder, 'BASS3B_V_202508_Velocity_Vertical.png')
#     plt.savefig(plot_file, dpi=150, bbox_inches='tight')
#     print(f"Saved plot: {plot_file}")
#     plt.close()




