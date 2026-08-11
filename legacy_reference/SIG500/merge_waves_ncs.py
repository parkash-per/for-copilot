# ### SIG500 processing

# #### Concatenating wave files from OceanContour - Burst_*VTC.WAVES.nc

import numpy as np
import xarray as xr
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob
import datetime


folder="/datasets/work/oa-srsalt/work/preqa/SWOT/cal_val/jason_calval/all_mooring_data/proc_1/rec_202508/BASS3B_PTSUVW_202508/SIG500_104424/Exported Data/S104424A001_BSS3B2408_0000/Burst"
os.chdir(folder)

fic1='Burst_001.VTC.WAVES.nc'
fic2='Burst_002.VTC.WAVES.nc'
filist = sorted(glob.glob("Burst_???.VTC.WAVES.nc"))


list_ind = range(len(filist)-1)
list_ind
for ific in list_ind :
    file_in = filist[ific+1]
    print(file_in)

# Open first file with decode_times=False to avoid timestamp overflow
# Waves data live in the nested group Data/Waves
file_in=filist[0]
ds = xr.open_dataset(file_in, group='Data/Waves', decode_times=False)

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
    ds2 = xr.open_dataset(file_in, group='Data/Waves', decode_times=False)
    
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

# Save to temporary file to avoid read/write conflicts
output_folder = "/datasets/work/oa-srsalt/work/preqa/SWOT/cal_val/jason_calval/all_mooring_data/proc_2/rec_202508/BASS3B_PTSUVW_202508/SIG500_104424"
output_file_temp = os.path.join(output_folder, "BASS3B_W_202508_temp.nc")
ds.to_netcdf(output_file_temp)


file_in = output_file_temp
ds_z = xr.open_dataset(file_in, decode_times=False)

# Create TIME as the primary dimension from TimeStamp if present, otherwise from time variable
if 'TimeStamp' in ds_z.variables or 'time' in ds_z.variables:
    if 'TimeStamp' in ds_z.variables:
        ts_sec_z = ds_z['TimeStamp'].values.astype('float64')
    else:
        ts_sec_z = ds_z['time'].values.astype('float64')  # seconds since 1970

    time_days_z = ts_sec_z / 86400.0 + days_offset_1950_to_1970

    # Keep original 'time' as 'time_instrument' (converted to days since 1950)
    if 'time' in ds_z.coords:
        time_days_instrument = ds_z['time'].values.astype('float64') / 86400.0 + days_offset_1950_to_1970
    else:
        time_days_instrument = time_days_z
    ds_z['time_instrument'] = (('time',), time_days_instrument)
    ds_z['time_instrument'].attrs.update({
        'units': 'days since 1950-01-01 00:00:00',
        'description': 'Instrument time dimension'
    })
    ds_z['time_instrument'].encoding = {'_FillValue': None}

    # Rename 'time' dimension to 'TIME' and reassign with converted values
    ds_z = ds_z.rename({'time': 'TIME'})
    ds_z = ds_z.assign_coords(TIME=time_days_z.astype('float64'))
    ds_z['TIME'].attrs.update({
        'units': 'days since 1950-01-01 00:00:00',
        'long_name': 'time',
        'short_name': 'time',
        'axis': 'T',
        'valid_min': '0.0',
        'valid_max': '90000.0',
        'calendar': 'gregorian'
    })
    ds_z['TIME'].encoding = {'_FillValue': None}

    # Drop TimeStamp if it existed
    if 'TimeStamp' in ds_z.variables:
        ds_z = ds_z.drop_vars('TimeStamp')

# Add MatlabTimeStamp as data variable: convert TIME to MATLAB datenum
matlab_offset_1950 = 712224  # datenum of 1950-01-01
if 'TIME' in ds_z.coords:
    matlab_datenum = ds_z['TIME'].values + matlab_offset_1950
    ds_z['MatlabTimeStamp'] = (('TIME',), matlab_datenum.astype('float64'))
    ds_z['MatlabTimeStamp'].attrs.update({
        'description': 'Instrument timestamp as MATLAB datenum (days)',
        'units': 'days since 1950-01-01 00:00:00'
    })
    ds_z['MatlabTimeStamp'].encoding = {'_FillValue': None}

# Add long_name and standard_name to all variables that don't have them
for var in ds_z.data_vars:
    if 'long_name' not in ds_z[var].attrs:
        ds_z[var].attrs['long_name'] = var
    if 'standard_name' not in ds_z[var].attrs:
        ds_z[var].attrs['standard_name'] = var

# Add long_name and standard_name to all coordinates that don't have them
for coord in ds_z.coords:
    if 'long_name' not in ds_z[coord].attrs:
        ds_z[coord].attrs['long_name'] = coord
    if 'standard_name' not in ds_z[coord].attrs:
        ds_z[coord].attrs['standard_name'] = coord

# Fix units attribute - replace "---" with empty string
for var in ds_z.data_vars:
    if 'units' in ds_z[var].attrs and ds_z[var].attrs['units'] == '---':
        ds_z[var].attrs['units'] = ''

for coord in ds_z.coords:
    if 'units' in ds_z[coord].attrs and ds_z[coord].attrs['units'] == '---':
        ds_z[coord].attrs['units'] = ''

# Save the processed file without CF time encoding to avoid overflow
output_file_z = os.path.join(output_folder, "BASS3B_W_202508_swapdims.nc")
ds_z.to_netcdf(output_file_z)
print(f"Saved: {output_file_z}")

# Clean up temporary file
try:
    os.remove(output_file_temp)
except OSError:
    pass

# ============================================================
## PLOTS ##
# ============================================================

# folder="/datasets/work/oa-srsalt/work/preqa/SWOT/cal_val/jason_calval/all_mooring_data/proc_2/rec_202508/BASS3B_PTSUVW_202508/SIG500_104424"
# os.chdir(folder)
# file_in = "BASS3B_W_202508_swapdims.nc"
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

# # Downsample for faster plotting - plot every Nth point
# downsample = 10  # Adjust this value - higher = faster but less detail

# # Create output folder for plots
# plot_folder = output_folder
# os.makedirs(plot_folder, exist_ok=True)

# plt.figure(figsize=(15,5))
# plt.plot(dsc.time[::downsample], dsc.Height_Hm0[::downsample])
# plt.xlabel('Time')
# plt.ylabel('Height_Hm0 (m)')
# plt.title('Significant Wave Height (Selected Period - Downsampled)')
# plt.grid(True)
# plot_file = os.path.join(plot_folder, 'BASS3B_W_202508_Hm0.png')
# plt.savefig(plot_file, dpi=150, bbox_inches='tight')
# print(f"Saved plot: {plot_file}")
# plt.close()

# # Remove bad data and plot with downsampling
# indbad=np.where(dsc.Height_H3.values<0)
# dsc_clean = dsc.copy()
# dsc_clean.Height_Hm0.values[indbad]=np.nan
# plt.figure(figsize=(30,5))
# plt.plot(dsc_clean.time[::downsample], dsc_clean.Height_Hm0[::downsample])
# plt.xlabel('Time')
# plt.ylabel('Height_Hm0 (m)')
# plt.title(f'Significant Wave Height (Cleaned - Every {downsample}th point)')
# plt.grid(True)
# plot_file = os.path.join(plot_folder, 'BASS3B_W_202508_Hm0_cleaned.png')
# plt.savefig(plot_file, dpi=150, bbox_inches='tight')
# print(f"Saved plot: {plot_file}")
# plt.close()



# # dsc


# # plt.rcParams.keys()


# # pr_st=5
# # pr_en=25
# # fr_en=1/pr_st
# # fr_st=1/pr_en

# # dscp=dsc.sel(WaveSpectra_Frequency=slice(fr_st, fr_en), DirectionalSpectra_Frequency=slice(fr_st, fr_en))
# # dscp


# # t_st=np.datetime64('2023-05-30')
# # t_en=np.datetime64('2023-06-02')
# # tav_st=np.datetime64('2023-05-31')
# # tav_en=np.datetime64('2023-06-01')


# # # figure('spectra_with_time')
# # plt.rcParams.update({'font.size': 26})
# # plt.rcParams.update({'font.weight': 'bold'})
# # fig,ax=plt.subplots(3,1, figsize=(30,15), sharex=True)

# # hm0_ymax=6.0
# # pr_max=20
# # ax[0].plot(dscp.time,dscp.Height_Hm0, lw=3)
# # ax[0].fill_between([t_st,t_en],[0,0],[hm0_ymax,hm0_ymax],alpha=0.2, color='pink')
# # ax[0].set_ylim(0,hm0_ymax)

# # img=ax[1].pcolormesh(dscp.time, 1/dscp.WaveSpectra_Frequency, dscp.WaveSpectra_Vel, shading='auto', cmap='jet', vmin=0, vmax=1)
# # ax[1].set_ylim(pr_st,pr_max)
# # ax[1].fill_between([t_st,t_en],[pr_st,pr_st],[pr_max,pr_max],alpha=0.4, color='pink')
# # box=ax[0].get_position()
# # ax[0].set_position([box.x0, box.y0, box.width * 0.8, box.height*1])
# # cbar=plt.colorbar(img)

# # imgaz=ax[2].pcolormesh(dscp.time, 1/dscp.DirectionalSpectra_Frequency, dscp.Direction, shading='auto', cmap='jet', vmin=0, vmax=360)
# # cbar=plt.colorbar(imgaz)
# # ax[2].set_ylim(pr_st,pr_max)
# # ax[2].set_ylabel('period (s)',fontweight='bold')
# # ax[1].set_ylabel('period (s)',fontweight='bold')
# # ax[1].set_title('spectral amplitude (m2/Hz)',fontweight='bold')
# # ax[2].set_title(' azimuth',fontweight='bold')
# # ax[0].set_title('SWH (m)',fontweight='bold')
# # ax[0].grid()
# # ax[1].grid()
# # ax[2].grid()

# # # Specify the filename and format (jpg)
# # plt.savefig('spectra_with_time.jpg', format='jpg')

# # # Close the plot (optional)
# # # plt.close()


# # fr_en=1/5
# # fr_st=1/30

# # t_st=np.datetime64('2023-05-30')
# # t_en=np.datetime64('2023-06-02')
# # tav_st=np.datetime64('2023-05-31')
# # tav_en=np.datetime64('2023-06-01')
# # dscaz=dsc.sel(time=slice(t_st,t_en),ASTSpectra_Frequency=slice(fr_st, fr_en))
# # dscazav=dsc.sel(time=slice(tav_st,tav_en),ASTSpectra_Frequency=slice(fr_st, fr_en))
# # period_azimuth_amplitude=dscazav.ASTSpectra_Energy.mean(axis=0)
# # # period_azimuth_amplitude=period_azimuth_amplitude
# # len(dscaz.ASTSpectra_Frequency)
# # len(dscaz.ASTSpectra_Frequency)


# # plt.rcParams.update({'font.size': 26})
# # plt.rcParams.update({'font.weight': 'bold'})

# # # figure('spectra_dir_with_time')
# # # fig,ax=plt.subplots(3,1, figsize=(30,15), sharex=True)
# # fig,ax=plt.subplots(3,1, figsize=(30,15))
# # fr_en=1/5
# # fr_st=1/30

# # # graph 1
# # hm0_ymax=3.5
# # ax[0].plot(dscaz.time,dscaz.Height_Hm0, lw=3)
# # box=ax[0].get_position()
# # ax[0].set_position([box.x0, box.y0, box.width * 0.8, box.height*1])
# # ax[0].set_ylim(0, hm0_ymax)
# # ax[0].set_xlim(t_st,t_en)
# # ax[0].set_title('SWH (m)',fontweight='bold')
# # ax[0].grid()
# # # ax[0].add_patch(plt.Rectangle(tav_st,0.5,(tav_en-tav_st),2.5))
# # ax[0].fill_between([tav_st,tav_en],[0,0],[hm0_ymax,hm0_ymax],alpha=0.2, color='pink')
# # # graph 2 
# # # dscp=dsc.sel(DirectionalSpectra_Frequency=slice(fr_st, fr_en))
# # img=ax[1].pcolormesh(dscaz.time, 1/dscaz.DirectionalSpectra_Frequency, dscaz.Direction, shading='auto', cmap='jet', vmin=0, vmax=360)
# # # box=ax[0].get_position()
# # # ax[1].set_position([box.x1, box.y1, box.width * 0.8, box.height*1])
# # ax[1].set_xlim(t_st,t_en)
# # cbar=plt.colorbar(img)
# # ax[1].set_ylim(1/fr_en, 1/fr_st)
# # ax[1].set_ylabel('period (s)',fontweight='bold')
# # ax[1].set_title('Azimuth',fontweight='bold')
# # ax[1].grid()

# # # graph 3
# # imgaz=ax[2].pcolormesh(1/dscaz.ASTSpectra_Frequency.values, dscaz.ASTSpectra_Direction.values,period_azimuth_amplitude, shading='auto', cmap='jet', norm=LogNorm())
# # cbar=plt.colorbar(imgaz)
# # ax[2].set_xlim(1/fr_en, 20)
# # ax[2].set_xlabel('period (s)',fontweight='bold')
# # ax[2].set_ylabel('azimuth (\u00b0)',fontweight='bold')
# # ax[2].set_title('spectral amplitude (m2/Hz)',fontweight='bold')
# # ax[2].grid()
# # # plt.tight_layout()

# # plt.savefig('directional_spectra_zoom.jpg', format='jpg')



# # # %matplotlib notebook
# # %matplotlib inline

# # fig,ax=plt.subplots(2,1, figsize=(30,15))
# # fr_en=1/2
# # fr_st=1/20
# # ax[0].plot(dscp.time,dscp.Height_Hm0)

# # dscp=dsc.sel(DirectionalSpectra_Frequency=slice(fr_st, fr_en))
# # img=ax[1].pcolormesh(dscp.time, 1/dscp.DirectionalSpectra_Frequency, dscp.Direction, shading='auto', cmap='jet', vmin=0, vmax=360)
# # ax[0].set_xlim(np.datetime64('2023-06-16'),np.datetime64('2023-06-18'))

# # cbar=plt.colorbar(img)






