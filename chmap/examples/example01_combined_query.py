"""
Short example to illustrate multi-spacecraft image query, selection, and download.

Here a time interval is built, segmented and used to get data.

The example should work for any number of spacecraft data pairs (1,2,3,4..).

ToDo: right now it only works when SDO spacecraft data is available. It should
be a short fix in the drms module to allow empty queries to be returned.
"""
import pandas as pd

from astropy.time import Time, TimeDelta
import astropy.units as u
from sunpy.time import TimeRange

from chmap.settings.app import App
from chmap.data.download import drms_helpers, vso_helpers
from chmap.data.download.euv_utils import get_image_set

# Get the data dir from the installed app settings.
data_dir = App.RAW_DATA_HOME

# Test a time interval
#  - The example code below requires data being available for SDO (for now).
#  - It will fail if not between 06/10/2010 and now).
time_start = Time('2014-04-13T18:04:00.000', scale='utc')
time_end = Time('2014-04-13T20:04:00.000', scale='utc')

# query parameters
interval_cadence = 2*u.hour
aia_search_cadence = 12*u.second
wave_aia = 193
wave_euvi = 195

# generate the list of time intervals
full_range = TimeRange(time_start, time_end)
time_ranges = full_range.window(interval_cadence, interval_cadence)

# initialize the jsoc drms helper for aia.lev1_euv_12
s12 = drms_helpers.S12(verbose=True)

# initialize the helper class for EUVI
euvi = vso_helpers.EUVI(verbose=True)

# pick a time_range to experiement with
time_range = time_ranges[0]

# query the jsoc for SDO/AIA
fs = s12.query_time_interval(time_range, wave_aia, aia_search_cadence)

# query the VSO for STA/EUVI and STB/EUVI
fa = euvi.query_time_interval(time_range, wave_euvi, craft='STEREO_A')
fb = euvi.query_time_interval(time_range, wave_euvi, craft='STEREO_B')

# merge the query results into one dataframe
df = pd.concat([fs, fa, fb], axis=0)

# Get central time the time_range, adding an optional offset
deltat = TimeDelta(0.0, format='sec')
time0 = Time(time_range.center, scale='utc', format='datetime') + deltat

# find the best "matching" set of images from this query (those near time0)
dfm = get_image_set(df, time0)

# print the selection, download if desired by iterating over the match dataframe.
download = True
overwrite = False
verbose = True
if download:
    for index, row in dfm.iterrows():
        print("Downloading : ", row['spacecraft'], row['instrument'], row['filter'], row['time'])
        if row['instrument'] == 'AIA':
            subdir, fname = s12.download_image_fixed_format(
                row, data_dir, update=True, overwrite=overwrite, verbose=verbose)
        elif row['instrument'] == 'EUVI':
            subdir, fname = euvi.download_image_fixed_format(
                row, data_dir, compress=True, overwrite=overwrite, verbose=verbose)
