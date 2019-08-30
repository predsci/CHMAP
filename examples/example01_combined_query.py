"""
Short example to illustrate multi-spacecraft image query, selection, and download.

Here a time interval is built, segmented and used to get data.

Right now the example only works for 3 spacecraft data being available. It should
be a short fix to make it work for arbitrary numbers/pairs.
"""
import numpy as np
from astropy.time import Time, TimeDelta
import astropy.units as u
from sunpy.time import TimeRange
from helpers import drms_helpers, vso_helpers
from settings.app import App

import pandas as pd


# Get the data dir from the installed app settings.
data_dir = App.RAW_DATA_HOME

# Test a time interval
#  - The example code below requires data being available for all three (for now).
#  - It will fail if not between 06/10/2010 and 08/18/2014).
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

# Get the decimal time of the center of the time_range (Julian Date)
deltat = TimeDelta(0.0, format='sec')
time0 = Time(time_range.center, scale='utc', format='datetime') + deltat
jd0 = time0.jd

print(time0)

# build a numpy array to hold all of the times and time_deltas
# first create a list of dataframes
results = [fs.jd.values, fa.jd.values, fb.jd.values]
sizes = []
for result in results:
    sizes.append(len(result))
time_delta = np.ndarray(tuple(sizes), dtype='float64')

# Now loop over all the image pairs to select the "perfect" group of images.
# Here we ,minimize an arbitrary weight function based on the time differences between the
# group (relative) and the requested central time (absolute).
# This loop needs to be rewritten to be agnostic to the spacecraft type and number of spacecraft.
pow1 = 2
pow2 = 2
if time_delta.ndim == 3:
    for idx, vals in np.ndenumerate(time_delta):
        vi = results[0][idx[0]]
        vj = results[1][idx[1]]
        vk = results[2][idx[2]]

        # this looks at the relative difference between the three pairs of images
        time_delta[idx] = abs(vi - vj)**pow1 + abs(vi - vk)**pow1 + abs(vj - vk)**pow1

        # this looks at the difference between a given image and the central time of the interval (jd0)
        time_delta[idx] = time_delta[idx] + abs(vi - jd0)**pow2 + abs(vj - jd0)**pow2 + abs(vk - jd0)**pow2

# Figure out the index
imins = np.where(time_delta == np.min(time_delta))

# print the selection, download if desired.
download = True
overwrite = False
verbose = True
for imin in range(0, len(imins[0])):
    i = imins[0][imin]
    j = imins[1][imin]
    k = imins[2][imin]
    print(i, j, k, fs.iloc[i].time, fa.iloc[j].time, fb.iloc[k].time, time_delta[i, j, k])
    if download:
        print("Downloading AIA: ", fs.iloc[i].time)
        subdir, fname = s12.download_image_fixed_format(fs.iloc[i], data_dir, update=True, overwrite=overwrite,
                                                        verbose=verbose)
        print("Downloading EUVI A: ", fa.iloc[j].time)
        subdir, fname = euvi.download_image_fixed_format(fa.iloc[j], data_dir, compress=True, overwrite=overwrite,
                                                         verbose=verbose)
        print("Downloading EUVI B: ", fb.iloc[k].time)
        subdir, fname = euvi.download_image_fixed_format(fb.iloc[k], data_dir, compress=True, overwrite=overwrite,
                                                         verbose=verbose)
