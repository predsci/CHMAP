import os
import numpy as np
from astropy.time import Time, TimeDelta
import astropy.units as u
from sunpy.time import TimeRange
from chmap.data.download import drms_helpers, vso_helpers

# define directories for testing
data_basedir=os.path.join(os.sep+'Users','cdowns','work','ch_evolution','data_tests')
if not os.path.isdir(data_basedir):
    print(data_basedir)
    raise RuntimeError('Base Data Directory Does not Exist!')
data_dir=os.path.join(data_basedir,'drms_rawdata')
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

# test a time interval
time_start=Time( '2013-02-02T23:04:00.000', scale='utc')
time_end  =Time( '2013-02-04T01:04:00.000', scale='utc')

#time_start=Time( '2017-02-01T11:00:00.000', scale='utc')
#time_end  =Time( '2017-02-04T13:00:00.000', scale='utc')

# query parameters
interval_cadence = 2*u.hour
aia_search_cadence = 12*u.second
wave_aia=193
wave_euvi=195

# generate the list of time intervals
full_range=TimeRange(time_start, time_end)
time_ranges=full_range.window(interval_cadence,interval_cadence)

# initialize the jsoc drms helper for aia.lev1_euv_12
s12 = drms_helpers.S12(verbose=True)

# initialize the helper class for EUVI
euvi = vso_helpers.EUVI(verbose=True)


# pick a time_range to experiement with
time_range=time_ranges[0]

#print(len(time_ranges))
#print(time_range)

# query the jsoc for AIA
fs = s12.query_time_interval(time_range, wave_aia, aia_search_cadence)

# query the VSO for EUVI
fa = euvi.query_time_interval(time_range, wave_euvi, craft='STEREO_A')
fb = euvi.query_time_interval(time_range, wave_euvi, craft='STEREO_B')

# get the center of the time_range
deltat=TimeDelta(0.0,format='sec')
time0 = Time(time_range.center, scale='utc', format='datetime')+deltat
jd0 = time0.jd

print(time0)

# build a numpy array to hold all of the time deltas
results = [fs.jd.values, fa.jd.values, fb.jd.values]
sizes = []
for result in results:
    sizes.append(len(result))
time_delta = np.ndarray(tuple(sizes), dtype='float64')

pow1=2
pow2=2

if time_delta.ndim == 3:
    for idx, vals in np.ndenumerate(time_delta):
        vi=results[0][idx[0]]
        vj=results[1][idx[1]]
        vk=results[2][idx[2]]
        #time_delta[idx] = abs(vi-vj) + abs(vi-vk) + abs(vj-vk)
        time_delta[idx] = abs(vi-vj)**pow1+ abs(vi-vk)**pow1 + abs(vj-vk)**pow1

        #time_delta[idx] = time_delta[idx] + abs(vi - jd0) + abs(vj - jd0) + abs(vk - jd0)
        time_delta[idx] = time_delta[idx] + abs(vi - jd0)**pow2 + abs(vj - jd0)**pow2 + abs(vk - jd0)**pow2
        #print(vi,vj,vk)
        #time_delta[idx] = results[0][i]

imins = np.where(time_delta == np.min(time_delta))


download=True

for imin in range(0,len(imins[0])):
    i = imins[0][imin]
    j = imins[1][imin]
    k = imins[2][imin]
    print(imin, fs.iloc[i].time, fa.iloc[j].time, fb.iloc[k].time, time_delta[i,j,k])
    if download:
        print("Downloading AIA: ", fs.iloc[i].time)
        subdir, fname = s12.download_image_fixed_format(fs.iloc[i], data_dir, update=True, verbose=True)
        print("Downloading EUVI A: ", fa.iloc[j].time)
        subdir, fname = euvi.download_image_fixed_format(fa.iloc[j], data_dir, compress=True, verbose=True)
        print("Downloading EUVI B: ", fb.iloc[k].time)
        subdir, fname = euvi.download_image_fixed_format(fb.iloc[k], data_dir, compress=True, verbose=True)



"""

for k in len(result)

print(len(fs))
print(len(fa))
print(len(fb))
"""
