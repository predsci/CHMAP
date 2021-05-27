import os
from astropy.time import Time
import astropy.units as u
from sunpy.time import TimeRange
from data.download import drms_helpers

# define directories for testing
data_basedir=os.path.join(os.sep+'Users','cdowns','work','ch_evolution','data_tests')
if not os.path.isdir(data_basedir):
    print(data_basedir)
    raise RuntimeError('Base Data Directory Does not Exist!')
data_dir=os.path.join(data_basedir,'drms_rawdata')
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

# test a time interval
time_start=Time( '2011-02-02T00:00:00.000', scale='utc')
time_end  =Time( '2011-02-04T00:00:00.000', scale='utc')

# query parametersr
interval_cadence = 2*u.min
image_search_cadence = 12*u.second
wavelength=193

# generate the list of time intervals
full_range=TimeRange(time_start, time_end)
time_ranges=full_range.window(interval_cadence,interval_cadence)

# pick a time_range to experiement with
time_range=time_ranges[0]

#print(len(time_ranges))
#print(time_range)

# initialize the jsoc drms helper for aia.lev1_euv_12
s12 = drms_helpers.S12(verbose=True)

# query the jsoc
ds = s12.query_time_interval(time_range, wavelength, image_search_cadence)

print(ds)

# download a file as a test
if len(ds)>0:
    index_get=0
    s12.download_image_fixed_format(ds.iloc[index_get], data_dir, update=True, verbose=True)


"""
# query the jsoc
keys, segs = s12.query_time_interval(time_range, wavelength, image_search_cadence)

# parse the output
timestrings, jds = drms_helpers.parse_query_times(keys)

print(timestrings)
#print(jds)

# download a file as a test
if len(keys)>0:
    index_get=0
    s12.download_image_fixed_format(keys.iloc[index_get], segs.iloc[index_get], data_dir, update=True, verbose=True)
"""


"""
if len(keys) > 0:
    timestrings = keys['T_OBS'].get_values().astype('str')
    time_array = Time(timestrings, scale='utc')
    jds = time_array.jd
else:
    timestrings = []
    time_array = []
    jds = []
"""

"""
# download result manually via the urllib
if len(keys) > 0:
    index_get=0
    url = drms_helpers.jsoc_url + segs['image'][index_get]
    dtime = time_array[index_get].datetime
    prefix='_'.join(s12.series.split('.'))
    postfix=str(wavelength)
    ext='fits'
    dir, fname = misc_helpers.construct_path_and_fname(data_dir, dtime, prefix, postfix, ext)
    fpath=dir+os.sep+fname
    misc_helpers.download_url(url, fpath)
"""

"""
# example of how to actually use the drms export function 
if len(keys) > 0:
    index_get=0
    time_str = timestrings[index_get]
    wave_str = str(wavelength)
    expsingle = '%s[%s][%s]{%s}' % (series, time_str, wave_str, ','.join(segments))
    print(expsingle)
    exp_request = client.export(expsingle, email=email, protocol='as-is')
    exp_request.download(data_dir)
"""


"""
print(keys)
print(segs)
print(timestrings)
print(jds)
"""

"""

print(get_jsoc_interval_format(0.9*astropy.units.s))
print(get_jsoc_interval_format(0.9*astropy.units.min))
print(get_jsoc_interval_format(0.9*astropy.units.hour))
print(get_jsoc_interval_format(0.9*astropy.units.day))
print(get_jsoc_interval_format(99*astropy.units.day))

"""


