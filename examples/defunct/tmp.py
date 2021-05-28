import drms
import os
import astropy
from astropy.time import Time
import astropy.units
import astropy.units as u
from sunpy.time import TimeRange

"""
Notes:
- 'Z' postfix on timestamps refers to UTC time-zone, and this seems to be the default format 
   of the JSOC. Be careful if interchanging with '_TAI' prefixes.
"""


data_basedir=os.path.join(os.sep+'Users','cdowns','work','ch_evolution','data_tests')
if not os.path.isdir(data_basedir):
    raise RuntimeError('Base Data Directory Does not Exist!')

data_dir=os.path.join(data_basedir,'drms_rawdata')
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)


# JSOC export protocol (as-is does not require official exports)
export_protocol = 'as-is'

email='cdowns@predsci.com'

series = 'aia.lev1_euv_12s'
zone_postfix='Z'
wavelength=193
interval_str = '/1m'
cadence_str = '@12s'

segments = ['image']
segstr = ', '.join(segments)

keys=['T_REC', 'DATE__OBS', 'T_OBS', 'WAVELNTH', 'EXPTIME', 'QUALITY', 'DATAMEAN']
keystr = ', '.join(keys)

filters=['QUALITY=0','EXPTIME>1.0']

client=drms.Client(verbose=True)

# Filter Query String
if len(filters) > 0:
    filter_str = '[? '+' ?][? '.join(filters)+' ?]'
else:
    filter_str=''

def get_jsoc_interval_format(time_interval):
    """
    Return a string with the jsoc style way of specifying a time interval (e.g. 12s, 2h, 2d)
    - the input time interval a length of time specified as an astropy unit type.
    """
    secs=time_interval.to(astropy.units.second).value
    if secs < 60.:
        return str(secs)+'s'
    elif secs < 3600.:
        return str(time_interval.to(astropy.units.min).value)+'m'
    elif secs < 86400.0:
        return str(time_interval.to(astropy.units.hour).value)+'h'
    else:
        return str(time_interval.to(astropy.units.day).value)+'d'

def jsoc_build_time_string_from_range(time_range, image_search_cadence):
    """
    Quick function to build a jsoc time string from time range
    """
    interval_str = '/'+get_jsoc_interval_format(time_range.seconds)
    cadence_str = '@'+get_jsoc_interval_format(image_search_cadence)
    time_start = Time( time_range.start, format='datetime', scale='utc')
    date_start = time_start.isot+zone_postfix
    return '%s%s%s' % (date_start, interval_str, cadence_str)


def jsoc_query_time_interval(time_range, wavelength, aia_search_cadence):
    """
    Quick function to query the JSOC for all matching images at a certain wavelength over
    a certain interval
    - returns the drms pandas dataframes of the keys and segments
    - if no images, len(keys) and len(segs) will be zero
    """
    wave_str=str(wavelength)
    interval_str = '/'+get_jsoc_interval_format(time_range.seconds)
    cadence_str = '@'+get_jsoc_interval_format(aia_search_cadence)
    time_start = Time( time_range.start, format='datetime', scale='utc')
    date_start = time_start.isot+zone_postfix

    time_str = '%s%s%s' % (date_start, interval_str, cadence_str)
    query_string = '%s[%s][%s]%s{%s}' % (series, time_str, wave_str, filter_str, ','.join(segments))

    keys, segs = client.query(query_string, key=keystr, seg=segstr)

    return keys, segs


# test a time interval
time_start=Time( '2011-02-01T00:00:00.000', scale='utc')
time_end  =Time( '2011-02-04T00:00:00.000', scale='utc')
interval_cadence = 2*u.min
aia_search_cadence = 12*u.second

# generate the list of time intervals
full_range=TimeRange(time_start, time_end)
time_ranges=full_range.window(interval_cadence,interval_cadence)

# pick a time range to experiement with
time_range=time_ranges[0]

print(len(time_ranges))
print(time_range)

#print(jsoc_build_time_string_from_range(time_range,aia_search_cadence))

keys, segs = jsoc_query_time_interval(time_range, 193, aia_search_cadence)

# = client.query(query_string, key=keystr, seg=segstr)

if len(keys) > 0:
    timestrings = keys['T_OBS'].get_values().astype('str')
    time_array = Time(timestrings, scale='utc')
    jds = time_array.jd
else:
    timestrings = []
    time_array = []
    jds = []

from utilities.file_io import io_helpers

# downlaod manually via the urllib
if len(keys) > 0:
    index_get=0
    url = 'http://jsoc.stanford.edu' + segs['image'][index_get]
    dtime = time_array[index_get].datetime
    prefix='_'.join(series.split('.'))
    postfix=str(wavelength)
    ext='fits'
    dir, fname = io_helpers.construct_path_and_fname(data_dir, dtime, prefix, postfix, ext)
    fpath=dir+os.sep+fname
    io_helpers.download_url(url, fpath)


"""
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


