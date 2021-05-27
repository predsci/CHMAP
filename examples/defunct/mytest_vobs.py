import os
from astropy.time import Time
import astropy.units as u
from sunpy.time import TimeRange
from data.download import vso_helpers

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
interval_cadence = 1*u.hour
image_search_cadence = 12*u.second
wavelength=195

# generate the list of time intervals
full_range=TimeRange(time_start, time_end)
time_ranges=full_range.window(interval_cadence,interval_cadence)

# pick a time_range to experiement with
time_range=time_ranges[0]

# initialize the helper class for EUVI
euvi = vso_helpers.EUVI(verbose=True)

fa = euvi.query_time_interval(time_range, wavelength, craft='STEREO_A')
fb = euvi.query_time_interval(time_range, wavelength, craft='STEREO_B')

print(fa)
print()
print(fb)

print()
if len(fa) > 0:
    index_get=0
    subdir, fname = euvi.download_image_fixed_format(fa.iloc[index_get], data_dir, compress=True, verbose=True)
    subdir, fname = euvi.download_image_fixed_format(fb.iloc[index_get], data_dir, compress=True, verbose=True)



"""
ta, ja, ua = euvi.query_time_interval(time_range, wavelength, craft='STEREO_A')
tb, jb, ub = euvi.query_time_interval(time_range, wavelength, craft='STEREO_B')

print(ta)
print(ja)
print(ua)


print(tb)
print(jb)
print(ub)
"""

"""
import pandas as pd
# make a dict out of the time results
da = {'time':ta, 'jd':ja, 'url':ua}
# turn this into a pandas dataframe
df = pd.DataFrame(data=da)
# slice it at a specific index
df.iloc[2]
"""


"""
instrument='euvi'

#result = Fido.search(a.Time(time_range), a.Instrument(instrument), a.Wavelength(195*u.angstrom), a.vso.Detector('STEREO_A'))
result = Fido.search(a.Time(time_range), a.Instrument(instrument), a.Wavelength(195*u.angstrom), vso.attrs.Source('STEREO_A'))

client = vso.VSOClient()
qr = client.query( vso.attrs.Time(time_range), vso.attrs.Instrument('SECCHI'), a.vso.Detector('STEREO_A'))

qr = client.search( vso.attrs.Time(time_range), vso.attrs.Detector('EUVI'), a.vso.Wavelength(195*u.angstrom), vso.attrs.Source('STEREO_A'))

result = Fido.search(a.Time(time_range), a.Instrument(instrument), a.Wavelength(195*u.angstrom), vso.attrs.Source('STEREO_A')))
"""


"""
# get the actual vso response from fido search here
d = result[0,:].get_response(0)


result[0,:].get_response(0)


tmp.build_table()['Start Time'].data
"""