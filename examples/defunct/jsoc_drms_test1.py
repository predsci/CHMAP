"""
This test is built off of the JSOC DRMS tutorial here: http://drms.readthedocs.io/en/stable/tutorial.html
Basically I want to see if i can manipulate the AIA queries and records the way i want

I installed it using conda with "conda-forge" enabled (like sunpy)
conda search drms
conda install drms
"""

import drms
import os
import sys


# Use 'as-is' instead of 'fits', if record keywords are not needed in the
# FITS header. This greatly reduces the server load!
#export_protocol = 'fits'
export_protocol = 'as-is'

series = 'aia.lev1_euv_12s'

wave_str='193'
#date0 = '2012-07-12T16:00:00_TAI'
time_fmt = 'Z' # String that indicates the time format (Z is UTC, can also be _TAI)
date0 = '2012-07-12T16:00:00Z'
interval_str = '/1m'
cadence_str = '@12s'

filters=['QUALITY=0','EXPTIME>1.0']

segments = ['image']
segstr = ', '.join(segments)

keys=['T_REC', 'DATE__OBS', 'T_OBS', 'WAVELNTH', 'EXPTIME', 'QUALITY', 'DATAMEAN']
keystr = ', '.join(keys)

print(keystr)



email='cdowns@predsci.com'

data_basedir=os.path.join(os.sep+'Users','cdowns','work','ch_evolution','data_tests')
if not os.path.isdir(data_basedir):
    raise RuntimeError('Base Data Directory Does not Exist!')

data_dir=os.path.join(data_basedir,'drms_rawdata')
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)




##c.check_email(email)
#if not c.check_email(email):
#    raise RuntimeError('Email address is not valid or not registered.')


c=drms.Client(verbose=True)


allkeys = c.keys(series)
print(allkeys)
#sys.exit()


# Time Query String
tstr = '%s%s%s' % (date0,interval_str,cadence_str)

# Filter Query String
if len(filters) > 0:
    fstr = '[? '+' ?][? '.join(filters)+' ?]'
else:
    fstr=''

# Full Query String
expstr = '%s[%s][%s]%s{%s}' % (series, tstr, wave_str, fstr, ','.join(segments))

query_str = '%s[%s]' % (series, tstr)

print(tstr)
print(fstr)
print(expstr)

print('')
print('Data export query:\n  %s\n' % expstr)

k, s = c.query(expstr, key=keystr, seg=segstr)
print(k)
print(s)

ind_select = 0
url = 'http://jsoc.stanford.edu' + s.image[ind_select]
print(url)

"""
# manually download the as-is file and check it out
import astropy.utils.data
tmpname = astropy.utils.data.download_file(url, cache=False)
print(tmpname)
import shutil
shutil.move(tmpname, os.path.join(data_dir,'aia_20120712T160001Z.193_drms_protocol_url.fits'))
"""


# now see if you can recover the correct header info
expsingle = '%s[%s][%s]{%s}' % (series, k.T_REC[ind_select], wave_str, ','.join(segments))
k = c.query(expsingle, key=allkeys)
print(k)

"""
print(k.T_REC)
print(k.T_REC[ind_select])

expsingle = '%s[%s][%s]{%s}' % (series, k.T_REC[ind_select], wave_str, ','.join(segments))
print(expsingle)
r = c.export( expstr, email=email, protocol='fits')


print(r)
print(r.data.filename)
print(r.urls.url[0])
r.download(data_dir, 0)

"""
