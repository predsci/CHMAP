"""
Example to illustrate image-database writing, querying, updating, and deleting.

The first portion of this code follows example01.  Then we download and log
to the database 9 images.  This establishes enough entries to demonstrate
database querying, updating, and deleting.

"""

import numpy as np
import datetime

from astropy.time import Time, TimeDelta
import astropy.units as u
from sunpy.time import TimeRange

from helpers import drms_helpers, vso_helpers
from settings.app import App
from modules.misc_funs import cluster_meth_1
from modules.DB_classes import *
from modules.DB_funs import init_DB_conn, query_euv_images, add_image2session, update_image_val, remove_euv_image

import pandas as pd

from sqlalchemy import and_


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

f_list = [fs, fa, fb]
# check for empty instrument results
f_list = [elem for elem in f_list if len(elem)>0]

# Get the decimal time of the center of the time_range (Julian Date)
deltat = TimeDelta(0.0, format='sec')
time0 = Time(time_range.center, scale='utc', format='datetime') + deltat
jd0 = time0.jd

print(time0)

# build a numpy array to hold all of the times and time_deltas
# first create a list of dataframes
results = [fs.jd.values, fa.jd.values, fb.jd.values]
# check for empty instrument results
results = [elem for elem in results if len(elem)>0]
sizes = []
for result in results:
    sizes.append(len(result))
time_delta = np.ndarray(tuple(sizes), dtype='float64')

# Now loop over all the image pairs to select the "perfect" group of images.
imins = cluster_meth_1(results=results, jd0=jd0)
# consider also returning a clustering-algorithm name to store in DB?
# maybe saving the constituent images of a map is enough?

# setup database connection
use_db = "sqlite"
sqlite_filename = "dbtest.db"
db_session = init_DB_conn(db_name=use_db, chd_base=Base, sqlite_fn=sqlite_filename)

# setup a longer list of imins to test multiple cluster download and enter into DB
    # three instruments
# test = [imins, ([155], [6], [6]), ([455], [18], [18])]
    # one instrument
test = [imins, ([155],), ([455], )]

# print the selection, download if desired.
download = True
overwrite = False
verbose = True
# download and enter into database
for imin in test:
    for ii in range(0, len(imin)):
        instrument = f_list[ii]['instrument'][0]
        spacecraft = f_list[ii]['spacecraft'][0]
        image_num = imin[ii][0]
        if download:
            if instrument=="AIA":
                print("Downloading AIA: ", f_list[ii].iloc[image_num].time)
                subdir, fname = s12.download_image_fixed_format(f_list[ii].iloc[image_num], data_dir, update=True,
                                                                overwrite=overwrite, verbose=verbose)
            elif instrument=="EUVI" and spacecraft=="STEREO_A":
                print("Downloading EUVI A: ", f_list[ii].iloc[image_num].time)
                subdir, fname = euvi.download_image_fixed_format(f_list[ii].iloc[image_num], data_dir, compress=True,
                                                                 overwrite=overwrite, verbose=verbose)
            elif instrument=="EUVI" and spacecraft=="STEREO_B":
                print("Downloading EUVI B: ", f_list[ii].iloc[image_num].time)
                subdir, fname = euvi.download_image_fixed_format(f_list[ii].iloc[image_num], data_dir, compress=True,
                                                                 overwrite=overwrite, verbose=verbose)
            else:
                print("Instrument ", instrument, " does not yet have a download function.  SKIPPING DOWNLOAD ")
                continue

            # use the downloaded image to extract metadata and write a row to the database (session)
            db_session = add_image2session(data_dir=data_dir, subdir=subdir, fname=fname, db_session=db_session)


# commit the changes to the DB, this also assigns auto-incrementing primekeys 'id'
db_session.commit()

# query_EUV_images function:
# requires time_min and time_max (datetime).  do we need to code 'jd' time option?
query_time_min = datetime.datetime(2014, 4, 13, 19, 35, 0)
query_time_max = datetime.datetime(2014, 4, 13, 19, 37, 0)
print("Query DB for downloaded images with timestamps between " + str(query_time_min) + " and " + str(query_time_max))
test_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)
# returns info on 3 images
print(test_pd)

# query specific instrument
query_time_min = datetime.datetime(2014, 4, 13, 10, 0, 0)
instrument = ("AIA", )
test_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max, instrument=instrument)

# query specific wavelength
wavelength = (195, )
test_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max, wavelength=wavelength)


# update_image_val function:
# currently requires a pandas-series object (like return of query_euv_images) to index
# into the DB. This semi-insures that there is not confusion about which row of the
# DB is to be updated.

# generate hdf file using some function like:
# hd_fname = process_image2hdf(test_pd.iloc[0])
hd_fname = "/Users/turtle/GitReps/CHD/test_data/hdf5/2014/04/13/sta_euvi_20140413T183530_195.hdf5"
# update database with file location
db_session = update_image_val(db_session=db_session, raw_series=test_pd.iloc[0], col_name="fname_hdf", new_val=hd_fname)

# also flag this image as 1 - 'verified good' (made-up example)
image_flag = 1
# update database with file location
db_session = update_image_val(db_session=db_session, raw_series=test_pd.iloc[0], col_name="flag", new_val=image_flag)


# remove_euv_image function:
# removes the files and then the corresponding DB row
exit_status, db_session = remove_euv_image(db_session=db_session, raw_series=test_pd.iloc[0])


db_session.close()