"""
Example to illustrate image-database writing, querying, updating, and deleting.

The first portion of this code follows example01.  Then we download and log
to the database 9 images.  This establishes enough entries to demonstrate
database querying, updating, and deleting.

"""

import os
import datetime
import numpy as np

from astropy.time import Time
import astropy.units as u

from helpers import drms_helpers, vso_helpers
from settings.app_JT_Q import App
from modules.misc_funs import cluster_meth_1, list_available_images
from database.db_classes import *
from database.db_funs import init_db_conn, query_euv_images, add_image2session


# Specify directories on mounted Q home
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
db_home_dir  = App.DATABASE_HOME


# setup database connection
use_db = "sqlite"
sqlite_filename = App.DATABASE_FNAME
sqlite_path = os.path.join(db_home_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=Base, sqlite_path=sqlite_path)

# initialize the jsoc drms helper for aia.lev1_euv_12
s12 = drms_helpers.S12(verbose=True)

# initialize the helper class for EUVI
euvi = vso_helpers.EUVI(verbose=True)

# query parameters
interval_cadence = 2*u.hour
aia_search_cadence = 12*u.second
wave_aia = 193
wave_euvi = 195

# --- to get 1-month data at 2-hr cadence, first loop over target times ---
# define download period
# period_start = Time('2011-01-01T00:00:00.000', scale='utc')
period_start = Time('2011-02-01T00:00:00.000', scale='utc')
period_end = Time('2012-01-01T00:00:00.000', scale='utc')
# define image search time-width
del_interval = 30*u.minute
# define target times over download period using interval_cadence
mean_times = Time(np.arange(period_start, period_end, interval_cadence))
print("First five target times: \n")
print(mean_times[0:5])

for time0 in mean_times:
    # Define a time interval for image searching
    time_start = time0 - del_interval
    time_end = time0 + del_interval
    # query various instrument repos for available images.
    f_list = list_available_images(time_start=time_start, time_end=time_end, euvi_interval_cadence=interval_cadence,
                                   aia_search_cadence=aia_search_cadence, wave_aia=wave_aia, wave_euvi=wave_euvi)

    jd0 = time0.jd
    print(time0)

    # Now loop over all the image pairs to select the "perfect" group of images.
    imin = cluster_meth_1(f_list=f_list, jd0=jd0)
    # consider also returning a clustering-algorithm name to store in DB?
    # maybe saving the constituent images of a map is enough?

    # print the selection, download if desired.
    download = True
    overwrite = False
    verbose = True
    # download and enter into database
    for ii in range(0, len(imin)):
        instrument = f_list[ii]['instrument'][0]
        spacecraft = f_list[ii]['spacecraft'][0]
        image_num = imin[ii][0]
        if download:
            if instrument=="AIA":
                print("Downloading AIA: ", f_list[ii].iloc[image_num].time)
                subdir, fname, download_flag = s12.download_image_fixed_format(f_list[ii].iloc[image_num], raw_data_dir,
                                                    update=True, overwrite=overwrite, verbose=verbose)
            elif instrument=="EUVI" and spacecraft=="STEREO_A":
                print("Downloading EUVI A: ", f_list[ii].iloc[image_num].time)
                subdir, fname, download_flag = euvi.download_image_fixed_format(f_list[ii].iloc[image_num],
                                                    raw_data_dir, compress=True, overwrite=overwrite, verbose=verbose)
            elif instrument=="EUVI" and spacecraft=="STEREO_B":
                print("Downloading EUVI B: ", f_list[ii].iloc[image_num].time)
                subdir, fname, download_flag = euvi.download_image_fixed_format(f_list[ii].iloc[image_num],
                                                    raw_data_dir, compress=True, overwrite=overwrite, verbose=verbose)
            else:
                print("Instrument ", instrument, " does not yet have a download function.  SKIPPING DOWNLOAD ")
                # update download results

                continue

            if fname is None:
                # download failed. do not attempt to add to DB
                continue
            # use the downloaded image to extract metadata and write a row to the database (session)
            db_session = add_image2session(data_dir=raw_data_dir, subdir=subdir, fname=fname, db_session=db_session)


    print("\nDownloads complete with all images added to DB session.  \nNow commit session changes to DB.\n")
    # commit the changes to the DB, this also assigns auto-incrementing prime-keys 'data_id'
    db_session.commit()

# query_EUV_images function:
# requires time_min and time_max (datetime).  do we need to code 'jd' time option?
query_time_min = datetime.datetime(2014, 4, 13, 19, 35, 0)
query_time_max = datetime.datetime(2014, 4, 13, 19, 37, 0)
print("\nQuery DB for downloaded images with timestamps between " + str(query_time_min) + " and " + str(query_time_max))
test_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)
# returns info on 3 images
print(test_pd)

# query specific instrument
query_time_min = datetime.datetime(2014, 4, 13, 10, 0, 0)
instrument = ("AIA", )
test_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max,
                           instrument=instrument)

# query specific wavelength
wavelength = (195, )
test_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max,
                           wavelength=wavelength)


db_session.close()
