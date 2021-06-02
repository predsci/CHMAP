"""
Example to illustrate image-database writing, querying, updating, and deleting.  Requires user to set
the location of data_dir to the reference_data/raw subdirectory.

This code relies on an SQLite DB to be present in the repo subdirectory 'reference_data/.
Also the corresponding fits files should be present in reference_data/raw/.
This code then demonstrate database querying and updating.

An example deletion is given, but full functionality cannot be demonstrated without deleting it.
"""
import sys
sys.path.append('/Users/tamarervin/work/chd')
import datetime
# import pandas as pd
import os

from settings.app import App
from chmap.database.db_classes import Base
from chmap.database.db_funs import init_db_conn, update_image_val, query_euv_images, pdseries_tohdf

# Assume that we are using the 'reference_data' setup supplied with repo
# manually set the data dir
raw_data_dir = os.path.join(App.APP_HOME, 'reference_data', 'raw')
hdf_data_dir = os.path.join(App.APP_HOME, 'reference_data', 'processed')
# manually set the database location using the installed app settings.
database_dir = os.path.join(App.APP_HOME, 'reference_data')


# setup database connection
use_db = "sqlite"
sqlite_filename = "dbtest.db"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=Base, sqlite_path=sqlite_path)

# query_EUV_images function:
# requires time_min and time_max (datetime).  do we need to code 'jd' time option?
query_time_min = datetime.datetime(2014, 4, 13, 19, 35, 0)
query_time_max = datetime.datetime(2014, 4, 13, 19, 37, 0)
print("\nQuery DB for downloaded images with timestamps between " + str(query_time_min) + " and " + str(query_time_max))
test_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)
# returns info on 3 images
print(test_pd)

# query specific instrument
print("\nQuery the reference database for all entries with instrument='AIA'.")
query_time_min = datetime.datetime(2014, 4, 13, 10, 0, 0)
instrument = ("AIA",)
test_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max,
                           instrument=instrument)
print(test_pd)

# query specific wavelength
wavelength = (195, 193, 12345)
print("\nQuery the reference database for all entries with wavelength==195 | wavelength==193 | wavelength==12345 .\n" +
      "This syntax can also be used for a multi-instrument search.")
test_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max,
                           wavelength=wavelength)
print(test_pd)

# query specific wavelength and instrument
wavelength = (195,)
instrument = ("EUVI-B",)
print("\nQuery for instrument='EUVI-B' and wavelength=195 .")
test_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max,
                           wavelength=wavelength, instrument=instrument)
print(test_pd)

# update_image_val function:
# currently requires a pandas-series object (like return of query_euv_images) to index
# into the DB. This semi-insures that there is not confusion about which row of the
# DB is to be updated.

# generate hdf file using some function like:
image_to_convert = test_pd.iloc[0]
hd_fname = pdseries_tohdf(image_to_convert)
#hd_fname = "2014/04/13/sta_euvi_20140413T183530_195.hdf5"
# update database with file location
db_session = update_image_val(db_session=db_session, raw_series=test_pd.iloc[0], col_name="fname_hdf", new_val=hd_fname)


# also flag this image as 1 - 'verified good' (made-up example)
image_flag = 1
# update database with file location
db_session = update_image_val(db_session=db_session, raw_series=test_pd.iloc[0], col_name="flag", new_val=image_flag)

# remove_euv_image function:
# removes the files and then the corresponding DB row
# this works, but has been commented because it will only work once
#exit_status, db_session = remove_euv_image(db_session=db_session, raw_series=test_pd.iloc[0], raw_dir=raw_data_dir, hdf_dir=hdf_data_dir)


db_session.close()
