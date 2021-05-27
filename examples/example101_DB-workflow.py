"""
Example to illustrate image-database writing, querying, updating, and deleting.

The first portion of this code follows example01.  Then we download and log
to the database 9 images.  This establishes enough entries to demonstrate
database querying, updating, and deleting.

"""
import sys
sys.path.append('/Users/tamarervin/work/chd')
import os
import datetime

from astropy.time import Time, TimeDelta
import astropy.units as u
from sunpy.time import TimeRange

from data.download import drms_helpers, vso_helpers
from settings.app import App
from modules.misc_funs import cluster_meth_1, list_available_images
from database.db_classes import Base
from database.db_funs import init_db_conn, query_euv_images, add_image2session, update_image_val, remove_euv_image, pdseries_tohdf


# to create the reference DB and files, set Create_Ref=True
Create_Ref = True

# Get the data dir from the installed app settings.
if Create_Ref:
    raw_data_dir = os.path.join(App.APP_HOME, "reference_data", "raw")
    hdf_data_dir = os.path.join(App.APP_HOME, "reference_data", "processed")
else:
    raw_data_dir = App.RAW_DATA_HOME
    hdf_data_dir = App.PROCESSED_DATA_HOME


# initialize the jsoc drms helper for aia.lev1_euv_12
s12 = drms_helpers.S12(verbose=True)

# initialize the helper class for EUVI
euvi = vso_helpers.EUVI(verbose=True)

# query parameters
interval_cadence = 2*u.hour
aia_search_cadence = 12*u.second
wave_aia = 193
wave_euvi = 195

# Define a time interval
time_start = Time('2011-02-14T18:04:00.000', scale='utc')
time_end = Time('2011-02-14T20:04:00.000', scale='utc')
# query various instrument repos for available images.
f_list = list_available_images(time_start=time_start, time_end=time_end, euvi_interval_cadence=interval_cadence,
                               aia_search_cadence=aia_search_cadence, wave_aia=wave_aia, wave_euvi=wave_euvi)

# Get the decimal time of the center of the time_range (Julian Date)
time_range = TimeRange(time_start, time_end)
deltat = TimeDelta(0.0, format='sec')
time0 = Time(time_range.center, scale='utc', format='datetime') + deltat
jd0 = time0.jd

print(time0)

# Now loop over all the image pairs to select the "perfect" group of images.
imins = cluster_meth_1(f_list=f_list, jd0=jd0)
# consider also returning a clustering-algorithm name to store in DB?
# maybe saving the constituent images of a map is enough?

# setup database connection
use_db = "sqlite"
sqlite_filename = "dbtest.db"
if Create_Ref:
    sqlite_path = os.path.join(App.APP_HOME, "test_data", sqlite_filename)
else:
    sqlite_path = os.path.join(App.DATABASE_HOME, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=Base, sqlite_path=sqlite_path)
print(db_session)

# setup a longer list of imins to test multiple cluster download and enter into DB
# Changing the time range above may cause this section to fail.
if len(f_list)==3:
    # three instruments
    test = [imins, ([155], [6], [6]), ([455], [18], [18])]
else:
    # one instrument (Assume that stereo A and B are down again)
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
                subdir, fname, exit_flag = s12.download_image_fixed_format(f_list[ii].iloc[image_num], raw_data_dir, update=True,
                                                                overwrite=overwrite, verbose=verbose)
            elif instrument=="EUVI" and spacecraft=="STEREO_A":
                print("Downloading EUVI A: ", f_list[ii].iloc[image_num].time)
                subdir, fname, exit_flag = euvi.download_image_fixed_format(f_list[ii].iloc[image_num], raw_data_dir, compress=True,
                                                                 overwrite=overwrite, verbose=verbose)
            elif instrument=="EUVI" and spacecraft=="STEREO_B":
                print("Downloading EUVI B: ", f_list[ii].iloc[image_num].time)
                subdir, fname, exit_flag = euvi.download_image_fixed_format(f_list[ii].iloc[image_num], raw_data_dir, compress=True,
                                                                 overwrite=overwrite, verbose=verbose)
            else:
                print("Instrument ", instrument, " does not yet have a download function.  SKIPPING DOWNLOAD ")
                continue

            # use the downloaded image to extract metadata and write a row to the database (session)
            db_session = add_image2session(data_dir=raw_data_dir, subdir=subdir, fname=fname, db_session=db_session)


print("\nDownloads complete with all images added to DB session.  \nNow commit session changes to DB.")
# commit the changes to the DB, this also assigns auto-incrementing primekeys 'image_id'
db_session.commit()

# query_EUV_images function:
# requires time_min and time_max (datetime).  do we need to code 'jd' time option?
query_time_min = datetime.datetime(2011, 2, 14, 19, 35, 0)
query_time_max = datetime.datetime(2011, 2, 14, 19, 37, 0)
print("\nQuery DB for downloaded images with timestamps between " + str(query_time_min) + " and " + str(query_time_max))
test_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)
# returns info on 3 images
print(test_pd)

# query specific instrument
query_time_min = datetime.datetime(2011, 2, 13, 19, 35, 0)
instrument = ("EUVI", )
test_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max, instrument=instrument)
print(test_pd)
# query specific wavelength
wavelength = (195, )
test_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max, wavelength=wavelength)
print(test_pd)

# update_image_val function:
# currently requires a pandas-series object (like return of query_euv_images) to index
# into the DB. This semi-insures that there is not confusion about which row of the
# DB is to be updated.

# generate hdf file using some function like:
hd_fname = pdseries_tohdf(test_pd.iloc[0])
print("test pd iloc", test_pd.iloc[0])
#hd_fname = "2014/04/13/sta_euvi_20140413T183530_195.hdf5"
# update database with file location
db_session = update_image_val(db_session=db_session, raw_series=test_pd.iloc[0], col_name="fname_hdf", new_val=hd_fname)

# also flag this image as 1 - 'verified good' (made-up example)
image_flag = 1
# update database with file location
db_session = update_image_val(db_session=db_session, raw_series=test_pd.iloc[0], col_name="flag", new_val=image_flag)

# remove_euv_image function:
if not Create_Ref:
    # first read entire DB
    test_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)
    print(test_pd)
    # removes the files and then the corresponding DB row
    print("\nRemoving one row from DB.")
    exit_status, db_session = remove_euv_image(db_session=db_session, raw_series=test_pd.iloc[0], raw_dir=raw_data_dir,
                                               hdf_dir=hdf_data_dir)
    # re-read entire DB
    test_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)
    print(test_pd)

    print("\nDB now has one less row.")


db_session.close()
