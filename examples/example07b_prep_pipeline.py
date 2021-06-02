"""
Here an image prep pipeline is demonstrated.
- The database is queried and a dataframe is built for prep.
- The prep can include many images or a specific set can be queried for prep.
- This also demonstrates how IDL will be used if STA/STB prep is called.
"""
import os

import pandas as pd
import datetime

from astropy.time import Time

from settings.app import App
from chmap.database.db_classes import Base
from chmap.database.db_funs import init_db_conn, query_euv_images
from chmap.data.download.euv_utils import get_image_set
from utilities.idl_connect import idl_helper
from chmap.data.corrections.image_prep import prep

# database location
database_dir = App.DATABASE_HOME

# raw and processed file locations (use the reference data dir in this example)
data_dir_raw = App.RAW_DATA_HOME
# data_dir_processed = os.path.join( App.APP_HOME, 'reference_data', 'processed')
data_dir_processed = App.PROCESSED_DATA_HOME

# designate which database to connect to
use_db = "mysql-Q"  # 'sqlite'  Use local sqlite file-based db
# 'mysql-Q' Use the remote MySQL database on Q
user = "cdowns"  # only needed for remote databases.
password = ""  # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)
    db_session = init_db_conn(db_name=use_db, chd_base=Base, sqlite_path=sqlite_path)
elif use_db == 'mysql-Q':
    # setup database connection to MySQL database on Q
    db_session = init_db_conn(db_name=use_db, chd_base=Base, user=user, password=password)

# setup the time range for the query
query_time_min = datetime.datetime(2014, 8, 13, 18, 0, 0)
query_time_max = datetime.datetime(2014, 8, 13, 20, 0, 0)
# query_time_min = datetime.datetime(2013,  1,  1,  0, 0, 0)
# query_time_max = datetime.datetime(2013, 12, 31, 23, 59, 59, 999999)

# query the database for each spacecraft type
fs = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max, instrument=('AIA',))
fa = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max, instrument=('EUVI-A',))
fb = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max, instrument=('EUVI-B',))

# merge the query results into one dataframe
df_all = pd.concat([fs, fa, fb], axis=0)

print(df_all)

# or build one big dataframe from a query
# df_all = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)

# or build a dataframe with only AIA (if ssw doesn't work)
# df_all = fs

# option to prep every file or prep an "image set"
prep_all = True

# pick a subset to prep
if not prep_all:
    time0 = Time(datetime.datetime(2014, 8, 13, 19, 5, 0))
    dfm = get_image_set(df_all, time0)  ## THIS IS BROKEN NOW BECAUSE DB DOESN"T SAVE JD ANYMORE
else:
    dfm = df_all
print(dfm)

# Prep options
deconvolve = True
write = True

# start the IDL session for STEREO A or B if necessary
if dfm['instrument'].str.contains('EUVI').any():
    idl_session = idl_helper.Session()
else:
    idl_session = None

# iterate over database entries, prep them
for index, row in dfm.iterrows():
    # build the filename
    fname_raw = row.fname_raw

    fits_infile = os.path.join(data_dir_raw, fname_raw)

    print(f'data_dir_raw: {data_dir_raw}')
    print(f'data_dir_processed: {data_dir_processed}')
    print(f'fname_raw: {fname_raw}')
    print(f'fits_infile: {fits_infile}')

    # prep the image
    subdir, fname, los = prep.prep_euv_image(
        fits_infile, data_dir_processed, write=write, idl_session=idl_session,
        deconvolve=deconvolve)

    # get the full path in case you want it
    full_path = os.path.join(data_dir_processed, subdir, fname)

    print(f'outfile: {full_path}')

# close the IDL session
if idl_session is not None:
    idl_session.end()

# read the file and plot it
# los = read_los_image(full_path)
#
# los.map.peek()
# los.map.fits_header
