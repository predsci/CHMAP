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
from modules.DB_classes import *
from modules.DB_funs import init_db_conn, query_euv_images
from modules.misc_funs import get_image_set
from helpers import idl_helper
from modules import prep

# database location
database_dir = os.path.join(App.APP_HOME, 'reference_data')

# raw and processed file locations (use the reference data dir in this example)
data_dir_raw = os.path.join(App.APP_HOME, 'reference_data', 'raw')
# data_dir_processed = os.path.join( App.APP_HOME, 'reference_data', 'processed')
data_dir_processed = App.PROCESSED_DATA_HOME

# setup database connection
use_db = "sqlite"
sqlite_filename = "dbtest.db"
sqlite_path = database_dir + "/" + sqlite_filename
db_session = init_db_conn(db_name=use_db, chd_base=Base, sqlite_path=sqlite_path)

# setup the time range for the query
query_time_min = datetime.datetime(2014, 4, 13, 18, 0, 0)
query_time_max = datetime.datetime(2014, 4, 13, 20, 0, 0)

# query the database for each spacecraft type
fs = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max, instrument=('AIA',))
fa = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max, instrument=('EUVI-A',))
fb = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max, instrument=('EUVI-B',))

# merge the query results into one dataframe
df_all = pd.concat([fs, fa, fb], axis=0)

# or build one big dataframe from a query
# df_all = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)

# or build a dataframe with only AIA (if ssw doesn't work)
# df_all = fs

# option to prep every file or prep an "image set"
prep_all = False

# pick a subset to prep
if not prep_all:
    time0 = Time(datetime.datetime(2014, 4, 13, 19, 5, 0))
    dfm = get_image_set(df_all, time0)
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

    # TEMPORARY, manually remove the non-relative part of the path in the database
    path_cut_string = '/Users/turtle/GitReps/CHD/reference_data/raw/'
    fname_raw = fname_raw.replace(path_cut_string, '')
    fits_infile = os.path.join(data_dir_raw, fname_raw)

    # prep the image
    subdir, fname, los = prep.prep_euv_image(
        fits_infile, data_dir_processed, write=write, idl_session=idl_session)

    # get the full path in case you want it
    full_path = os.path.join(data_dir_processed, subdir, fname)

# close the IDL session
if idl_session is not None:
    idl_session.end()
