"""
Framework for processing all raw files in DB (with no 'fname_hdf')
to hdf format.  Also record new hdf filename in DB.
- This was used to process the 1st year of data we tested (2011-2012).
"""

import os
import pandas as pd
import datetime
import warnings
from h5py.h5py_warnings import H5pyDeprecationWarning

from settings.app import App
import modules.DB_classes as DBClass
from modules.DB_funs import init_db_conn, update_image_val, query_euv_images

from helpers import idl_helper
from modules import prep

# designate which database to connect to
use_db = "mysql-Q"  # 'sqlite'  Use local sqlite file-based db
# 'mysql-Q' Use the remote MySQL database on Q
user = "cdowns"  # only needed for remote databases.
password = ""  # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

# Establish connection to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)
    db_session = init_db_conn(db_name=use_db, chd_base=DBClass.Base, sqlite_path=sqlite_path)
elif use_db == 'mysql-Q':
    # setup database connection to MySQL database on Q
    db_session = init_db_conn(db_name=use_db, chd_base=DBClass.Base, user=user, password=password)

# Flag to do everything that hasn't been processed yet or a specific query for testing
do_all_unprocessed = True

# search for images in database that have no processed fname
if do_all_unprocessed:
    query_result = pd.read_sql(
        db_session.query(DBClass.EUV_Images).filter(DBClass.EUV_Images.fname_hdf == "").statement,
        db_session.bind)

    # sort it by time so that it is easy to track progression in a physical way
    query_result.sort_values(by=['date_obs'], inplace=True)

# or query the database for each spacecraft type for a specific time range
else:
    period_start = datetime.datetime(2013, 3, 3, 0, 0, 0)
    period_end = datetime.datetime(2013, 3, 4, 0, 0, 0)
    fs = query_euv_images(db_session=db_session, time_min=period_start, time_max=period_end, instrument=('AIA',))
    fa = query_euv_images(db_session=db_session, time_min=period_start, time_max=period_end, instrument=('EUVI-A',))
    fb = query_euv_images(db_session=db_session, time_min=period_start, time_max=period_end, instrument=('EUVI-B',))
    query_result = pd.concat([fs, fa, fb], axis=0)

# use default print for the data frame to see start and end
print(query_result)

# print out the number of records to prep
print(f'### Query Returned {len(query_result)} images to prep')

# Prep options
deconvolve = True
write = True

# start the IDL session for STEREO A or B if necessary
if query_result['instrument'].str.contains('EUVI').any():
    idl_session = idl_helper.Session()
else:
    idl_session = None

# disable hdf5 depreciation warnings for the prep step
warnings.filterwarnings("ignore", category=H5pyDeprecationWarning)

# use a manual counter since sorting can change the index order in iterrows()
ifile = 1

# loop through images and process to hdf
for index, row in query_result.iterrows():
    # --- Debug/testing -----
    # can set row = query_results.iloc[0, ] to test without looping
    # row = query_result.iloc[100,]

    # Create file path to .fits
    raw_data_file = os.path.join(App.RAW_DATA_HOME, row.fname_raw)
    # prep the image
    print('')
    print('---------------------------------------------------------')
    print(f'Working on file {ifile} of {len(query_result)}')
    print('---------------------------------------------------------')
    print(f'  query row index:  {index}')
    print(f'  database image_id:  {row.image_id}')
    print('  Raw File:  ' + raw_data_file)
    subdir, fname, los = prep.prep_euv_image(
        raw_data_file, App.PROCESSED_DATA_HOME, write=write, idl_session=idl_session, deconvolve=deconvolve)

    # return a relative filename for the hdf
    hdf_rel_path = os.path.join(subdir, fname)
    hdf_full_path = os.path.join(App.PROCESSED_DATA_HOME, subdir, fname)

    # update DB to reflect the new filename
    print('  Committing processed path to database: ' + hdf_rel_path)
    db_session = update_image_val(db_session, row, "fname_hdf", hdf_rel_path)

    ifile = ifile + 1

# close the IDL session
if idl_session is not None:
    idl_session.end()

db_session.close()
