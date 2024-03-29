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

import chmap.database.db_classes as DBClass
from chmap.database.db_funs import init_db_conn_old, update_image_val, query_euv_images

from chmap.utilities.idl_connect import idl_helper
from chmap.data.corrections.image_prep import prep

# test setting umask for writing to Q over AFP
os.umask(0o002)

# Paths to the database filesystem
raw_data_home = '/Volumes/extdata2/CHD_DB/raw_images'
processed_data_home = '/Volumes/extdata2/CHD_DB/processed_images'

# designate which database to connect to
use_db = "mysql-Q"  # 'sqlite'  Use local sqlite file-based db
# 'mysql-Q' Use the remote MySQL database on Q
user = "cdowns"  # only needed for remote databases.
password = ""  # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn_old() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

# Establish connection to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)
    db_session = init_db_conn_old(db_name=use_db, chd_base=DBClass.Base, sqlite_path=sqlite_path)
elif use_db == 'mysql-Q':
    # setup database connection to MySQL database on Q
    db_session = init_db_conn_old(db_name=use_db, chd_base=DBClass.Base, user=user, password=password)

# Flag to do everything that hasn't been processed yet or a specific query for testing
do_all_unprocessed = True

# search for images in database that have no processed fname
if do_all_unprocessed:
    query_result = pd.read_sql(
        db_session.query(DBClass.Data_Files, DBClass.EUV_Images.instrument).filter(
            DBClass.Data_Files.fname_hdf == "", DBClass.Data_Files.flag == 0,
            DBClass.Data_Files.type == "EUV_Image",
            DBClass.Data_Files.data_id == DBClass.EUV_Images.data_id).statement,
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

# test = True
# if test:
#     query_result = query_result.iloc[0:5]

# loop through images and process to hdf
for index, row in query_result.iterrows():
    # --- Debug/testing -----
    # can set row = query_results.iloc[0, ] to test without looping
    # row = query_result.iloc[100,]

    # Create file path to .fits
    raw_data_file = os.path.join(raw_data_home, row.fname_raw)
    # prep the image
    print('')
    print('---------------------------------------------------------')
    print(f'Working on file {ifile} of {len(query_result)}')
    print('---------------------------------------------------------')
    print(f'  query row index:  {index}')
    print(f'  database data_id:  {row.data_id}')
    print('  Raw File:  ' + raw_data_file)
    subdir, fname, los = prep.prep_euv_image(
        raw_data_file, processed_data_home, write=write, idl_session=idl_session, deconvolve=deconvolve)

    # return a relative filename for the hdf
    hdf_rel_path = os.path.join(subdir, fname)
    hdf_full_path = os.path.join(processed_data_home, subdir, fname)

    # update DB to reflect the new filename
    print('  Committing processed path to database: ' + hdf_rel_path)
    db_session = update_image_val(db_session, row, "fname_hdf", hdf_rel_path)

    ifile = ifile + 1

# close the IDL session
if idl_session is not None:
    idl_session.end()

db_session.close()
