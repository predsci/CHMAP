"""
Framework for deleting all processed files in a given time-range.
Also remove hdf filename in DB.
"""

import os
import pandas as pd
import datetime
import warnings
from h5py.h5py_warnings import H5pyDeprecationWarning

import chmap.database.db_classes as DBClass
from chmap.database.db_funs import init_db_conn_old, update_image_val, query_euv_images, remove_euv_image

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
user = "turtle"  # only needed for remote databases.
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


# query the database for each spacecraft type for a specific time range
# period_start = datetime.datetime(2024, 11, 1, 0, 0, 0)
# period_end = datetime.datetime(2025, 5, 1, 0, 0, 0)
period_start = datetime.datetime(2025, 4, 1, 1, 0, 0)
period_end = datetime.datetime(2025, 7, 1, 3, 0, 0)
# query_result = query_euv_images(db_session=db_session, time_min=period_start, time_max=period_end, instrument=('AIA',))
flag = 0
query_result = query_euv_images(db_session=db_session, time_min=period_start, time_max=period_end, instrument=('AIA',),
                                flag=flag)

# use default print for the data frame to see start and end
print(query_result)

# print out the number of records to prep
print(f'### Query Returned {len(query_result)} images to delete')

# disable hdf5 depreciation warnings for the prep step
warnings.filterwarnings("ignore", category=H5pyDeprecationWarning)

# use a manual counter since sorting can change the index order in iterrows()
ifile = 1

# loop through images and process to hdf
for index, row in query_result.iterrows():
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

    exit_flag, db_session = remove_euv_image(db_session, row, raw_data_home, processed_data_home, proc_only=True)

    # reset image flag to 0
    if flag == -1:
        db_session = update_image_val(db_session, row, 'flag', 0)

    ifile = ifile + 1

db_session.close()
