"""
Framework for processing all raw files in DB (with no 'fname_hdf')
to hdf format.  Also record new hdf filename in DB.
- This was used to process the 1st year of data we tested (2011-2012).
"""

import os
import pandas as pd
import datetime

from settings.app import App
import modules.DB_classes as DBClass
from modules.DB_funs import init_db_conn, update_image_val

from helpers import idl_helper
from modules import prep

# Specify a period of time for query
period_start = datetime.datetime(2011, 1, 1, 0, 0, 0)
period_end = datetime.datetime(2012, 1, 1, 0, 0, 0)

# Establish connection to database
use_db = "sqlite"
sqlite_path = os.path.join(App.DATABASE_HOME, App.DATABASE_FNAME)
db_session = init_db_conn(db_name=use_db, chd_base=DBClass.Base, sqlite_path=sqlite_path)

# search for images in database that have no processed fname
query_result = pd.read_sql(db_session.query(DBClass.EUV_Images).filter(DBClass.EUV_Images.fname_hdf == "").statement,
                           db_session.bind)

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
    print(f'Working on index {index} of {len(query_result)}')
    print('---------------------------------------------------------')
    print('Raw File:  ' + raw_data_file)
    subdir, fname, los = prep.prep_euv_image(
        raw_data_file, App.PROCESSED_DATA_HOME, write=write, idl_session=idl_session)

    # return a relative filename for the hdf
    hdf_rel_path = os.path.join(subdir, fname)
    hdf_full_path = os.path.join(App.PROCESSED_DATA_HOME, subdir, fname)

    # update DB to reflect the new filename
    db_session = update_image_val(db_session, row, "fname_hdf", hdf_rel_path)

# close the IDL session
if idl_session is not None:
    idl_session.end()

db_session.close()
