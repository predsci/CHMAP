"""
Framework for processing all raw files in DB (with no 'fname_hdf')
to hdf format.  Also record new hdf filename in DB.
"""


import os
import pandas as pd
import datetime

from settings.app_JT_Q import App
import modules.DB_classes_v2 as DBClass
from modules.DB_funs_v2 import init_db_conn, update_image_val

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

# loop through images and process to hdf
for index, row in query_result.iterrows():
    # --- Debug/testing -----
    # can set row = query_results.iloc[0, ] to test without looping

    # Create file path to .fits
    raw_data_file = os.path.join(App.RAW_DATA_HOME, row.fname_raw)
    # perform some function on raw_data_file

    # return a relative filename for the hdf
    hdf_rel_path = ""
    # update DB to reflect the new filename
    db_session = update_image_val(db_session, row, "fname_hdf", hdf_rel_path)


db_session.close()

