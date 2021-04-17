
# Delete all maps in a timeframe

import os
os.environ["OMP_NUM_THREADS"] = "4"  # limit number of threads numpy can spawn
import numpy as np
import datetime
import time
# import pandas as pd

from settings.app import App
import modules.DB_classes as db_class
import modules.DB_funs as db_funcs
import analysis.chd_analysis.CHD_pipeline_funcs as chd_funcs
import modules.map_manip as map_manip

# -------- parameters --------- #
# TIME RANGE FOR QUERYING
query_time_min = datetime.datetime(2007, 3, 1, 0, 0, 0)
query_time_max = datetime.datetime(2012, 1, 1, 0, 0, 0)
# define map interval cadence and width
map_freq = 2  # number of hours
interval_delta = 30  # number of minutes
del_interval_dt = datetime.timedelta(minutes=interval_delta)
del_interval = np.timedelta64(interval_delta, 'm')

# INITIALIZE DATABASE CONNECTION
# DATABASE PATHS
map_data_dir = App.MAP_FILE_HOME
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME
# designate which database to connect to
use_db = "mysql-Q" # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
                        # 'mysql-Q_test' Use the development database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

# Establish connection to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = db_funcs.init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
elif use_db in ('mysql-Q', 'mysql-Q_test'):
    # setup database connection to MySQL database on Q
    db_session = db_funcs.init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user, password=password)


# query the maps to be deleted
map_info, data_info, method_info, image_assoc = db_funcs.query_euv_maps(
    db_session, mean_time_range=(query_time_min, query_time_max)
)

db_session = db_funcs.remove_euv_map(db_session, map_info, map_data_dir)

db_session.close()
