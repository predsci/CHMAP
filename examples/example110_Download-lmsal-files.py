
import os
import time
import numpy as np
import datetime

import database.db_classes as DBClass
from helpers import lmsal_helpers
from helpers import misc_helpers
from settings.app import App

data_type = "magnetic map"
data_provider = "lmsal"

# declare local raw map path
map_path = os.path.join(App.DATABASE_HOME, "raw_maps")

# define a vector of target times
    # first declare start and end times
min_datetime = datetime.datetime(2014, 1, 1, 0, 0, 0)
max_datetime = datetime.datetime(2021, 1, 1, 0, 0, 0)

# define image search interval cadence and width
interval_cadence = datetime.timedelta(hours=2)
del_interval = datetime.timedelta(minutes=30)

# data-file dirs
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
# database location
database_dir = App.DATABASE_HOME
# give the sqlite file a unique name
sqlite_filename = App.DATABASE_FNAME

# designate which database to connect to
use_db = "mysql-Q"      # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
                        # 'mysql-Q_test' Use the development database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

start_time = time.time()

# Establish connection to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = db_funs.init_db_conn(db_name=use_db, chd_base=DBClass.Base, sqlite_path=sqlite_path)
elif use_db in ('mysql-Q', 'mysql-Q_test'):
    # setup database connection to MySQL database on Q
    db_session = db_funs.init_db_conn(db_name=use_db, chd_base=DBClass.Base, user=user, password=password)



# define target times over download period using interval_cadence (image times in astropy Time() format)
target_times = np.arange(min_datetime, max_datetime, interval_cadence).astype(datetime.datetime).tolist()

# query lmsal index
url_df = lmsal_helpers.query_lmsal_index(min_datetime, max_datetime)

# insert map time-selection algorithm here
# for now, maps exist every 6 hours, so just download all

# loop through magnetic maps
for index, row in url_df.iterrows():
    download_url = row.full_url
    # construct file path and name
    full_dir, fname = misc_helpers.construct_path_and_fname(map_path, row.datetime, data_provider, data_type, "h5")
    full_path = os.path.join(full_dir, fname)
    rel_path = os.path.relpath(full_path, map_path)
    # download file
    download_flag = misc_helpers.download_url(download_url, full_path)

    # add record to session
    date_obs = row.datetime.to_pydatetime()
    fname_hdf = ""
    db_session, write_flag = db_funs.add_datafile2session(
        db_session, date_obs=date_obs, data_provider=data_provider,
        data_type=data_type, fname_raw=rel_path, fname_hdf=fname_hdf
    )

    if write_flag == 0:
        db_session.commit()
        print("Database record added to Data_Files for provider", data_provider,
              ", data_type", data_type, ", and observation date", date_obs, ".\n")

end_time = time.time()
print(end_time-start_time, " total seconds elapsed.")

# close connection to DB
db_session.close()
