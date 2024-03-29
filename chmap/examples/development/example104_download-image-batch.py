"""
Specify times for synchronic image download.
Query available images and download best matches.
"""

import os
import numpy as np
import pandas as pd
from astropy.time import Time
import astropy.units as u

import chmap.database.db_classes as DBClass
from chmap.database.db_funs import init_db_conn_old
from chmap.data.download.image_download import synchronic_euv_download

# Specify a vector of synchronic times
period_start = Time('2021-04-29T00:00:00.000', scale='utc')
period_end = Time('2023-01-01T00:00:00.000', scale='utc')
# define image search interval cadence and width
interval_cadence = 2*u.hour
del_interval = 30*u.minute
# define target times over download period using interval_cadence (image times in astropy Time() format)
target_times = Time(np.arange(period_start, period_end, interval_cadence))
# generate DataFrame that defines synchronic target times as well as min/max limits
synch_times = pd.DataFrame({'target_time': target_times, 'min_time': target_times - del_interval,
                            'max_time': target_times + del_interval})

# specify path and filename for download_results file
download_results_filename = "download_results_" + period_start.__str__()
pickle_file = os.path.join("/Users/turtle/Dropbox/MyNACD", "test_data", download_results_filename)

# data-file dirs
raw_data_dir = "/Volumes/extdata2/CHD_DB/raw_images"
# database location (only for sqlite)
database_dir = None
# give the sqlite file a unique name (only for sqlite)
sqlite_filename = None

# designate which database to connect to
use_db = "mysql-Q"       # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn_old() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

# Establish connection to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = init_db_conn_old(db_name=use_db, chd_base=DBClass.Base, sqlite_path=sqlite_path)
elif use_db in ['mysql-Q', 'mysql-Q_test']:
    # setup database connection to MySQL database on Q
    db_session = init_db_conn_old(db_name=use_db, chd_base=DBClass.Base, user=user, password=password)

# query for images, download, and log to database
download_result = synchronic_euv_download(synch_times, raw_data_dir, db_session, download=True, overwrite=False,
                                          verbose=True)

download_result.to_pickle(pickle_file)

# print a summary of results
print("Summary of download resutls:")
print(download_result.result_desc.value_counts())

db_session.close()



