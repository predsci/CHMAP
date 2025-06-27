"""
Determine most recent raw images in database.
Query available (newer) images and download best matches.
"""
import datetime
import os
import pandas as pd
import astropy.time
import astropy.units as u
from sqlalchemy import func

import chmap.database.db_classes as DBClass
from chmap.database.db_funs import init_db_conn_old
from chmap.data.download.image_download import synchronic_euv_download, get_synch_times

# --- Establish connection to database ---------------------------------
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

# --- Query database for latest image --------------------------------
image_max_date = db_session.query(func.max(DBClass.EUV_Images.date_obs)).one()[0]
# AIA availability lags Stereo A by approx 6 days. To be safe, start queries
# 10 days before
period_start_dt = image_max_date - datetime.timedelta(days=10)

# --- Set a series of target synchronic times ------------------------
# Specify a vector of synchronic times
period_start = astropy.time.Time(period_start_dt, scale='utc')
period_end = astropy.time.Time.now()
# period_start = astropy.time.Time('2023-03-30T00:00:00', format='isot', scale='utc')
# period_end = astropy.time.Time('2023-04-07T00:00:00', format='isot', scale='utc')
# define image search interval cadence and width
interval_cadence = 2*u.hour
del_interval = 30*u.minute
# define target times over download period using interval_cadence (image times in astropy Time() format)
target_times = get_synch_times(period_start, period_end, interval_cadence)
# generate DataFrame that defines synchronic target times as well as min/max limits
synch_times = pd.DataFrame({'target_time': target_times, 'min_time': target_times - del_interval,
                            'max_time': target_times + del_interval})

# specify path and filename for download_results file
download_results_filename = "download_results_" + period_start.__str__().replace(" ", "T") + ".pkl"
pickle_file = os.path.join("/Users/turtle/Dropbox/MyNACD", "test_data", download_results_filename)

# --- Start querying/downloading images for each synchronic target time -----------------
# query for images, download, and log to database
download_result = synchronic_euv_download(synch_times, raw_data_dir, db_session, download=True, overwrite=False,
                                          verbose=True)
# --- Save download results and wrap-up -------------------------------------------------
download_result.to_pickle(pickle_file)

# print a summary of results
print("Summary of download results:")
print(download_result.result_desc.value_counts())

db_session.close()
