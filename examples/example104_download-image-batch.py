"""
Specify times for synchronic image download.
Query available images and download best matches.
"""

import sys
import os
import numpy as np
import pandas as pd
from astropy.time import Time
import astropy.units as u

from settings.app import App
import modules.DB_classes as DBClass
from modules.DB_funs import init_db_conn
from modules.image_download import synchronic_euv_download

# Specify a vector of synchronic times
period_start = Time('2013-01-01T00:00:00.000', scale='utc')
period_end = Time('2014-01-01T00:00:00.000', scale='utc')
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
pickle_file = os.path.join(App.APP_HOME, "test_data", download_results_filename)

# Establish connection to database
use_db = "sqlite"
sqlite_path = os.path.join(App.DATABASE_HOME, App.DATABASE_FNAME)
db_session = init_db_conn(db_name=use_db, chd_base=DBClass.Base, sqlite_path=sqlite_path)

# query for images, download, and log to database
download_result = synchronic_euv_download(synch_times, App.RAW_DATA_HOME, db_session, download=True, overwrite=False,
                                          verbose=True)

download_result.to_pickle(pickle_file)

# print a summary of results
print("Summary of download resutls:")
print(download_result.result_desc.value_counts())

db_session.close()



