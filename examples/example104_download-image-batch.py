"""
Specify times for synchronic image download.
Query available images and download best matches.
"""


import os
import numpy as np
import pandas as pd
from astropy.time import Time
import astropy.units as u

from settings.app_JT_Q import App
import modules.DB_classes_v2 as DBClass
from modules.DB_funs_v2 import init_db_conn
from modules.image_download import synchronic_euv_download

# Specify a vector of synchronic times
period_start = Time('2011-01-01T00:00:00.000', scale='utc')
period_end = Time('2012-01-01T00:00:00.000', scale='utc')
# define image search interval cadence and width
interval_cadence = 2*u.hour
del_interval = 30*u.minute
# define target times over download period using interval_cadence (image times in astropy Time() format)
target_times = Time(np.arange(period_start, period_end, interval_cadence))
# generate DataFrame that defines synchronic target times as well as min/max limits
synch_times = pd.DataFrame({'target_time': target_times, 'min_time': target_times - del_interval,
                            'max_time': target_times + del_interval})

# specify path and filename for download_results file
pickle_file = "/Users/turtle/GitReps/CHD/test_data/download_results_2011.pkl"

# Establish connection to database
use_db = "sqlite"
sqlite_path = os.path.join(App.DATABASE_HOME, App.DATABASE_FNAME)
db_session = init_db_conn(db_name=use_db, chd_base=DBClass.Base, sqlite_path=sqlite_path)

# query for images, download, and log to database
download_result = synchronic_euv_download(synch_times, App.RAW_DATA_HOME, db_session, download=True, overwrite=False,
                                          verbose=True)

download_result.to_pickle(pickle_file)

db_session.close()



