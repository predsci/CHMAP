"""
Assuming we want regularly spaced image pairs/trios for synchronic map building,
determine where images are missing from local DB for each instrument.  Then plot
"""


import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from settings.app_JT_Q import App
from modules.DB_classes import *
from modules.DB_funs import init_db_conn, query_euv_images

# query parameters
interval_cadence = datetime.timedelta(hours=2.)
interval_width   = datetime.timedelta(hours=1.)
wave_aia = 193
wave_euvi = 195

# Specify a vector of synchronic times
period_start = datetime.datetime(2011, 1, 1, 0, 0, 0)
period_end = datetime.datetime(2012, 1, 1, 0, 0, 0)
# define target times over download period using interval_cadence
mean_times = np.arange(period_start, period_end, interval_cadence)

query_start = period_start - interval_width
query_end   = period_end + interval_width

# Specify directories on mounted Q home
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
db_home_dir  = App.DATABASE_HOME

# setup database connection
use_db = "sqlite"
sqlite_filename = App.DATABASE_FNAME
sqlite_path = os.path.join(db_home_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=Base, sqlite_path=sqlite_path)

# query a list of images in query range for each instrument. sort by time
aia_images = query_euv_images(db_session, time_min=query_start, time_max=query_end, instrument=("AIA", ),
                              wavelength=(wave_aia, ))
aia_images = aia_images.sort_values(by="date_obs")

euvia_images = query_euv_images(db_session, time_min=query_start, time_max=query_end, instrument=("EUVI-A", ),
                              wavelength=(wave_euvi, ))
euvia_images = euvia_images.sort_values(by="date_obs")

euvib_images = query_euv_images(db_session, time_min=query_start, time_max=query_end, instrument=("EUVI-B", ),
                              wavelength=(wave_euvi, ))
euvib_images = euvib_images.sort_values(by="date_obs")

# --- determine the time between consecutive images ---
# add phantom point at begining
aia_times = pd.concat([pd.Series(period_start - interval_cadence), aia_images.date_obs], ignore_index=True)
# diff datetime Series
aia_diff = aia_times.diff()
# remove phantom point
aia_diff = aia_diff.drop(0)
# approximate the number of missed observations before this point
aia_missed = aia_diff/interval_cadence - 1
aia_missed = aia_missed.round(decimals=0).astype(int)


# add phantom point at begining
euvia_times = pd.concat([pd.Series(period_start - interval_cadence), euvia_images.date_obs], ignore_index=True)
# diff datetime Series
euvia_diff = euvia_times.diff()
# remove phantom point
euvia_diff = euvia_diff.drop(0)
# approximate the number of missed observations before this point
euvia_missed = euvia_diff/interval_cadence - 1
euvia_missed = euvia_missed.round(decimals=0).astype(int)


# add phantom point at begining
euvib_times = pd.concat([pd.Series(period_start - interval_cadence), euvib_images.date_obs], ignore_index=True)
# diff datetime Series
euvib_diff = euvib_times.diff()
# remove phantom point
euvib_diff = euvib_diff.drop(0)
# approximate the number of missed observations before this point
euvib_missed = euvib_diff/interval_cadence - 1
euvib_missed = euvib_missed.round(decimals=0).astype(int)


# --- Plot missed images -------------------
plot_max = max([aia_missed.max(), euvia_missed.max(), euvib_missed.max()])

plt.scatter(aia_images.date_obs, aia_missed, c='blue')
plt.scatter(euvia_images.date_obs, euvia_missed, c='red')
plt.scatter(euvib_images.date_obs, euvib_missed, c='green')
plt.xlabel("Synchronic DateTime")
plt.ylabel("Consecutive missed observations")
plt.legend(loc="upper left", labels=("AIA", "EUVI-A", "EUVI-B"))


