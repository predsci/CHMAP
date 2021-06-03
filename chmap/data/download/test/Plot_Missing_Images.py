"""
Assuming we want regularly spaced image pairs/trios for synchronic map building,
determine where images are missing from local DB for each instrument.  Then plot
"""


import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chmap.settings.app import App
import chmap.database.db_classes as db_class
from chmap.database.db_funs import init_db_conn_old, query_euv_images

# query parameters
interval_cadence = datetime.timedelta(hours=2.)
interval_width   = datetime.timedelta(hours=1.)
wave_aia = 193
wave_euvi = 195

# Specify a vector of synchronic times
period_start = datetime.datetime(2007, 4, 1, 0, 0, 0)
period_end = datetime.datetime(2021, 1, 1, 0, 0, 0)
# define target times over download period using interval_cadence
mean_times = np.arange(period_start, period_end, interval_cadence)

query_start = period_start - interval_width
query_end   = period_end + interval_width

# Specify directories on mounted Q home
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
db_home_dir  = App.DATABASE_HOME

# INITIALIZE DATABASE CONNECTION
# DATABASE PATHS
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME

create = False

# designate which database to connect to
use_db = "mysql-Q"       # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn_old() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

# connect to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = init_db_conn_old(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
elif use_db == 'mysql-Q':
    # setup database connection to MySQL database on Q
    db_session = init_db_conn_old(db_name=use_db, chd_base=db_class.Base, user=user, password=password)

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



# plot Stereo A only (log y-axis)
euvia_missed_log = euvia_missed
euvia_missed_log[euvia_missed == 0] = .1


plt.figure(1)

plt.scatter(euvia_images.date_obs, euvia_missed_log, c='red', marker="|")
plt.yscale("log")
plt.ylim((10**-1.1, 10**3.4))
plt.xlabel("Image Observation Time")
plt.ylabel("Consecutive missed observations (trailing)")
plt.title(r'Stereo-A EUV (195$\AA$)')

plot_fname = "/Users/turtle/Dropbox/MyNACD/analysis/missing_images/StereoA.pdf"
plt.savefig(plot_fname)

plt.close(1)


# plot Stereo B only (log y-axis)
euvib_missed_log = euvib_missed
euvib_missed_log[euvib_missed == 0] = .1


plt.figure(2)

plt.scatter(euvib_images.date_obs, euvib_missed_log, c='green', marker="|")
plt.yscale("log")
plt.ylim((10**-1.1, 10**1.5))
plt.xlabel("Image Observation Time")
plt.ylabel("Consecutive missed observations (trailing)")
plt.title(r'Stereo-B EUV (195$\AA$)')

plot_fname = "/Users/turtle/Dropbox/MyNACD/analysis/missing_images/StereoB.pdf"
plt.savefig(plot_fname)

plt.close(2)


# plot AIA only (log y-axis)
aia_missed_log = aia_missed
aia_missed_log[aia_missed == 0] = .1


plt.figure(3)

plt.scatter(aia_images.date_obs, aia_missed_log, c='blue', marker="|")
plt.yscale("log")
plt.ylim((10**-1.1, 10**2.2))
plt.xlabel("Image Observation Time")
plt.ylabel("Consecutive missed observations (trailing)")
plt.title(r'AIA EUV (193$\AA$)')

plot_fname = "/Users/turtle/Dropbox/MyNACD/analysis/missing_images/AIA.pdf"
plt.savefig(plot_fname)

plt.close(3)





db_session.close()

