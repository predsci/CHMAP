
import os
import pandas as pd
import numpy as np
import datetime

from helpers import lmsal_helpers
from helpers import misc_helpers
from settings.app import App

mmap_type = "lmsal"

# declare local raw map path
map_path = os.path.join(App.DATABASE_HOME, "raw_mmaps")

# define a vector of target times
    # first declare start and end times
min_datetime = datetime.datetime(2019, 1, 1, 0, 0, 0)
max_datetime = datetime.datetime(2019, 1, 2, 0, 0, 0)

# define image search interval cadence and width
interval_cadence = datetime.timedelta(hours=2)
del_interval = datetime.timedelta(minutes=30)
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
    full_dir, fname = misc_helpers.construct_path_and_fname(map_path, row.datetime, mmap_type, "", "h5")
    full_path = os.path.join(full_dir, fname)
    # download file
    download_flag = misc_helpers.download_url(download_url, full_path)


