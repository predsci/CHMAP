#!/usr/bin/env python
"""
outline to create combination EUV maps
- this method doesn't automatically save individual image maps to database, bc storage
1. Select images
2. Apply pre-processing corrections
    a. Limb-Brightening
    b. Inter-Instrument Transformation
3. Coronal Hole Detection
4. Convert to Map
5. Combine Maps and Save to DB
"""

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
query_time_min = datetime.datetime(2007, 6, 2, 0, 0, 0)
query_time_max = datetime.datetime(2008, 1, 1, 0, 0, 0)
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

# INSTRUMENTS
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
# CORRECTION PARAMETERS
n_intensity_bins = 200
R0 = 1.01

# DETECTION PARAMETERS
# region-growing threshold parameters
thresh1 = 0.95
thresh2 = 1.35
# consecutive pixel value
nc = 3
# maximum number of iterations
iters = 1000

# MINIMUM MERGE MAPPING PARAMETERS
del_mu = None  # optional between this method and mu_merge_cutoff method
mu_cutoff = 0.0  # lower mu cutoff value
mu_merge_cutoff = 0.4  # mu cutoff in overlap areas

# MAP PARAMETERS
x_range = [0, 2 * np.pi]
y_range = [-1, 1]
reduce_map_nycoord = 160
reduce_map_nxcoord = 400
full_map_nycoord = 2048
full_map_nxcoord = 2048*2
# del_y = (y_range[1] - y_range[0]) / (map_nycoord - 1)
# map_nxcoord = (np.floor((x_range[1] - x_range[0]) / del_y) + 1).astype(int)
# generate map x,y grids. y grid centered on equator, x referenced from lon=0
full_map_y = np.linspace(y_range[0], y_range[1], full_map_nycoord, dtype='<f4')
full_map_x = np.linspace(x_range[0], x_range[1], full_map_nxcoord, dtype='<f4')
reduce_map_y = np.linspace(y_range[0], y_range[1], reduce_map_nycoord, dtype='<f4')
reduce_map_x = np.linspace(x_range[0], x_range[1], reduce_map_nxcoord, dtype='<f4')

### --------- NOTHING TO UPDATE BELOW -------- ###
# Establish connection to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = db_funcs.init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
elif use_db in ('mysql-Q', 'mysql-Q_test'):
    # setup database connection to MySQL database on Q
    db_session = db_funcs.init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user, password=password)

#### STEP ONE: SELECT IMAGES ####
start_time = time.time()
# 1.) query some images
query_pd = db_funcs.query_euv_images(db_session=db_session, time_min=query_time_min - del_interval_dt,
                                     time_max=query_time_max + del_interval_dt)

# 2.) generate a dataframe to record methods
# methods_list = db_funcs.generate_methdf(query_pd)

#### STEP TWO: APPLY PRE-PROCESSING CORRECTIONS ####
# 1.) get dates
moving_avg_centers = chd_funcs.get_dates(time_min=query_time_min, time_max=query_time_max, map_freq=map_freq)

# 3.) loop through center dates
for date_ind, center in enumerate(moving_avg_centers):
    # choose which images to use in the same way we choose images for synchronic download
    synch_images = chd_funcs.select_synchronic_images(center, del_interval, query_pd, inst_list)
    if synch_images is None:
        # no images fall in the appropriate range, skip
        continue
    # apply corrections to those images
    date_pd, los_list, iit_list, use_indices, methods_list, ref_alpha, ref_x = \
        chd_funcs.apply_ipp_2(db_session, center, synch_images, inst_list, hdf_data_dir,
                              n_intensity_bins, R0)

    #### STEP THREE: CORONAL HOLE DETECTION ####
    if los_list[0] is not None:
        chd_image_list = chd_funcs.chd_2(iit_list, los_list, use_indices, thresh1,
                                         thresh2, ref_alpha, ref_x, nc, iters)

        # Need to choose when to reduce image/map precision
        # 1. Reduce single-image maps - this allows current tracking of 'origin_image'
        #    to continue for each pixel.
        # 2. Reduce merged maps - this allows high-precision minimum intensity merge,
        #    but interrupts 'origin_image' tracking.

        #### STEP FOUR: CONVERT TO MAP ####
        map_list, chd_map_list, methods_list, data_info, map_info = \
            chd_funcs.create_singles_maps_2(synch_images, iit_list, chd_image_list,
                                            methods_list, full_map_x, full_map_y, R0)

        #### STEP SIX: REDUCE MAP PIXEL GRID ####
        reduced_maps = [None, ] * map_list.__len__()
        for ii in range(map_list.__len__()):
            # first combine chd and image into a single map object
            map_list[ii].chd = chd_map_list[ii].data
            print("Reducing resolution on single image/chd map of", map_list[ii].data_info.instrument[0],
                  "at", map_list[ii].data_info.date_obs[0])
            # perform map reduction
            reduced_maps[ii] = map_manip.downsamp_reg_grid(map_list[ii], reduce_map_y,
                reduce_map_x, single_origin_image=map_list[ii].data_info.data_id[0])

        #### STEP FIVE: CREATE COMBINED MAPS ####
        # euv_combined, chd_combined = chd_funcs.create_combined_maps(
        #     db_session, map_data_dir, map_list, chd_map_list,
        #     methods_list, data_info, map_info, mu_merge_cutoff=mu_merge_cutoff,
        #     mu_cutoff=mu_cutoff)
        if reduced_maps.__len__() == 1:
            synchronic_map = reduced_maps[0]
        else:
            synchronic_map = chd_funcs.create_combined_maps_2(
                reduced_maps, mu_merge_cutoff=mu_merge_cutoff, mu_cutoff=mu_cutoff)

        #### STEP SEVEN: SAVE MAP TO DATABASE ####
        db_session = synchronic_map.write_to_file(map_data_dir, map_type='synchronic',
                                                  db_session=db_session)

end_time = time.time()
print("Total elapsed time: ", end_time-start_time)
