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

# This can be a computationally intensive process.
# To limit number of threads numpy can spawn:
# os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import datetime
import time
# import pandas as pd

from settings.app import App
import database.db_classes as db_class
import database.db_funs as db_funcs
import data.corrections.apply_lbc_iit as apply_lbc_iit
import chmap.coronal_holes.detection.chd_funcs as chd_funcs
import maps.util.map_manip as map_manip
import utilities.datatypes.datatypes as datatypes
import maps.image2map as image2map
import maps.midm as midm
import maps.synchronic.synch_utils as synch_utils

# -------- parameters --------- #
# TIME RANGE FOR QUERYING
query_time_min = datetime.datetime(2018, 1, 1, 2, 0, 0)
query_time_max = datetime.datetime(2018, 1, 1, 3, 0, 0)
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
del_mu = 0.6  # optional between this method and mu_merge_cutoff method (not both)
mu_cutoff = 0.0  # lower mu cutoff value
mu_merge_cutoff = None  # mu cutoff in overlap areas
EUV_CHD_sep = False  # Do separate minimum intensity merges for image and CHD

# MAP PARAMETERS
x_range = [0, 2 * np.pi]
y_range = [-1, 1]
reduce_map_nycoord = 640
reduce_map_nxcoord = 1600
full_map_nycoord = 2048
full_map_nxcoord = 2048*2
low_res_nycoord = 160
low_res_nxcoord = 400
# del_y = (y_range[1] - y_range[0]) / (map_nycoord - 1)
# map_nxcoord = (np.floor((x_range[1] - x_range[0]) / del_y) + 1).astype(int)
# generate map x,y grids. y grid centered on equator, x referenced from lon=0
full_map_y = np.linspace(y_range[0], y_range[1], full_map_nycoord, dtype='<f4')
full_map_x = np.linspace(x_range[0], x_range[1], full_map_nxcoord, dtype='<f4')
reduce_map_y = np.linspace(y_range[0], y_range[1], reduce_map_nycoord, dtype='<f4')
reduce_map_x = np.linspace(x_range[0], x_range[1], reduce_map_nxcoord, dtype='<f4')
low_res_y = np.linspace(y_range[0], y_range[1], low_res_nycoord, dtype='<f4')
low_res_x = np.linspace(x_range[0], x_range[1], low_res_nxcoord, dtype='<f4')


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
moving_avg_centers = synch_utils.get_dates(time_min=query_time_min, time_max=query_time_max, map_freq=map_freq)

# 3.) loop through center dates
for date_ind, center in enumerate(moving_avg_centers):
    # choose which images to use in the same way we choose images for synchronic download
    synch_images, cluster_method = synch_utils.select_synchronic_images(
        center, del_interval, query_pd, inst_list)
    if synch_images is None:
        # no images fall in the appropriate range, skip
        continue
    # apply corrections to those images
    date_pd, los_list, iit_list, use_indices, methods_list, ref_alpha, ref_x = \
        apply_lbc_iit.apply_ipp_2(db_session, center, synch_images, inst_list, hdf_data_dir,
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
            image2map.create_singles_maps_2(synch_images, iit_list, chd_image_list,
                                                 methods_list, full_map_x, full_map_y, R0)

        #### STEP SIX: REDUCE MAP PIXEL GRID ####
        reduced_maps = [datatypes.PsiMap()]*map_list.__len__()
        for ii in range(map_list.__len__()):
            # first combine chd and image into a single map object
            map_list[ii].chd = chd_map_list[ii].data.astype('float16')
            print("Reducing resolution on single image/chd map of", map_list[ii].data_info.instrument[0],
                  "at", map_list[ii].data_info.date_obs[0])
            # perform map reduction
            reduced_maps[ii] = map_manip.downsamp_reg_grid(map_list[ii], reduce_map_y,
                                                           reduce_map_x, single_origin_image=map_list[ii].data_info.data_id[0])
            # reduced_maps[ii], row_mat, col_mat = map_manip.downsamp_reg_grid(
            #     map_list[ii], reduce_map_y, reduce_map_x,
            #     single_origin_image=map_list[ii].data_info.data_id[0])

        #### STEP FIVE: CREATE COMBINED MAPS ####
        # if reduced_maps.__len__() == 1:
        #     # synchronic_map = reduced_maps[0].__copy__()
        #     synchronic_map = reduced_maps[0]
        # else:
        synchronic_map = midm.create_combined_maps_2(
            reduced_maps.copy(), mu_merge_cutoff=mu_merge_cutoff, del_mu=del_mu,
            mu_cutoff=mu_cutoff, EUV_CHD_sep=EUV_CHD_sep)
        # add synchronic clustering method to final map
        synchronic_map.append_method_info(cluster_method)

        #### STEP SEVEN: SAVE MAP TO DATABASE ####
        db_session = synchronic_map.write_to_file(map_data_dir, map_type='synchronic',
                                                  db_session=db_session)

        #### SAVE LOW-RES MAP AS WELL ####
        low_res_maps = [datatypes.PsiMap()]*map_list.__len__()
        for ii in range(map_list.__len__()):
            # first combine chd and image into a single map object
            map_list[ii].chd = chd_map_list[ii].data.astype('float16')
            print("Reducing resolution on single image/chd map of", map_list[ii].data_info.instrument[0],
                  "at", map_list[ii].data_info.date_obs[0])
            # perform map reduction
            low_res_maps[ii] = map_manip.downsamp_reg_grid(
                map_list[ii], low_res_y, low_res_x,
                single_origin_image=map_list[ii].data_info.data_id[0],
                uniform_no_data=False)
        low_synch_map = midm.create_combined_maps_2(
            low_res_maps.copy(), mu_merge_cutoff=mu_merge_cutoff, del_mu=del_mu,
            mu_cutoff=mu_cutoff, EUV_CHD_sep=EUV_CHD_sep)
        # add synchronic clustering method to final map
        low_synch_map.append_method_info(cluster_method)
        db_session = low_synch_map.write_to_file(map_data_dir, map_type='synchronic',
                                                 db_session=db_session)

end_time = time.time()
print("Total elapsed time: ", end_time-start_time)
