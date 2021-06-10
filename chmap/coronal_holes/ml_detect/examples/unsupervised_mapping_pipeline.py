#!/usr/bin/env python

"""
outline to create combination EUV/CHD maps using ML Algorithm
1. Select images
2. Apply pre-processing corrections
    a. Limb-Brightening
    b. Inter-Instrument Transformation
3. Coronal Hole Detection using ML Algorithm
4. Convert to Map
5. Combine Maps and Save to DB
"""


import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import chmap.utilities.plotting.psi_plotting as Plotting
import chmap.database.db_classes as db_class
import chmap.database.db_funs as db_funcs
import chmap.data.corrections.apply_lbc_iit as apply_lbc_iit
import chmap.coronal_holes.ml_detect.tools.ml_functions as ml_funcs
import chmap.maps.image2map as image2map
import chmap.maps.midm as midm
import chmap.maps.synchronic.synch_utils as synch_utils

# -------- parameters --------- #
# TIME RANGE FOR QUERYING
query_time_min = datetime.datetime(2011, 8, 16, 0, 0, 0)
query_time_max = datetime.datetime(2011, 8, 18, 0, 0, 0)
# # define map interval cadence and width
map_freq = 2  # number of hours
interval_delta = 30  # number of minutes
del_interval_dt = datetime.timedelta(minutes=interval_delta)
del_interval = np.timedelta64(interval_delta, 'm')

# INITIALIZE DATABASE CONNECTION
# DATABASE PATHS
map_data_dir = 'path/to/map/directory'
raw_data_dir = 'path/to/raw_data/directory'
hdf_data_dir = 'path/to/processed_data/directory'
database_dir = 'path/to/database/directory'
sqlite_filename = 'path/to/database/filename'
# initialize database connection
# using mySQL
use_db = "mysql-Q"
user = "tervin"
password = ""
# using sqlite
# use_db = "sqlite"
# sqlite_path = os.path.join(database_dir, sqlite_filename)

# INSTRUMENTS
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
# CORRECTION PARAMETERS
n_intensity_bins = 200
R0 = 1.01
N_CLUSTERS = 14
weight = 1.4

# MINIMUM MERGE MAPPING PARAMETERS
del_mu = None  # optional between this method and mu_merge_cutoff method
mu_cutoff = 0.0  # lower mu cutoff value
mu_merge_cutoff = 0.4  # mu cutoff in overlap areas

# MAP PARAMETERS
x_range = [0, 2 * np.pi]
y_range = [-1, 1]
map_nycoord = 360
map_nxcoord = 900

# generate map x,y grids. y grid centered on equator, x referenced from lon=0
map_y = np.linspace(y_range[0], y_range[1], map_nycoord, dtype='<f4')
map_x = np.linspace(x_range[0], x_range[1], map_nxcoord, dtype='<f4')

# generate K-Means x, y grids
idx = np.indices((map_nxcoord, map_nycoord))
idx_row = idx[0]
idx_row = idx_row / np.max(idx_row)
idx_col = idx[1]
idx_col = idx_col / np.max(idx_col)
# flatten arrays
idx_col_flt = idx_col.flatten()
idx_row_flt = idx_row.flatten()

### --------- NOTHING TO UPDATE BELOW -------- ###
# Establish connection to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = db_funcs.init_db_conn_old(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
elif use_db in ('mysql-Q', 'mysql-Q_test'):
    # setup database connection to MySQL database on Q
    db_session = db_funcs.init_db_conn_old(db_name=use_db, chd_base=db_class.Base, user=user, password=password)

#### STEP ONE: SELECT IMAGES ####
# 1.) query some images
query_pd = db_funcs.query_euv_images(db_session=db_session, time_min=query_time_min - del_interval_dt,
                                     time_max=query_time_max + del_interval_dt)

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
        #### STEP FOUR: CONVERT TO MAP ####
        map_list, methods_list, data_info, map_info = \
            image2map.create_singles_maps_2(synch_images, iit_list, chd_image_list=None,
                                            methods_list=methods_list, map_x=map_x, map_y=map_y, R0=R0)

        #### STEP FIVE: CREATE COMBINED MAPS AND SAVE TO DB ####
        euv_combined = midm.create_combined_maps_2(
            map_list, mu_merge_cutoff=mu_merge_cutoff, del_mu=del_mu,
            mu_cutoff=mu_cutoff, EUV_CHD_sep=False)

        #### STEP SIX: APPLY CH AND AR DETECTION ####img = img_array[i]
        map = np.where(euv_combined.data == -9999, 0, euv_combined.data)
        map2 = np.log(map)
        map2 = np.where(map2 == -np.inf, 0, map2)

        arr = np.zeros((map_nxcoord * map_nycoord, 3))
        arr[:, 0] = idx_col_flt * weight
        arr[:, 1] = idx_row_flt * weight
        arr[:, 2] = map2.flatten() * 2

        psi_chd_map, psi_ar_map, chd_labeled, ar_labeled = ml_funcs.kmeans_detection(euv_combined.data, map2, arr, N_CLUSTERS,
                                                                                     map_nycoord, map_nxcoord, map_x, map_y)

        title = 'Minimum Intensity Merge: Unsupervised Detection Map\nDate: ' + str(center)
        Plotting.PlotMap(psi_chd_map, title=title, nfig=date_ind)
        Plotting.PlotMap(psi_chd_map, map_type='Contour', title=title, nfig=date_ind)
        Plotting.PlotMap(psi_ar_map, map_type='Contour1', title=title, nfig=date_ind)




