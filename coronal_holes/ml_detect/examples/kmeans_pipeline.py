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

import sys

sys.path.append("/Users/tamarervin/CH_Project/CHD")
import os
import pandas as pd
import numpy as np
import datetime
from skimage import measure
from settings.app import App
import modules.DB_classes as db_class
import modules.DB_funs as db_funcs
import matplotlib.pyplot as plt
import modules.Plotting as Plotting
import analysis.chd_analysis.CHD_pipeline_funcs as chd_funcs
import analysis.ml_analysis.ch_detect.ml_functions as ml_funcs

# -------- parameters --------- #
# TIME RANGE FOR QUERYING
query_time_min = datetime.datetime(2011, 8, 16, 0, 0, 0)
query_time_max = datetime.datetime(2011, 8, 18, 0, 0, 0)
map_freq = 2  # number of hours

# INITIALIZE DATABASE CONNECTION
# DATABASE PATHS
map_data_dir = App.MAP_FILE_HOME
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME
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

# CHD Model
model_h5 = '/Volumes/CHD_DB/model_unet_FINAL.h5'

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
# connect to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = db_funcs.init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
elif use_db == 'mysql-Q':
    # setup database connection to MySQL database on Q
    db_session = db_funcs.init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user, password=password)

#### STEP ONE: SELECT IMAGES AND LOAD MODEL ####
# 1.) query some images
query_pd = db_funcs.query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)

# 2.) generate a dataframe to record methods
methods_list = db_funcs.generate_methdf(query_pd)

# 3.) load unet neural network
model = ml_funcs.load_model(model_h5, IMG_SIZE=2048, N_CHANNELS=3)

#### STEP TWO: APPLY PRE-PROCESSING CORRECTIONS ####
# 1.) get dates
moving_avg_centers = chd_funcs.get_dates(time_min=query_time_min, time_max=query_time_max, map_freq=map_freq)

# 2.) get instrument combos
lbc_combo_query, iit_combo_query = chd_funcs.get_inst_combos(db_session, inst_list, time_min=query_time_min,
                                                             time_max=query_time_max)

# 3.) loop through center dates
for date_ind, center in enumerate(moving_avg_centers):
    date_pd, los_list, iit_list, use_indices, methods_list, ref_alpha, ref_x = chd_funcs.apply_ipp(db_session, center,
                                                                                                   query_pd,
                                                                                                   inst_list,
                                                                                                   hdf_data_dir,
                                                                                                   lbc_combo_query,
                                                                                                   iit_combo_query,
                                                                                                   methods_list,
                                                                                                   n_intensity_bins,
                                                                                                   R0)

    #### STEP THREE: CORONAL HOLE DETECTION ####
    if los_list[0] is not None:
        #### STEP FOUR: CONVERT TO MAP ####
        map_list, methods_list, data_info, map_info = chd_funcs.create_singles_maps(inst_list, date_pd,
                                                                                    iit_list,
                                                                                    chd_image_list=None,
                                                                                    methods_list=methods_list, map_x=map_x,
                                                                                    map_y=map_y, R0=R0)
        #### STEP FIVE: CREATE COMBINED MAPS AND SAVE TO DB ####
        euv_combined = chd_funcs.create_combined_maps_2(map_list, mu_merge_cutoff=mu_merge_cutoff, del_mu=None, mu_cutoff=0.0,
                               EUV_CHD_sep=False, low_int_filt=1.)

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
                # + '\nNumber of detected CH: ' + str(chd_labeled[1]) + ', Number of detected AR: ' + str(ar_labeled[1])
        Plotting.PlotMap(psi_chd_map, title=title, nfig=date_ind)
        Plotting.PlotMap(psi_chd_map, map_type='Contour', title=title, nfig=date_ind)
        Plotting.PlotMap(psi_ar_map, map_type='Contour1', title=title, nfig=date_ind)
        plt.savefig('/Volumes/CHD_DB/pred_maps/kmeans/map2_' + str(date_ind))




