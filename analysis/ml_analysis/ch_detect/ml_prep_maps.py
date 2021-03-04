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
import sys
sys.path.append("/Users/tamarervin/CH_Project/CHD")
import os
import numpy as np
import datetime
import h5py as h5
import matplotlib.colors as colors
import matplotlib as mpl

from settings.app import App
import modules.DB_classes as db_class
import modules.DB_funs as db_funcs
import analysis.chd_analysis.CHD_pipeline_funcs as chd_funcs

# -------- parameters --------- #
# TIME RANGE FOR QUERYING
query_time_min = datetime.datetime(2011, 5, 1, 0, 0, 0)
query_time_max = datetime.datetime(2011, 6, 1, 0, 0, 0)
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
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = db_funcs.init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user, password=password)

# initialize database connection
# use_db = "sqlite"
# sqlite_path = os.path.join(database_dir, sqlite_filename)
# db_session = db_funcs.init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

# hdf file info
h5_filename = '/Volumes/CHD_DB/map_data.h5'
# CHD Model
model_h5 = '/Volumes/CHD_DB/model_unet_FINAL.h5'

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
map_nycoord = 120
map_nxcoord = 300
# del_y = (y_range[1] - y_range[0]) / (map_nycoord - 1)
# map_nxcoord = (np.floor((x_range[1] - x_range[0]) / del_y) + 1).astype(int)
# generate map x,y grids. y grid centered on equator, x referenced from lon=0
map_y = np.linspace(y_range[0], y_range[1], map_nycoord, dtype='<f4')
map_x = np.linspace(x_range[0], x_range[1], map_nxcoord, dtype='<f4')

### --------- NOTHING TO UPDATE BELOW -------- ###
#### STEP ONE: SELECT IMAGES ####
# 1.) query some images
query_pd = db_funcs.query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)

# 2.) generate a dataframe to record methods
methods_list = db_funcs.generate_methdf(query_pd)

# 3.) load unet neural network
# model = ml_funcs.load_model(model_h5, IMG_SIZE=2048, N_CHANNELS=3)

#### STEP TWO: APPLY PRE-PROCESSING CORRECTIONS ####
# 1.) get dates
moving_avg_centers = chd_funcs.get_dates(time_min=query_time_min, time_max=query_time_max, map_freq=map_freq)

# 2.) get instrument combos
lbc_combo_query, iit_combo_query = chd_funcs.get_inst_combos(db_session, inst_list, time_min=query_time_min,
                                                             time_max=query_time_max)

# 3.) loop through center dates
h5file = h5.File(h5_filename, 'w')
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
        chd_image_list = chd_funcs.chd(iit_list, los_list, use_indices, inst_list, thresh1, thresh2, ref_alpha, ref_x,
                                       nc, iters)
        #### STEP FOUR: CONVERT TO MAP ####
        map_list, methods_list, image_info, map_info = chd_funcs.create_singles_maps(inst_list, date_pd,
                                                                                     iit_list,
                                                                                     chd_image_list,
                                                                                     methods_list, map_x,
                                                                                     map_y, R0)
        #### STEP FIVE: CREATE COMBINED MAPS AND SAVE TO DB ####
        euv_combined = chd_funcs.create_combined_maps(db_session, map_data_dir, map_list,
                                                      methods_list, image_info, map_info,
                                                      mu_merge_cutoff=mu_merge_cutoff, mu_cutoff=mu_cutoff)
        g = h5file.create_group(str(center))
        # create euv image in file
        scalarMap = mpl.cm.ScalarMappable(norm=colors.LogNorm(vmin=1.0, vmax=np.max(euv_combined.data)),
                                          cmap='sohoeit195')
        colorVal = scalarMap.to_rgba(euv_combined.data, norm=True)
        g.create_dataset("euv_image", data=colorVal[:, :, :3])
        # create chd mask in file
        arr = euv_combined.chd
        arr3D = np.repeat(arr[..., None], 1, axis=2)
        g.create_dataset("chd_data", data=arr3D[:, :, :3])
h5file.close()
