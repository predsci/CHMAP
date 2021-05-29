"""
outline to create combination EUV maps and equivalent quality maps
1. Select images
2. Apply pre-processing corrections
    a. Limb-Brightening
    b. Inter-Instrument Transformation
3. Coronal Hole Detection
4. Convert to Map
5. Combine Maps
6. Create Mu Dependent Quality Map
"""

import os
import numpy as np
import datetime

import data.corrections.apply_lbc_iit as apply_lbc_iit
import maps.image2map as image2map
import maps.midm as midm
import maps.synchronic.synch_utils as synch_utils
from settings.app import App
import database.db_classes as db_class
import database.db_funs as db_funcs
import coronal_holes.detection.chd_funcs as chd_funcs
from data_products.DP_funs import quality_map

# -------- parameters --------- #
# TIME RANGE FOR QUERYING
query_time_min = datetime.datetime(2011, 8, 16, 20, 0, 0)
query_time_max = datetime.datetime(2011, 8, 17, 0, 0, 0)
map_freq = 2  # number of hours

# INITIALIZE DATABASE CONNECTION
# DATABASE PATHS
map_data_dir = App.MAP_FILE_HOME
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME
# initialize database connection
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = db_funcs.init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

# INSTRUMENTS
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
color_list = ["Blues", "Greens", "Reds", "Oranges", "Purples"]
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
del_mu = 0.2  # optional between this method and mu_merge_cutoff method
mu_cutoff = 0.0  # lower mu cutoff value
mu_merge_cutoff = 0.4  # mu cutoff in overlap areas

# MAP PARAMETERS
x_range = [0, 2 * np.pi]
y_range = [-1, 1]
map_nycoord = 1600
del_y = (y_range[1] - y_range[0]) / (map_nycoord - 1)
map_nxcoord = (np.floor((x_range[1] - x_range[0]) / del_y) + 1).astype(int)
# generate map x,y grids. y grid centered on equator, x referenced from lon=0
map_y = np.linspace(y_range[0], y_range[1], map_nycoord, dtype='<f4')
map_x = np.linspace(x_range[0], x_range[1], map_nxcoord, dtype='<f4')

### --------- NOTHING TO UPDATE BELOW -------- ###
#### STEP ONE: SELECT IMAGES ####
# 1.) query some images
query_pd = db_funcs.query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)

# 2.) generate a dataframe to record methods
methods_list = db_funcs.generate_methdf(query_pd)

#### LOOP THROUGH CENTERS ####
# 1.) get dates
moving_avg_centers = synch_utils.get_dates(time_min=query_time_min, time_max=query_time_max, map_freq=map_freq)

# 2.) get instrument combos
lbc_combo_query, iit_combo_query = apply_lbc_iit.get_inst_combos(db_session, inst_list, time_min=query_time_min,
                                                                                  time_max=query_time_max)

for date_ind, center in enumerate(moving_avg_centers):
    #### STEP TWO: APPLY PRE-PROCESSING CORRECTIONS ####
    date_pd, los_list, iit_list, use_indices, methods_list, ref_alpha, ref_x = apply_lbc_iit.apply_ipp(db_session, center,
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
        map_list, chd_map_list, methods_list, data_info, map_info = image2map.create_singles_maps(inst_list, date_pd,
                                                                                                       iit_list,
                                                                                                       chd_image_list,
                                                                                                       methods_list, map_x,
                                                                                                       map_y, R0)
        #### STEP FIVE: CREATE COMBINED MAPS AND SAVE TO DB ####
        euv_combined, chd_combined = midm.create_combined_maps(db_session, map_data_dir, map_list, chd_map_list,
                                                                    methods_list, data_info, map_info,
                                                                    mu_merge_cutoff=mu_merge_cutoff,
                                                                    mu_cutoff=mu_cutoff)

        #### STEP SIX: CREATE A QUALITY MAP FOR CORRESPONDING CHD MAP ####
        quality_map(db_session, map_data_dir, inst_list, query_pd, euv_combined, chd_combined, color_list=color_list)



