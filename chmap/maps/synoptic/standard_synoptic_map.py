"""
pipeline for creation of a "standard" synoptic map
weighting over a CR favoring central longitudes
"""

import os
import numpy as np
import datetime

import chmap.data.corrections.apply_lbc_iit as apply_lbc_iit
import chmap.database.db_classes as db_class
import chmap.database.db_funs as db_funcs
import chmap.maps.synoptic.synoptic_funcs as cr_funcs
import chmap.maps.time_averaged.dp_funcs as dp_funcs

#### INPUT PARAMETERS ####

#### ----- choose one of the below time range options ----- #
# CARRINGTON ROTATION
cr_rot = None
# TIME RANGE FOR QUERYING
# query_time_min = datetime.datetime(2011, 5, 1, 0, 0, 0)
query_time_min = None
# query_time_max = datetime.datetime(2011, 5, 4, 0, 0, 0)
query_time_max = None
# INTEREST DATE TO CARRINGTION ROTATION
interest_date = datetime.datetime(2011, 5, 1, 0, 0, 0)
ref_inst = "AIA"  # reference instrument for creating full CR
center = True  # true if interest date is center date, false if start date

#### ----- map creation parameters ----- ####
# INSTRUMENTS
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
# COLOR LIST FOR INSTRUMENT QUALITY MAPS
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
del_mu = None  # optional between this method and mu_merge_cutoff method
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

# INITIALIZE DATABASE CONNECTION
# DATABASE PATHS
map_data_dir = 'path/to/map/directory'
raw_data_dir = 'path/to/raw_data/directory'
hdf_data_dir = 'path/to/processed_data/directory'
database_dir = 'path/to/database/directory'
sqlite_filename = 'path/to/database/filename'

# designate which database to connect to
use_db = "mysql-Q" # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
                        # 'mysql-Q_test' Use the development database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn_old() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

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
query_pd = cr_funcs.query_datebase_cr(db_session, query_time_min, query_time_max, interest_date, center, ref_inst,
                                      cr_rot)

# 2.) generate a dataframe to record methods
methods_list = db_funcs.generate_methdf(query_pd)

# 3.) get instrument combos
lbc_combo_query, iit_combo_query = apply_lbc_iit.get_inst_combos(db_session, inst_list, time_min=query_time_min,
                                                                 time_max=query_time_max)

#### LOOP THROUGH IMAGES ####
euv_combined = None
chd_combined = None
data_info = []
map_info = []
for row in query_pd.iterrows():
    #### STEP TWO: APPLY PRE-PROCESSING CORRECTIONS ####
    los_image, iit_image, methods_list, use_indices = cr_funcs.apply_ipp(db_session, hdf_data_dir, inst_list, row,
                                                                         methods_list, lbc_combo_query,
                                                                         iit_combo_query,
                                                                         n_intensity_bins=n_intensity_bins, R0=R0)

    #### STEP THREE: CORONAL HOLE DETECTION ####
    chd_image = cr_funcs.chd(db_session, inst_list, los_image, iit_image, use_indices, iit_combo_query, thresh1=thresh1,
                             thresh2=thresh2, nc=nc, iters=iters)

    #### STEP FOUR: CONVERT TO MAP ####
    euv_map, chd_map = cr_funcs.create_map(iit_image, chd_image, methods_list, row, map_x=map_x, map_y=map_y, R0=R0)

    #### STEP FIVE: CREATE COMBINED MAPS ####
    euv_combined, chd_combined, combined_method, chd_combined_method = cr_funcs.cr_map(euv_map, chd_map, euv_combined,
                                                                                       chd_combined, data_info,
                                                                                       map_info,
                                                                                       mu_cutoff=mu_cutoff,
                                                                                       mu_merge_cutoff=mu_merge_cutoff)

#### STEP SIX: PLOT COMBINED MAP AND SAVE TO DATABASE ####
cr_funcs.save_maps(db_session, map_data_dir, euv_combined, chd_combined, data_info, map_info, methods_list,
                   combined_method, chd_combined_method)

#### CREATE QUALITY MAPS
dp_funcs.quality_map(db_session, map_data_dir, inst_list, query_pd, euv_combined,
                     chd_combined=None, color_list=color_list)



