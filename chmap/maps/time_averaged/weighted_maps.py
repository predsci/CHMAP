"""
weight maps based off dates
smooth weighting curve around date you care about
"""

import os
import datetime

import chmap.data.corrections.apply_lbc_iit as apply_lbc_iit
from chmap.settings.app import App
import chmap.database.db_classes as db_class
import chmap.database.db_funs as db_funcs
import chmap.maps.synoptic.synoptic_funcs as cr_funcs
import chmap.maps.time_averaged.dp_funcs as dp_funcs
import numpy as np

# TIME PARAMETERS
# center date
center_time = datetime.datetime(2011, 5, 16, 12, 0, 0)
# map frequency, in hours
map_freq = 2
# timescale
timescale = datetime.timedelta(weeks=4, days=3)

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
sigma = 0.15  # sigma value for time weighted gaussian distribution

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
# USER MUST INITIALIZE
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

# 1.) get instrument combos based on timescale
query_time_min = center_time - (timescale / 2)
query_time_max = center_time + (timescale / 2)
lbc_combo_query, iit_combo_query = apply_lbc_iit.get_inst_combos(db_session, inst_list, time_min=query_time_min,
                                                                 time_max=query_time_max)

#### STEP ONE: SELECT IMAGES ####
# 1.) query some images
query_pd = db_funcs.query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)

# 2.) generate a dataframe to record methods
methods_list = db_funcs.generate_methdf(query_pd)

# 3.) generate normal distribution
norm_dist = dp_funcs.gauss_time(query_pd, sigma)

#### LOOP THROUGH IMAGES ####
euv_combined = None
chd_combined = None
data_info = []
map_info = []
sum_wgt = 0
for row in query_pd.iterrows():
    # determine correct weighting based off date
    date_time = row[1].date_obs
    weight = norm_dist[row[0]]  # normal distribution at row index
    #### STEP TWO: APPLY PRE-PROCESSING CORRECTIONS ####
    los_image, iit_image, methods_list, use_indices = cr_funcs.apply_ipp(db_session, hdf_data_dir, inst_list, row,
                                                                         methods_list, lbc_combo_query,
                                                                         iit_combo_query,
                                                                         n_intensity_bins=n_intensity_bins, R0=R0)

    #### STEP THREE: CORONAL HOLE DETECTION ####
    chd_image = cr_funcs.chd(db_session, inst_list, los_image, iit_image, use_indices, iit_combo_query,
                             thresh1=thresh1,
                             thresh2=thresh2, nc=nc, iters=iters)

    #### STEP FOUR: CONVERT TO MAP ####
    euv_map, chd_map = cr_funcs.create_map(iit_image, chd_image, methods_list, row, map_x=map_x, map_y=map_y, R0=R0)

    #### STEP FIVE: CREATE COMBINED MAPS ####
    euv_combined, chd_combined, sum_wgt, combined_method = dp_funcs.time_wgt_map(euv_map, chd_map, euv_combined,
                                                                                 chd_combined, data_info, map_info,
                                                                                 weight, sum_wgt, sigma, mu_cutoff)


#### STEP SIX: SAVE TO DATABASE ####
dp_funcs.save_gauss_time_maps(db_session, map_data_dir, euv_combined, chd_combined, data_info, map_info, methods_list,
                               combined_method)

# ### TESTERS
# # UPPER TIME SERIES
# date_time = center_time
# sum = 0
# while query_time_max >= date_time >= query_time_min:
#     date_time += datetime.timedelta(hours=map_freq)
#
# # LOWER TIME SERIES
# date_time = center_time
# while query_time_max >= date_time >= query_time_min:
#     # determine correct image row
#     date_time -= datetime.timedelta(hours=map_freq)
#     row = query_pd[query_pd.date_obs == date_time]
#
# import numpy as np
# from scipy.stats import norm
# import matplotlib.pyplot as plt
# #x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
# x = np.arange(0, 2, len(query_pd))
# norm_dist = norm.pdf(x, loc=1, scale=0.15)
# plt.plot(x, norm_dist)
