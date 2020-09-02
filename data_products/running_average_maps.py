"""
create maps of different time scales then create a weighted average
combination map
"""

import os
import numpy as np
import datetime

from settings.app import App
import modules.DB_classes as db_class
import modules.DB_funs as db_funcs
import analysis.chd_analysis.CHD_pipeline_funcs as chd_funcs
import analysis.chd_analysis.CR_mapping_funcs as cr_funcs

# -------- UPDATEABLE PARAMETERS --------- #

# TIMESCALES
center_time = datetime.datetime(2011, 7, 1, 0, 0, 0)
map_freq = 2  # number of hours between each map
# list of timedelta timescales
timescales = [datetime.timedelta(days=1), datetime.timedelta(weeks=1), datetime.timedelta(weeks=2), datetime.timedelta(weeks=4)]
# weighting of timescales, None for even weighting
timescale_weight = None

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
map_nycoord = 1600
del_y = (y_range[1] - y_range[0]) / (map_nycoord - 1)
map_nxcoord = (np.floor((x_range[1] - x_range[0]) / del_y) + 1).astype(int)
# generate map x,y grids. y grid centered on equator, x referenced from lon=0
map_y = np.linspace(y_range[0], y_range[1], map_nycoord, dtype='<f4')
map_x = np.linspace(x_range[0], x_range[1], map_nxcoord, dtype='<f4')

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

### --------- NOTHING TO UPDATE BELOW -------- ###

# create even weights
if timescale_weight is None:
    timescale_weight = [1.0 / len(timescales)] * len(timescales)

# check map weights add to 1
weight_sum = sum(timescale_weight)
if weight_sum != 1:
    raise ValueError("The timescale weights do not sum to 1. Please reenter weights or change to None "
                     "for even weighting of maps.")

# initialize map lists
euv_timescale = [None] * len(timescales)
chd_timescale = [None] * len(timescales)
image_info_timescale = [None] * len(timescales)
map_info_timescale = [None] * len(timescales)

# get instrument combos based on largest timescale
max_time = max(timescales)
max_time_min = center_time - (max_time / 2)
max_time_max = center_time + (max_time / 2)
lbc_combo_query, iit_combo_query = chd_funcs.get_inst_combos(db_session, inst_list, time_min=max_time_min,
                                                             time_max=max_time_max)

for time_ind, timescale in enumerate(timescales):
    print("Starting map creation for maps of timescale:", timescale, "\n")
    time_weight = timescale_weight[time_ind]
    query_time_min = center_time - (timescale / 2)
    query_time_max = center_time + (timescale / 2)

    #### STEP ONE: SELECT IMAGES ####
    # 1.) query some images
    query_pd = db_funcs.query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)

    # 2.) generate a dataframe to record methods
    methods_list = db_funcs.generate_methdf(query_pd)

    #### LOOP THROUGH IMAGES ####
    euv_combined = None
    chd_combined = None
    image_info = []
    map_info = []
    for row in query_pd.iterrows():
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
        euv_combined, chd_combined, combined_method = cr_funcs.cr_map(euv_map, chd_map, euv_combined,
                                                                      chd_combined, image_info, map_info,
                                                                      mu_cutoff=mu_cutoff,
                                                                      mu_merge_cutoff=mu_merge_cutoff)
    # add maps and info to timescale lists
    euv_timescale[time_ind] = euv_combined
    chd_timescale[time_ind] = chd_combined
    image_info_timescale[time_ind] = image_info
    map_info_timescale[time_ind] = map_info

#### STEP SIX: COMBINE TIMESCALE MAPS AND SAVE TO DATABASE ####



import numpy as np
import modules.datatypes as psi_d_types
from settings.info import DTypes

map_list = euv_timescale
chd_map_list = chd_timescale
timescale_weights = [0.1, 0.2, 0.3, 0.4]




import modules.Plotting as Plotting

Plotting.PlotMap(euv_time_combined, nfig="EUV Map Timescale Weighted",
                 title="EUV Map Timescale Weighted\nTime Min: " + str(max_time_min) + "\nTime Max: " + str(
                     max_time_max))
Plotting.PlotMap(euv_time_combined, "CHD Map Timescale Weighted", title="CHD Map Timescale Weighted\nTime Min: " + str(max_time_min) + "\nTime Max: " + str(
                     max_time_max))
Plotting.PlotMap(chd_time_combined, nfig="CHD Map Timescale Weighted", title="CHD Map Timescale Weighted\nTime Min: " +
                                                                             str(max_time_min) + "\nTime Max: " + str(
    max_time_max), map_type='CHD')

Plotting.PlotMap(chd_timescale[0], nfig="CR CHD Map",
                 title="Minimum Intensity CR CHD Merge Map\nTime Min: " + str(max_time_min) + "\nTime Max: " + str(
                     max_time_max), map_type='CHD')
Plotting.PlotMap(chd_timescale[1], nfig="CR CHD Map1",
                 title="Minimum Intensity CR CHD Merge Map\nTime Min: " + str(max_time_min) + "\nTime Max: " + str(
                     max_time_max), map_type='CHD')

Plotting.PlotMap(euv_timescale[0], nfig="EUV Map0",
                 title="Minimum Intensity EUV Merge Map\nTime Min: " + str(max_time_min) + "\nTime Max: " + str(
                     max_time_max) + "Timescale: 1 Day")
Plotting.PlotMap(euv_timescale[1], nfig="EUV Map1",
                 title="Minimum Intensity EUV Merge Map\nTime Min: " + str(max_time_min) + "\nTime Max: " + str(
                     max_time_max)+ "Timescale: 5 Days")


for ind, map in enumerate(euv_timescale):
    Plotting.PlotMap(euv_timescale[ind], nfig="EUV Map "+str(ind),
                     title="Minimum Intensity EUV Merge Map\nTime Min: " + str(max_time_min) + "\nTime Max: " + str(
                         max_time_max))
    Plotting.PlotMap(chd_timescale[ind], nfig="CR CHD Map "+str(ind),
                     title="Minimum Intensity CR CHD Merge Map\nTime Min: " + str(max_time_min) + "\nTime Max: " + str(
                         max_time_max), map_type='CHD')


