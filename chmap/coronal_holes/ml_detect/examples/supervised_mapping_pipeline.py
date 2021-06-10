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
import time

import chmap.utilities.plotting.psi_plotting as Plotting
import chmap.coronal_holes.ml_detect.tools.ml_functions as ml_funcs
import chmap.database.db_classes as db_class
import chmap.database.db_funs as db_funcs
import chmap.data.corrections.apply_lbc_iit as apply_lbc_iit
import chmap.maps.image2map as image2map
import chmap.maps.midm as midm
import chmap.maps.synchronic.synch_utils as synch_utils

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
# USER MUST INITIALIZE
map_data_dir = 'path/to/map/directory'
raw_data_dir = 'path/to/raw_data/directory'
hdf_data_dir = 'path/to/processed_data/directory'
database_dir = 'path/to/database/directory'
sqlite_filename = 'path/to/database/filename'
# path to CHD Model
model_h5 = '/chmap/coronal_holes/ml_detect/tools/model_unet.h5'

# initialize database connection
# using mySQL
use_db = "mysql-Q"
user = "tervin"
password = ""

# using sqlite
# use_db = "sqlite"

# INSTRUMENTS
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
# CORRECTION PARAMETERS
n_intensity_bins = 200
R0 = 1.01


# MINIMUM MERGE MAPPING PARAMETERS
del_mu = None  # optional between this method and mu_merge_cutoff method
mu_cutoff = 0.0  # lower mu cutoff value
mu_merge_cutoff = 0.4  # mu cutoff in overlap areas
EUV_CHD_sep = False  # Do separate minimum intensity merges for image and CHD

# MAP PARAMETERS
x_range = [0, 2 * np.pi]
y_range = [-1, 1]
map_nycoord = 720
map_nxcoord = 1800

# generate map x,y grids. y grid centered on equator, x referenced from lon=0
map_y = np.linspace(y_range[0], y_range[1], map_nycoord, dtype='<f4')
map_x = np.linspace(x_range[0], x_range[1], map_nxcoord, dtype='<f4')

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
start_time = time.time()
# 1.) query some images
query_pd = db_funcs.query_euv_images(db_session=db_session, time_min=query_time_min - del_interval_dt,
                                     time_max=query_time_max + del_interval_dt)

# 3.) load unet neural network
model = ml_funcs.load_model(model_h5, IMG_SIZE=2048, N_CHANNELS=3)

#### STEP TWO: APPLY PRE-PROCESSING CORRECTIONS ####
# 1.) get dates
moving_avg_centers = synch_utils.get_dates(time_min=query_time_min, time_max=query_time_max, map_freq=map_freq)

# 2.) loop through center dates
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
        chd_image_list = ml_funcs.ml_chd(model, iit_list, los_list, use_indices, inst_list)

        #### STEP FOUR: CONVERT TO MAP ####
        map_list, chd_map_list, methods_list, data_info, map_info = \
            image2map.create_singles_maps_2(synch_images, iit_list, chd_image_list,
                                            methods_list=methods_list, map_x=map_x, map_y=map_y, R0=R0)
        # create one data object with EUV & CHD map
        for ii in range(map_list.__len__()):
            # first combine chd and image into a single map object
            map_list[ii].chd = chd_map_list[ii].data.astype('float16')

        #### STEP FIVE: CREATE COMBINED MAPS ####
        synchronic_map = midm.create_combined_maps_2(map_list, mu_merge_cutoff=mu_merge_cutoff, del_mu=del_mu,
            mu_cutoff=mu_cutoff, EUV_CHD_sep=EUV_CHD_sep)
        # add synchronic clustering method to final map
        synchronic_map.append_method_info(cluster_method)

        #### STEP SIX: PLOT SYNCHRONIC MAPS ####
        synchronic_map.chd = np.where(synchronic_map.chd > 0, 1, -9999)
        title = 'Minimum Intensity Merge: Supervised (CNN) Detection Map\nDate: ' + str(center)
        Plotting.PlotMap(synchronic_map, nfig="Supervised Map",
                         title=title, map_type='EUV')
        Plotting.PlotMap(synchronic_map, nfig="Supervised Map", title=title, map_type='Contour')


end_time = time.time()
print("Total elapsed time: ", end_time - start_time)
