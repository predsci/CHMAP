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
import numpy as np
import datetime
from chmap.settings.app import App
import modules.DB_classes as db_class
import modules.DB_funs as db_funcs
import modules.Plotting as Plotting
import analysis.chd_analysis.CHD_pipeline_funcs as chd_funcs
import analysis.ml_analysis.ch_detect.ml_functions as ml_funcs

# -------- parameters --------- #
# TIME RANGE FOR QUERYING
query_time_min = datetime.datetime(2011, 8, 16, 0, 0, 0)
query_time_max = datetime.datetime(2011, 8, 16, 3, 0, 0)
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

# CHD Model
model_h5 = '/Volumes/CHD_DB/model_unet_FINAL.h5'

# MINIMUM MERGE MAPPING PARAMETERS
del_mu = None  # optional between this method and mu_merge_cutoff method
mu_cutoff = 0.0  # lower mu cutoff value
mu_merge_cutoff = 0.4  # mu cutoff in overlap areas

# MAP PARAMETERS
x_range = [0, 2 * np.pi]
y_range = [-1, 1]
map_nycoord = 720
map_nxcoord = 1800

# generate map x,y grids. y grid centered on equator, x referenced from lon=0
map_y = np.linspace(y_range[0], y_range[1], map_nycoord, dtype='<f4')
map_x = np.linspace(x_range[0], x_range[1], map_nxcoord, dtype='<f4')

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
        chd_image_list = ml_funcs.ml_chd(model, iit_list, los_list, use_indices, inst_list)

        #### STEP FOUR: CONVERT TO MAP ####
        map_list, methods_list, data_info, map_info = chd_funcs.create_singles_maps(inst_list, date_pd,
                                                                                    iit_list,
                                                                                    chd_image_list,
                                                                                    methods_list, map_x,
                                                                                    map_y, R0)
        #### STEP FIVE: CREATE COMBINED MAPS AND SAVE TO DB ####
        euv_combined = chd_funcs.create_combined_maps_2(map_list, mu_merge_cutoff=mu_merge_cutoff, del_mu=None, mu_cutoff=0.0,
                               EUV_CHD_sep=True, low_int_filt=1.)
        # create combined maps
        # euv_maps = []
        # chd_maps = []
        # for euv_map in map_list:
        #     if len(euv_map.data) != 0:
        #         euv_maps.append(euv_map)
        # for chd_map in chd_map_list:
        #     if len(chd_map.data) != 0:
        #         chd_maps.append(chd_map)
        #
        # euv_combined = combine_maps(euv_maps, mu_merge_cutoff=mu_merge_cutoff,
        #                                           mu_cutoff=mu_cutoff)
        # chd_combined = combine_maps(chd_maps, mu_merge_cutoff=mu_merge_cutoff,
        #                             mu_cutoff=mu_cutoff)
        # combined_method = {'meth_name': ("Min-Int-Merge-mu_merge", "Min-Int-Merge-mu_merge"), 'meth_description':
        #     ["Minimum intensity merge for synchronic map: based on Caplan et. al."] * 2,
        #                    'var_name': ("mu_cutoff", "mu_merge_cutoff"),
        #                    'var_description': ("lower mu cutoff value",
        #                                        "mu cutoff value in areas of "
        #                                        "overlap"),
        #                    'var_val': (mu_cutoff, mu_merge_cutoff)}
        #
        # # generate a record of the method and variable values used for interpolation
        # euv_combined.append_method_info(methods_list)
        # euv_combined.append_method_info(pd.DataFrame(data=combined_method))
        # euv_combined.append_data_info(data_info)
        # euv_combined.append_map_info(map_info)
        #
        #
        # # plot maps
        # # area constraint
        # chd_clustered = euv_combined.chd
        # chd_labeled = measure.label(chd_clustered, connectivity=2, background=0, return_num=True)
        #
        # # get area
        # chd_area = [props.area for props in measure.regionprops(chd_labeled[0])]
        #
        # # remove CH with less than 7 pixels in area
        # chd_good_area = np.where(np.array(chd_area) > 2)
        # indices = []
        # chd_plot = np.zeros(chd_labeled[0].shape)
        # for val in chd_good_area[0]:
        #     val_label = val + 1
        #     indices.append(np.logical_and(chd_labeled[0] == val_label, val in chd_good_area[0]))
        # for idx in indices:
        #     chd_plot[idx] = chd_labeled[0][idx] + 1
        #
        # chd_plot = np.where(chd_plot > 0, 1, 0)
        # chd_labeled = measure.label(chd_plot, connectivity=2, background=0, return_num=True)
        # euv_combined.chd = chd_plot
        # psi_chd_map = psi_d_types.PsiMap(data=euv_combined.data, chd=chd_plot, x=map_x, y=map_y)
        euv_combined.chd = np.where(euv_combined.chd > 0, 1, -9999)
        title = 'Minimum Intensity Merge: Supervised (CNN) Detection Map\nDate: ' + str(center)
        Plotting.PlotMap(euv_combined, nfig="Supervised Map",
                         title=title, map_type='EUV')
        Plotting.PlotMap(euv_combined, nfig="Supervised Map", title=title, map_type='Contour')
        #
        #
