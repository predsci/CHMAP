"""
methods to create full CR maps with an unknown number of images
inputs needed: min and max time, instrument list
"""

import os
import numpy as np
import datetime
import pandas as pd

from settings.app import App
from modules.map_manip import combine_maps
import modules.Plotting as Plotting
import ezseg.ezsegwrapper as ezsegwrapper
import modules.datatypes as datatypes
import modules.DB_classes as db_class
import modules.DB_funs as db_funcs
import analysis.chd_analysis.CHD_pipeline_funcs as chd_funcs
import analysis.lbcc_analysis.LBCC_theoretic_funcs as lbcc_funcs
import analysis.iit_analysis.IIT_pipeline_funcs as iit_funcs

# -------- parameters --------- #
# TIME RANGE FOR QUERYING
query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
query_time_max = datetime.datetime(2011, 4, 10, 0, 0, 0)
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
del_mu = None  # optional between this method and mu_cut_over method
mu_cutoff = 0.0  # lower mu cutoff value
mu_cut_over = 0.4  # mu cutoff in overlap areas

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

#### STEP TWO: APPLY PRE-PROCESSING CORRECTIONS ####
# 1.) get dates
moving_avg_centers = chd_funcs.get_dates(time_min=query_time_min, time_max=query_time_max, map_freq=map_freq)

# 2.) get instrument combos
lbc_combo_query, iit_combo_query = chd_funcs.get_inst_combos(db_session, inst_list, time_min=query_time_min,
                                                             time_max=query_time_max)

#### LOOP THROUGH IMAGES ####
euv_combined = None
chd_combined = None
image_info = []
map_info = []
for index, row in query_pd.iterrows():
    #### STEP TWO: APPLY PRE-PROCESSING CORRECTIONS ####
    inst_ind = inst_list.index(row.instrument)
    # apply LBC
    los_image, lbcc_image, mu_indices, use_ind, theoretic_query = lbcc_funcs.apply_lbc(db_session, hdf_data_dir,
                                                                                       lbc_combo_query[inst_ind],
                                                                                       image_row=row,
                                                                                       n_intensity_bins=n_intensity_bins,
                                                                                       R0=R0)
    # apply IIT
    lbcc_image, iit_image, use_indices, alpha, x = iit_funcs.apply_iit(db_session, iit_combo_query[inst_ind],
                                                                       lbcc_image, use_ind, los_image, R0=R0)
    # add methods to dataframe
    ipp_method = {'meth_name': ("LBCC", "IIT"), 'meth_description': ["LBCC Theoretic Fit Method", "IIT Fit Method"],
                  'var_name': ("LBCC", "IIT"), 'var_description': (" ", " ")}
    methods_list[index] = methods_list[index].append(pd.DataFrame(data=ipp_method), sort=False)

    #### STEP THREE: CORONAL HOLE DETECTION ####
    # reference alpha, x for threshold
    sta_ind = inst_list.index('EUVI-A')
    ref_alpha, ref_x = db_funcs.query_var_val(db_session, meth_name='IIT', date_obs=row.date_obs,
                                              inst_combo_query=iit_combo_query[sta_ind])
    image_data = iit_image.iit_data
    use_chd = use_indices.astype(int)
    use_chd = np.where(use_chd == 1, use_chd, -9999)
    nx = iit_image.x.size
    ny = iit_image.y.size
    t1 = thresh1 * ref_alpha + ref_x
    t2 = thresh2 * ref_alpha + ref_x
    ezseg_output, iters_used = ezsegwrapper.ezseg(np.log10(image_data), use_chd, nx, ny, t1, t2, nc, iters)
    chd_result = np.logical_and(ezseg_output == 0, use_chd == 1)
    chd_result = chd_result.astype(int)
    chd_image = datatypes.create_chd_image(los_image, chd_result)
    chd_image.get_coordinates()

    #### STEP FOUR: CONVERT TO MAP ####
    # EUV map
    euv_map = iit_image.interp_to_map(R0=R0, map_x=map_x, map_y=map_y, image_num=row.image_id)
    # CHD map
    chd_map = chd_image.interp_to_map(R0=R0, map_x=map_x, map_y=map_y, image_num=row.image_id)
    # record image and map info
    euv_map.append_image_info(row)
    chd_map.append_image_info(row)
    image_info.append(row)
    map_info.append(euv_map.map_info)

    # generate a record of the method and variable values used for interpolation
    interp_method = {'meth_name': ("Im2Map_Lin_Interp_1",), 'meth_description':
        ["Use SciPy.RegularGridInterpolator() to linearly interpolate from an Image to a Map"] * 1,
                     'var_name': ("R0",), 'var_description': ("Solar radii",), 'var_val': (R0,)}
    # add to the methods dataframe for this map
    methods_list[index] = methods_list[index].append(pd.DataFrame(data=interp_method), sort=False)

    # incorporate the methods dataframe into the map object
    euv_map.append_method_info(methods_list[index])
    chd_map.append_method_info(methods_list[index])

    #### STEP FIVE: CREATE COMBINED MAPS ####
    # create map lists
    euv_maps = [euv_map, ]
    chd_maps = [chd_map, ]
    if euv_combined is not None:
        euv_maps.append(euv_combined)
    if chd_combined is not None:
        chd_maps.append(chd_combined)

    # combine maps with minimum intensity merge
    if del_mu is not None:
        euv_combined, chd_combined = combine_maps(euv_maps, chd_maps, del_mu=del_mu, mu_cutoff=mu_cutoff)
        combined_method = {'meth_name': ("Min-Int-Merge_1", "Min-Int-Merge_1"), 'meth_description':
            ["Minimum intensity merge: using del mu"] * 2,
                           'var_name': ("mu_cutoff", "del_mu"), 'var_description': ("lower mu cutoff value",
                                                                                    "max acceptable mu range"),
                           'var_val': (mu_cutoff, del_mu)}
    else:
        euv_combined, chd_combined = combine_maps(euv_maps, chd_maps, mu_cut_over=mu_cut_over, mu_cutoff=mu_cutoff)
        combined_method = {'meth_name': ("Min-Int-Merge_2", "Min-Int-Merge_2"), 'meth_description':
            ["Minimum intensity merge: based on Caplan et. al."] * 2,
                           'var_name': ("mu_cutoff", "mu_cut_over"), 'var_description': ("lower mu cutoff value",
                                                                                         "mu cutoff value in areas of "
                                                                                         "overlap"),
                           'var_val': (mu_cutoff, mu_cut_over)}

#### STEP SIX: PLOT COMBINED MAP AND SAVE TO DATABASE ####
# generate a record of the method and variable values used for interpolation
euv_combined.append_method_info(methods_list)
euv_combined.append_method_info(pd.DataFrame(data=combined_method))
euv_combined.append_image_info(image_info)
euv_combined.append_map_info(map_info)
chd_combined.append_method_info(methods_list)
chd_combined.append_method_info(pd.DataFrame(data=combined_method))
chd_combined.append_image_info(image_info)
chd_combined.append_map_info(map_info)

# plot maps
Plotting.PlotMap(euv_combined, nfig="EUV Combined map for: " + str(euv_combined.image_info.date_obs[0]),
                 title="Minimum Intensity Merge Map\nDate: " + str(euv_combined.image_info.date_obs[0]))
Plotting.PlotMap(euv_combined, nfig="EUV/CHD Combined map for: " + str(euv_combined.image_info.date_obs[0]),
                 title="Minimum Intensity EUV/CHD Merge Map\nDate: " + str(euv_combined.image_info.date_obs[0]))
Plotting.PlotMap(chd_combined, nfig="EUV/CHD Combined map for: " + str(chd_combined.image_info.date_obs[0]),
                 title="Minimum Intensity EUV/CHD Merge Map\nDate: " + str(chd_combined.image_info.date_obs[0]),
                 map_type='CHD')
