#!/usr/bin/env python

"""
create grid of CHD Contour maps to test Threshold values
"""

import os
import datetime

import sys
sys.path.append("/Users/tamarervin/CH_Project/CHD")
from chmap.settings.app import App
import chmap.database.db_classes as db_class
import chmap.database.db_funs as db_funcs
import chmap.coronal_holes.detection.chd_funcs as chd_funcs

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import chmap.data.corrections.apply_lbc_iit as apply_lbc_iit
from chmap.maps.util.map_manip import combine_maps
import chmap.maps.image2map as image2map

# -------- parameters --------- #
# TIME RANGE FOR QUERYING
query_times = [datetime.datetime(2011, 4, 1, 0, 0, 0), datetime.datetime(2011, 5, 1, 0, 0, 0),
               datetime.datetime(2011, 6, 1, 0, 0, 0),
               datetime.datetime(2011, 7, 1, 0, 0, 0), datetime.datetime(2011, 8, 1, 0, 0, 0)]
# query_times = [datetime.datetime(2011, 4, 1, 0, 0, 0), datetime.datetime(2011, 5, 1, 0, 0, 0)]
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
# region-growing threshold parameter lists
threshold_values1 = [0.90, 0.925, 0.95, 0.975]
threshold_values2 = [1.30, 1.325, 1.35, 1.375]
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
map_nycoord = 720
# del_y = (y_range[1] - y_range[0]) / (map_nycoord - 1)
# map_nxcoord = (np.floor((x_range[1] - x_range[0]) / del_y) + 1).astype(int)
map_nxcoord = 1800
# generate map x,y grids. y grid centered on equator, x referenced from lon=0
map_y = np.linspace(y_range[0], y_range[1], map_nycoord, dtype='<f4')
map_x = np.linspace(x_range[0], x_range[1], map_nxcoord, dtype='<f4')

### --------- NOTHING TO UPDATE BELOW -------- ###
### determine subplots and lists needed
#  fig, axes = plt.subplots(len(threshold_values1), sharex=True, sharey=True)
# map_list = [None] * len(threshold_values1)
# chd_map_list = [None] * len(threshold_values1)
# euv_combined = [[datatypes.PsiMap() for j in range(len(query_times))] for i in range(len(threshold_values1))]
# chd_combined = [[datatypes.PsiMap() for j in range(len(query_times))] for i in range(len(threshold_values1))]

### get instrument combos
lbc_combo_query, iit_combo_query = apply_lbc_iit.get_inst_combos(db_session, inst_list,
                                                                 time_min=min(query_times),
                                                                 time_max=max(query_times))

### LOOP THROUGH EACH OF THE DATES ###
for index, date in enumerate(query_times):
    print("Creating synchronic maps for", date)
    # times
    query_time_min = date - datetime.timedelta(hours=map_freq / 2, minutes=5)
    query_time_max = date + datetime.timedelta(hours=map_freq / 2, minutes=5)
    #### STEP ONE: SELECT IMAGES ####
    # 1.) query some images
    query_pd = db_funcs.query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)

    # 2.) generate a dataframe to record methods
    methods_list = db_funcs.generate_methdf(query_pd)

    #### STEP TWO: APPLY PRE-PROCESSING CORRECTIONS ####
    date_pd, los_list, iit_list, use_indices, methods_list, ref_alpha, ref_x = apply_lbc_iit.apply_ipp(db_session, date,
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
        for t1_index, thresh1 in enumerate(threshold_values1):
            for t2_index, thresh2 in enumerate(threshold_values2):
                plt.figure(str(date.date) + " Thresh 1: " + str(thresh1) + " Thresh 2: " +  str(thresh2))
                chd_image = chd_funcs.chd(iit_list, los_list, use_indices, inst_list, thresh1,
                                          thresh2,
                                          ref_alpha, ref_x, nc, iters)
                #### STEP FOUR: CONVERT TO MAP ####
                map_list, chd_map_list, methods_list, data_info, map_info = image2map.create_singles_maps(inst_list,
                                                                                                          date_pd,
                                                                                                          iit_list,
                                                                                                          chd_image,
                                                                                                          methods_list,
                                                                                                          map_x,
                                                                                                          map_y, R0)

                #### STEP FIVE: CREATE COMBINED MAPS####
                # create combined maps
                euv_maps = []
                chd_maps = []
                for euv_map in map_list:
                    if len(euv_map.data) != 0:
                        euv_maps.append(euv_map)
                for chd_map in chd_map_list:
                    if len(chd_map.data) != 0:
                        chd_maps.append(chd_map)
                if del_mu is not None:
                    euv_combined, chd_combined = combine_maps(euv_maps, chd_maps,
                                                              del_mu=del_mu,
                                                              mu_cutoff=mu_cutoff)
                    combined_method = {'meth_name': ("Min-Int-Merge-del_mu", "Min-Int-Merge-del_mu"),
                                       'meth_description':
                                           ["Minimum intensity merge for synchronic map: using del mu"] * 2,
                                       'var_name': ("mu_cutoff", "del_mu"), 'var_description': ("lower mu cutoff value",
                                                                                                "max acceptable mu range"),
                                       'var_val': (mu_cutoff, del_mu)}
                else:
                    euv_combined, chd_combined = combine_maps(euv_maps, chd_maps,
                                                              mu_merge_cutoff=mu_merge_cutoff,
                                                              mu_cutoff=mu_cutoff)
                    combined_method = {'meth_name': ("Min-Int-Merge-mu_merge", "Min-Int-Merge-mu_merge"),
                                       'meth_description':
                                           ["Minimum intensity merge for synchronic map: based on Caplan et. al."] * 2,
                                       'var_name': ("mu_cutoff", "mu_merge_cutoff"),
                                       'var_description': ("lower mu cutoff value",
                                                           "mu cutoff value in areas of "
                                                           "overlap"),
                                       'var_val': (mu_cutoff, mu_merge_cutoff)}

                # generate a record of the method and variable values used for interpolation
                euv_combined.append_method_info(methods_list)
                euv_combined.append_method_info(pd.DataFrame(data=combined_method))
                euv_combined.append_data_info(data_info)
                euv_combined.append_map_info(map_info)
                chd_combined.append_method_info(methods_list)
                chd_combined.append_method_info(pd.DataFrame(data=combined_method))
                chd_combined.append_data_info(data_info)
                chd_combined.append_map_info(map_info)

                # plot maps
                euv_map_plot = euv_combined
                chd_map_plot = chd_combined
                # convert map x-extents to degrees
                x_range = [180 * euv_map_plot.x.min() / np.pi, 180 * euv_map_plot.x.max() / np.pi]
                # setup xticks
                xticks = np.arange(x_range[0], x_range[1] + 1, 30)
                # color maps
                chd_cmap = plt.get_cmap('Purples')
                euv_cmap = plt.get_cmap('sohoeit195')
                norm = mpl.colors.LogNorm(vmin=1.0, vmax=np.nanmax(euv_map_plot.data))
                # plot onto subplots
                x_extent = np.linspace(x_range[0], x_range[1], len(chd_map_plot.x))

                # plt.contour(x_extent, chd_map_plot.y, chd_map_plot.data, origin="lower", cmap=chd_cmap,
                # extent=[x_range[0], x_range[1], chd_map_plot.y.min(), chd_map_plot.y.max()], linewidths=0.5, levels=1)
                # plt.imshow(chd_map_plot.data, extent=[x_range[0], x_range[1], euv_map_plot.y.min(),
                # euv_map_plot.y.max()], origin="lower", cmap=euv_cmap, aspect=90.0, norm=norm)

                plt.imshow(euv_map_plot.data,
                           extent=[x_range[0], x_range[1], euv_map_plot.y.min(), euv_map_plot.y.max()],
                           origin="lower", cmap=euv_cmap, aspect=90.0, norm=norm)
                plt.title("Date: " + str(date.date()) + '\nThresholds: ' + str(thresh1) + ', ' + str(thresh2))
                #
                # , bbox_inches="tight"

                image = chd_map_plot
                maskimg = np.zeros(image.data.shape, dtype='int')
                maskimg[image.data > 0] = 3

                mapimg = (maskimg == 3)

                # a vertical line segment is needed, when the pixels next to each other horizontally
                #   belong to different groups (one is part of the mask, the other isn't)
                # after this ver_seg has two arrays, one for row coordinates, the other for column coordinates
                ver_seg = np.where(mapimg[:, 1:] != mapimg[:, :-1])

                # the same is repeated for horizontal segments
                hor_seg = np.where(mapimg[1:, :] != mapimg[:-1, :])

                # in order to draw a discontinuous line, we add Nones in between segments
                l = []
                for p in zip(*hor_seg):
                    l.append((p[1], p[0] + 1))
                    l.append((p[1] + 1, p[0] + 1))
                    l.append((np.nan, np.nan))

                # and the same for vertical segments
                for p in zip(*ver_seg):
                    l.append((p[1] + 1, p[0]))
                    l.append((p[1] + 1, p[0] + 1))
                    l.append((np.nan, np.nan))

                # now we transform the list into a numpy array of Nx2 shape
                segments = np.array(l)

                # now we need to know something about the image which is shown
                #   at this point let's assume it has extents (x0, y0)..(x1,y1) on the axis
                #   drawn with origin='lower'
                # with this information we can rescale our points
                x0 = x_range[0]
                x1 = x_range[1]
                y0 = np.min(image.y)
                y1 = np.max(image.y)
                segments[:, 0] = x0 + (x1 - x0) * segments[:, 0] / mapimg.shape[1]
                segments[:, 1] = y0 + (y1 - y0) * segments[:, 1] / mapimg.shape[0]

                # and now there isn't anything else to do than plot it
                plt.plot(segments[:, 0], segments[:, 1], color="red", linewidth=0.3)
                plt.savefig('threshold/' + str(date.date()) + "/" + str(thresh1) + "_" + str(thresh2) + '.png')


