#!/usr/bin/env python
"""
Tamar Ervin
Date: June 9, 2021
comparison of unsupervised learning algorithm
with the ezseg algorithm by comparing areas
"""

import os
import sys
sys.path.append("/Users/tamarervin/CH_Project/CHD")
# This can be a computationally intensive process.
# To limit number of threads numpy can spawn:
# os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from skimage import measure
import chmap.database.db_classes as db_class
import chmap.database.db_funs as db_funcs
import chmap.data.corrections.apply_lbc_iit as apply_lbc_iit
import chmap.coronal_holes.detection.chd_funcs as chd_funcs
import chmap.coronal_holes.ml_detect.tools.ml_functions as ml_funcs
import chmap.maps.util.map_manip as map_manip
import chmap.utilities.datatypes.datatypes as datatypes
import chmap.maps.image2map as image2map
import chmap.maps.midm as midm
import chmap.maps.synchronic.synch_utils as synch_utils
import chmap.utilities.plotting.psi_plotting as Plotting

# -------- parameters --------- #
# TIME RANGE FOR QUERYING
query_time_min = datetime.datetime(2011, 10, 16, 0, 0, 0)
query_time_max = datetime.datetime(2012, 1, 16, 0, 0, 0)
# define map interval cadence and width
map_freq = 6  # number of hours
interval_delta = 30  # number of minutes
del_interval_dt = datetime.timedelta(minutes=interval_delta)
del_interval = np.timedelta64(interval_delta, 'm')

# INITIALIZE DATABASE CONNECTION
# DATABASE PATHS
# map_data_dir = App.MAP_FILE_HOME
# raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = '/Volumes/CHD_DB/processed_images'
database_dir = '/Volumes/CHD_DB'
sqlite_filename = '/Volumes/CHD_DB/CHD_DB.db'
# designate which database to connect to
use_db = "mysql-Q"  # 'sqlite'  Use local sqlite file-based db
# use_db = 'sqlite'
# 'mysql-Q' Use the remote MySQL database on Q
# 'mysql-Q_test' Use the development database on Q
user = "tervin"  # only needed for remote databases.
password = "luv2runFH"  # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn_old() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

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
# k-means parameters
N_CLUSTERS = 14
weight = 1.4

# MINIMUM MERGE MAPPING PARAMETERS
del_mu = None  # optional between this method and mu_merge_cutoff method (not both)
mu_cutoff = 0.0  # lower mu cutoff value
mu_merge_cutoff = 0.4  # mu cutoff in overlap areas
EUV_CHD_sep = False  # Do separate minimum intensity merges for image and CHD

# MAP PARAMETERS
x_range = [0, 2 * np.pi]
y_range = [-1, 1]
reduce_map_nycoord = 640
reduce_map_nxcoord = 1600
full_map_nycoord = 640
full_map_nxcoord = 1600
low_res_nycoord = 160
low_res_nxcoord = 400

# generate map x,y grids. y grid centered on equator, x referenced from lon=0
full_map_y = np.linspace(y_range[0], y_range[1], full_map_nycoord, dtype='<f4')
full_map_x = np.linspace(x_range[0], x_range[1], full_map_nxcoord, dtype='<f4')
reduce_map_y = np.linspace(y_range[0], y_range[1], reduce_map_nycoord, dtype='<f4')
reduce_map_x = np.linspace(x_range[0], x_range[1], reduce_map_nxcoord, dtype='<f4')
low_res_y = np.linspace(y_range[0], y_range[1], low_res_nycoord, dtype='<f4')
low_res_x = np.linspace(x_range[0], x_range[1], low_res_nxcoord, dtype='<f4')

# generate K-Means x, y grids
idx = np.indices((full_map_nxcoord, full_map_nycoord))
idx_row = idx[0]
idx_row = idx_row / np.max(idx_row)
idx_col = idx[1]
idx_col = idx_col / np.max(idx_col)
# flatten arrays
idx_col_flt = idx_col.flatten()
idx_row_flt = idx_row.flatten()

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
query_pd = db_funcs.query_euv_images(db_session=db_session, time_min=query_time_min - del_interval_dt,
                                     time_max=query_time_max + del_interval_dt)

#### STEP TWO: APPLY PRE-PROCESSING CORRECTIONS ####
# 1.) get dates
moving_avg_centers = synch_utils.get_dates(time_min=query_time_min, time_max=query_time_max, map_freq=map_freq)

chd_area_ezseg = np.zeros(len(moving_avg_centers))
chd_area_kmeans = np.zeros(len(moving_avg_centers))

# 3.) loop through center dates
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
        chd_image_list = chd_funcs.chd_2(iit_list, los_list, use_indices, thresh1,
                                         thresh2, ref_alpha, ref_x, nc, iters)

        #### STEP FOUR: CONVERT TO MAP ####
        map_list, chd_map_list, methods_list, data_info, map_info = \
            image2map.create_singles_maps_2(synch_images, iit_list, chd_image_list,
                                            methods_list, full_map_x, full_map_y, R0)

        #### STEP SIX: COMBINE EUV AND CHD MAP INTO ONE OBJECT ####
        for ii in range(map_list.__len__()):
            # combine chd and image into a single map object
            map_list[ii].chd = chd_map_list[ii].data.astype('float16')

        #### STEP FIVE: CREATE COMBINED MAPS ####
        synchronic_map = midm.create_combined_maps_2(
            map_list.copy(), mu_merge_cutoff=mu_merge_cutoff, del_mu=del_mu,
            mu_cutoff=mu_cutoff, EUV_CHD_sep=EUV_CHD_sep)
        # add synchronic clustering method to final map
        synchronic_map.append_method_info(cluster_method)

        # area constraint
        chd_labeled = measure.label(synchronic_map.chd, connectivity=2, background=0, return_num=True)

        # get area
        chd_area = [props.area for props in measure.regionprops(chd_labeled[0])]
        chd_area_ezseg[date_ind] = np.sum(chd_area)

        #### STEP SIX: K-MEANS DETECTION ####
        map = np.where(synchronic_map.data == -9999, 0, synchronic_map.data)
        map2 = np.log(map)
        map2 = np.where(map2 == -np.inf, 0, map2)

        arr = np.zeros((full_map_nxcoord * full_map_nycoord, 3))
        arr[:, 0] = idx_col_flt * weight
        arr[:, 1] = idx_row_flt * weight
        arr[:, 2] = map2.flatten() * 2
        psi_chd_map, psi_ar_map, chd_labeled, ar_labeled = \
            ml_funcs.kmeans_detection(synchronic_map.data, map2, arr, N_CLUSTERS,
                                      full_map_nycoord, full_map_nxcoord, full_map_x, full_map_y)
        chd_area = [props.area for props in measure.regionprops(chd_labeled[0])]
        chd_area_kmeans[date_ind] = np.sum(chd_area)

        # # plot maps
        # title_ezseg = 'Minimum Intensity Merge: EZSEG Detection Map\nDate: ' + str(center)
        # title_kmeans = 'Minimum Intensity Merge: Unsupervised Detection Map\nDate: ' + str(center)
        # title_ar = 'Minimum Intensity Merge: Unsupervised CH/AR Detection Map\nDate: ' + str(center)
        # Plotting.PlotMap(synchronic_map, title=title_ezseg, nfig='ezseg' + str(date_ind))
        # synchronic_map.chd = np.where(synchronic_map.chd > 0, 1, 0)
        # Plotting.PlotMap(synchronic_map, map_type='Contour', title=title_ezseg, nfig='ezseg' + str(date_ind))
        # plt.savefig('/Volumes/CHD_DB/unsupervised/ezseg_' + str(date_ind))
        # Plotting.PlotMap(psi_chd_map, title=title_kmeans, nfig='chd' + str(date_ind))
        # Plotting.PlotMap(psi_chd_map, map_type='Contour', title=title_kmeans, nfig='chd' + str(date_ind))
        # plt.savefig('/Volumes/CHD_DB/unsupervised/kmeans_' + str(date_ind))
        # Plotting.PlotMap(psi_chd_map, title=title_ar, nfig='ar' + str(date_ind))
        # Plotting.PlotMap(psi_chd_map, map_type='Contour', title=title_ar, nfig='ar' + str(date_ind))
        # Plotting.PlotMap(psi_ar_map, map_type='AR_Contour', title=title_ar, nfig='ar' + str(date_ind))
        # plt.savefig('/Volumes/CHD_DB/unsupervised/kmeans_ar_' + str(date_ind))

# plot area comparison
dates= list(moving_avg_centers)
with open('/Volumes/CHD_DB/unsupervised/area_dates4.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % date for date in dates)

with open('/Volumes/CHD_DB/unsupervised/area_ezseg4.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % area for area in chd_area_ezseg)

with open('/Volumes/CHD_DB/unsupervised/area_kmeans4.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % area for area in chd_area_kmeans)

plt.scatter(dates, chd_area_ezseg/10000, color='Black', label='EZSEG CH Area')
plt.scatter(dates, chd_area_kmeans/10000, color='Blue', label='Unsupervised CH Area')
plt.title("Comparison of CH Areas between EZSEG and \n Unsupervised Learning Detection Methods")
plt.xlabel("Dates: " + str(moving_avg_centers[0]) + " to " + str(moving_avg_centers[-1]))
plt.ylabel("Pixel Area (10^4)")
plt.xticks(color='w')
plt.legend()
plt.savefig('/Volumes/CHD_DB/unsupervised/comparison_plot4')

### build new comparison plot from lists in text files
dates_file = open('/Volumes/CHD_DB/unsupervised/area_dates4.txt', 'r')
dates_list = dates_file.readlines()

ezseg_file = open('/Volumes/CHD_DB/unsupervised/area_ezseg4.txt', 'r')
ezseg_list = ezseg_file.readlines()

kmeans_file = open('/Volumes/CHD_DB/unsupervised/area_kmeans4.txt', 'r')
kmeans_list = kmeans_file.readlines()

list1 = np.zeros(len(kmeans_list))
for i, area in enumerate(ezseg_list):
    list1[i] = float(area)

list2 = np.zeros(len(kmeans_list))
for i, area in enumerate(kmeans_list):
    list2[i] = float(area)

plt.plot(list(range(0, len(dates_list))), list1/10000, color='Black', label='EZSEG CH Area')
plt.plot(list(range(0, len(dates_list))), list2/10000, color='Blue', label='Unsupervised CH Area')
plt.title("Comparison of CH Areas between EZSEG and \n Unsupervised Learning Detection Methods")
plt.xlabel("Dates: " + str(dates_list[0]) + " to " + str(dates_list[-1]))
plt.xticks(color='w')
plt.ylabel("Pixel Area (10^4)")
plt.legend()