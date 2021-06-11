#!/usr/bin/env python
"""
Tamar Ervin
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

import os
import numpy as np
import datetime
import h5py as h5

import chmap.database.db_classes as db_class
import chmap.database.db_funs as db_funcs
import chmap.data.corrections.apply_lbc_iit as apply_lbc_iit
import chmap.maps.synchronic.synch_utils as synch_utils
import chmap.coronal_holes.detection.chd_funcs as chd_funcs


# -------- parameters --------- #
# TIME RANGE FOR QUERYING
query_time_min = datetime.datetime(2011, 12, 1, 0, 0, 0)
query_time_max = datetime.datetime(2011, 12, 1, 12, 0, 0)
# define map interval cadence and width
map_freq = 6  # number of hours
interval_delta = 30  # number of minutes
del_interval_dt = datetime.timedelta(minutes=interval_delta)
del_interval = np.timedelta64(interval_delta, 'm')

# INITIALIZE DATABASE CONNECTION
# USER MUST INITIALIZE
map_data_dir = 'path/to/map/directory'
raw_data_dir = 'path/to/raw_data/directory'
hdf_data_dir = 'path/to/processed_data/directory'
database_dir = 'path/to/database/directory'
sqlite_filename = 'path/to/database/filename'
# designate which database to connect to
use_db = "mysql-Q"  # 'sqlite'  Use local sqlite file-based db
# use_db = 'sqlite'
# 'mysql-Q' Use the remote MySQL database on Q
# 'mysql-Q_test' Use the development database on Q
user = "tervin"  # only needed for remote databases.
password = ""  # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn_old() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

# hdf file info
h5_filename = '/Volumes/CHD_DB/data_images_small.h5'

# INSTRUMENTS
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
# CORRECTION PARAMETERS
n_intensity_bins = 200
R0 = 1.01

# DETECTION PARAMETERS
# test size of dataset
test_size = 0.33
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

#### STEP ONE: SELECT IMAGES ####
# 1.) query some images
query_pd = db_funcs.query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)

# 2.) generate a dataframe to record methods
methods_list = db_funcs.generate_methdf(query_pd)

#### STEP TWO: APPLY PRE-PROCESSING CORRECTIONS ####
# 1.) get dates
moving_avg_centers = synch_utils.get_dates(time_min=query_time_min, time_max=query_time_max, map_freq=map_freq)

# 2.) loop through center dates
h5file = h5.File(h5_filename, 'w')
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

    chd_image_list = chd_funcs.chd(iit_list, los_list, use_indices, inst_list, thresh1, thresh2,
                                   ref_alpha, ref_x, nc, iters)
    for i, iit_image in enumerate(iit_list):
        if iit_image is not None:
            g = h5file.create_group(str(center) + "_" + str(iit_image.instrument))
            # create euv image in file
            # scalarMap = mpl.cm.ScalarMappable(norm=colors.LogNorm(vmin=1.0, vmax=np.max(iit_image.iit_data)),
            #                                   cmap='sohoeit195')
            # colorVal = scalarMap.to_rgba(iit_image.iit_data, norm=True)
            g.create_dataset("euv_image", data=iit_image.iit_data)

            # create chd mask in file
            # arr = chd_image_list[i].data
            # arr3D = np.repeat(arr[..., None], 1, axis=2)
            g.create_dataset("chd_data", data=chd_image_list[i].data)
h5file.close()



