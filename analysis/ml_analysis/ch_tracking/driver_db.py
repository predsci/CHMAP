"""Apply Coronal Hole Tracking Algorithm to a particular test case. Importing Detection images from Q
(see Jamie's examples/examples112_query_maps.py for more information on how to access the database).

Last Modified: April 27th, 2021 (Opal)"""

import os
import datetime
import numpy as np

from modules import DB_funs
import modules.DB_classes as DBClass
import modules.datatypes as psi_datatype
from settings.app import App
from analysis.ml_analysis.ch_tracking.src import CoronalHoleDB
from analysis.ml_analysis.ch_tracking.classification import classify_grey_scaled_image
from analysis.ml_analysis.ch_tracking.plots import plot_coronal_hole
from modules.map_manip import MapMesh
import modules.Plotting as EasyPlot
import pickle
import matplotlib.pyplot as plt

# ================================================================================================================
# Step 1: Choose a test case - time interval
# ================================================================================================================
# define map query start and end times
query_start = datetime.datetime(year=2010, month=12, day=20, hour=23, minute=0, second=0)
query_end = datetime.datetime(year=2011, month=4, day=1, hour=1, minute=0, second=0)

# initialize coronal hole tracking database.
ch_lib = CoronalHoleDB()

# ================================================================================================================
# Step 2: Read in detected images from the database.
# ================================================================================================================
# --- User Parameters ----------------------
map_dir = App.MAP_FILE_HOME

# define map type and grid to query
map_methods = ['Synch_Im_Sel', 'GridSize_sinLat', 'MIDM-Comb-del_mu']

# here we specify methods for synchronic image selection, a sine(lat) axis, and
# del_mu driven minimum intensity merge.
grid_size = (400, 160)  # low resolution images.
# parameter values are stored as floats in the DB, so input a range to query for each.
# 'n_phi' and 'n_SinLat' are number of grid points. We also want a del_mu of 0.6
map_vars = {"n_phi": [grid_size[0] - 0.1, grid_size[0] + 0.1],
            "n_SinLat": [grid_size[1] - 0.1, grid_size[1] + 0.1],
            "del_mu": [0.59, 0.61]}

# designate which database to connect to
use_db = "mysql-Q"  # 'sqlite'  Use local sqlite file-based db
# 'mysql-Q' Use the remote MySQL database on Q
# 'mysql-Q_test' Use the development database on Q
user = "opalissan"  # only needed for remote databases.
password = ""  # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

# Establish connection to database
db_session = DB_funs.init_db_conn(db_name=use_db, chd_base=DBClass.Base, user=user,
                                  password=password)

# --- Begin execution ----------------------
# query maps in time range
map_info, data_info, method_info, image_assoc = DB_funs.query_euv_maps(
    db_session, mean_time_range=(query_start, query_end), methods=map_methods,
    var_val_range=map_vars)

# the query returns multiple dataframes that together describe the map-making
# process and constituent images.  Here we are mostly interested in the map_info
# dataframe.  It contains one row per map with a number of information columns:
map_info.keys()

# iterate through the rows of map_info
for row_index, row in map_info.iterrows():
    print("Processing map for:", row.date_mean)
    # load map (some older maps have a leading '/' that messes with os.path.join
    if row.fname[0] == "/":
        rel_path = row.fname[1:]
    else:
        rel_path = row.fname
    map_path = os.path.join(map_dir, rel_path)
    my_map = psi_datatype.read_psi_map(map_path)

    # ================================================================================================================
    # Step 3: Input image, coordinates, mesh spacing, and timestamp.
    # ================================================================================================================
    # note that the grid has been reduced since the coronal
    # hole detection was performed, so values are floats between 0. and 1.
    chd_data = my_map.chd.astype('float32')

    # restrict chd_data to be positive.
    chd_data[chd_data < 0] = 0
    chd_data[chd_data > 1] = 1

    # image coordinates (latitude and longitude).
    phi_coords = my_map.x
    sinlat_coords = my_map.y
    theta_coords = np.pi / 2 + np.arcsin(sinlat_coords)
    mean_timestamp = row.T[2]

    # save mesh map
    ch_lib.Mesh = MapMesh(p=phi_coords, t=theta_coords)

    # ================================================================================================================
    # Step 3: Latitude Weighted Dilation +
    #         Compute all contour features +
    #         Force periodicity and delete small contours.
    # ================================================================================================================
    # get list of contours.
    contour_list_pruned = classify_grey_scaled_image(greyscale_image=chd_data,
                                                     lat_coord=ch_lib.Mesh.t,
                                                     lon_coord=ch_lib.Mesh.p,
                                                     AreaThreshold=ch_lib.AreaThreshold,
                                                     frame_num=ch_lib.frame_num,
                                                     BinaryThreshold=ch_lib.BinaryThreshold,
                                                     gamma=ch_lib.gamma)

    # ================================================================================================================
    # Step 4: Match coronal holes detected to previous frame detections.
    # ================================================================================================================
    ch_lib.assign_new_coronal_holes(contour_list=contour_list_pruned,
                                    timestamp=str(mean_timestamp))

    # ================================================================================================================
    # Step 5: Plot results.
    # ================================================================================================================
    # plot connectivity sub - graphs.
    dir_name = "/Users/opalissan/desktop/CHT_RESULTS/"
    folder_name = "2010-12-20-to-2011-04-01/"
    graph_file_name = "graph_frame_" + str(ch_lib.frame_num) + ".png"
    image_file_name = "classified_frame_" + str(ch_lib.frame_num) + ".png"

    # plot coronal holes in the latest frame.
    plot_coronal_hole(ch_list=ch_lib.window_holder[-1].contour_list, n_t=ch_lib.Mesh.n_t, n_p=ch_lib.Mesh.n_p,
                      title="Frame: " + str(ch_lib.frame_num) + ", Time: " + str(mean_timestamp),
                      filename=dir_name + folder_name + image_file_name)

    if ch_lib.frame_num > 20:
        ch_lib.Graph.plot_num_subgraphs = 5
    elif ch_lib.frame_num > 10:
        ch_lib.Graph.plot_num_subgraphs = 8

    ch_lib.Graph.create_plots(save_dir=dir_name + folder_name + graph_file_name)
    plt.show()

    # iterate over frame number.
    ch_lib.frame_num += 1

    # save in pickle file
    if ch_lib.frame_num == 50:
        # # save object to pickle file.
        with open(dir_name + folder_name + "graph_pickle.pkl", 'wb') as f:
            pickle.dump(ch_lib.Graph, f)

        with open(dir_name + folder_name + "ch_dict.pkl", 'wb') as f:
            pickle.dump(ch_lib.ch_dict, f)

        break

# close database connection
db_session.close()


