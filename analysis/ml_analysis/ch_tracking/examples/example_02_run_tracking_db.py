""" This is a tutorial/example on how to run the coronal hole tracking algorithm on CH Maps
from in the database (db).

This Module includes the following operations:

1. Importing Detection images (CH Maps - low res) from Q
(see Jamie's examples/examples112_query_maps.py for more information on how to access the database).

2. Save list of coronal holes for each frame (pickle file format).

3. Save a connectivity graph of the coronal hole evolution in time (pickle file format).

4. Save image of the coronal hole detected frame in each iteration + save a plot of the graph then create a side
    by side (.mov)

Last Modified: May 6th, 2021 (Opal)
"""


import os
import datetime
import numpy as np
import cv2
from modules import DB_funs
import modules.DB_classes as DBClass
import modules.datatypes as psi_datatype
from settings.app import App
from analysis.ml_analysis.ch_tracking.src.main import CoronalHoleDB
from analysis.ml_analysis.ch_tracking.src.classification import classify_grey_scaled_image
from analysis.ml_analysis.ch_tracking.tools.plots import plot_coronal_hole
from modules.map_manip import MapMesh

import pickle
import matplotlib.pyplot as plt


# ================================================================================================================
# Step 1: Choose a test case - time interval
# ================================================================================================================
# define map query start and end times
query_start = datetime.datetime(year=2011, month=1, day=1, hour=23, minute=0, second=0)
query_end = datetime.datetime(year=2011, month=2, day=1, hour=1, minute=0, second=0)

# initialize coronal hole tracking database.
ch_lib = CoronalHoleDB()

# ================================================================================================================
# Step 2: Initialize directory and folder to save results (USER PARAMETERS)
# ================================================================================================================
# --- User Parameters ----------------------
dir_name = "/Users/opalissan/desktop/CHT_RESULTS/"
folder_name = "2011-01-01-2011-02-01/"

# ================================================================================================================
# Step 3: Read in detected images from the database.
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
    # Step 4: Input image, coordinates, mesh spacing, and timestamp.
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

    # TODO: ASK JAMIE WHAT IS THE CORRECT TIMESTAMP?
    mean_timestamp = row.T[2]

    # save mesh map
    ch_lib.Mesh = MapMesh(p=phi_coords, t=theta_coords)

    # ================================================================================================================
    # Step 5: Latitude Weighted Dilation (in longitude) + Uniform dilation (in latitude)
    #         Compute all contour features +
    #         Force periodicity and delete small contours.
    # ================================================================================================================
    # get list of contours.
    contour_list_pruned = classify_grey_scaled_image(greyscale_image=chd_data,
                                                     lat_coord=ch_lib.Mesh.t,
                                                     lon_coord=ch_lib.Mesh.p,
                                                     AreaThreshold=ch_lib.AreaThreshold,
                                                     frame_num=ch_lib.frame_num,
                                                     frame_timestamp=mean_timestamp,
                                                     BinaryThreshold=ch_lib.BinaryThreshold,
                                                     gamma=ch_lib.gamma,
                                                     beta=ch_lib.beta)

    # ================================================================================================================
    # Step 6: Match coronal holes detected to previous frame detections.
    # ================================================================================================================
    ch_lib.assign_new_coronal_holes(contour_list=contour_list_pruned,
                                    timestamp=mean_timestamp)

    # ================================================================================================================
    # Step 7: Save Frame list of coronal holes and graph in pickle file.
    # ================================================================================================================
    # save the contours found in the latest frame as a pickle file.
    with open(os.path.join(dir_name + folder_name + str(mean_timestamp) + ".pkl"), 'wb') as f:
        pickle.dump(ch_lib.window_holder[-1], f)

    # save in pickle file
    if ch_lib.frame_num == 100:
        # save object to pickle file.
        with open(os.path.join(dir_name + folder_name + str(mean_timestamp) + "_graph" + ".pkl"), 'wb') as f:
            pickle.dump(ch_lib.Graph, f)

    # ================================================================================================================
    # Step 8: Plot results.
    # ================================================================================================================
    # plot connectivity sub-graphs.
    graph_file_name = "graph_frame_" + str(ch_lib.frame_num) + ".png"
    image_file_name = "classified_frame_" + str(ch_lib.frame_num) + ".png"

    # plot coronal holes in the latest frame.
    plot_coronal_hole(ch_list=ch_lib.window_holder[-1].contour_list, n_t=ch_lib.Mesh.n_t, n_p=ch_lib.Mesh.n_p,
                      title="Frame: " + str(ch_lib.frame_num) + ", Time: " + str(mean_timestamp),
                      filename=dir_name + folder_name + image_file_name, plot_rect=True, plot_circle=True,
                      fontscale=0.3, circle_radius=80, thickness_rect=1, thickness_circle=1)

    # plot current graph in the latest window.
    ch_lib.Graph.create_plots(save_dir=dir_name + folder_name + graph_file_name)
    # plt.show()

    # iterate over frame number.
    ch_lib.frame_num += 1

# close database connection
db_session.close()


# ======================================================================================================================
# Step 9: Save to video.
# ======================================================================================================================
SaveVid = True

if SaveVid:
    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(dir_name + folder_name + "tracking_vid_combined.mov", fourcc, 1, (640 * 2, 480))

    for j in range(1, ch_lib.frame_num - 1):
        graph_file_name = "graph_frame_" + str(j) + ".png"
        image_file_name = "classified_frame_" + str(j) + ".png"
        img1 = cv2.imread(dir_name + folder_name + image_file_name)
        img2 = cv2.imread(dir_name + folder_name + graph_file_name)
        video.write(np.hstack((img1, img2)))

    cv2.destroyAllWindows()
    video.release()