""" This is a tutorial/example on how to run the coronal hole tracking algorithm on CH Maps
from in the database (db).

This Module includes the following operations:

1. Importing Detection images (CH Maps - low res) from Q
(see Jamie's examples/examples112_query_maps.py for more information on how to access the database).

2. Save list of coronal holes for each frame (pickle file format).

3. Save a connectivity graph of the coronal hole evolution in time (pickle file format).

4. Save image of the coronal hole detected frame in each iteration + save a plot of the graph then create a side
    by side (.mov)

Last Modified: June 6th, 2021 (Opal)
"""


import os
import datetime
import numpy as np
import cv2
import pickle
import chmap.database.db_classes as db_class
import chmap.database.db_funs as db_funcs
import chmap.utilities.datatypes.datatypes as psi_datatype
from chmap.coronal_holes.tracking.src.main import CoronalHoleDB
from chmap.coronal_holes.tracking.src.classification import classify_grey_scaled_image
from chmap.coronal_holes.tracking.tools.plots import plot_coronal_hole
from chmap.maps.util.map_manip import MapMesh

# ================================================================================================================
# Step 1: Choose a test case - time interval
# ================================================================================================================
# define map query start and end times
# paper test case: Dec 29th 2010 to April 8th 2011.
query_start = datetime.datetime(year=2007, month=3, day=1, hour=0, minute=0, second=0)
query_end = datetime.datetime(year=2020, month=7, day=30, hour=0, minute=0, second=0)


# ================================================================================================================
# Step 2: Initialize directory and folder to save results (USER PARAMETERS)
# ================================================================================================================
# --- User Parameters ----------------------
dir_name = "/Users/osissan/desktop/CHT_RESULTS/"
folder_name = "2007to2020/"
graph_folder = "graphs/"
frame_folder = "frames/"
pickle_folder = "pkl/"

# ================================================================================================================
# Step 3: Algorithm Hyper Parameters
# ================================================================================================================
# specify hyper parameters.
# contour binary threshold.
CoronalHoleDB.BinaryThreshold = 0.7
# coronal hole area threshold.
CoronalHoleDB.AreaThreshold = 5E-3
# window to match coronal holes.
CoronalHoleDB.window = 80
# parameter for longitude dilation (this should be changed for larger image dimensions).
CoronalHoleDB.gamma = 12
# parameter for latitude dilation (this should be changed for larger image dimensions).
CoronalHoleDB.beta = 8
# connectivity threshold.
CoronalHoleDB.ConnectivityThresh = 0.2
# connectivity threshold.
CoronalHoleDB.AreaMatchThresh = 0.1
# knn k hyper parameter
CoronalHoleDB.kHyper = 15
# knn thresh
CoronalHoleDB.kNNThresh = 0


# initialize coronal hole tracking database.
ch_lib = CoronalHoleDB()
# ================================================================================================================
# Step 3: Read in detected images from the database.
# ================================================================================================================
# --- User Parameters ----------------------
# INITIALIZE DATABASE CONNECTION
# Database paths
map_data_dir = "/Users/osissan/desktop/CH_DB"
db_type = "mysql"
user = "opalissan"
password = ""
cred_dir = "/Users/opalissan/PycharmProjects/CHMAP/chmap/settings"
db_loc = "q.predsci.com"
mysql_db_name = "chd"

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

# Establish connection to database
db_session = db_funcs.init_db_conn(db_type, db_class.Base, db_loc, db_name=mysql_db_name,
                                   user=user, password=password, cred_dir=cred_dir)

# --- Begin execution ----------------------
# query maps in time range
map_info, data_info, method_info, image_assoc = db_funcs.query_euv_maps(
    db_session, mean_time_range=(query_start, query_end), methods=map_methods,
    var_val_range=map_vars)

# the query returns multiple dataframes that together describe the map-making
# process and constituent images.  Here we are mostly interested in the map_info
# dataframe.  It contains one row per map with a number of information columns:
map_info.keys()


# iterate through the rows of map_info
for row_index, row in map_info.iterrows():
    print("Processing map for: " + str(row.date_mean) + ", Frame num = " + str(ch_lib.frame_num))
    # load map (some older maps have a leading '/' that messes with os.path.join
    if row.fname[0] == "/":
        rel_path = row.fname[1:]
    else:
        rel_path = row.fname
    map_path = os.path.join(map_data_dir, rel_path)
    # if os.path.isfile(map_path):
    my_map = psi_datatype.read_psi_map(map_path)
    # ================================================================================================================
    # Step 4: Input image, coordinates, mesh spacing, and timestamp.
    # ================================================================================================================
    # note that the grid has been reduced since the coronal
    # hole detection was performed, so values are floats between 0. and 1.
    chd_data = my_map.chd.astype('float32')
    # flip image to correspond 0 - 0 and pi - n_t
    input_image = cv2.flip(chd_data, 0)

    # restrict chd_data to be positive.
    input_image[input_image < 0] = 0
    input_image[input_image > 1] = 1

    # image coordinates (latitude and longitude).
    phi_coords = my_map.x
    sinlat_coords = my_map.y
    theta_coords = np.pi / 2 + np.arcsin(sinlat_coords)
    mean_timestamp = row.T[2]

    # save mesh map
    ch_lib.Mesh = MapMesh(p=phi_coords, t=theta_coords)

    # ================================================================================================================
    # Step 5: Latitude Weighted Dilation (in longitude) + Uniform dilation (in latitude)
    #         Compute all contour features +
    #         Force periodicity and delete small contours.
    # ================================================================================================================
    # get list of contours.
    contour_list_pruned = classify_grey_scaled_image(greyscale_image=input_image,
                                                     lat_coord=ch_lib.Mesh.t,
                                                     lon_coord=ch_lib.Mesh.p,
                                                     AreaThreshold=ch_lib.AreaThreshold,
                                                     frame_num=ch_lib.frame_num,
                                                     frame_timestamp=mean_timestamp,
                                                     BinaryThreshold=ch_lib.BinaryThreshold,
                                                     gamma=ch_lib.gamma,
                                                     beta=ch_lib.beta,
                                                     db_session=db_session,
                                                     map_dir=map_data_dir)

    # ================================================================================================================
    # Step 6: Match coronal holes detected to previous frame detections.
    # ================================================================================================================
    ch_lib.assign_new_coronal_holes(contour_list=contour_list_pruned,
                                    timestamp=mean_timestamp)

    # ================================================================================================================
    # Step 7: Save Frame list of coronal holes.
    # ================================================================================================================
    # save the contours found in the latest frame as a pickle file.
    file_name_pkl = str(mean_timestamp).replace(':', '-')
    file_name_pkl = file_name_pkl.replace(' ', '-')
    with open(os.path.join(dir_name + folder_name + pickle_folder + file_name_pkl + ".pkl"), 'wb') as f:
        pickle.dump(ch_lib.window_holder[-1], f)

    # ================================================================================================================
    # Step 8: Plot results.
    # ================================================================================================================
    # plot connectivity sub-graphs.
    # graph_file_name = "graph_frame_" + str(ch_lib.frame_num) + ".png"
    image_file_name = file_name_pkl + ".png"

    # plot coronal holes in the latest frame.
    plot_coronal_hole(ch_list=ch_lib.window_holder[-1].contour_list, n_t=ch_lib.Mesh.n_t, n_p=ch_lib.Mesh.n_p,
                      title="Frame: " + str(ch_lib.frame_num) + ", Time: " + str(mean_timestamp),
                      filename=dir_name + folder_name + frame_folder + image_file_name, plot_rect=False,
                      plot_circle=True, fontscale=0.3, circle_radius=80, thickness_rect=1, thickness_circle=1)

    # plot current graph in the latest window.
    # ch_lib.Graph.create_plots(save_dir=dir_name + folder_name + graph_folder + graph_file_name)
    # plt.show()
    # save the connectivity graph every 1000 frames...
    if ch_lib.frame_num % 1000 == 0:
        # save object to pickle file.
        with open(os.path.join(dir_name + folder_name + "connectivity_graph_" +
                               str(file_name_pkl) + ".pkl"), 'wb') as f:
            pickle.dump(ch_lib.Graph, f)

    # iterate over frame number.
    ch_lib.frame_num += 1
# close database connection
db_session.close()

# ================================================================================================================
# Step 9: Save Connectivity Graph.
# ================================================================================================================
# save object to pickle file.
with open(os.path.join(dir_name + folder_name + "connectivity_graph" + ".pkl"), 'wb') as f:
    pickle.dump(ch_lib.Graph, f)

# ======================================================================================================================
# Step 10: Save to video.
# ======================================================================================================================
SaveVid = False

if SaveVid:
    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(dir_name + folder_name + "tracking_vid_combined.mov", fourcc, 30, (1280 * 2, 960))

    for j in range(1, ch_lib.frame_num - 1):
        graph_file_name = "graph_frame_" + str(j) + ".png"
        image_file_name = "classified_frame_" + str(j) + ".png"
        img1 = cv2.imread(dir_name + folder_name + frame_folder + image_file_name)
        img2 = cv2.imread(dir_name + folder_name + graph_folder + graph_file_name)
        video.write(np.hstack((img1, img2)))

    cv2.destroyAllWindows()
    video.release()