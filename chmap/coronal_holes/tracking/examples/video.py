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
query_start = datetime.datetime(year=2010, month=1, day=1, hour=0, minute=0, second=0)
query_end = datetime.datetime(year=2010, month=12, day=30, hour=0, minute=0, second=0)


# ================================================================================================================
# Step 2: Initialize directory and folder to save results (USER PARAMETERS)
# ================================================================================================================
# --- User Parameters ----------------------
dir_name = "/Users/osissan/desktop/CHT_RESULTS/"
folder_name = "2010/"
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
ch_lib.frame_num = 4638
# ======================================================================================================================
# Step 10: Save to video.
# ======================================================================================================================
SaveVid = True

if SaveVid:
    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(dir_name + folder_name + "tracking_vid_2010.mov", fourcc, 30, (1280, 960))

    for j in range(1, ch_lib.frame_num - 1):
        print("frame num = ", j)
        graph_file_name = "graph_frame_" + str(j) + ".png"
        image_file_name = "classified_frame_" + str(j) + ".png"
        img1 = cv2.imread(dir_name + folder_name + frame_folder + image_file_name)
        video.write(img1)

    cv2.destroyAllWindows()
    video.release()