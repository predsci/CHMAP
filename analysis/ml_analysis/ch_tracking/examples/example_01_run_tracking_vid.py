"""This is a tutorial/example on how to run the coronal hole tracking algorithm from an
example video saved in data folder called: maps_r101_chm_low_res_1.mov

This Module includes the following operations:

1. Importing maps_r101_chm_low_res_1.mov using opencv library.

2. Save list of coronal holes for each frame (pickle file format).

3. Save a connectivity graph of the coronal hole evolution in time (pickle file format).

4. Save image of the coronal hole detected frame in each iteration + save a plot of the graph then create a side
    by side (.mov)

Last Modified: May 6th, 2021 (Opal)
"""

import cv2
import numpy as np
from analysis.ml_analysis.ch_tracking.contour import Contour
import pickle
import json
import os
from modules.map_manip import MapMesh
from analysis.ml_analysis.ch_tracking.src import CoronalHoleDB
from analysis.ml_analysis.ch_tracking.tools.plots import plot_coronal_hole
from analysis.ml_analysis.ch_tracking.classification import classify_grey_scaled_image
import matplotlib.pyplot as plt
from astropy.time import Time

# ================================================================================================================
# Step 1: Initialize directory and folder to save results (USER PARAMETERS)
# ================================================================================================================
# --- User Parameters ----------------------
dir_name = "/Users/opalissan/desktop/CHT_RESULTS/"
folder_name = "2011-02-17-2011-10-27/"

# Upload coronal hole video.
cap = cv2.VideoCapture("../data/maps_r101_chm_low_res_1.mov")

# time interval
times = ['2011-02-17T18:00:30', '2011-10-27T23:55:30']
t = Time(times)
dt = t[1] - t[0]
times = t[0] + dt * np.linspace(0., 1., 67)

# cut out the axis and title.
t, b, r, l = 47, -55, 110, -55

# coronal hole video database.
ch_lib = CoronalHoleDB()

# initialize frame index.
ch_lib.frame_num = 1

# loop over each frame.
while ch_lib.frame_num <= 66:
    # ================================================================================================================
    # Step 2: Read in first frame.
    # ================================================================================================================
    # read in frame by frame.
    success, img = cap.read()

    # cut out the image axis and title.
    image = img[t:b, r:l, :]

    # ================================================================================================================
    # Step 3: Convert Image to Greyscale.
    # ================================================================================================================
    # gray scale: coronal holes are close to 1 other regions are close to 0.
    image = (255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) / 255

    # update the image dimensions.
    n_t, n_p = np.shape(image)
    ch_lib.Mesh = MapMesh(p=np.linspace(0, 2 * np.pi, n_p),
                          t=(np.pi / 2 + np.arcsin(np.linspace(-1, 1, n_t))))

    # ================================================================================================================
    # Step 4: Latitude Weighted Dilation +
    #         Compute all contour features +
    #         Force periodicity and delete small contours.
    # ================================================================================================================
    # get list of contours.
    contour_list_pruned = classify_grey_scaled_image(greyscale_image=image,
                                                     lat_coord=ch_lib.Mesh.t,
                                                     lon_coord=ch_lib.Mesh.p,
                                                     AreaThreshold=ch_lib.AreaThreshold,
                                                     frame_num=ch_lib.frame_num,
                                                     frame_timestamp=times[ch_lib.frame_num - 1],
                                                     BinaryThreshold=ch_lib.BinaryThreshold,
                                                     gamma=ch_lib.gamma,
                                                     beta=ch_lib.beta)

    # ================================================================================================================
    # Step 5: Match coronal holes detected to previous frame detections.
    # ================================================================================================================
    ch_lib.assign_new_coronal_holes(contour_list=contour_list_pruned,
                                    timestamp=times[ch_lib.frame_num - 1])

    # ================================================================================================================
    # Step 6: Save Frame list of coronal holes.
    # ================================================================================================================
    # save the contours found in the latest frame as a pickle file.
    with open(os.path.join(dir_name + folder_name + str(times[ch_lib.frame_num - 1]) + ".pkl"), 'wb') as f:
        pickle.dump(ch_lib.window_holder[-1], f)

    # ================================================================================================================
    # Step 8: Plot results.
    # ================================================================================================================
    # plot connectivity sub-graphs.
    graph_file_name = "graph_frame_" + str(ch_lib.frame_num) + ".png"
    image_file_name = "classified_frame_" + str(ch_lib.frame_num) + ".png"

    # plot coronal holes in the latest frame.
    plot_coronal_hole(ch_list=ch_lib.window_holder[-1].contour_list, n_t=ch_lib.Mesh.n_t, n_p=ch_lib.Mesh.n_p,
                      title="Frame: " + str(ch_lib.frame_num) + ", Time: " + str(times[ch_lib.frame_num - 1]),
                      filename=dir_name + folder_name + image_file_name, plot_rect=True, plot_circle=True,
                      fontscale=0.7, circle_radius=100, thickness_rect=1)

    # plot current graph in the latest window.
    ch_lib.Graph.create_plots(save_dir=dir_name + folder_name + graph_file_name)
    plt.show()

    # print diagnostic.
    print("Frame num = ", ch_lib.frame_num)

    # update frame iteration.
    ch_lib.frame_num += 1

# ================================================================================================================
# Step 9: Save connectivity graph.
# ================================================================================================================
# save graph in pickle file
with open(os.path.join(dir_name + folder_name + "connectivity_graph" + ".pkl"), 'wb') as f:
    pickle.dump(ch_lib.Graph, f)


# ======================================================================================================================
# Step 10: Save to video.
# ======================================================================================================================
SaveVid = True

if SaveVid:
    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video = cv2.VideoWriter("results/images/testervid/coronalhole" + str(ID) + ".mov", fourcc, 1, (640, 480))
    video = cv2.VideoWriter(dir_name + folder_name + "tracking_vid_combined.mov", fourcc, 1, (640 * 2, 480))

    for j in range(1, 49):
        graph_file_name = "graph_frame_" + str(j) + ".png"
        image_file_name = "classified_frame_" + str(j) + ".png"
        img1 = cv2.imread(dir_name + folder_name + image_file_name)
        img2 = cv2.imread(dir_name + folder_name + graph_file_name)
        video.write(np.hstack((img1, img2)))

    cv2.destroyAllWindows()
    video.release()