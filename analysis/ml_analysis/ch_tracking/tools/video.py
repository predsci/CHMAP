"""Create a video analyzing the tracking algorithm results. """

import cv2
import numpy as np
from analysis.ml_analysis.ch_tracking.tools.plots import set_up_plt_figure
import os

# ======================================================================================================================
# Step 1: User Parameters.
# ======================================================================================================================
dir_name = "/Users/opalissan/desktop/CHT_RESULTS/"
folder_name = "2011-02-17-2011-04-01/"

# ======================================================================================================================
# Step 2: Save to video.
# ======================================================================================================================
SaveVid = True

# Upload coronal hole video.
cap = cv2.VideoCapture("../data/maps_r101_chm_low_res_1_Trim.mp4")

# cut out the axis and title.
t, b, r, l = 47, -55, 110, -55


if SaveVid:
    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(dir_name + folder_name + "classified_and_input.mov", fourcc, 1, (640 * 2, 480))

    for j in range(1, 331):
        input_file_name = "input/input_frame_" + str(j) + ".png"
        image_file_name = "classified_frame_" + str(j) + ".png"
        img1 = cv2.imread(dir_name + folder_name + image_file_name)
        img2 = cv2.imread(dir_name + folder_name + input_file_name)
        video.write(np.hstack((img1, img2)))

    cv2.destroyAllWindows()
    video.release()