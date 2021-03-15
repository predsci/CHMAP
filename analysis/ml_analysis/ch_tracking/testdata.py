"""Test Data saved in results/first_frame_test.json/.pkl

Author: Opal Issan, March 15th, 2021.

The purpose of this module is to read in various frames and their corresponding coronal holes and to implement KNN to
match coronal holes between frames. See mkdocs website for more details about KNN algorithm. """

import numpy as np
import matplotlib.pyplot as plt
from analysis.ml_analysis.ch_tracking.ch_db import CoronalHoleDB
from sklearn.neighbors import KNeighborsClassifier
from analysis.ml_analysis.ch_tracking.contour import Contour
from analysis.ml_analysis.ch_tracking.plots import plot_coronal_hole
import json
import cv2
import pickle


# ======================================================================================================================
# Read Pickle file saved in "results" folder as an object of coronalhole.py.
# ======================================================================================================================
ReadFile = True

if ReadFile:
    file_name = "results/first_frame_test.pkl"
    ch_db = pickle.load(open(file_name, "rb"))
    print(ch_db)


# ======================================================================================================================
# save png files to create a video of coronal hole #1.
# ======================================================================================================================
SavePng = False

# coronal hole ID number
ID = 7

# image size (n_t, n_p)
n_t = 398
n_p = 644

if SavePng and ReadFile:
    # read in a list of coronal holes that were classified as 1.
    ch_list = ch_db.ch_dict[ID].contour_list

    # loop over all the coronal holes in the same class and plot their pixel points, centroid, and frame number.
    # save to results file.
    for ch in ch_list:

        # plot the single coronal hole.
        plot_coronal_hole(ch=ch, n_t=n_t, n_p=n_p,
                          title="Coronal Hole #" + str(ch.id) + ", Frame #" + str(ch.frame_num),
                          filename="results/images/tester/"+"ch_"+str(ch.id)+"_frame_"+str(ch.frame_num))

# ======================================================================================================================
# Save to video.
# ======================================================================================================================
SaveVid = False

if SaveVid:
    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter("results/images/testervid/coronalhole" + str(ID) + ".mov", fourcc, 1, (640, 480))

    for j in range(1, 20):
        filename = "results/images/tester/"+"ch_"+str(ID)+"_frame_"+str(j)
        img = cv2.imread(filename+'.png')
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


# ======================================================================================================================
# Save Coronal Holes centroid and label.
# ======================================================================================================================

for ch in ch_db:
    print(ch)