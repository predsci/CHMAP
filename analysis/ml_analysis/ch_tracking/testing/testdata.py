"""Test Data saved in results/first_frame_test.json/.pkl

Author: Opal Issan, March 15th, 2021.

The purpose of this module is to read in various frames and their corresponding coronal holes and to implement KNN to
match coronal holes between frames. See mkdocs website for more details about KNN algorithm. """

import numpy as np
import matplotlib.pyplot as plt
from analysis.ml_analysis.ch_tracking.src import CoronalHoleDB
from sklearn.neighbors import KNeighborsClassifier
from analysis.ml_analysis.ch_tracking.contour import Contour
from analysis.ml_analysis.ch_tracking.plots import plot_coronal_hole
from analysis.ml_analysis.ch_tracking.knn import KNN
from modules.map_manip import MapMesh
from analysis.ml_analysis.ch_tracking.areaoverlap import area_overlap
import json
import cv2
import pickle

Verbose = True

# ======================================================================================================================
# Read Pickle file saved in "results" folder as an object of coronalhole.py.
# ======================================================================================================================
ReadFile = False

if ReadFile:
    file_name = "results/first_frame_test_knn.pkl"
    ch_db = pickle.load(open(file_name, "rb"))
    if Verbose:
        print(ch_db)

    # ======================================================================================================================
    # Plot the feature evaluation of coronal hole 1 and 5.
    # ======================================================================================================================
    ID = 4
    ch_list = ch_db.ch_dict[ID].contour_list

    frame_list, area_list = [], []
    for ch in ch_list:
        # plot feature.
        if frame_list.count(ch.frame_num) > 0:
            area_list[-1] += ch.area
        else:
            frame_list.append(ch.frame_num)
            area_list.append(ch.area)

    plt.scatter(frame_list, area_list, c="b")
    plt.plot(frame_list, area_list, c="k")
    plt.xlabel("Frame")
    plt.ylabel("Area")
    plt.title("Coronal Hole # " + str(ID))

    plt.savefig("figures/" + "ch_area_" + str(ID) + ".png")
    plt.show()

# ======================================================================================================================
# save png files to create a video of coronal hole #1.
# ======================================================================================================================
SavePng = False
# coronal hole ID number
ID = 1

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
                          filename="results/images/tester/" + "ch_" + str(ch.id) + "_frame_" + str(ch.frame_num))

# ======================================================================================================================
# Save to video.
# ======================================================================================================================
SaveVid = True

if SaveVid:
    dir_name = "/Users/opalissan/desktop/CHT_RESULTS/"
    folder_name = "2010-12-20-to-2011-04-01/"
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

# ======================================================================================================================
# Save Coronal Holes centroid and label.
# ======================================================================================================================
# frame number treated as "new frame"
testKNN = False

if testKNN:
    frame_number = 5
    # window size.
    window = 4

    # initialize the X and Y lists with centroid location and label respectively.
    # X spherical location is (theta, phi) or (lat, lon)
    X_train = []
    Y_train = []

    X_test = []

    # save first and second frame.
    for ch_class, ch_list in ch_db.ch_dict.items():
        for ch in ch_list.contour_list:
            if ch.frame_num == 1:
                X_train.append(ch.phys_centroid)
                Y_train.append(ch.id)
            if ch.frame_num == 2:
                X_train.append(ch.phys_centroid)
                Y_train.append(ch.id)
            if ch.frame_num == 3:
                X_test.append(ch.phys_centroid)

    plot = False

    if plot:
        ii = 0
        for t, p in X_train:
            plt.scatter(p, t, label=str(Y_train[ii]), s=5)
            ii += 1

        for t, p in X_test:
            plt.scatter(p, t, c="k", s=5)

        plt.scatter(p, t, label="Frame2", c="k", s=5)

        plt.legend()
        plt.gca().invert_yaxis()
        plt.title("Match Centroid using KNN")
        plt.savefig("figures/knn_test.png")

    # ======================================================================================================================
    #  Apply KNN algorithm to classify the coronal holes based n their centroid proximity.
    #
    #  * Note, for WKNN (weighted based on distance 1/d) small perturbations in the hyper-parameter
    #  (K) will barely effect the results.
    #
    #  See knn.py for implementation.
    # ======================================================================================================================

    # apply KNN Algorithm and analyze the results.
    classifier = KNN(X_train=X_train, X_test=X_test, Y_train=Y_train, K=6, thresh=0.2)

    # probability results.
    res = classifier.X_test_results

    # if proba > thresh check its overlap of pixels area.
    area_check_list = classifier.check_list

    if Verbose:
        print("label list = ", Y_train)
        print("results = ", res)
        print("Area check list = ", area_check_list)

    # ======================================================================================================================
    #  Based on the results above (knn), find the area overlap between the new coronal holes and the coronal holes that
    #  have a high probability to be associated with. This operation can be computationally expensive, therefore, it is
    #  preferred to compare the overlapping regions with as few coronal holes as possible. Therefore, we will pick the
    #  latest say m frames (approximately 5 to 10) and compare the area overlap in that domain.
    #
    #  * See areaoverlap.py for functions implementation.
    # ======================================================================================================================

    # area matrix
    mesh = MapMesh(p=np.linspace(0, 2 * np.pi, n_p), t=np.linspace(0, np.pi, n_t))
    da = mesh.da

    if plot:
        plt.matshow(da.T)
        plt.colorbar()
        plt.title("$\Delta A$")
        plt.show()

    # initialize probability matrix
    proba_mat = []

    # prepare data-set
    X_test = []
    X_train = []
    for ch_class, ch_list in ch_db.ch_dict.items():
        for ch in ch_list.contour_list:
            if ch.frame_num == 1:
                X_train.append(ch)
            if ch.frame_num == 2:
                X_train.append(ch)
            if ch.frame_num == 3:
                X_test.append(ch)

    for ii, ch_list in enumerate(area_check_list):
        prob_list = []
        for id in ch_list:
            coronal_hole_list = ch_db.ch_dict[id]
            p = []
            for ch in coronal_hole_list.contour_list:
                p1, p2 = area_overlap(ch1=ch, ch2=X_test[ii], da=da)
                p.append((p1 + p2) / 2)
            prob_list.append(np.mean(p))
        proba_mat.append(prob_list)

    print(proba_mat)
