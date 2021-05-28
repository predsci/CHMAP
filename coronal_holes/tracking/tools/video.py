"""Create a video analyzing the tracking algorithm results. """

import cv2
import numpy as np

ex1 = False
ex2 = True
if ex1:
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


if ex2:
    import numpy as np
    import cv2
    # ================================================================================================================
    # Step 1: Initialize directory and folder to save results (USER PARAMETERS)
    # ================================================================================================================
    # --- User Parameters ----------------------
    dir_name = "/Users/opalissan/desktop/CHT_RESULTS/"
    folder_name = "2010-12-29-2011-04-08c4hr/"
    graph_folder = "graphs/"
    frame_folder = "frames/"
    pickle_folder = "pkl/"

    # ======================================================================================================================
    # Step 2: Save to video.
    # ======================================================================================================================
    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(dir_name + folder_name + "tracking_vid_combined.mov", fourcc, 1, (640 * 2, 480))

    for j in range(1, 463):
        graph_file_name = "graph_frame_" + str(j) + ".png"
        image_file_name = "classified_frame_" + str(j) + ".png"
        img1 = cv2.imread(dir_name + folder_name + frame_folder + image_file_name)
        img2 = cv2.imread(dir_name + folder_name + graph_folder + graph_file_name)
        video.write(np.hstack((img1, img2)))

    cv2.destroyAllWindows()
    video.release()