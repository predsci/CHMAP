"""
Tracking Algorithm steps (overview):
-----------------------------
1. input image in lat-lon projection. /

2. latitude weighted dilation. /

3. find contours + color the areas arbitrarily. /

4. multiply mask on original binary image. /

5. delete small contours. /

6. force periodicity. /

7. match coronal holes to previous coronal holes in *window* consecutive frames. /

8. add contours to database - which can be saved to JSON file. /

9. graph connectivity - keep track of splitting & merging of coronal holes. /
------------------------------

Last Modified: April 13th, 2021 (Opal).
"""

import cv2
import numpy as np
from analysis.ml_analysis.ch_tracking.contour import Contour
import pickle
import json
from modules.map_manip import MapMesh
from analysis.ml_analysis.ch_tracking.src import CoronalHoleDB
from analysis.ml_analysis.ch_tracking.plots import plot_coronal_hole
from analysis.ml_analysis.ch_tracking.classification import classify_grey_scaled_image
import matplotlib.pyplot as plt
from astropy.time import Time

# Upload coronal hole video.
cap = cv2.VideoCapture("example_vid/maps_r101_chm_low_res_1.mov")

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
while ch_lib.frame_num <= 67:
    # ================================================================================================================
    # Step 1: Read in first frame.
    # ================================================================================================================
    # read in frame by frame.
    success, img = cap.read()

    # cut out the image axis and title.
    image = img[t:b, r:l, :]

    # ================================================================================================================
    # Step 2: Convert Image to Greyscale.
    # ================================================================================================================
    # gray scale: coronal holes are close to 255 other regions are close to 0.
    image = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # update the image dimensions.
    n_t, n_p = np.shape(image)
    ch_lib.Mesh = MapMesh(p=np.linspace(0, 2 * np.pi, n_p),
                          t=(np.pi/2 + np.arcsin(np.linspace(-1, 1, n_t))))

    # ================================================================================================================
    # Step 3: Latitude Weighted Dilation +
    #         Compute all contour features +
    #         Force periodicity and delete small contours.
    # ================================================================================================================
    contour_list_pruned = classify_grey_scaled_image(greyscale_image=image,
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
                                    timestamp=times[ch_lib.frame_num - 1])

    # ================================================================================================================
    # Step 5: Plot results.
    # ================================================================================================================
    # plot coronal holes in the latest frame.
    plot_coronal_hole(ch_list=ch_lib.window_holder[-1].contour_list, n_t=ch_lib.Mesh.n_t, n_p=ch_lib.Mesh.n_p,
                      title="Frame: " + str(ch_lib.frame_num) + ", Time: " + str(times[ch_lib.frame_num])[:10],
                      filename=False)

    # plot connectivity sub - graphs.
    file_name = "results/images/tester/frames/" + "graph_frame_" + str(ch_lib.frame_num) + ".png"
    ch_lib.Graph.create_plots()
    plt.show()

    # wait time between frames.
    # cv2.waitKey(1000)

    # # iterate over frame number.
    ch_lib.frame_num += 1
    print(ch_lib.frame_num)

# save ch library to json file.
# with open('results/first_frame_test.json', 'w') as f:
#     f.write(ch_lib.__str__())

# # save object to pickle file.
# with open('results/first_frame_test_knn.pkl', 'wb') as f:
#    pickle.dump(ch_lib, f)
