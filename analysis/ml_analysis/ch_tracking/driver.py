"""
Author: Opal Issan. Date: Jan 19th, 2020.
Assumptions:
-------------------
1. Coronal hole that does not appear after 5 frame is no longer present. -- hence center list is reinitialized after
each frame.
2. a coronal hole center will be somewhat closer to its previous frame center than to other coronal hole centers. -
distance is measured in euclidean.

The primary assumption of the centroid tracking algorithm is that a given object will potentially move
in between subsequent frames, but the distance between the centroids for frames F_t and F_{t + 1} will
be smaller than all other distances between objects.


# TODO:
1. match coronal holes based on previous 5 frames!
2. add periodicity...
3. print ch_db to a log file.
4. compute the difference between pixels in 2 projection. - verify we do not loose
information when converting to a new projection and back.
5. set a metric to match coronal holes based on the percentage of their bounding rectangles overlap.
"""

import cv2
import numpy as np
from analysis.ml_analysis.ch_tracking.contour import Contour
import pickle
import matplotlib.pyplot as plt
from analysis.ml_analysis.ch_tracking.ch_db import CoronalHoleDB

# Upload coronal hole video.
cap = cv2.VideoCapture("example_vid/maps_r101_chm_low_res_1.mov")

# cut out the axis and title.
t, b, r, l = 47, -55, 110, -55

# coronal hole video database.
ch_lib = CoronalHoleDB()

# initialize frame index.
ch_lib.frame_num = 1

# loop over each frame.
while ch_lib.frame_num <= 10:

    # read in frame by frame.
    success, img = cap.read()

    # cut out the image axis and title.
    image = img[t:b, r:l, :]

    # gray scale.
    imgray_polar = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # map to polar projection.
    imgray_polar = ch_lib.map_new_polar_projection(gray_image=imgray_polar)

    # update the image dimensions.
    Contour.n_t, Contour.n_p = np.shape(imgray_polar)

    # add coronal holes to data base.
    ch_lib.find_contours(imgray_polar)

    # plot the contours.
    if ch_lib.frame_num == 1:
        # initialize first frame.
        ch_lib.first_frame_initialize()
    else:
        # find the match.at-lon
        ch_lib.match_coronal_holes()

    # plot each contour in our current frame. cv2.FILLED colors the whole contour in its unique color,
    # and save the image.
    map_back_contour = ch_lib.map_back_to_long_lat_rbg(input_image=ch_lib.fill_contours())

    # compute coronal hole features, including: center, area, bounding box, etc...
    ch_lib.update_coronal_hole_features(rbg_image=map_back_contour)

    # iterate over each coronal hole and plot its color + id number.
    for c in ch_lib.p1.contour_list:
        # plot each contour in our current frame. cv2.FILLED colors the whole contour in its unique color.
        # cv2.drawContours(image=image,
        #                  contours=[np.array([np.array([a]) for a in zip(c.contour_pixels[1], c.contour_pixels[0])])],
        #                  contourIdx=0, color=c.color, thickness=1)
        # plot the contours center.
        cv2.circle(img=map_back_contour, center=c.lat_lon_pixel_centroid, radius=3, color=(0, 0, 0), thickness=-1)
        # plot bounding box c.straight box returns top left x, y, w, h.
        cv2.rectangle(img=map_back_contour, pt1=(c.straight_box[0], c.straight_box[1]),
                      pt2=(c.straight_box[0] + c.straight_box[2], c.straight_box[1] + c.straight_box[3]),
                      color=(0, 255, 0), thickness=2)
        # plot the contour's ID number.
        cv2.putText(img=map_back_contour, text="ch #" + str(c.id), org=tuple(np.add(c.lat_lon_pixel_centroid, (-20, 15))),
                    fontFace=cv2.Formatter_FMT_DEFAULT, fontScale=0.5, color=(0, 0, 0), thickness=1)

    # iterate over frame number.
    ch_lib.frame_num += 1
    # show tracking image.
    cv2.imshow("Coronal Hole Tracking", map_back_contour)
    # show original image.
    cv2.imshow("Original Image", image)
    # save image.
    cv2.imwrite("ch_tracking_jan19.png", map_back_contour)
    # wait time between frames.
    cv2.waitKey(3000)

cv2.waitKey(0)

