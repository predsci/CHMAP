"""
Author: Opal Issan. Date: Jan 7th, 2020.
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
1. New map projection to avoid classifying multiple coronal holes at the poles...
2. set a metric to match coronal holes based on the percentage of their bounding rectangles overlap.


"""

import cv2
import numpy as np
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
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # add coronal holes to data base.
    ch_lib.find_contours(imgray)

    # plot the contours.
    if ch_lib.frame_num == 1:
        # initialize first frame.
        ch_lib.first_frame_initialize()

    else:
        # find the match.
        ch_lib.match_coronal_holes()

    # iterate over each coronal hole and plot its color + id number.
    for c in ch_lib.p1.contour_list:
        # plot each contour in our current frame. cv2.FILLED colors the whole contour in its unique color.
        cv2.drawContours(image=image, contours=[c.contour], contourIdx=0, color=c.color, thickness=cv2.FILLED)
        # plot the contours center.
        cv2.circle(img=image, center=c.pixel_centroid, radius=3, color=(0, 0, 0), thickness=-1)
        # plot the contour's ID number.
        cv2.putText(img=image, text="ch #" + str(c.id), org=tuple(np.add(c.pixel_centroid, (-20, 15))),
                    fontFace=cv2.Formatter_FMT_DEFAULT, fontScale=0.5, color=(0, 0, 0), thickness=1)

    ch_lib.frame_num += 1
    cv2.imshow("Coronal Hole Tracking", image)
    cv2.imwrite("figures/CoronalHoleTracking.png", image)
    cv2.waitKey(1000)

cv2.waitKey(0)

