"""
Author: Opal Issan. Date: Jan 7th, 2020.
Assumptions:
-------------------
1. Coronal hole that does not appear after one frame is no longer present. -- hence center list is reinitialized after
each frame.
2. a coronal hole center will be somewhat closer to its previous frame center than to other coronal hole centers. -
distance is measured in euclidean.

The primary assumption of the centroid tracking algorithm is that a given object will potentially move
in between subsequent frames, but the distance between the centroids for frames F_t and F_{t + 1} will
be smaller than all other distances between objects.

"""

import cv2
import numpy as np
from scipy.spatial import distance as dist

# Upload coronal hole video.
cap = cv2.VideoCapture("/Users/opalissan/CH_Project/CHD_DB/processed_images/maps_r101_chm_low_res_1.mov")

# cut out the axis and title.
t, b, r, l = 47, -55, 110, -55

# save contour dict of the video.
centroid_dict = dict()
# save coronal hole centers of the previous frame.
centroid_input = []
centroid_prev = []

# initialize frame index.
frame_idx = 1

while frame_idx <= 100:
    # read in frame by frame.
    success, img = cap.read()

    # cut out the image axis and title.
    image = img[t:b, r:l, :]

    # gray scale.
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # find contours.
    ret, thresh = cv2.threshold(imgray, 55, 255, 0)
    contours, hierarchy = cv2.findContours(cv2.bitwise_not(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("\nNumber of contours detected initially = %d" % len(contours))

    # do not count small contours.
    AreaThreshold = 50
    saved_contour = []
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > AreaThreshold:
            saved_contour.append(contours[i])

    print("Number of contours detected after elimination = %d" % len(saved_contour))
    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

    # finding connections between old frame and new frame.
    for c in saved_contour:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # measure the euclidean distance between
        centroid_input.append((cX, cY))

    # specific an index for each contour.
    # find the circle with the closes proximity from the previous frame.
    # check that this is not the first frame.
    if frame_idx == 1:
        # specific an index for each contour.
        idx = 1
        for cX, cY in centroid_input:
            # plot the circle and ID number.
            cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)
            cv2.putText(image, "ch #" + str(idx), (cX - 20, cY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            idx += 1
        centroid_prev = centroid_input.copy()

    else:
        D = dist.cdist(centroid_prev, centroid_input)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        priority_queue = (list(zip(rows, cols)))
        priority_queue = [(a, b) for i, [a, b] in enumerate(priority_queue) if not any(c == b for _, c in priority_queue[:i])]
        print("length of priority queue = ", len(priority_queue))
        print(priority_queue)

        centroid_copy = centroid_input.copy()
        if len(centroid_prev) > len(centroid_input):
            centroid_ordered = centroid_prev.copy()
        else:
            centroid_ordered = centroid_input.copy()

        for obj, new in priority_queue:
            cX, cY = centroid_input[new]
            cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)
            cv2.putText(image, "ch #" + str(obj+1), (cX - 20, cY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            centroid_ordered[obj] = centroid_input[new]
        centroid_prev = centroid_ordered.copy()

    # reinitialize.
    frame_idx += 1
    centroid_input = []
    # show the image
    cv2.imshow("Image", image)
    cv2.imwrite("ID.png", image)
    cv2.waitKey(10)

cv2.waitKey(0)
