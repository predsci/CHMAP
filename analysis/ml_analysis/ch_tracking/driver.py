"""
Author: Opal Issan. Date: Feb 3rd, 2021.
Assumptions:
-------------------
Tracking Algorithm steps:

1. Input image in lat -lon projection
2. remove fake detections by opening - erode then dilate.
3. find contours + color the areas arbitrarily - TODO: verify that they are unique by using a uniform vector.
4. multiply mask on original binary image.
5. delete small contours.
6. force periodicity.
7. match coronal holes to previous coronal holes in 5 consecutive frames. TODO: This is done by centroid distance and
TODO: can also be done by using another feature such as area or bounding box.
8. add contours to database - which can be saved to JSON file.
"""

import cv2
import numpy as np
from analysis.ml_analysis.ch_tracking.contour import Contour
import pickle
from scipy import ndimage
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
while ch_lib.frame_num <= 25:

    # read in frame by frame.
    success, img = cap.read()

    # cut out the image axis and title.
    image = img[t:b, r:l, :]

    # gray scale: coronal holes are close to 255 other regions are close to 0.
    image = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # update the image dimensions.
    Contour.n_t, Contour.n_p = np.shape(image)

    # pass one extra erode and dilate before latitude weighted dilation.
    img = cv2.erode(image, np.ones((3, 3), dtype=np.uint8), iterations=1)
    img = cv2.dilate(img, np.ones((3, 3), dtype=np.uint8), iterations=4)

    # theta array.
    theta = np.linspace(0, np.pi, Contour.n_t)

    # latitude weighted dilation.
    for ii in range(Contour.n_t):
        # build the flat structuring element.
        width = int(((1 - np.sin(theta[ii])) ** 3) * Contour.n_p)
        if width > 0:
            kernel = np.ones(width, dtype=np.uint8)
            # save dilated strip.
            img[ii, :] = np.reshape(cv2.dilate(img[ii, :], kernel, iterations=1), Contour.n_p)
        else:
            img[ii, :] = img[ii, :]

    # add coronal holes to data base.
    rbg_dilated, color_list = ch_lib.find_contours(img)

    # create a threshold mask.
    ret, thresh = cv2.threshold(image, CoronalHoleDB.BinaryThreshold, 1, 0)

    # multiply mask.
    classified_img = (rbg_dilated.transpose(2, 0, 1) * thresh).transpose(1, 2, 0)

    # save contour pixels of each coronal hole in the classified image.
    contour_list = ch_lib.save_contour_pixel_locations(classified_img, color_list)

    # force periodicity and delete small segments.
    contour_list_pruned = ch_lib.prune_coronal_hole_list(contour_list=contour_list)

    # TODO: Match coronal holes detected to previous ones.
    ch_lib.match_coronal_holes(contour_list=contour_list_pruned)

    # iterate over each coronal hole and plot its color + id number.
    image = np.zeros(classified_img.shape, dtype=np.uint8)
    for c in ch_lib.p1.contour_list:
        # TODO: color coronal hole.
        image[c.contour_pixels_row, c.contour_pixels_column, :] = c.color
        # plot the contours center.
        cv2.circle(img=image, center=c.pixel_centroid, radius=3, color=(0, 0, 0), thickness=-1)
        # check if its has multiple bounding boxes.
        ii = 0
        while ii < len(c.straight_box) / 4:
            # plot bounding box c.straight box returns top left x, y, w, h.
            cv2.rectangle(img=image, pt1=(c.straight_box[4 * ii + 0], c.straight_box[4 * ii + 1]),
                          pt2=(c.straight_box[4 * ii + 0] + c.straight_box[4 * ii + 2], c.straight_box[4 * ii + 1] +
                               c.straight_box[4 * ii + 3]),
                          color=(0, 255, 0), thickness=2)
            ii += 1
        # plot the contour's ID number.
        cv2.putText(img=image, text="ch #" + str(c.id), org=tuple(np.add(c.pixel_centroid, (-20, 15))),
                    fontFace=cv2.Formatter_FMT_DEFAULT, fontScale=0.5, color=(0, 0, 0), thickness=1)

    # iterate over frame number.
    ch_lib.frame_num += 1

    if 8 <= ch_lib.frame_num <= 9:
        # print ch_lib.
        print(ch_lib)
        # show tracking image.
        cv2.imshow("Coronal Hole Tracking, Frame = " + str(ch_lib.frame_num), classified_img)
        # show original image.
        cv2.imshow("Original Image, Frame = " + str(ch_lib.frame_num), image)
        # plot using matplotlib.
        plt.imshow(classified_img)
        plt.title("Coronal Hole Tracking, $w(\u03F4) = (1-\sin(\u03F4)) n_p$")
        plt.show()
        # wait time between frames.
        cv2.waitKey(10000)

cv2.waitKey(0)
