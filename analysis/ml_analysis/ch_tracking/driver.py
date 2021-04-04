"""
Author: Opal Issan. Date: Feb 3rd, 2021.
Assumptions:
-------------------
Tracking Algorithm steps:

1. Input image in lat-lon projection
2. latitude weighted dilation.
3. find contours + color the areas arbitrarily - TODO: verify that they are unique by using a uniform vector.
4. multiply mask on original binary image.
5. delete small contours.
6. force periodicity.
7. match coronal holes to previous coronal holes in *window* consecutive frames.
8. add contours to database - which can be saved to JSON file.
9. graph connectivity - keep track of splitting & merging of coronal holes.
"""

import cv2
import numpy as np
from analysis.ml_analysis.ch_tracking.contour import Contour
import pickle
import json
from modules.map_manip import MapMesh
from analysis.ml_analysis.ch_tracking.src import CoronalHoleDB
import matplotlib.pyplot as plt


# Upload coronal hole video.
cap = cv2.VideoCapture("example_vid/maps_r101_chm_low_res_1.mov")

# cut out the axis and title.
t, b, r, l = 47, -55, 110, -55

# coronal hole video database.
ch_lib = CoronalHoleDB()

# initialize frame index.
ch_lib.frame_num = 1

# loop over each frame.
while ch_lib.frame_num <= 40:
    # ================================================================================================================
    # Step 1: Read in first frame.
    # ================================================================================================================
    # read in frame by frame.
    success, img = cap.read()

    # cut out the image axis and title.
    image = img[t:b, r:l, :]

    # image = cv2.imread('example_vid/various_shapes_0.jpg')
    # ================================================================================================================
    # Step 2: Convert Image to Greyscale.
    # ================================================================================================================
    # gray scale: coronal holes are close to 255 other regions are close to 0.
    image = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # update the image dimensions.
    Contour.n_t, Contour.n_p = np.shape(image)
    Contour.Mesh = MapMesh(p=np.linspace(0, 2*np.pi, Contour.n_p), t=np.linspace(0, np.pi, Contour.n_t))
    # ================================================================================================================
    # Step 3: Latitude Weighted Dilation.
    # ================================================================================================================
    # pass one extra erode and dilate before latitude weighted dilation.
    img = ch_lib.lat_weighted_dilation(grey_scale_image=image)

    # ================================================================================================================
    # Step 4: Plot the contours found in the dilated image and multiply mask.
    # ================================================================================================================
    # add coronal holes to data base.
    rbg_dilated, color_list = ch_lib.find_contours(img)

    # create a threshold mask.
    ret, thresh = cv2.threshold(image, CoronalHoleDB.BinaryThreshold, 1, 0)

    # multiply mask.
    classified_img = (rbg_dilated.transpose(2, 0, 1) * thresh).transpose(1, 2, 0)

    # save contour pixels of each coronal hole in the classified image.
    contour_list = ch_lib.save_contour_pixel_locations(classified_img, color_list)

    # ================================================================================================================
    # Step 5: Force periodicity and remove small detections.
    # ================================================================================================================
    # force periodicity and delete small segments.
    contour_list_pruned = ch_lib.prune_coronal_hole_list(contour_list=contour_list)

    # ================================================================================================================
    # Step 6: Match coronal holes detected to previous frame detections.
    # ================================================================================================================
    ch_lib.assign_new_coronal_holes(contour_list=contour_list_pruned)

    # ================================================================================================================
    # Step 7: Plot results.
    # ================================================================================================================
    # iterate over each coronal hole and plot its color + id number.
    final_image = np.ones(classified_img.shape, dtype=np.uint8) * 255
    for c in ch_lib.window_holder[-1].contour_list:
        # plot contour pixels.
        final_image[c.contour_pixels_theta, c.contour_pixels_phi, :] = c.color
        # plot the contours center.
        cv2.circle(img=final_image, center=(c.pixel_centroid[1], c.pixel_centroid[0]),
                   radius=4, color=(0, 0, 0), thickness=-1)

        cv2.circle(img=final_image, center=(c.pixel_centroid[1], c.pixel_centroid[0]),
                   radius=int(100*c.area), color=(255, 20, 147), thickness=2)

        # check if its has multiple bounding boxes.
        ii = 0
        while ii < len(c.straight_box) / 4:
            # plot bounding box c.straight box returns top left x, y, w, h.
            cv2.rectangle(img=final_image, pt1=(c.straight_box[4 * ii + 0], c.straight_box[4 * ii + 1]),
                          pt2=(c.straight_box[4 * ii + 0] + c.straight_box[4 * ii + 2], c.straight_box[4 * ii + 1] +
                               c.straight_box[4 * ii + 3]),
                          color=(0, 255, 0), thickness=2)
            ii += 1

        # draw rotated box.
        cv2.drawContours(final_image, [c.rot_box_corners], 0, (0, 0, 255), 2)

        # plot the contour's ID number.
        cv2.putText(img=final_image, text="#" + str(c.id),
                    org=tuple(np.add((c.pixel_centroid[1], c.pixel_centroid[0]), (-15, 15))),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)

        # # rotated box angle.
        # cv2.putText(img=final_image, text="a=" + str(round(c.pca_tilt, 2)),
        #             org=tuple(np.add((c.pixel_centroid[1], c.pixel_centroid[0]), (30, 10))),
        #             fontFace=cv2.Formatter_FMT_DEFAULT, fontScale=0.3, color=(0, 0, 0), thickness=1)
        #
        # # rotated box corners
        # cv2.putText(img=final_image, text="0",
        #             org=(c.rot_box_corners[0][0], c.rot_box_corners[0][1]),
        #             fontFace=cv2.Formatter_FMT_DEFAULT, fontScale=0.5, color=(255, 0, 0), thickness=1)
        #
        # cv2.putText(img=final_image, text="1",
        #             org=(c.rot_box_corners[1][0], c.rot_box_corners[1][1]),
        #             fontFace=cv2.Formatter_FMT_DEFAULT, fontScale=0.5, color=(255, 0, 0), thickness=1)
        #
        # cv2.putText(img=final_image, text="2",
        #             org=(c.rot_box_corners[2][0], c.rot_box_corners[2][1]),
        #             fontFace=cv2.Formatter_FMT_DEFAULT, fontScale=0.5, color=(255, 0, 0), thickness=1)

    if 1 <= ch_lib.frame_num <= 40:
        # # print ch_lib.
        # print(ch_lib)
        # # show tracking image.
        # cv2.imshow("Coronal Hole Tracking Final, Frame = " + str(ch_lib.frame_num), final_image)
        # # show original image.
        # cv2.imshow("Original Image, Frame = " + str(ch_lib.frame_num), image)
        # # show dilated image.
        # cv2.imshow("Dilated Image, Frame = " + str(ch_lib.frame_num), img)
        # # show image before forcing periodicity and deleting small coronal holes.
        # cv2.imshow("Classified Image before forcing periodicity and deleting small contours"
        #            ", Frame = " + str(ch_lib.frame_num), classified_img)

        # in order to access the zoom feature of an image, we can also plot using matplotlib gui.
        # plot using matplotlib.
        plt.imshow(final_image)

        # pixel coordinates + set ticks.
        p_pixel = np.linspace(0, Contour.n_p, 5)
        t_pixel = np.linspace(0, Contour.n_t, 5)

        plt.xticks(p_pixel, ["0", "$90$", "$180$", "$270$", "$360$"])
        plt.yticks(t_pixel, ["1", "$\dfrac{1}{2}$", "$0$", "-$\dfrac{1}{2}$", "-$1$"])

        # axis label.
        plt.xlabel("Longitude (Deg.)")
        plt.ylabel("Sin(Lat.)")

        # title
        plt.title("Coronal Hole Tracking, Frame #" + str(ch_lib.frame_num))
        # plt.show()
        # plt.savefig("results/images/tester/frames/" + "frame_" + str(ch_lib.frame_num) + ".png")

        # plot connectivity subgraphs.
        ch_lib.Graph.create_plots()

        # wait time between frames.
    cv2.waitKey(1000)

    # # iterate over frame number.
    ch_lib.frame_num += 1

# save ch library to json file.
# with open('results/first_frame_test.json', 'w') as f:
#     f.write(ch_lib.__str__())

# # save object to pickle file.
# with open('results/first_frame_test_knn.pkl', 'wb') as f:
#    pickle.dump(ch_lib, f)

cv2.waitKey(0)
