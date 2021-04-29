"""Classify coronal holes based on pixel proximity on a sphere.
Input: Greyscale image.
Output: list of contours and their attributes: coronal hole centroid, area, tilt, bounding box, etc ...

Last modified: April 26th, 2021 (Opal)
"""

import cv2
from analysis.ml_analysis.ch_tracking.dilation import latitude_weighted_dilation, find_contours, \
    get_list_of_contours_from_rbg, extra_3_by_3_kernel_dilation
from analysis.ml_analysis.ch_tracking.periodicity import prune_coronal_hole_list
from modules.map_manip import MapMesh
import matplotlib.pyplot as plt


def classify_grey_scaled_image(greyscale_image, lat_coord, lon_coord, frame_num=0,
                               gamma=10, BinaryThreshold=200, AreaThreshold=5e-3):
    """

    Parameters
    ----------
    lon_coord: (numpy array)
            longitude coordinate 1d mesh grid, Interval=[0, 2pi] (radians)
    lat_coord: (numpy array)
            latitude coordinate 1d mesh grid, Interval=[0, pi] (radians)
    greyscale_image: (numpy array)
            Greyscale image with pixel values between 0 and 255, such that
            coronal holes are close to 255 other regions are close to 0.
    AreaThreshold: (float)
            Area threshold to remove small contours from the returned list pf contours.
    BinaryThreshold: (int)
            Binary Threshold for finding contour on the latitude weighted image.
    gamma: (int)
            The width of the dilation 1d structuring element at the equator.
    frame_num: (int)
            identification number of the frame - will be assigned to coronal hole contour objects as an attribute.

    Returns
    -------
        list of contours.
    """

    # update the image dimensions.
    Mesh = MapMesh(p=lon_coord, t=lat_coord)

    # ================================================================================================================
    # Step 3: Latitude Weighted Dilation.
    # ================================================================================================================
    # pass one extra erode and dilate before latitude weighted dilation.
    w_dilated_img = latitude_weighted_dilation(grey_scale_image=greyscale_image,
                                               theta=Mesh.t,
                                               gamma=gamma,
                                               n_p=Mesh.n_p)

    dilated_img = extra_3_by_3_kernel_dilation(image=w_dilated_img)

    # ================================================================================================================
    # Step 4: Plot the contours found in the dilated image and multiply mask.
    # ================================================================================================================
    # add coronal holes to data base.
    rbg_dilated, color_list = find_contours(image=dilated_img, thresh=BinaryThreshold, Mesh=Mesh)

    # create a threshold mask.
    ret, thresh = cv2.threshold(greyscale_image, BinaryThreshold, 1, 0)

    # multiply mask.
    classified_img = (rbg_dilated.transpose(2, 0, 1) * thresh).transpose(1, 2, 0)

    # save contour pixels of each coronal hole in the classified image.
    full_contour_list = get_list_of_contours_from_rbg(rbg_image=classified_img, color_list=color_list, Mesh=Mesh,
                                                      frame_num=frame_num)

    # ================================================================================================================
    # Step 5: Force periodicity and remove small detections.
    # ================================================================================================================
    # force periodicity and delete small segments.
    pruned_contour_list = prune_coronal_hole_list(contour_list=full_contour_list, Mesh=Mesh,
                                                  area_threshold=AreaThreshold)

    return pruned_contour_list


if __name__ == "__main__":
    import numpy as np
    from analysis.ml_analysis.ch_tracking.dilation import generate_ch_color
    from analysis.ml_analysis.ch_tracking.plots import plot_coronal_hole

    # read in an example image
    image = cv2.imread("example_vid/various_shapes_0.jpg")

    # example image dimensions.
    n_t, n_p, color_dim = np.shape(image)
    p = np.linspace(0, 2 * np.pi, n_p)
    t = np.pi / 2 + np.arcsin(np.linspace(-1, 1, n_t))

    # force grey scale
    image_grey = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # get list of contours.
    contour_list = classify_grey_scaled_image(greyscale_image=image_grey, lat_coord=t, lon_coord=p, frame_num=0)

    # generate coronal hole random color.
    for ch in contour_list:
        ch.color = generate_ch_color()

    # plot all contours on a blank image.
    plot_coronal_hole(ch_list=contour_list, n_t=len(t), n_p=len(p), title="tester", filename="test_classification_fun")
    plt.show()
