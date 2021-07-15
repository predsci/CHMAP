"""Classify coronal holes based on pixel proximity on a sphere.
Input: Greyscale image.
Output: list of contours and their attributes: coronal hole centroid, area, tilt, bounding box, etc ...

Last modified: June 7th, 2021 (Opal)
"""

import cv2
from chmap.coronal_holes.tracking.src.dilation import latitude_weighted_dilation, find_contours, \
    get_list_of_contours_from_rbg, uniform_dilation_in_latitude
from chmap.coronal_holes.tracking.src.periodicity import prune_coronal_hole_list
from chmap.maps.util.map_manip import MapMesh
import matplotlib.pyplot as plt


def classify_grey_scaled_image(greyscale_image, lat_coord, lon_coord, map_dir, db_session, gamma=10, beta=10,
                               BinaryThreshold=0.7, AreaThreshold=5e-3, frame_num=0, frame_timestamp=None):
    """ Find a list of contours (coronal hole objects) with coronal hole attributes such as
        pixel location, area, centroid, bounding box, tilt, etc... from a CH greyscale/binary map.

    Parameters
    ----------
    lon_coord: (numpy array)
            longitude coordinate 1d mesh grid, Interval=[0, 2pi] (radians)

    lat_coord: (numpy array)
            latitude coordinate 1d mesh grid, Interval=[0, pi] (radians)

    greyscale_image: (numpy array)
            Greyscale image with pixel values between 0 and 1, such that
            coronal holes are close to 1 other regions are close to 0.

    AreaThreshold: (float) - Optional
            Area threshold to remove small contours from the returned list of contours.
            Default is 5E-3.

    BinaryThreshold: (int) - Optional
            Binary Threshold for finding contour on the latitude weighted image.
            Default is 0.7.

    gamma: (int) - Optional
            The width of the dilation 1d structuring element at the equator (longitude dilation).
            Default is 10.

    beta: (int) - Optional
            The height of the dilation 1d structuring element (latitude dilation).
            Default is 10.

    frame_num: (int) - Optional
            identification number of the frame - will be assigned to
            coronal hole contour objects as an attribute.

    frame_timestamp: (str) - Optional
            Synchronic CH Map time-stamp. This is used as an attribute of the contour object.

    map_dir: (str)
            path to directory with magnetic data to compute the flux.

    db_session:
            database session information.

    Returns
    -------
        list of contours.
    """

    # update the image dimensions.
    Mesh = MapMesh(p=lon_coord, t=lat_coord)

    # ================================================================================================================
    # Step 3: Latitude Weighted Dilation (in longitude) + Uniform dilation (in latitude).
    # ================================================================================================================
    # latitude weighted dilation in longitude direction.
    latitude_w_dilated_img = latitude_weighted_dilation(grey_scale_image=greyscale_image,
                                                        theta=Mesh.t,
                                                        gamma=gamma,
                                                        n_p=Mesh.n_p)
    # uniform dilation in latitude.
    dilated_img = uniform_dilation_in_latitude(image=latitude_w_dilated_img, beta=beta)

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
                                                      frame_num=frame_num, frame_timestamp=frame_timestamp,
                                                      db_session=db_session, map_dir=map_dir)

    # ================================================================================================================
    # Step 5: Force periodicity and remove small detections.
    # ================================================================================================================
    # force periodicity and delete small segments.
    pruned_contour_list = prune_coronal_hole_list(contour_list=full_contour_list, Mesh=Mesh,
                                                  area_threshold=AreaThreshold)

    return pruned_contour_list


if __name__ == "__main__":
    import numpy as np
    from chmap.coronal_holes.tracking.src import generate_ch_color
    from chmap.coronal_holes.tracking.tools.plots import plot_coronal_hole

    # read in an example image
    image = cv2.imread("data/example_frame.png")

    # example image dimensions.
    n_t, n_p, color_dim = np.shape(image)
    p = np.linspace(0, 2 * np.pi, n_p)
    t = np.pi / 2 + np.arcsin(np.linspace(-1, 1, n_t))

    # force grey scale
    image_grey = (255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))/255

    # get list of contours.
    contour_list = classify_grey_scaled_image(greyscale_image=image_grey, lat_coord=t, lon_coord=p, frame_num=0)

    # generate coronal hole random color.
    for ch in contour_list:
        ch.color = generate_ch_color()

    # plot all contours on a blank image.
    plot_coronal_hole(ch_list=contour_list, n_t=len(t), n_p=len(p), title="tester", filename="test_classification_fun")
    plt.show()
