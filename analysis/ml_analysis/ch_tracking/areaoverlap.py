"""Match Coronal Holes based on their area overlap. """

import numpy as np


def area_overlap(ch1, ch2, da):
    """Area overlap of two coronal holes.

    Parameters
    ----------
    da: area matrix of each pixel in the image (spherical coordinates).
    ch1: Coronal Hole 1.
        Contour() object.
    ch2: Coronal Hole 2.
        Contour() object.

    Returns
    -------
        areaoverlap/ch1area, areaoverlap/ch2area
    """
    # zip ch1 pixel location.
    ch1_location = list(zip(ch1.contour_pixels_phi, ch1.contour_pixels_theta))

    # zip ch2 pixel location.
    ch2_location = set(zip(ch2.contour_pixels_phi, ch2.contour_pixels_theta))

    # find intersection - [phi, theta].
    pixel_intersection = np.array(list(ch2_location.intersection(ch1_location)))

    if len(pixel_intersection) == 0:
        intersection = 0
    else:
        # area of intersection on a sphere.
        intersection = np.sum(da[pixel_intersection[:, 0], pixel_intersection[:, 1]])

    return intersection / ch1.area, intersection / ch2.area


def classification_results(area_overlap_list, thresh=0.5):
    """ Based on the area_overlap results, we will identify if the coronal hole is associated with previously found
    coronal hole or will be classified as a new coronal hole and will be assigned a unique ID number.

    Parameters
    ----------
    area_overlap_list: (list)
        results of the new frame in order of identified coronal holes.

    thresh: (float)
        threshold to decide if the overlap is significant.
        Default is 0.5

    Returns
    -------
        (ndarray) with resulting class. "0" means "below the thresh", "1" means "above the thresh"
    """
    for ii, cls in enumerate(area_overlap_list):
        area_overlap_list[ii] = np.where(np.asarray(cls) > thresh, 1, 0).tolist()
    return area_overlap_list
