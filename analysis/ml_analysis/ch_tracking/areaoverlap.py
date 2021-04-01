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


def max_area_overlap(area_check_list, area_overlap_results, threshold=0.5):
    """ Return the ID with the *most* area overlap.

    Parameters
    ----------
    threshold: (float) between 0 and 1
        Default is 0.5
        Threshold for area overlap ratio to be a match.

    area_check_list: (list)
        results of KNN algorithm.

    area_overlap_results: (list)
        area overlap average ratio.

    Returns
    -------
        list of id corresponding to the contour list "0" means new class.
    """
    # initialize the returned list.
    match_list = [None] * len(area_overlap_results)

    for ii, res in enumerate(area_overlap_results):
        # find the maximum area overlap average ratio.
        max_val = max(res)

        # if it is below the threshold then assign as a "new" coronal hole - labeled with zero.
        if max_val < threshold:
            match_list[ii] = 0

        # otherwise, assign the ID corresponding to the max area overlap.
        else:
            # find the index of maximum overlap.
            max_index = res.index(max_val)

            # assign the corresponding ID number.
            match_list[ii] = area_check_list[ii][max_index]

    return match_list
