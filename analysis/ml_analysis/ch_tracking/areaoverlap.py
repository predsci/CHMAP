"""Match Coronal Holes based on their area overlap. """

import numpy as np
from analysis.ml_analysis.ch_tracking.contour import Contour


def area_overlap(ch1, ch2, da):
    """Area overlap of two coronal holes.
    TODO: Optimize.

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
    # create a dummy matrix
    mat1 = np.zeros(da.shape)
    mat2 = np.zeros(da.shape)

    # create ch1 mat
    mat1[ch1.contour_pixels_phi, ch1.contour_pixels_theta] = 1

    # create ch2 mat
    mat2[ch2.contour_pixels_phi, ch2.contour_pixels_theta] = 1

    # multiply mask
    overlap = np.multiply(mat1, mat2)

    # where is the overlapping region.
    index = np.where(overlap)

    # compute the area on a sphere of the overlapped region.
    intersection = np.sum(da[index[0], index[1]])

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
