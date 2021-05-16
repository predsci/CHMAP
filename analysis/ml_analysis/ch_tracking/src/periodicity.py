"""Force periodicity on contour list
Input: Contour list, MeshMap
Output: pruned list.

Last Modified: April 26th, 2021. (Opal)

# todo: fix merging rotated box ...
"""
import numpy as np


def prune_coronal_hole_list(contour_list, Mesh, area_threshold=0):
    """Remove small coronal holes and force periodicity.

    Parameters
    ----------
    Mesh: (MeshMap)
                object containing the image coordinates.
    area_threshold: (float)
                area threshold to delete small contours.
    contour_list: (list)
                list of all contours.

    Returns
    -------
    pruned contour list.
    """
    # remove small coronal holes.
    contour_list = remove_small_coronal_holes(contour_list=contour_list, area_threshold=area_threshold)
    # force periodicity.
    return force_periodicity(contour_list=contour_list, Mesh=Mesh)


def force_periodicity(contour_list, Mesh):
    """Force periodicity.

    Parameters
    ----------
    Mesh: (MeshMap)
                object containing the image coordinates.
    contour_list: (list)
                list of all contours.

    Returns
    -------
    updated contour list.
    """
    # loop over each coronal hole and check if it is on the periodic border.
    ii = 0
    while ii <= len(contour_list) - 2:
        c1 = contour_list[ii]
        # check if it overlaps phi=0.
        if c1.periodic_at_zero:
            # check for all other periodic 2pi.
            jj = ii + 1
            while jj <= len(contour_list) - 1 and len(contour_list) > 0:
                c2 = contour_list[jj]
                if c2.periodic_at_2pi:
                    # get interval of latitude at 0.
                    t1, t2 = c1.lat_interval_at_zero()
                    # get interval of latitude at 2pi.
                    t3, t4 = c2.lat_interval_at_2_pi(Mesh=Mesh)
                    # check if intervals overlap.
                    if interval_overlap(t1, t2, t3, t4):
                        # merge the two contours by appending c2 to c1.
                        contour_list[ii] = merge_contours(c1=c1, c2=c2, Mesh=Mesh)
                        c1 = contour_list[ii]
                        contour_list.remove(c2)
                        ii += -1
                jj += 1

        # check if it overlaps phi=2pi.
        if c1.periodic_at_2pi:
            # check for all other periodic 0.
            jj = ii + 1
            while jj <= len(contour_list) - 1 and len(contour_list) > 0:
                c2 = contour_list[jj]
                if c2.periodic_at_zero:
                    # get interval of latitude at 2pi.
                    t1, t2 = c1.lat_interval_at_2_pi(Mesh=Mesh)
                    # get interval of latitude at 0.
                    t3, t4 = c2.lat_interval_at_zero()
                    # check if intervals overlap.
                    if interval_overlap(t1, t2, t3, t4):
                        # merge the two contours by appending c2 to c1.
                        contour_list[ii] = merge_contours(c1=c1, c2=c2, Mesh=Mesh)
                        contour_list.remove(c2)
                        ii += -1
                jj += 1
        ii += 1
    return contour_list


def merge_contours(c1, c2, Mesh):
    """Merge c2 onto c1.
        # TODO: update rotated box features.
    Parameters
    ----------
    Mesh: (MeshMap)
                object containing the image coordinates.
    c1: (Contour)
                Contour object see contour.py
    c2: (Contour)
                Contour object see contour.py

    Returns
    -------
    c1 modified.
    """
    # append c2 pixel locations to c1.
    c1.contour_pixels_theta = np.append(c2.contour_pixels_theta, c1.contour_pixels_theta)
    c1.contour_pixels_phi = np.append(c2.contour_pixels_phi, c1.contour_pixels_phi)

    # update c1 periodic label.
    if c2.periodic_at_2pi:
        c1.periodic_at_2pi = True
    if c2.periodic_at_zero:
        c1.periodic_at_zero = True

    # update c1 area.
    c1.area = c1.area + c2.area

    # update c1 pixel centroid.
    c1.pixel_centroid = c1.compute_pixel_centroid(Mesh=Mesh)

    # update bounding box.
    c1.straight_box = np.append(c1.straight_box, c2.straight_box)

    # update bounding box area.

    c1.straight_box_area = c1.straight_box_area + c2.straight_box_area

    # todo: fix this....
    # c1.rot_box = np.append(c1.rot_box, c2.rot_box)

    # save rot box corners.
    c1.rot_box_corners = np.vstack((c1.rot_box_corners, c2.rot_box_corners))

    # save rot box angle with respect to north.
    c1.rot_box_angle = np.append(c1.rot_box_angle, c2.rot_box_angle)

    # compute the rotate box area.
    c1.rot_box_area = c1.rot_box_area + c2.rot_box_area

    # compute the tilt of the coronal hole in spherical coordinates using PCA.
    c1.pca_tilt, c1.sig_tilt = c1.compute_coronal_hole_tilt_pca(Mesh=Mesh)

    return c1


def remove_small_coronal_holes(contour_list, area_threshold):
    """Remove all contours that are smaller than AreaThreshold.

    Parameters
    ----------
    area_threshold: (float)
                area threshold to delete small contours.
    contour_list: (list)
                list of all contours.

    Returns
    -------
    pruned contour list.
    """
    ii = 0
    while ii < len(contour_list):
        if contour_list[ii].area < area_threshold:
            contour_list.remove(contour_list[ii])
            ii += -1
        ii += 1
    return contour_list


def interval_overlap(t1, t2, t3, t4):
    """check if two intervals overlap.

    Parameters
    ----------
    t1: int
    t2: int
    t3: int
    t4: int

    Returns
    -------
    Boolean (True/False)
    The two intervals are built from [t1,t2] and [t3,t4] assuming t1 <= t2 and t3 <=t4.
    If the two intervals overlap: return True, otherwise False.
    """
    if t1 <= t3 <= t2:
        return True
    elif t1 <= t4 <= t2:
        return True
    elif t3 <= t1 <= t4:
        return True
    elif t3 <= t2 <= t4:
        return True
    return False
