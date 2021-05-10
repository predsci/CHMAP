"""
Functions to manipulate and combine maps
"""

import sys
import numpy as np
from scipy.interpolate import interp1d
from skimage.measure import block_reduce
import time

import modules.datatypes as psi_d_types
from settings.info import DTypes
from modules.coord_manip import s2c
import modules.lbcc_funs as lbcc_funs



class MapMesh:
    """Simple class to hold mesh information for a phi/theta map.

    The purpose of this class is to facilitate map/mesh based computations such as integral
    averaging or mesh spacing calculations.

    This holds things like:
    - the exterior (main) mesh locations.
    - the interior (half) mesh locations.
    - the area factors at the main mesh points (using an approximation for the edges).
    - 2D arrays with the cartesian locations of each point (for easy averaging calculations).
    - interpolators to get the cell index given a phi or theta value
    - interpolators to get the local mesh spacing for a given phi or theta value

    The phi and theta can be non-uniformly spaced.

    This class is not as flexible/general as what is in our fortran code remesh_general
    but it is suitable for our purposes.

    We can add other interpolators / interpolation methods for maps to this later if needed.
    """

    def __init__(self, p, t):
        """
        Generate the mesh object based on 1D arrays of phi and theta locations
        """
        # define a floating point tolerance for checking if at 0, pi, or 2pi
        eps = 2e-7

        # check to see if it is periodic (ASSUME ONE REPEATED POINT)
        periodic = False
        if abs(p[0]) <= eps and abs(p[-1] - 2*np.pi) <= eps:
            periodic = True

        # get the 1D mesh properties for phi and theta:
        #   this is: half mesh locations, main mesh spacing, half mesh spacing
        ph, dp, dph = get_1d_mesh_properties(p)
        th, dt, dth = get_1d_mesh_properties(t)

        # get a special version of dp for periodic interpolation (put the whole cell width vs. half)
        # need 2 versions since for area, you want the half cell!
        dp_alt = np.copy(dp)
        if periodic:
            dp_alt[0] = dp[0] + dp[-1]
            dp_alt[-1] = dp_alt[0]

        # Get the coordinates, assuming radius = 1 (photosphere)
        # here i want to make an array that is phi, theta ordered on purpose
        t2d, p2d = np.meshgrid(t, p)
        dt2d, dp2d = np.meshgrid(dt, dp)
        r2d = np.ones_like(t2d)

        # get the cartesian coordinates (for convenience in later computations)
        x2d, y2d, z2d = s2c(r2d, t2d, p2d)

        # get the area of each pixel, but modify t2d to account for polar points (shift a quarter point)
        if abs(t[0]) <= eps:
            t2d[:, 0] = 0.5*(t[0] + th[0])
        if abs(t[-1] - np.pi) <= eps:
            t2d[:, -1] = 0.5*(t[-1] + th[-1])
        da = np.sin(t2d)*dt2d*dp2d

        # now build interpolators that will be helpful
        interp_p2index = interp1d(p, np.arange(len(p)), fill_value=(0, len(p) - 1), bounds_error=False)
        interp_t2index = interp1d(t, np.arange(len(t)), fill_value=(0, len(t) - 1), bounds_error=False)
        interp_p2dp = interp1d(p, dp_alt, fill_value=(dp_alt[0], dp_alt[-1]), bounds_error=False)
        interp_t2dt = interp1d(t, dt, fill_value=(dt[0], dt[-1]), bounds_error=False)

        # add these as attributes to the class
        self.n_p = len(p)
        self.n_t = len(t)
        self.p = p
        self.t = t
        self.dp = dp
        self.dt = dt
        self.ph = ph
        self.th = th
        self.dph = dph
        self.dth = dth
        self.da = da
        self.periodic = periodic
        self.interp_p2index = interp_p2index
        self.interp_t2index = interp_t2index
        self.interp_p2dp = interp_p2dp
        self.interp_t2dt = interp_t2dt
        self.x2d = x2d
        self.y2d = y2d
        self.z2d = z2d


def get_1d_mesh_properties(x):
    """
    Return mesh properties based on an array of 1D mesh locations (assume monotonic).

    Assuming that the input x is the bounding exterior (main) mesh, this returns
    the interior (half) mesh locations and the cell centered spacings for each.

    The spacing BCs for cells at the boundary assume you are doing intergrals --> use half.

    :param x: 1D numpy array of grid locations.
    """
    # Compute the interior (half) mesh positions
    xh = 0.5*(x[1:] + x[0:-1])

    # Compute the spacing centered around the interior half mesh
    dxh = x[1:] - x[0:-1]

    # Compute the spacing centered on the bounding (main) mesh, interior first
    dx = xh[1:] - xh[0:-1]

    # Add the boundary points (half spacing)
    dx = np.concatenate([[xh[0] - x[0]], dx, [x[-1] - xh[-1]]])

    return xh, np.abs(dx), np.abs(dxh)


def combine_maps(map_list, mu_cutoff=0.0, mu_merge_cutoff=None, del_mu=None):
    """
    Take a list of Psi_Map objects and do minimum intensity merge to a single map.
    Using mu_merge_cutoff: based off the two cutoff algorithm from Caplan et. al. 2016.
    Using del_mu: based off maximum mu value from list
    :param map_list: List of Psi_Map objects
    :param mu_cutoff: data points/pixels with a mu less than mu_cutoff will be discarded before
    merging.
    :param mu_merge_cutoff: mu cutoff value for discarding pixels in areas of instrument overlap
    :param del_mu: For a given data point/pixel of the map first find the maximum mu from map_list.
    :return: Psi_Map object resulting from merge.
    """
    # determine number of maps. if only one, do nothing
    nmaps = len(map_list)
    if nmaps == 1:
        return map_list[0]

    # check that all maps have the same x and y grids
    same_grid = all(map_list[0].x == map_list[1].x) and all(map_list[0].y == map_list[1].y)
    if nmaps > 2:
        for ii in range(1, nmaps - 1):
            same_temp = all(map_list[ii].x == map_list[ii + 1].x) and all(map_list[ii].y == map_list[ii + 1].y)
            same_grid = same_grid & same_temp
            if not same_grid:
                break

    if same_grid:
        if mu_cutoff > 0.0:
            # for all pixels with mu < mu_cutoff, set intensity to no_data_val
            for ii in range(nmaps):
                map_list[ii].data[map_list[ii].mu < mu_cutoff] = map_list[ii].no_data_val

        # construct arrays of mu's and data
        mat_size = map_list[0].mu.shape
        mu_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_MU)
        data_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_DATA)
        image_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_ORIGIN_IMAGE)
        if map_list[0].chd is not None:
            chd_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_CHD)
            for ii in range(nmaps):
                mu_array[:, :, ii] = map_list[ii].mu
                data_array[:, :, ii] = map_list[ii].data
                chd_array[:, :, ii] = map_list[ii].chd
                image_array[:, :, ii] = map_list[ii].origin_image
        else:
            for ii in range(nmaps):
                mu_array[:, :, ii] = map_list[ii].mu
                data_array[:, :, ii] = map_list[ii].data
                image_array[:, :, ii] = map_list[ii].origin_image

        float_info = np.finfo(map_list[0].data.dtype)
        good_index = np.ones(shape=mat_size + (nmaps,), dtype=bool)
        if mu_merge_cutoff is not None:
            overlap = np.ndarray(shape=mat_size + (nmaps,), dtype=bool)
            for ii in range(nmaps):
                for jj in range(nmaps):
                    if ii != jj:
                        overlap[:, :, ii] = np.logical_and(data_array[:, :, ii] != map_list[0].no_data_val,
                                                           data_array[:, :, jj] != map_list[0].no_data_val)
            for ii in range(nmaps):
                good_index[:, :, ii] = np.logical_or(np.logical_and(overlap[:, :, ii],
                                                                    mu_array[:, :, ii] >= mu_merge_cutoff),
                                                     np.logical_and(
                                                         data_array[:, :, ii] != map_list[0].no_data_val,
                                                         mu_array[:, :, ii] >= mu_cutoff))
            good_index2D = np.any(good_index, axis=2)
            # make poor mu pixels unusable to merge
            # data_array[np.logical_not(good_index)] = float_info.max
        elif del_mu is not None:
            max_mu = mu_array.max(axis=2)
            for ii in range(nmaps):
                good_index[:, :, ii] = np.logical_and(
                    mu_array[:, :, ii] > (max_mu - del_mu),
                    data_array[:, :, ii] >= 0.)
            good_index2D = np.any(good_index, axis=2)
            # make poor mu pixels unusable to merge
            # data_array[np.logical_not(good_index)] = float_info.max

        good_data = data_array.copy()
        # make poor mu pixels unusable to merge
        good_data[~good_index] = float_info.max
        good_data_winner = np.argmin(good_data, axis=2)

        # for less-good data, make no_data_vals unusable to merge
        # data_array[data_array == map_list[0].no_data_val] = float_info.max
        data_array[data_array < 0.] = float_info.max
        # find minimum intensity of remaining pixels
        all_winners = np.argmin(data_array, axis=2)
        # return bad-mu and no-data pixels to no_data_val
        data_array[data_array == float_info.max] = map_list[0].no_data_val

        # correct indices to create maps
        col_index, row_index = np.meshgrid(range(mat_size[1]), range(mat_size[0]))
        keep_mu = mu_array[:, :, 0].copy()
        keep_mu[good_index2D] = mu_array[row_index, col_index, good_data_winner][good_index2D]
        keep_mu[~good_index2D] = mu_array[row_index, col_index, all_winners][~good_index2D]

        keep_data = data_array[:, :, 0].copy()
        keep_data[good_index2D] = data_array[row_index, col_index, good_data_winner][good_index2D]
        keep_data[~good_index2D] = data_array[row_index, col_index, all_winners][~good_index2D]

        keep_image = image_array[:, :, 0].copy()
        keep_image[good_index2D] = image_array[row_index, col_index, good_data_winner][good_index2D]
        keep_image[~good_index2D] = image_array[row_index, col_index, all_winners][~good_index2D]

        if map_list[0].chd is not None:
            keep_chd = chd_array[:, :, 0].copy()
            keep_chd[good_index2D] = chd_array[row_index, col_index, good_data_winner][good_index2D]
            keep_chd[~good_index2D] = chd_array[row_index, col_index, all_winners][~good_index2D]
        else:
            keep_chd = None

        # Generate new EUV map
        euv_map = psi_d_types.PsiMap(keep_data, map_list[0].x, map_list[0].y, mu=keep_mu,
                                     origin_image=keep_image, chd=keep_chd, no_data_val=map_list[0].no_data_val)
    else:
        raise ValueError("'map_list' maps have different grids. This is not yet supported in " +
                         "map_manip.combine_maps()")

    return euv_map


def combine_cr_maps(n_images, map_list, mu_cutoff=0.0, mu_merge_cutoff=0.4):
    """
        Take an already combined map, and a single image map, and do minimum intensity merge to a single map.
        Using mu_merge_cutoff: based off the two cutoff algorithm from Caplan et. al. 2016.
        Using del_mu: based off maximum mu value from list
        :param n_images: number of images in original map
        :param map_list: List of Psi_Map objects (single_euv_map, combined_euv_map)
        :param mu_cutoff: data points/pixels with a mu less than mu_cutoff will be discarded before
        merging.
        :param mu_merge_cutoff: mu cutoff value for discarding pixels in areas of instrument overlap
        :return: Psi_Map object resulting from merge.
        """
    # determine number of maps. if only one, do nothing
    nmaps = len(map_list)
    if nmaps == 1:
        return map_list[0]

    # check that all maps have the same x and y grids
    same_grid = all(map_list[0].x == map_list[1].x) and all(map_list[0].y == map_list[1].y)
    if nmaps > 2:
        for ii in range(1, nmaps - 1):
            same_temp = all(map_list[ii].x == map_list[ii + 1].x) and all(map_list[ii].y == map_list[ii + 1].y)
            same_grid = same_grid & same_temp
            if not same_grid:
                break

    else:
        # check that all maps have the same x and y grids
        same_grid = all(map_list[0].x == map_list[1].x) and all(map_list[0].y == map_list[1].y)
        if nmaps > 2:
            for ii in range(1, nmaps - 1):
                same_temp = all(map_list[ii].x == map_list[ii + 1].x) and all(map_list[ii].y == map_list[ii + 1].y)
                same_grid = same_grid & same_temp
                if not same_grid:
                    break

    if same_grid:
        if mu_cutoff > 0.0:
            # for all pixels with mu < mu_cutoff, set intensity to no_data_val
            for ii in range(nmaps):
                map_list[ii].data[map_list[ii].mu < mu_cutoff] = map_list[ii].no_data_val

        # construct arrays of mu's and data
        mat_size = map_list[0].mu.shape
        mu_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_MU)
        data_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_DATA)
        use_data = np.ndarray(shape=mat_size, dtype=DTypes.MAP_DATA)
        image_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_ORIGIN_IMAGE)
        for ii in range(nmaps):
            mu_array[:, :, ii] = map_list[ii].mu
            data_array[:, :, ii] = map_list[ii].data
            image_array[:, :, ii] = map_list[ii].origin_image

        # find overlap indices
        float_info = np.finfo(map_list[0].data.dtype)
        good_index = np.ndarray(shape=mat_size + (nmaps,), dtype=bool)
        overlap = np.logical_and(data_array[:, :, 0] != map_list[0].no_data_val,
                                 data_array[:, :, 1] != map_list[0].no_data_val)
        # insert data in no-overlap areas
        for ii in range(nmaps):
            good_mu = np.logical_and(overlap, mu_array[:, :, ii] >= mu_merge_cutoff)
            use_data[good_mu] = data_array[:, :, ii][good_mu]
            # added the data != 0 to try and get rid of the white parts
            good_index[:, :, ii] = np.logical_or(good_mu, np.logical_or(data_array[:, :, ii] != map_list[0].no_data_val,
                                                                        data_array[:, :, ii] != 0))

        # make poor mu pixels unusable to merge
        data_array[np.logical_not(good_index)] = float_info.max
        # make no_data_vals unusable to merge
        data_array[data_array == map_list[0].no_data_val] = float_info.max
        # find minimum intensity of remaining pixels
        map_index = np.argmin(data_array, axis=2)
        # return bad-mu and no-data pixels to no_data_val
        data_array[data_array == float_info.max] = map_list[0].no_data_val

        # determine viable non-overlap pixels
        euv_new = np.logical_and(map_list[0].data != map_list[0].no_data_val, np.logical_not(overlap))
        euv_org = np.logical_and(map_list[1].data != map_list[0].no_data_val, np.logical_not(overlap))

        # insert other non-overlap data
        use_data[euv_new] = data_array[:, :, 0][euv_new]
        use_data[euv_org] = data_array[:, :, 1][euv_org]

        # correct indices to create maps
        col_index, row_index = np.meshgrid(range(mat_size[1]), range(mat_size[0]))
        map_index[euv_new] = 0
        map_index[euv_org] = 1
        keep_mu = mu_array[row_index, col_index, map_index]
        keep_data = data_array[row_index, col_index, map_index]
        keep_image = image_array[row_index, col_index, map_index]

        # Generate new CHD map
        if map_list[0].chd is not None:
            chd_data = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_DATA)
            chd_array = np.ndarray(shape=mat_size, dtype=DTypes.MAP_DATA)
            for ii in range(nmaps):
                chd_data[:, :, ii] = map_list[ii].chd
            # determine overlap pixels
            use_chd = np.logical_and(map_list[0].data != map_list[0].no_data_val,
                                     map_list[1].data != map_list[0].no_data_val)
            # average chd data to get probability
            chd_array[use_chd] = ((n_images - 1) * chd_data[:, :, 1][use_chd] + chd_data[:, :, 0][
                use_chd]) / n_images

            # determine viable non-overlap pixels
            use_new = np.logical_and(map_list[0].data != map_list[0].no_data_val, np.logical_not(use_chd))
            use_org = np.logical_and(map_list[1].data != map_list[0].no_data_val, np.logical_not(use_chd))
            # insert other non-overlap data
            chd_array[use_new] = chd_data[:, :, 0][use_new]
            chd_array[use_org] = chd_data[:, :, 1][use_org]
            # choose correct chd data to use
            keep_chd = chd_array[row_index, col_index]
        else:
            keep_chd = None
        # Generate new EUV map
        euv_combined = psi_d_types.PsiMap(keep_data, map_list[0].x, map_list[0].y, mu=keep_mu,
                                          origin_image=keep_image, chd=keep_chd, no_data_val=map_list[0].no_data_val)

    else:
        raise ValueError("'map_list' maps have different grids. This is not yet supported in " +
                         "map_manip.combine_cr_maps()")

    return euv_combined


def combine_mu_maps(n_images, map_list, mu_cutoff=0.0, mu_merge_cutoff=0.4):
    """
        Take an already combined map, and a single image map, and do minimum intensity merge to a single map.
        Using mu_merge_cutoff: based off the two cutoff algorithm from Caplan et. al. 2016.
        Using del_mu: based off maximum mu value from list
        :param n_images: number of images in original map
        :param map_list: List of Psi_Map objects (single_euv_map, combined_euv_map)
        :param mu_cutoff: data points/pixels with a mu less than mu_cutoff will be discarded before
        merging.
        :param mu_merge_cutoff: mu cutoff value for discarding pixels in areas of instrument overlap
        :return: Psi_Map object resulting from merge.
        """
    # determine number of maps. if only one, do nothing
    nmaps = len(map_list)
    if nmaps == 1:
        if map_list[0].chd is not None:
            mu_values = map_list[0].mu
            mu_values = np.where(mu_values > 0, mu_values, 0.01)
            map_list[0].chd = map_list[0].chd * mu_values
            return map_list[0]
        else:
            return map_list[0]

    # check that all maps have the same x and y grids
    same_grid = all(map_list[0].x == map_list[1].x) and all(map_list[0].y == map_list[1].y)
    if nmaps > 2:
        for ii in range(1, nmaps - 1):
            same_temp = all(map_list[ii].x == map_list[ii + 1].x) and all(map_list[ii].y == map_list[ii + 1].y)
            same_grid = same_grid & same_temp
            if not same_grid:
                break

    else:
        # check that all maps have the same x and y grids
        same_grid = all(map_list[0].x == map_list[1].x) and all(map_list[0].y == map_list[1].y)
        if nmaps > 2:
            for ii in range(1, nmaps - 1):
                same_temp = all(map_list[ii].x == map_list[ii + 1].x) and all(map_list[ii].y == map_list[ii + 1].y)
                same_grid = same_grid & same_temp
                if not same_grid:
                    break

    if same_grid:
        if mu_cutoff > 0.0:
            # for all pixels with mu < mu_cutoff, set intensity to no_data_val
            for ii in range(nmaps):
                map_list[ii].data[map_list[ii].mu < mu_cutoff] = map_list[ii].no_data_val

        # construct arrays of mu's and data
        mat_size = map_list[0].mu.shape
        mu_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_MU)
        data_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_DATA)
        use_data = np.ndarray(shape=mat_size, dtype=DTypes.MAP_DATA)
        image_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_ORIGIN_IMAGE)
        for ii in range(nmaps):
            mu_array[:, :, ii] = map_list[ii].mu
            data_array[:, :, ii] = map_list[ii].data
            image_array[:, :, ii] = map_list[ii].origin_image
        # mu values less than 0 become 0
        mu_array = np.where(mu_array > 0, mu_array, 0.01)
        # weight by mu value
        # data_array[:, :, 0] = data_array[:, :, 0] * mu_array[:, :, 0]
        # find overlap indices
        float_info = np.finfo(map_list[0].data.dtype)
        good_index = np.ndarray(shape=mat_size + (nmaps,), dtype=bool)
        overlap = np.logical_and(data_array[:, :, 0] != map_list[0].no_data_val,
                                 data_array[:, :, 1] != map_list[0].no_data_val)
        # insert data in no-overlap areas
        for ii in range(nmaps):
            good_mu = np.logical_and(overlap, mu_array[:, :, ii] >= mu_merge_cutoff)
            use_data[good_mu] = data_array[:, :, ii][good_mu]
            # added the data != 0 to try and get rid of the white parts
            good_index[:, :, ii] = np.logical_or(good_mu, np.logical_or(data_array[:, :, ii] != map_list[0].no_data_val,
                                                                        data_array[:, :, ii] != 0))

        # make poor mu pixels unusable to merge
        data_array[np.logical_not(good_index)] = float_info.max
        # make no_data_vals unusable to merge
        data_array[data_array == map_list[0].no_data_val] = float_info.max
        # find minimum intensity of remaining pixels
        map_index = np.argmin(data_array, axis=2)
        # return bad-mu and no-data pixels to no_data_val
        data_array[data_array == float_info.max] = map_list[0].no_data_val

        # determine viable non-overlap pixels
        euv_new = np.logical_and(map_list[0].data != map_list[0].no_data_val, np.logical_not(overlap))
        euv_org = np.logical_and(map_list[1].data != map_list[0].no_data_val, np.logical_not(overlap))

        # insert other non-overlap data
        use_data[euv_new] = data_array[:, :, 0][euv_new]
        use_data[euv_org] = data_array[:, :, 1][euv_org]

        # correct indices to create maps
        col_index, row_index = np.meshgrid(range(mat_size[1]), range(mat_size[0]))
        map_index[euv_new] = 0
        map_index[euv_org] = 1
        keep_mu = mu_array[row_index, col_index, map_index]
        keep_data = data_array[row_index, col_index, map_index]
        keep_image = image_array[row_index, col_index, map_index]

        # Generate new CHD map
        if map_list[0].chd is not None:
            chd_data = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_DATA)
            chd_array = np.ndarray(shape=mat_size, dtype=DTypes.MAP_DATA)
            for ii in range(nmaps):
                chd_data[:, :, ii] = map_list[ii].chd
            # add mu weighting to new data
            chd_data[:, :, 0] = chd_data[:, :, 0] * mu_array[:, :, 0]
            # determine overlap pixels
            use_chd = np.logical_and(map_list[0].data != map_list[0].no_data_val,
                                     map_list[1].data != map_list[0].no_data_val)
            # average chd data to get probability
            chd_array[use_chd] = ((n_images - 1) * chd_data[:, :, 1][use_chd] + chd_data[:, :, 0][
                use_chd]) / n_images

            # determine viable non-overlap pixels
            use_new = np.logical_and(map_list[0].data != map_list[0].no_data_val, np.logical_not(use_chd))
            use_org = np.logical_and(map_list[1].data != map_list[0].no_data_val, np.logical_not(use_chd))
            # insert other non-overlap data
            chd_array[use_new] = chd_data[:, :, 0][use_new]
            chd_array[use_org] = chd_data[:, :, 1][use_org]
            # choose correct chd data to use
            keep_chd = chd_array[row_index, col_index]
        else:
            keep_chd = None
        # Generate new EUV map
        euv_combined = psi_d_types.PsiMap(keep_data, map_list[0].x, map_list[0].y, mu=keep_mu,
                                          origin_image=keep_image, chd=keep_chd, no_data_val=map_list[0].no_data_val)

    else:
        raise ValueError("'map_list' maps have different grids. This is not yet supported in " +
                         "map_manip.combine_mu_maps()")

    return euv_combined


def combine_timescale_maps(timescale_weights, map_list):
    """
       Take a list of combined maps of varying timescales and do weighted minimum intensity merge to a single map.
        :param timescale_weights: weighting list for timescales
        :param map_list: List of Psi_Map objects (single_euv_map, combined_euv_map)
        :return: Psi_Map object resulting from merge.
        """

    # determine number of maps. if only one, do nothing
    nmaps = len(map_list)

    if nmaps == 1:
        return map_list[0]

    # check that all maps have the same x and y grids
    same_grid = all(map_list[0].x == map_list[1].x) and all(map_list[0].y == map_list[1].y)
    if nmaps > 2:
        for ii in range(1, nmaps - 1):
            same_temp = all(map_list[ii].x == map_list[ii + 1].x) and all(map_list[ii].y == map_list[ii + 1].y)
            same_grid = same_grid & same_temp
            if not same_grid:
                break

    else:
        # check that all maps have the same x and y grids
        same_grid = all(map_list[0].x == map_list[1].x) and all(map_list[0].y == map_list[1].y)

    if same_grid:
        # construct arrays of mu's and data
        mat_size = map_list[0].mu.shape
        mu_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_MU)
        data_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_DATA)
        image_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_ORIGIN_IMAGE)
        if map_list[0].chd is not None:
            chd_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_DATA)
            for ii in range(nmaps):
                mu_array[:, :, ii] = map_list[ii].mu
                data_array[:, :, ii] = map_list[ii].data
                chd_array[:, :, ii] = map_list[ii].chd
                image_array[:, :, ii] = map_list[ii].origin_image
        else:
            for ii in range(nmaps):
                mu_array[:, :, ii] = map_list[ii].mu
                data_array[:, :, ii] = map_list[ii].data
                image_array[:, :, ii] = map_list[ii].origin_image

        # average EUV data based on timescale weights
        # TODO: currently assumes that all data is "good" - need to figure out how to implement "good index"
        col_index, row_index = np.meshgrid(range(mat_size[1]), range(mat_size[0]))
        keep_data = np.ndarray(shape=mat_size, dtype=DTypes.MAP_DATA)
        keep_chd = np.ndarray(shape=mat_size, dtype=DTypes.MAP_DATA)
        keep_mu = np.ndarray(shape=mat_size, dtype=DTypes.MAP_DATA)
        sum_wgt = 0
        for wgt_ind, weight in enumerate(timescale_weights):
            sum_wgt += weight
            keep_data = (keep_data + data_array[row_index, col_index, wgt_ind] * weight) / sum_wgt
            keep_mu = (keep_mu + mu_array[row_index, col_index, wgt_ind] * weight) / sum_wgt
            if map_list[0].chd is not None:
                keep_chd = (keep_chd + chd_array[row_index, col_index, wgt_ind] * weight) / sum_wgt
            else:
                keep_chd = None

        # generate EUV map
        euv_time_combined = psi_d_types.PsiMap(keep_data, map_list[0].x, map_list[0].y, mu=keep_mu,
                                               origin_image=None, chd=keep_chd, no_data_val=map_list[0].no_data_val)

    else:
        raise ValueError("'map_list' maps have different grids. This is not yet supported in " +
                         "map_manip.combine_maps()")

    return euv_time_combined


def combine_timewgt_maps(weight, sum_wgt, map_list, mu_cutoff=0.0):
    nmaps = len(map_list)
    if nmaps == 1:
        sum_wgt += weight
        return map_list[0], sum_wgt

    else:
        # check that all maps have the same x and y grids
        same_grid = all(map_list[0].x == map_list[1].x) and all(map_list[0].y == map_list[1].y)
        if nmaps > 2:
            for ii in range(1, nmaps - 1):
                same_temp = all(map_list[ii].x == map_list[ii + 1].x) and all(map_list[ii].y == map_list[ii + 1].y)
                same_grid = same_grid & same_temp
                if not same_grid:
                    break

        if same_grid:
            if mu_cutoff > 0.0:
                # for all pixels with mu < mu_cutoff, set intensity to no_data_val
                for ii in range(nmaps):
                    map_list[ii].data[map_list[ii].mu < mu_cutoff] = map_list[ii].no_data_val

            # construct arrays of mu's and data
            mat_size = map_list[0].mu.shape
            data_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_DATA)
            use_data = np.ndarray(shape=mat_size, dtype=DTypes.MAP_DATA)
            for ii in range(nmaps):
                data_array[:, :, ii] = map_list[ii].data

            # find overlap indices
            overlap = np.logical_and(data_array[:, :, 0] != map_list[0].no_data_val,
                                     data_array[:, :, 1] != map_list[0].no_data_val)
            # weight euv data by gaussian distribution
            use_data[overlap] = (sum_wgt * data_array[:, :, 1][overlap] + data_array[:, :, 0][
                overlap] * weight) / (sum_wgt + weight)
            # determine viable non-overlap pixels
            use_new = np.logical_and(map_list[0].data != map_list[0].no_data_val, np.logical_not(overlap))
            use_org = np.logical_and(map_list[1].data != map_list[0].no_data_val, np.logical_not(overlap))
            # insert other non-overlap data
            use_data[use_new] = (data_array[:, :, 0][use_new] * weight) / (sum_wgt + weight)
            use_data[use_org] = (data_array[:, :, 1][use_org] * sum_wgt) / (sum_wgt + weight)

            # Generate new CHD map
            if map_list[0].chd is not None:
                chd_data = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_DATA)
                chd_array = np.ndarray(shape=mat_size, dtype=DTypes.MAP_DATA)
                for ii in range(nmaps):
                    chd_data[:, :, ii] = map_list[ii].chd
                # weight chd data by gaussian distribution
                chd_array[overlap] = (sum_wgt * chd_data[:, :, 1][overlap] + chd_data[:, :, 0][
                    overlap] * weight) / (sum_wgt + weight)
                # insert other non-overlap data
                chd_array[use_new] = (chd_data[:, :, 0][use_new] * weight) / (sum_wgt + weight)
                chd_array[use_org] = (chd_data[:, :, 1][use_org] * sum_wgt) / (sum_wgt + weight)

            else:
                chd_array = None
            # Generate new EUV map
            euv_combined = psi_d_types.PsiMap(use_data, map_list[0].x, map_list[0].y, mu=None,
                                              origin_image=None, chd=chd_array, no_data_val=map_list[0].no_data_val)
            # add weight to the sum
            sum_wgt += weight

    return euv_combined, sum_wgt


def downsamp_mean_unifgrid(map, new_y_size, new_x_size, chd_flag=True):

    # calculate block size
    x_block_size = np.ceil(map.x.len()/new_x_size)
    y_block_size = np.ceil(map.y.len()/new_y_size)
    block_size = (x_block_size, y_block_size)
    # replace no-data-vals with NaNs

    # use the mean to downsample
    new_data = block_reduce(map.data, block_size=block_size, func=np.nanmean,
                            cval=np.nan)
    # convert NaNs back to no-data-vals

    # calculate new x and y centers

    # generate a copy of the map and update values

    if chd_flag:
        # also downsample the coronal hole detection grid
        # replace no-data-vals with NaNs

        # use the mean to downsample
        new_data = block_reduce(map.data, block_size=block_size, func=np.nanmean,
                                cval=np.nan)
        # convert NaNs back to no-data-vals

        # update map

    else:
        # assign downsampled map.chd to Null
        pass

    return new_map


def downsamp_reg_grid_orig(map, new_y, new_x, image_method=0, chd_method=0, periodic_x=False):
    # Input a PSI-map object and re-sample to the new x, y coords
    # Assume grids are regular, but non-uniform

    # check that new coord range does not exceed old coord range
    x_in_range = (map.x.min() <= new_x.min()) & (map.x.max() >= new_x.max())
    y_in_range = (map.y.min() <= new_y.min()) & (map.y.max() >= new_y.max())
    if not x_in_range or not y_in_range:
        raise ValueError("The new grid defined by 'new_x' and 'new_y' exceeds the"
                         "range of the existing grid defined by map.x and map.y.")

    # generate MapMesh object for both grids
    new_mesh = MapMesh(new_x, np.arcsin(new_y))
    old_mesh = MapMesh(map.x, np.arcsin(map.y))
    # generate bin edges for each grid
    new_y_interior_edges = (new_y[0:-1]+new_y[1:])/2
    # new_y_edges = np.concatenate([[new_y[0] - (new_y[1]-new_y[0])/2], new_y_interior_edges,
    #                              [new_y[-1] + (new_y[-1]-new_y[-2])/2]])
    new_y_edges = np.concatenate([[new_y[0]], new_y_interior_edges, [new_y[-1]]])
    new_x_interior_edges = (new_x[0:-1] + new_x[1:])/2
    # new_x_edges = np.concatenate([[new_x[0] - (new_x[1]-new_x[0])/2], new_x_interior_edges,
    #                              [new_x[-1] + (new_x[-1]-new_x[-2])/2]])
    new_x_edges = np.concatenate([[new_x[0]], new_x_interior_edges, [new_x[-1]]])
    old_y_interior_edges = (map.y[0:-1] + map.y[1:])/2
    # old_y_edges = np.concatenate([[map.y[0] - (map.y[1]-map.y[0])/2], old_y_interior_edges,
    #                               [map.y[-1] + (map.y[-1]-map.y[-2])/2]])
    old_y_edges = np.concatenate([[map.y[0]], old_y_interior_edges, [map.y[-1]]])
    old_x_interior_edges = (map.x[0:-1] + map.x[1:])/2
    # old_x_edges = np.concatenate([[map.x[0] - (map.x[1]-map.x[0])/2], old_x_interior_edges,
    #                               [map.x[-1] + (map.x[-1]-map.x[-2])/2]])
    old_x_edges = np.concatenate([[map.x[0]], old_x_interior_edges, [map.x[-1]]])

    # determine overlap weights for each row and column of new grid
    #   include area-weighting in row associations
    new_y_n = len(new_y)
    old_y_n = len(map.y)
    start_time = time.time()
    row_weight_mat = np.ndarray((new_y_n, old_y_n), dtype=float)
    row_weight_info = {'index': np.zeros(1), 'weights': np.zeros(1)}
    row_weight_list = [row_weight_info, ] * new_y_n
    for new_y_index in range(new_y_n):
        # determine linear row-weighting of original pixels to new pixels
        new_hist = np.ones(1)
        temp_edges = new_y_edges[new_y_index:(new_y_index+2)]
        old_hist = lbcc_funs.hist_integration(new_hist, temp_edges, old_y_edges)
        bin_indices = np.where(old_hist > 0.)
        bin_weights = old_hist[bin_indices]
        # also weight by pixel area on surface of sphere
        area_weights = old_mesh.da[0, bin_indices]
        bin_weights = bin_weights*area_weights
        # normalize
        bin_weights = bin_weights/bin_weights.sum()
        # store indices and weights for each row
        row_weight_mat[new_y_index, bin_indices] = bin_weights
        row_weight_list[new_y_index] = {'index': bin_indices, 'weights': bin_weights}

    end_time = time.time()

    # repeat for columns
    new_x_n = len(new_x)
    old_x_n = len(map.x)
    start_time = time.time()
    col_weight_info = {'index': np.zeros(1), 'weights': np.zeros(1)}
    col_weight_list = [col_weight_info, ]*new_x_n
    column_weight_mat = np.ndarray((old_x_n, new_x_n), dtype=float)
    for new_x_index in range(new_x_n):
        # determine linear row-weighting of original pixels to new pixels
        # new_hist = np.zeros(new_x_n)
        # new_hist[new_x_index] = 1
        # old_hist = lbcc_funs.hist_integration(new_hist, new_x_edges, old_x_edges)
        new_hist = np.ones(1)
        temp_edges = new_x_edges[new_x_index:(new_x_index + 2)]
        old_hist = lbcc_funs.hist_integration(new_hist, temp_edges, old_x_edges)
        bin_indices = np.where(old_hist > 0.)
        bin_weights = old_hist[bin_indices]
        # normalize
        bin_weights = bin_weights/bin_weights.sum()
        # store indices and weights for each column
        column_weight_mat[bin_indices, new_x_index] = bin_weights
        # col_weight_list[new_x_index]['index'] = bin_indices
        # col_weight_list[new_x_index]['weights'] = bin_weights
        col_weight_list[new_x_index] = {'index': bin_indices, 'weights': bin_weights}
    end_time = time.time()

    # prepare (de-linked) data for weighted-averaging
    full_data = map.data.copy()
    full_data[full_data == map.no_data_val] = np.nan
    # Method 1: apply the row and column reduction by matrix multiplication
    #   - does not work with NaNs or no_data_vals
    row_reduced_data = np.matmul(row_weight_mat, map.data)
    reduced_data = np.matmul(row_reduced_data, column_weight_mat)

    # Method 2: loop through new pixels to do np.nanmean on each
    start_time = time.time()
    new_data = np.zeros((new_y_n, new_x_n))
    for new_y_index in range(new_y_n):
        y_indices = row_weight_list[new_y_index]['index']
        y_weights = row_weight_list[new_y_index]['weights']
        for new_x_index in range(new_x_n):
            x_indices = col_weight_list[new_x_index]['index']
            x_weights = col_weight_list[new_x_index]['weights']
            outer_weights = np.outer(y_weights, x_weights)
            temp_data = full_data[y_indices[0]:(y_indices[-1]+1),
                                  x_indices[0]:(x_indices[-1]+1)]
            nan_index = np.isnan(temp_data)
            if np.any(nan_index):
                outer_weights[nan_index] = 0.
                sum_weights = outer_weights.sum()
                if sum_weights < 0.5:
                    # consider this point to be 'outside' existing data boundary
                    new_data[new_y_index, new_x_index] = np.nan
                else:
                    # re-normalize weights matrix
                    outer_weights = outer_weights/outer_weights.sum()
                    # complete weighted average by dot product and sum
                    new_data[new_y_index, new_x_index] = \
                        np.multiply(outer_weights, temp_data).sum()
            else:
                # complete weighted average by element-wise product and sum
                new_data[new_y_index, new_x_index] = \
                    np.multiply(outer_weights, temp_data).sum()

    end_time = time.time()
    print(end_time-start_time, " seconds elapsed.\n")
    # quick plot testing (remove at clean-up)
    import matplotlib.pyplot as plt
    plt.figure(0)
    plt.imshow(full_data, origin='lower')
    plt.title("Original Map")
    plt.figure(1)
    plt.imshow(new_data, origin='lower')
    plt.title("Reduced Map")

    # reset NaNs to no_data_val
    new_data[np.isnan(new_data)] = map.no_data_val

    # generate new map object and fill
    new_map = psi_d_types.PsiMap(data=reduced_data, x=new_x, y=new_y, mu=None,
                                 origin_image=None, map_lon=None, chd=None,
                                 no_data_val=map.no_data_val)


def downsamp_reg_grid(full_map, new_y, new_x, image_method=0, chd_method=0, periodic_x=True,
                      y_units='sinlat', uniform_poles=True, single_origin_image=None,
                      uniform_no_data=True):
    """
    Input a PSI-map object (full_map) and re-sample to the new x, y coords.
    Assumes grids are regular, but non-uniform.

    :param full_map: PsiMap object
                The full-map (covers full sphere) to be downsampled
    :param new_y: Array like
                  Vector of pixel centers. Units can be specified with y_units.
                  Default is sin(lat).
    :param new_x: Array like
                  Vector of pixel centers. Currently only support longitude in
                  radians (phi).
    :param image_method: integer
                         0 - Average across overlapped pixels to downsample
                         1 - Use random sampling to downsample (not yet supported)
    :param chd_method: integer
                       0 - Average across overlapped pixels to downsample
    :param periodic_x: True/False
                       Should the left and right edge be treated as periodic
    :param y_units: character string
                    Latitude units: 'sinlat', 'lat_rad', 'lat_deg', 'theta'
    :param uniform_poles: True/False
                          Enforce uniformity at the north and south pole (across
                          the top and bottom edge of the map)
    :param single_origin_image: integer (default: None)
                                If the map contains information from only one image,
                                this value will be used to fill 'origin_image' in
                                the output map. If None, 'origin_image' will be set
                                to None.
    :param uniform_no_data: True/False
                            The algorithm is significantly sped-up if 'no_data' pixel
                            locations are consistent across all map grids (data, chd,
                            mu).  When 'no_data' pixels vary from grid to grid, set
                            to False.
    :return: PsiMap object
             The new map object with grid defined by new_x and new_y.
    """
    #

    # check that new coord range does not exceed old coord range
    x_in_range = (full_map.x.min() <= new_x.min()) & (full_map.x.max() >= new_x.max())
    y_in_range = (full_map.y.min() <= new_y.min()) & (full_map.y.max() >= new_y.max())
    if not x_in_range or not y_in_range:
        raise ValueError("The new grid defined by 'new_x' and 'new_y' exceeds the"
                         "range of the existing grid defined by full_map.x and full_map.y.")

    start_time = time.time()
    # generate MapMesh object for both grids
    # new_theta = -np.arcsin(new_y) + np.pi/2
    # new_mesh = MapMesh(new_x, new_theta)
    if y_units == "sinlat":
        old_theta = -np.arcsin(full_map.y) + np.pi/2
        sin_lat = full_map.y
        new_sin_lat = new_y
    elif y_units == "lat_rad":
        old_theta = -full_map.y + np.pi/2
        sin_lat = np.sin(full_map.y)
        new_sin_lat = np.sin(new_y)
    elif y_units == "lat_deg":
        old_theta = -(np.pi/180)*full_map.y + np.pi/2
        sin_lat = np.sin((np.pi/180)*full_map.y)
        new_sin_lat = np.sin((np.pi/180)*new_y)
    else:
        old_theta = full_map.y
        sin_lat = np.sin(-full_map.y + np.pi/2)
        new_sin_lat = np.sin(-new_y + np.pi/2)
    old_mesh = MapMesh(full_map.x, old_theta)
    # the theta grid is reversed from what MapMesh expects so change sign on da
    da = np.transpose(-old_mesh.da.copy())
    # new_da = np.transpose(-new_mesh.da.copy())
    # generate bin edges for each grid
    new_y_interior_edges = (new_sin_lat[0:-1]+new_sin_lat[1:])/2
    new_y_edges = np.concatenate([[new_sin_lat[0]], new_y_interior_edges, [new_sin_lat[-1]]])
    new_x_interior_edges = (new_x[0:-1] + new_x[1:])/2
    new_x_edges = np.concatenate([[new_x[0]], new_x_interior_edges, [new_x[-1]]])
    old_y_interior_edges = (sin_lat[0:-1] + sin_lat[1:])/2
    old_y_edges = np.concatenate([[sin_lat[0]], old_y_interior_edges, [sin_lat[-1]]])
    old_x_interior_edges = (full_map.x[0:-1] + full_map.x[1:])/2
    old_x_edges = np.concatenate([[full_map.x[0]], old_x_interior_edges, [full_map.x[-1]]])

    # determine overlap weights for each row and column of new grid
    #   include area-weighting in row associations
    new_y_n = len(new_sin_lat)
    old_y_n = len(sin_lat)
    old_y_widths = np.diff(old_y_edges)
    row_weight_mat = np.zeros((new_y_n, old_y_n), dtype=float)
    row_da_weight  = np.zeros((new_y_n, old_y_n), dtype=float)
    for new_y_index in range(new_y_n):
        # determine linear row-weighting of original pixels to new pixels
        # new_hist = np.ones(1)
        temp_edges = new_y_edges[new_y_index:(new_y_index+2)]
        # old_hist = lbcc_funs.hist_integration(new_hist, temp_edges, old_y_edges)
        pixel_portions = pixel_portion_overlap1D(temp_edges, old_y_edges)
        bin_indices = np.where(pixel_portions > 0.)
        bin_weights = pixel_portions[bin_indices]
        row_da_weight[new_y_index, bin_indices] = bin_weights
        # also weight by pixel width(height).
        # area_weights = da[bin_indices, 1]
        area_weights = old_y_widths[bin_indices]
        bin_weights = bin_weights*area_weights
        # normalize
        bin_weights = bin_weights/bin_weights.sum()
        # store indices and weights for each row
        row_weight_mat[new_y_index, bin_indices] = bin_weights

    # repeat for columns
    new_x_n = len(new_x)
    old_x_n = len(full_map.x)
    old_x_widths = np.diff(old_x_edges)
    column_weight_mat = np.zeros((old_x_n, new_x_n), dtype=float)
    col_da_weight = np.zeros((old_x_n, new_x_n), dtype=float)
    for new_x_index in range(new_x_n):
        # determine linear row-weighting of original pixels to new pixels
        # new_hist = np.ones(1)
        temp_edges = new_x_edges[new_x_index:(new_x_index + 2)]
        # old_hist = lbcc_funs.hist_integration(new_hist, temp_edges, old_x_edges)
        pixel_portions = pixel_portion_overlap1D(temp_edges, old_x_edges)
        bin_indices = np.where(pixel_portions > 0.)
        bin_weights = pixel_portions[bin_indices]
        col_da_weight[bin_indices, new_x_index] = bin_weights
        # multiply by pixel widths
        bin_weights = bin_weights * old_x_widths[bin_indices]
        # normalize
        bin_weights = bin_weights/bin_weights.sum()
        # store indices and weights for each column
        column_weight_mat[bin_indices, new_x_index] = bin_weights

    # prepare (de-linked) data for weighted-averaging
    full_data = full_map.data.copy()
    no_data_index = full_data == full_map.no_data_val
    full_data[no_data_index] = 0.
    # apply the row and column reduction by matrix multiplication
    row_reduced_data = np.matmul(row_weight_mat, full_data)
    reduced_data = np.matmul(row_reduced_data, column_weight_mat)
    # also calculate da in the new grid
    reduced_grid_da = np.matmul(np.matmul(row_da_weight, da), col_da_weight)
    no_data_da = da.copy()
    no_data_da[no_data_index] = 0.
    reduced_no_data_da = np.matmul(np.matmul(row_da_weight, no_data_da),
                                   col_da_weight)
    # use the area ratio to improve intensity estimate at data boundaries (and
    # better estimate the boundary)
    da_ratio = reduced_no_data_da/reduced_grid_da
    new_no_data_index = da_ratio < 0.5
    reduced_data[new_no_data_index] = full_map.no_data_val
    reduced_data[~new_no_data_index] = reduced_data[~new_no_data_index]/ \
        da_ratio[~new_no_data_index]

    if single_origin_image is not None:
        origin_image = np.zeros(reduced_data.shape)
        origin_image[~new_no_data_index] = single_origin_image
    else:
        origin_image = None

    # now apply reduction to chd map
    if full_map.chd is not None:
        chd_data = full_map.chd.copy()
        if not uniform_no_data:
            # recalculate the no_data locations and area ratio
            no_chd_index = chd_data == full_map.no_data_val
            reduced_grid_da = np.matmul(np.matmul(row_da_weight, da), col_da_weight)
            no_data_da = da.copy()
            no_data_da[no_chd_index] = 0.
            reduced_no_data_da = np.matmul(np.matmul(row_da_weight, no_data_da),
                                           col_da_weight)
            # use the area ratio to improve intensity estimate at data boundaries (and
            # better estimate the boundary)
            da_ratio = reduced_no_data_da/reduced_grid_da
            new_no_chd_index = da_ratio < 0.5
        else:
            no_chd_index = no_data_index
            new_no_chd_index = new_no_data_index

        chd_data[no_chd_index] = 0.
        # apply the row and column reduction by matrix multiplication
        row_reduced_chd = np.matmul(row_weight_mat, chd_data)
        reduced_chd = np.matmul(row_reduced_chd, column_weight_mat)
        # use the area ratio to improve chd estimate at data boundaries (and
        # better estimate the boundary)
        reduced_chd[new_no_chd_index] = full_map.no_data_val
        reduced_chd[~new_no_chd_index] = reduced_chd[~new_no_chd_index] / \
            da_ratio[~new_no_chd_index]
    else:
        reduced_chd = None

    # now apply reduction to mu values
    if full_map.mu is not None:
        mu_data = full_map.mu.copy()
        # if not uniform_no_data:
        #     # recalculate the no_data locations and area ratio
        #     no_data_index = chd_data == full_map.no_data_val
        #     reduced_grid_da = np.matmul(np.matmul(row_da_weight, da), col_da_weight)
        #     no_data_da = da.copy()
        #     no_data_da[no_data_index] = 0.
        #     reduced_no_data_da = np.matmul(np.matmul(row_da_weight, no_data_da),
        #                                    col_da_weight)
        #     # use the area ratio to improve intensity estimate at data boundaries (and
        #     # better estimate the boundary)
        #     da_ratio = reduced_no_data_da/reduced_grid_da
        #     new_no_data_index = da_ratio < 0.5
        #
        # mu_data[no_data_index] = 0.
        # apply the row and column reduction by matrix multiplication
        row_reduced_mu = np.matmul(row_weight_mat, mu_data)
        reduced_mu = np.matmul(row_reduced_mu, column_weight_mat)
        # use the area ratio to improve mu estimate at data boundaries (and
        # better estimate the boundary)
        # reduced_mu[new_no_data_index] = full_map.no_data_val
        # reduced_mu[~new_no_data_index] = reduced_mu[~new_no_data_index]/ \
        #     da_ratio[~new_no_data_index]
    else:
        reduced_mu = None

    # now apply reduction to map_lon values (?)
    if full_map.map_lon is not None:
        map_lon = None
    else:
        map_lon = None

    if uniform_poles:
        no_data_vec = reduced_data[0, ] == full_map.no_data_val
        if np.any(~no_data_vec):
            reduced_data[0, ] = np.mean(reduced_data[0, ~no_data_vec])
        no_data_vec = reduced_data[-1, ] == full_map.no_data_val
        if np.any(~no_data_vec):
            reduced_data[-1, ] = np.mean(reduced_data[-1, ~no_data_vec])
        if chd_data is not None:
            no_data_vec = chd_data[0, ] == full_map.no_data_val
            if np.any(~no_data_vec):
                chd_data[0, ] = np.mean(chd_data[0, ~no_data_vec])
            no_data_vec = chd_data[-1, ] == full_map.no_data_val
            if np.any(~no_data_vec):
                chd_data[-1, ] = np.mean(chd_data[-1, ~no_data_vec])
        if reduced_mu is not None:
            no_data_vec = reduced_mu[0, ] == full_map.no_data_val
            if np.any(~no_data_vec):
                reduced_mu[0, ] = np.mean(reduced_mu[0, ~no_data_vec])
            no_data_vec = reduced_mu[-1, ] == full_map.no_data_val
            if np.any(~no_data_vec):
                reduced_mu[-1, ] = np.mean(reduced_mu[-1, ~no_data_vec])

    if periodic_x:
        new_x_widths = np.diff(new_x_edges)
        # average half-pixels from left and right edge
        reduced_data[:, 0], reduced_data[:, -1] = periodic_x_avg(
            reduced_data[:, 0], reduced_data[:, -1], new_x_widths, full_map.no_data_val)

        if reduced_mu is not None:
            reduced_mu[:, 0], reduced_mu[:, -1] = periodic_x_avg(
                reduced_mu[:, 0], reduced_mu[:, -1], new_x_widths, full_map.no_data_val)

        if chd_data is not None:
            chd_data[:, 0], chd_data[:, -1] = periodic_x_avg(
                chd_data[:, 0], chd_data[:, -1], new_x_widths, full_map.no_data_val)

    end_time = time.time()
    # print(end_time - start_time, " seconds elapsed.\n")

    # quick plot testing (remove at clean-up)
    # import matplotlib.pyplot as plt
    # plt.figure(0)
    # plt.imshow(full_map.data, origin='lower')
    # plt.title("Original Map")
    # plt.figure(1)
    # plt.imshow(reduced_data, origin='lower')
    # plt.title("Reduced Map")

    # alter method 'GridSize_sinLat' to new resolution
    method_info = full_map.method_info.copy()
    y_index = method_info.meth_name.eq('GridSize_sinLat') & \
        method_info.var_name.eq('n_SinLat')
    method_info.loc[y_index, 'var_val'] = new_y_n
    x_index = method_info.meth_name.eq('GridSize_sinLat') & \
        method_info.var_name.eq('n_phi')
    method_info.loc[x_index, 'var_val'] = new_x_n

    # generate new map object and fill
    new_map = psi_d_types.PsiMap(data=reduced_data, x=new_x, y=new_y, mu=reduced_mu,
                                 origin_image=origin_image, map_lon=map_lon, chd=reduced_chd,
                                 no_data_val=full_map.no_data_val)
    new_map.append_method_info(method_info)
    new_map.append_map_info(full_map.map_info)
    new_map.append_data_info(full_map.data_info)

    return new_map


def pixel_portion_overlap1D(edges, new_edges):
    # function to calc new pixel overlap with a single pixel
    """

    :param edges: list-like with two entries (sorted increasing)
           Original pixel edges
    :param new_edges: list-like (sorted increasing)
           New pixel edges
    :return: np.ndarray vector with length=len(new_edges)-1
             The portion of each new pixel that overlaps the original pixel.
    """
    # initiate results vector
    n_bins = len(new_edges) - 1
    out_vec = np.zeros(n_bins)

    left_edges = new_edges[0:-1]
    right_edges = new_edges[1:]
    left_in = left_edges < edges[1]
    left_out = left_edges < edges[0]
    right_in = right_edges > edges[0]
    right_out = right_edges > edges[1]

    # for extremely large arrays (approx n_bins>2E5), this is probably faster
    # temp_index = np.searchsorted(left_edges, edges[1], side='right')
    # left_in = np.zeros(n_bins, dtype=bool)
    # left_in[:temp_index] = True

    consume_bin = left_out & right_out
    # check for single new pixel that surrounds old pixel
    if any(consume_bin):
        # calculate portion of surrounding bin that overlaps old pixel
        out_vec[consume_bin] = (edges[1] - edges[0])/(right_edges[consume_bin] -
                                                      left_edges[consume_bin])
        return out_vec

    # check left overlap for partial overlap
    left_overlap = left_out & right_in
    out_vec[left_overlap] = (right_edges[left_overlap] - edges[0]) / \
                            (right_edges[left_overlap] - left_edges[left_overlap])

    # check for partial right overlap
    right_overlap = right_out & left_in
    out_vec[right_overlap] = (edges[1] - left_edges[right_overlap]) / \
                             (right_edges[right_overlap] - left_edges[right_overlap])

    # remaining overlap pixels fall inside original pixel
    full_overlap = ~left_out & ~right_out
    out_vec[full_overlap] = 1.

    return out_vec


def periodic_x_avg(l_vec, r_vec, x_edges, no_data_val):

    # average half-pixels from left and right edge
    left_no_data = l_vec == no_data_val
    right_no_data = r_vec == no_data_val

    both_index = ~left_no_data & ~right_no_data
    average_vec = (x_edges[0]*l_vec[both_index] +
                   x_edges[-1]*r_vec[both_index])/ \
                  (x_edges[0] + x_edges[-1])
    l_vec[both_index] = average_vec
    r_vec[both_index] = average_vec

    right_index = left_no_data & ~right_no_data
    l_vec[right_index] = r_vec[right_index]
    left_index = ~left_no_data & right_no_data
    r_vec[left_index] = l_vec[left_index]

    return l_vec, r_vec


def chdmap_br_flux(br_map, chd_map):
    """
    Calculate the magnetic flux in the coronal holes of a map.

    Overlay a coronal hole map with a radial magnetic field map and sum the flux
    associated with coronal holes (except where there is no data).
    :param chd_map: PsiMap
                     A map object with chd_map.chd being an array that defines
                     one or more coronal hole regions. Values should be between
                     0.0 and 1.0 except where 'no_data' is designated by a value
                     of chd_map.no_data_val.
    :param br_map: PsiMap
                   A map object with br_map.data being an array that contains
                   radial magnetic field magnitudes and with shape/mesh matching
                   the chd_data array.
    :return: float
             Scalar sum of flux in the coronal hole regions of chd_map.chd.
    """
    # verify that maps have same grids
    same_x = (len(chd_map.x) == len(br_map.x)) and all(chd_map.x == br_map.x)
    same_y = (len(chd_map.y) == len(br_map.y)) and all(chd_map.y == br_map.y)
    if ~(same_x and same_y):
        sys.exit("Input maps for modules/map_manip.chdmap_br_flux() do not have identical axes.")

    # calculate the flux
    chd_flux = chdarray_br_flux(br_map, chd_map, chd_map.no_data_val)

    return chd_flux


def chdarray_br_flux(br_map, chd_data, chd_no_data_val):
    """
    Calculate the magnetic flux in the coronal holes of a map.

    Overlay a coronal hole map with a radial magnetic field map and sum the flux
    associated with coronal holes (except where there is no data).
    :param chd_data: numpy.ndarray
                     An array that defines one or more coronal hole regions.
                     Values should be between 0.0 and 1.0 except where
                     'no_data' is designated by a value of chd_no_data_val.
    :param br_map: PsiMap
                   A map object with br_map.data being an array that contains
                   radial magnetic field magnitudes and with shape/mesh matching
                   the chd_data array.
    :param chd_no_data_val: float
                            The value in chd_data that designates 'no_data'
    :return: float
             Scalar sum of flux in the coronal hole regions of chd_map.chd.
    """
    # calc theta from sin(lat). increase float precision to reduce numeric
    # error from np.arcsin()
    theta_y = np.pi/2 - np.arcsin(np.flip(br_map.y.astype('float64')))
    # generate a mesh
    map_mesh = MapMesh(br_map.x, theta_y)
    # convert area characteristic of mesh back to map grid
    map_da = np.flip(map_mesh.da.transpose(), axis=0)
    # do not evaluate at 'no_data' points
    no_data_index = chd_data == chd_no_data_val.astype(DTypes.MAP_CHD)
    chd_data[no_data_index] == 0.
    # sum total flux
    chd_flux = np.sum(chd_data * br_map.data * map_da)

    return chd_flux
