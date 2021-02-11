"""
Functions to manipulate and combine maps
"""

import numpy as np
from scipy.interpolate import interp1d

import modules.datatypes as psi_d_types
from settings.info import DTypes
from modules.coord_manip import s2c


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

    return xh, dx, dxh


def combine_maps(map_list, chd_map_list=None, mu_cutoff=0.0, mu_merge_cutoff=None, del_mu=None):
    """
    Take a list of Psi_Map objects and do minimum intensity merge to a single map.
    Using mu_merge_cutoff: based off the two cutoff algorithm from Caplan et. al. 2016.
    Using del_mu: based off maximum mu value from list
    :param map_list: List of Psi_Map objects
    :param chd_map_list: List of Psi_Map objects of CHD data
    :param mu_cutoff: data points/pixels with a mu less than mu_cutoff will be discarded before
    merging.
    :param mu_merge_cutoff: mu cutoff value for discarding pixels in areas of instrument overlap
    :param del_mu: For a given data point/pixel of the map first find the maximum mu from map_list.
    :return: Psi_Map object resulting from merge.
    """
    # determine number of maps. if only one, do nothing
    nmaps = len(map_list)
    if nmaps == 1:
        if chd_map_list is not None:
            return map_list[0], chd_map_list[0]
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
        if chd_map_list is not None:
            chd_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_DATA)
            for ii in range(nmaps):
                mu_array[:, :, ii] = map_list[ii].mu
                data_array[:, :, ii] = map_list[ii].data
                chd_array[:, :, ii] = chd_map_list[ii].data
                image_array[:, :, ii] = map_list[ii].origin_image
        else:
            for ii in range(nmaps):
                mu_array[:, :, ii] = map_list[ii].mu
                data_array[:, :, ii] = map_list[ii].data
                image_array[:, :, ii] = map_list[ii].origin_image

        float_info = np.finfo(map_list[0].data.dtype)
        if mu_merge_cutoff is not None:
            good_index = np.ndarray(shape=mat_size + (nmaps,), dtype=bool)
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
            # make poor mu pixels unusable to merge
            data_array[np.logical_not(good_index)] = float_info.max
        elif del_mu is not None:
            max_mu = mu_array.max(axis=2)
            good_index = np.ndarray(shape=mat_size + (nmaps,), dtype=bool)
            for ii in range(nmaps):
                good_index[:, :, ii] = mu_array[:, :, ii] > (max_mu - del_mu)
            # make poor mu pixels unusable to merge
            data_array[np.logical_not(good_index)] = float_info.max

        # make no_data_vals unusable to merge
        data_array[data_array == map_list[0].no_data_val] = float_info.max
        # find minimum intensity of remaining pixels
        map_index = np.argmin(data_array, axis=2)
        # return bad-mu and no-data pixels to no_data_val
        data_array[data_array == float_info.max] = map_list[0].no_data_val

        # correct indices to create maps
        col_index, row_index = np.meshgrid(range(mat_size[1]), range(mat_size[0]))
        keep_mu = mu_array[row_index, col_index, map_index]
        keep_data = data_array[row_index, col_index, map_index]
        keep_image = image_array[row_index, col_index, map_index]

        # Generate new CHD map
        if chd_map_list is not None:
            keep_chd = chd_array[row_index, col_index, map_index]
            chd_map = psi_d_types.PsiMap(keep_chd, map_list[0].x, map_list[0].y, mu=keep_mu,
                                         origin_image=keep_image, no_data_val=map_list[0].no_data_val)
        else:
            chd_map = None
        # Generate new EUV map
        euv_map = psi_d_types.PsiMap(keep_data, map_list[0].x, map_list[0].y, mu=keep_mu,
                                     origin_image=keep_image, no_data_val=map_list[0].no_data_val)

    else:
        raise ValueError("'map_list' maps have different grids. This is not yet supported in " +
                         "map_manip.combine_maps()")

    return euv_map, chd_map


def combine_cr_maps(n_images, map_list, chd_map_list=None, mu_cutoff=0.0, mu_merge_cutoff=0.4):
    """
        Take an already combined map, and a single image map, and do minimum intensity merge to a single map.
        Using mu_merge_cutoff: based off the two cutoff algorithm from Caplan et. al. 2016.
        Using del_mu: based off maximum mu value from list
        :param n_images: number of images in original map
        :param map_list: List of Psi_Map objects (single_euv_map, combined_euv_map)
        :param chd_map_list: List of Psi_Map objects of CHD data (single_chd_map, combined_chd_map)
        :param mu_cutoff: data points/pixels with a mu less than mu_cutoff will be discarded before
        merging.
        :param mu_merge_cutoff: mu cutoff value for discarding pixels in areas of instrument overlap
        :return: Psi_Map object resulting from merge.
        """
    # determine number of maps. if only one, do nothing
    nmaps = len(map_list)
    if nmaps == 1:
        if chd_map_list is not None:
            return map_list[0], chd_map_list[0]
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
        if chd_map_list is not None:
            chd_data = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_DATA)
            chd_array = np.ndarray(shape=mat_size, dtype=DTypes.MAP_DATA)
            for ii in range(nmaps):
                chd_data[:, :, ii] = chd_map_list[ii].data
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
            chd_combined = psi_d_types.PsiMap(keep_chd, map_list[0].x, map_list[0].y, mu=keep_mu,
                                              origin_image=keep_image, no_data_val=map_list[0].no_data_val)
        else:
            chd_combined = None
        # Generate new EUV map
        euv_combined = psi_d_types.PsiMap(keep_data, map_list[0].x, map_list[0].y, mu=keep_mu,
                                          origin_image=keep_image, no_data_val=map_list[0].no_data_val)

    else:
        raise ValueError("'map_list' maps have different grids. This is not yet supported in " +
                         "map_manip.combine_cr_maps()")

    return euv_combined, chd_combined


def combine_mu_maps(n_images, map_list, chd_map_list=None, mu_cutoff=0.0, mu_merge_cutoff=0.4):
    """
        Take an already combined map, and a single image map, and do minimum intensity merge to a single map.
        Using mu_merge_cutoff: based off the two cutoff algorithm from Caplan et. al. 2016.
        Using del_mu: based off maximum mu value from list
        :param n_images: number of images in original map
        :param map_list: List of Psi_Map objects (single_euv_map, combined_euv_map)
        :param chd_map_list: List of Psi_Map objects of CHD data (single_chd_map, combined_chd_map)
        :param mu_cutoff: data points/pixels with a mu less than mu_cutoff will be discarded before
        merging.
        :param mu_merge_cutoff: mu cutoff value for discarding pixels in areas of instrument overlap
        :return: Psi_Map object resulting from merge.
        """
    # determine number of maps. if only one, do nothing
    nmaps = len(map_list)
    if nmaps == 1:
        if chd_map_list is not None:
            mu_values = map_list[0].mu
            mu_values = np.where(mu_values > 0, mu_values, 0.01)
            # map_list[0].data = map_list[0].data * mu_values
            chd_map_list[0].data = chd_map_list[0].data * mu_values
            return map_list[0], chd_map_list[0]
        else:
            mu_values = map_list[0].mu
            mu_values = np.where(mu_values > 0, mu_values, 0.01)
            # map_list[0].data = map_list[0].data * mu_values
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
        if chd_map_list is not None:
            chd_data = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_DATA)
            chd_array = np.ndarray(shape=mat_size, dtype=DTypes.MAP_DATA)
            for ii in range(nmaps):
                chd_data[:, :, ii] = chd_map_list[ii].data
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
            chd_combined = psi_d_types.PsiMap(keep_chd, map_list[0].x, map_list[0].y, mu=keep_mu,
                                              origin_image=keep_image, no_data_val=map_list[0].no_data_val)
        else:
            chd_combined = None
        # Generate new EUV map
        euv_combined = psi_d_types.PsiMap(keep_data, map_list[0].x, map_list[0].y, mu=keep_mu,
                                          origin_image=keep_image, no_data_val=map_list[0].no_data_val)

    else:
        raise ValueError("'map_list' maps have different grids. This is not yet supported in " +
                         "map_manip.combine_mu_maps()")

    return euv_combined, chd_combined


def combine_timescale_maps(timescale_weights, map_list, chd_map_list=None):
    """
       Take a list of combined maps of varying timescales and do weighted minimum intensity merge to a single map.
        :param timescale_weights: weighting list for timescales
        :param map_list: List of Psi_Map objects (single_euv_map, combined_euv_map)
        :param chd_map_list: List of Psi_Map objects of CHD data (single_chd_map, combined_chd_map)
        :param mu_cutoff: data points/pixels with a mu less than mu_cutoff will be discarded before
         merging.
        :param mu_merge_cutoff: mu cutoff value for discarding pixels in areas of instrument overlap
        :return: Psi_Map object resulting from merge.
        """

    # determine number of maps. if only one, do nothing
    nmaps = len(map_list)

    if nmaps == 1:
        if chd_map_list is not None:
            return map_list[0], chd_map_list[0]
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

    if same_grid:
        # construct arrays of mu's and data
        mat_size = map_list[0].mu.shape
        mu_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_MU)
        data_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_DATA)
        image_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_ORIGIN_IMAGE)
        if chd_map_list is not None:
            chd_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_DATA)
            for ii in range(nmaps):
                mu_array[:, :, ii] = map_list[ii].mu
                data_array[:, :, ii] = map_list[ii].data
                chd_array[:, :, ii] = chd_map_list[ii].data
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
            if chd_map_list is not None:
                keep_chd = (keep_chd + chd_array[row_index, col_index, wgt_ind] * weight) / sum_wgt

        # Generate new CHD map
        if chd_map_list is not None:
            chd_time_combined = psi_d_types.PsiMap(keep_chd, map_list[0].x, map_list[0].y, mu=keep_mu,
                                                   origin_image=None, no_data_val=map_list[0].no_data_val)
        else:
            chd_time_combined = None
        # generate EUV map
        euv_time_combined = psi_d_types.PsiMap(keep_data, map_list[0].x, map_list[0].y, mu=keep_mu,
                                               origin_image=None, no_data_val=map_list[0].no_data_val)

    else:
        raise ValueError("'map_list' maps have different grids. This is not yet supported in " +
                         "map_manip.combine_maps()")

    return euv_time_combined, chd_time_combined


def combine_timewgt_maps(weight, sum_wgt, map_list, chd_map_list=None, mu_cutoff=0.0):
    nmaps = len(map_list)
    if nmaps == 1:
        sum_wgt += weight
        if chd_map_list is not None:
            return map_list[0], chd_map_list[0], sum_wgt
        else:
            return map_list[0], None, sum_wgt

    else:
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
            if chd_map_list is not None:
                chd_data = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_DATA)
                chd_array = np.ndarray(shape=mat_size, dtype=DTypes.MAP_DATA)
                for ii in range(nmaps):
                    chd_data[:, :, ii] = chd_map_list[ii].data
                # weight chd data by gaussian distribution
                chd_array[overlap] = (sum_wgt * chd_data[:, :, 1][overlap] + chd_data[:, :, 0][
                    overlap] * weight) / (sum_wgt + weight)
                # insert other non-overlap data
                chd_array[use_new] = (chd_data[:, :, 0][use_new] * weight) / (sum_wgt + weight)
                chd_array[use_org] = (chd_data[:, :, 1][use_org] * sum_wgt) / (sum_wgt + weight)
                # choose correct chd data to use
                chd_combined = psi_d_types.PsiMap(chd_array, map_list[0].x, map_list[0].y, mu=None,
                                                  origin_image=None, no_data_val=map_list[0].no_data_val)
            else:
                chd_combined = None
            # Generate new EUV map
            euv_combined = psi_d_types.PsiMap(use_data, map_list[0].x, map_list[0].y, mu=None,
                                              origin_image=None, no_data_val=map_list[0].no_data_val)
            # add weight to the sum
            sum_wgt += weight

    return euv_combined, chd_combined, sum_wgt
