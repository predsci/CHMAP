"""
Functions to manipulate and combine maps
"""

import numpy as np

import modules.datatypes as psi_d_types
from settings.info import DTypes


def combine_maps(map_list, mu_cutoff=0.0, del_mu=None):
    """
    Take a list of Psi_Map objects and do minimum intensity merge to a single map.  When mu_cutoff
    is 0.0 and del_mu is None, this reverts to a simple minimum intensity merge.
    :param map_list: List of Psi_Map objects
    :param mu_cutoff: data points/pixels with a mu less than mu_cutoff will be discarded before
    merging.
    :param del_mu: For a given data point/pixel of the map first find the maximum mu from map_list.
    Pixels with mu < max_mu-mu_cutoff are discarded before merge.
    :return: Psi_Map object resulting from merge.
    """
    # determine number of maps. if only one, do nothing
    nmaps = len(map_list)
    if nmaps == 1:
        # need to also record merge parameters
        # map_out.mthod_par = {'mu_cutoff': mu_cutoff, 'del_mu': del_mu}
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

        # first construct arrays of mu's and data
        mat_size = map_list[0].mu.shape
        mu_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_MU)
        data_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_DATA)
        image_array = np.ndarray(shape=mat_size + (nmaps,), dtype=DTypes.MAP_ORIGIN_IMAGE)
        for ii in range(nmaps):
            mu_array[:, :, ii] = map_list[ii].mu
            data_array[:, :, ii] = map_list[ii].data
            image_array[:, :, ii] = map_list[ii].origin_image

        keep_mu = np.ndarray(shape=mat_size)
        keep_data = np.ndarray(shape=mat_size)
        keep_imag = np.ndarray(shape=mat_size, dtype=int)
        float_info = np.finfo(map_list[0].data.dtype)

        if del_mu is not None:
            max_mu = mu_array.max(axis=2)
            good_index = np.ndarray(shape=mat_size + (nmaps,), dtype=bool)
            for ii in range(nmaps):
                good_index[:, :, ii] = mu_array[:, :, ii] > (max_mu - del_mu)
            # make poor mu pixels unuseable to merge
            data_array[np.logical_not(good_index)] = float_info.max
            # make no_data_vals unuseable to merge
            data_array[data_array == map_list[0].no_data_val] = float_info.max
            # find minimum intensity of remaining pixels
            map_index = np.argmin(data_array, axis=2)

            # for ii in range(mat_size[0]):
            #     for jj in range(mat_size[1]):
            #         good_index = mu_array[ii, jj, :] > (max_mu[ii, jj] - del_mu)
            #         arg_min    = data_array[ii, jj, good_index].argmin()
            #         map_index  = good_index.nonzero()[0][arg_min]
            #         keep_mu[ii, jj] = mu_array[ii, jj, map_index]
            #         keep_data[ii, jj] = data_array[ii, jj, map_index]
            #         keep_imag[ii, jj] = image_array[ii, jj, map_index]
        else:
            # make no_data_vals unuseable to merge
            data_array[data_array == map_list[0].no_data_val] = float_info.max
            # find minimum intensity of remaining pixels
            map_index = np.argmin(data_array, axis=2)

        # return bad-mu and no-data pixels to no_data_val
        data_array[data_array == float_info.max] = map_list[0].no_data_val

        col_index, row_index = np.meshgrid(range(mat_size[1]), range(mat_size[0]))
        keep_mu = mu_array[row_index, col_index, map_index]
        keep_data = data_array[row_index, col_index, map_index]
        keep_imag = image_array[row_index, col_index, map_index]

        # Generate new map object
        map_out = psi_d_types.PsiMap(keep_data, map_list[0].x, map_list[0].y, mu=keep_mu,
                                     origin_image=keep_imag, no_data_val=map_list[0].no_data_val)
        # need to also record merge parameters
        # map_out.mthod_par = {'mu_cutoff': mu_cutoff, 'del_mu': del_mu}

    else:
        raise ValueError("'map_list' maps have different grids. This is not yet supported in " +
                         "map_manip.combine_maps()")

    return map_out
