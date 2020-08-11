"""
Functions to manipulate and combine maps
"""

import numpy as np
import modules.datatypes as psi_d_types
from settings.info import DTypes


def combine_maps(map_list, chd_map_list=None, mu_cutoff=0.0, mu_cut_over=None, del_mu=None):
    """
    Take a list of Psi_Map objects and do minimum intensity merge to a single map.
    Using mu_cut_over: based off the two cutoff algorithm from Caplan et. al. 2016.
    Using del_mu: based off maximum mu value from list
    :param map_list: List of Psi_Map objects
    :param chd_map_list: List of Psi_Map objects of CHD data
    :param mu_cutoff: data points/pixels with a mu less than mu_cutoff will be discarded before
    merging.
    :param mu_cut_over: mu cutoff value for discarding pixels in areas of instrument overlap
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
        if mu_cut_over is not None:
            good_index = np.ndarray(shape=mat_size + (nmaps,), dtype=bool)
            overlap = np.ndarray(shape=mat_size + (nmaps,), dtype=bool)
            for ii in range(nmaps):
                for jj in range(nmaps):
                    if ii != jj:
                        overlap[:, :, ii] = np.logical_and(data_array[:, :, ii] != map_list[0].no_data_val,
                                                           data_array[:, :, jj] != map_list[0].no_data_val)
            for ii in range(nmaps):
                good_index[:, :, ii] = np.logical_or(np.logical_and(overlap[:, :, ii],
                                                                    mu_array[:, :, ii] >= mu_cut_over), np.logical_and(
                    data_array[:, :, ii] != map_list[0].no_data_val, mu_array[:, :, ii] >= mu_cutoff))
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


def combine_cr_maps(n_images, map_list, chd_map_list=None, mu_cutoff=0.0, mu_cut_over=None, del_mu=None):
    """
        Take an already combined map, and a single image map, and do minimum intensity merge to a single map.
        Using mu_cut_over: based off the two cutoff algorithm from Caplan et. al. 2016.
        Using del_mu: based off maximum mu value from list
        :param map_list: List of Psi_Map objects (single_euv_map, combined_euv_map)
        :param chd_map_list: List of Psi_Map objects of CHD data (single_chd_map, combined_chd_map)
        :param mu_cutoff: data points/pixels with a mu less than mu_cutoff will be discarded before
        merging.
        :param mu_cut_over: mu cutoff value for discarding pixels in areas of instrument overlap
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
        if mu_cut_over is not None:
            good_index = np.ndarray(shape=mat_size + (nmaps,), dtype=bool)
            overlap = np.ndarray(shape=mat_size + (nmaps,), dtype=bool)
            for ii in range(nmaps):
                for jj in range(nmaps):
                    if ii != jj:
                        overlap[:, :, ii] = np.logical_and(data_array[:, :, ii] != map_list[0].no_data_val,
                                                           data_array[:, :, jj] != map_list[0].no_data_val)
            for ii in range(nmaps):
                good_index[:, :, ii] = np.logical_or(np.logical_and(overlap[:, :, ii],
                                                                    mu_array[:, :, ii] >= mu_cut_over), np.logical_and(
                    data_array[:, :, ii] != map_list[0].no_data_val, mu_array[:, :, ii] >= mu_cutoff))
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
            chd_array[np.logical_not(good_index)] = np.max(chd_array)
            chd_array[data_array == map_list[0].no_data_val] = np.max(chd_array)
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