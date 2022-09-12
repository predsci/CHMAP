import time

import numpy as np
import pandas as pd

from chmap.utilities.datatypes import datatypes as datatypes


def create_singles_maps(inst_list, date_pd, iit_list, chd_image_list, methods_list, map_x=None, map_y=None, R0=1.01):
    """
    function to map single instrument images to a Carrington map
    @param inst_list: instrument list
    @param date_pd: dataframe of EUV Images for specific date time
    @param iit_list: list of IIT Images for mapping
    @param chd_image_list: list of CHD Images for mapping
    @param methods_list: methods dataframe list
    @param map_x: 1D array of x coordinates
    @param map_y: 1D array of y coordinates
    @param R0: radius
    @return: list of euv maps, list of chd maps, methods list, image info, and map info
    """
    start = time.time()
    data_info = []
    map_info = []
    map_list = [datatypes.PsiMap()]*len(inst_list)
    chd_map_list = [datatypes.PsiMap()]*len(inst_list)

    for inst_ind, instrument in enumerate(inst_list):
        if iit_list[inst_ind] is not None:
            # query correct image combos
            index = np.where(date_pd['instrument'] == instrument)[0][0]
            inst_image = date_pd[date_pd['instrument'] == instrument]
            image_row = inst_image.iloc[0]
            # EUV map
            map_list[inst_ind] = iit_list[inst_ind].interp_to_map(R0=R0, map_x=map_x, map_y=map_y,
                                                                  image_num=image_row.data_id)
            # CHD map
            chd_map_list[inst_ind] = chd_image_list[inst_ind].interp_to_map(R0=R0, map_x=map_x, map_y=map_y,
                                                                            image_num=image_row.data_id)
            # record image and map info
            chd_map_list[inst_ind].append_data_info(image_row)
            map_list[inst_ind].append_data_info(image_row)
            data_info.append(image_row)
            map_info.append(map_list[inst_ind].map_info)

            # generate a record of the method and variable values used for interpolation
            interp_method = {'meth_name': ("Im2Map_Lin_Interp_1",), 'meth_description':
                ["Use SciPy.RegularGridInterpolator() to linearly interpolate from an Image to a Map"] * 1,
                             'var_name': ("R0",), 'var_description': ("Solar radii",), 'var_val': (R0,)}
            # add to the methods dataframe for this map
            methods_list[index] = methods_list[index].append(pd.DataFrame(data=interp_method), sort=False)

            # also record a method for map grid size
            grid_method = {'meth_name': ("GridSize_sinLat", "GridSize_sinLat"), 'meth_description':
                           ["Map number of grid points: phi x sin(lat)"] * 2, 'var_name': ("n_phi", "n_SinLat"),
                           'var_description': ("Number of grid points in phi", "Number of grid points in sin(lat)"),
                           'var_val': (len(map_x), len(map_y))}
            methods_list[index] = methods_list[index].append(pd.DataFrame(data=grid_method), sort=False)

            # incorporate the methods dataframe into the map object
            map_list[inst_ind].append_method_info(methods_list[index])
            chd_map_list[inst_ind].append_method_info(methods_list[index])

    end = time.time()
    print("Images interpolated to maps in", end - start, "seconds.")
    return map_list, chd_map_list, methods_list, data_info, map_info


def create_singles_maps_2(date_pd, iit_list, chd_image_list, methods_list, map_x=None, map_y=None, R0=1.01):
    """
    Function to map single images to a Carrington map.

    New in version 2: will do a coronal hole detection for all images in iit_list, rather
    than assuming that each list entry corresponds to an instrument.
    @param date_pd: dataframe of EUV Images for specific date time
    @param iit_list: list of IIT Images for mapping
    @param chd_image_list: list of CHD Images for mapping
    @param methods_list: methods dataframe list
    @param map_x: 1D array of x coordinates
    @param map_y: 1D array of y coordinates
    @param R0: radius
    @return: list of euv maps, list of chd maps, methods list, image info, and map info
    """
    start = time.time()
    data_info = []
    map_info = []
    map_list = [datatypes.PsiMap()]*len(iit_list)
    chd_map_list = [datatypes.PsiMap()]*len(iit_list)

    for iit_ind in range(iit_list.__len__()):
        if iit_list[iit_ind] is not None:
            # query correct image combos
            image_row = date_pd.iloc[iit_ind]
            # EUV map
            map_list[iit_ind] = iit_list[iit_ind].interp_to_map(R0=R0, map_x=map_x, map_y=map_y,
                                                                no_data_val=iit_list[iit_ind].no_data_val,
                                                                interp_field="iit_data", image_num=image_row.data_id,
                                                                helio_proj=True)
            # CHD map
            chd_map_list[iit_ind] = chd_image_list[iit_ind].interp_to_map(R0=R0, map_x=map_x, map_y=map_y,
                                                                          image_num=image_row.data_id)
            # record image and map info
            image_row_pd = date_pd.iloc[[iit_ind]]
            chd_map_list[iit_ind].append_data_info(image_row_pd)
            map_list[iit_ind].append_data_info(image_row_pd)
            data_info.append(image_row)
            # data_info = pd.concat([data_info, image_row_pd], sort=False, ignore_index=True)
            map_info.append(map_list[iit_ind].map_info)
            # map_info = pd.concat([map_info, map_list[iit_ind].map_info], sort=False, ignore_index=True)

            # generate a record of the method and variable values used for interpolation
            interp_method = {'meth_name': ("Im2Map_Lin_Interp_1",), 'meth_description':
                ["Use SciPy.RegularGridInterpolator() to linearly interpolate from an Image to a Map"] * 1,
                             'var_name': ("R0",), 'var_description': ("Solar radii",), 'var_val': (R0,)}
            # add to the methods dataframe for this map
            # methods_list[iit_ind] = methods_list[iit_ind].append(pd.DataFrame(data=interp_method), sort=False)
            methods_list[iit_ind] = pd.concat([methods_list[iit_ind], pd.DataFrame(data=interp_method)],
                                              sort=False, ignore_index=True)

            # also record a method for map grid size
            grid_method = {'meth_name': ("GridSize_sinLat", "GridSize_sinLat"), 'meth_description':
                           ["Map number of grid points: phi x sin(lat)"] * 2, 'var_name': ("n_phi", "n_SinLat"),
                           'var_description': ("Number of grid points in phi", "Number of grid points in sin(lat)"),
                           'var_val': (len(map_x), len(map_y))}
            # methods_list[iit_ind] = methods_list[iit_ind].append(pd.DataFrame(data=grid_method), sort=False)
            methods_list[iit_ind] = pd.concat([methods_list[iit_ind], pd.DataFrame(data=grid_method)],
                                              sort=False, ignore_index=True)

            # incorporate the methods dataframe into the map object
            map_list[iit_ind].append_method_info(methods_list[iit_ind])
            chd_map_list[iit_ind].append_method_info(methods_list[iit_ind])

    end = time.time()
    print("Images interpolated to maps in", end - start, "seconds.")
    return map_list, chd_map_list, methods_list, data_info, map_info