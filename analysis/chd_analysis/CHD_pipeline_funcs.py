
"""
functions to create EUV/CHD maps and save to the database
1. Select images
2. Apply pre-processing corrections
    a. Limb-Brightening
    b. Inter-Instrument Transformation
3. Coronal Hole Detection
4. Convert to Map
5. Combine Maps and Save to DB
"""

import time
import pandas as pd
import numpy as np
import datetime

import utilities.plotting.psi_plotting as Plotting
import software.ezseg.ezsegwrapper as ezsegwrapper
import database.db_funs as db_funcs
from maps.util.map_manip import combine_maps
import utilities.datatypes.datatypes as datatypes
import data.corrections.lbcc.LBCC_theoretic_funcs as lbcc_funcs
import data.corrections.iit.IIT_pipeline_funcs as iit_funcs
from data.download.euv_utils import cluster_meth_1
from settings.info import DTypes


#### STEP ONE: SELECT IMAGES ####
# this step uses database functions from modules/DB_funs
# 1.) query some images
# query_pd = db_funcs.query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)
# 2.) generate a dataframe to record methods
# methods_list = db_funcs.generate_methdf(query_pd)


#### STEP TWO: APPLY PRE-PROCESSING CORRECTIONS ####
# 1.) get dates
def get_dates(time_min, time_max, map_freq=2):
    """
    function to create moving average dates based on hourly frequency of map creation
    @param time_min: minimum datetime value for querying
    @param time_max: maximum datetime value for querying
    @param map_freq: integer value representing hourly cadence for map creation
    @return: list of center dates
    """
    map_frequency = int((time_max - time_min).total_seconds() / 3600 / map_freq)
    moving_avg_centers = np.array(
        [np.datetime64(str(time_min)) + ii * np.timedelta64(map_freq, 'h') for ii in range(map_frequency + 1)])
    return moving_avg_centers


# 2.) get instrument combos
def get_inst_combos(db_session, inst_list, time_min, time_max):
    """
    function to create instrument based lists of combo queries for image pre-processing
    @param db_session: database session to query image combos from
    @param inst_list: list of instruments
    @param time_min: minimum query time for image combos
    @param time_max: maximum query time for image combos
    @return: inst list of lbc combo queries, inst list of iit combo queries
    """
    start = time.time()
    print("Querying Combo IDs from the database. This only needs to be done once.")
    # query for combo ids within date range
    lbc_combo_query = [None] * len(inst_list)
    iit_combo_query = [None] * len(inst_list)
    for inst_index, instrument in enumerate(inst_list):
        lbc_combo = db_funcs.query_inst_combo(db_session, time_min - datetime.timedelta(days=180),
                                              time_max + datetime.timedelta(days=180),
                                              meth_name='LBCC', instrument=instrument)
        iit_combo = db_funcs.query_inst_combo(db_session, time_min - datetime.timedelta(days=180),
                                              time_max + datetime.timedelta(days=180), meth_name='IIT',
                                              instrument=instrument)
        lbc_combo_query[inst_index] = lbc_combo
        iit_combo_query[inst_index] = iit_combo
        print("Combo IDs have been queried for", instrument)
    end = time.time()
    print("Combo IDs have been queried from the database in", end - start, "seconds.")
    return lbc_combo_query, iit_combo_query


# 3.) apply IIP
def apply_ipp(db_session, center_date, query_pd, inst_list, hdf_data_dir, lbc_combo_query,
              iit_combo_query, methods_list, n_intensity_bins=200, R0=1.01):
    """
    function to apply image pre-processing (limb-brightening, inter-instrument transformation) corrections
    to EUV images for creation of maps
    @param db_session: database session from which to query correction variable values
    @param center_date: date for querying
    @param query_pd: pandas dataframe of euv_images
    @param inst_list: instrument list
    @param hdf_data_dir: directory of hdf5 files
    @param lbc_combo_query: list (of length number of instruments) of lbc image combo queries
    @param iit_combo_query: list (of length number of instruments) of iit image combo queries
    @param methods_list: methods dataframe
    @param n_intensity_bins: number of intensity bins
    @param R0: radius
    @return: image dataframe, list of los images, list of iit images, indices used for correction, methods list,
             ref alpha, ref x
    """
    start = time.time()
    # create image lists
    image_pd = [None] * len(inst_list)
    los_list = [None] * len(inst_list)
    iit_list = [None] * len(inst_list)
    use_indices = [(2048, 2048)] * len(inst_list)
    # convert date to correct format
    print("\nStarting corrections for", center_date, "images:")
    date_time = np.datetime64(center_date).astype(datetime.datetime)
    # alpha, x for threshold
    sta_ind = inst_list.index('EUVI-A')
    ref_alpha, ref_x = db_funcs.query_var_val(db_session, meth_name='IIT', date_obs=date_time,
                                              inst_combo_query=iit_combo_query[sta_ind])
    # create dataframe for date
    hist_date = query_pd['date_obs']
    date_pd = query_pd[
        (hist_date >= np.datetime64(date_time - datetime.timedelta(minutes=10))) &
        (hist_date <= np.datetime64(date_time + datetime.timedelta(minutes=10)))]
    if len(date_pd) == 0:
        print("No Images to Process for this date.")
    else:
        for inst_ind, instrument in enumerate(inst_list):
            # get image row
            image_pd[inst_ind] = date_pd[date_pd['instrument'] == instrument]
            inst_image = date_pd[date_pd['instrument'] == instrument]
            if len(inst_image) == 0:
                print("No", instrument, "image to process for this date.")
            else:
                image_row = inst_image.iloc[0]
                index = np.where(date_pd['instrument'] == instrument)[0][0]
                print("Processing image number", image_row.data_id, "for LBC and IIT Corrections.")
                # apply LBC
                los_list[inst_ind], lbcc_image, mu_indices, use_ind, theoretic_query = lbcc_funcs.apply_lbc(db_session,
                                                                                                            hdf_data_dir,
                                                                                                            lbc_combo_query[
                                                                                                                inst_ind],
                                                                                                            image_row=image_row,
                                                                                                            n_intensity_bins=n_intensity_bins,
                                                                                                            R0=R0)
                # generate a record of the method and variable values used for LBC
                #  lbc_method = {'meth_name': ("LBCC", "LBCC", "LBCC", "LBCC", "LBCC", "LBCC"), 'meth_description':
                #                 ["LBCC Theoretic Fit Method"] * 6, 'var_name': ("TheoVar0", "TheoVar1", "TheoVar2", "TheoVar3", "TheoVar4", "TheoVar5"),
                #                           'var_description': ("Theoretic fit parameter at index 0", "Theoretic fit parameter at index 1", "Theoretic fit parameter at index 2",
                #                                               "Theoretic fit parameter at index 3", "Theoretic fit parameter at index 4", "Theoretic fit parameter at index 5"),
                #                           'var_val': (theoretic_query[0], theoretic_query[1], theoretic_query[2], theoretic_query[3],
                #                                       theoretic_query[4], theoretic_query[5])}
                # apply IIT
                lbcc_image, iit_list[inst_ind], use_indices[inst_ind], alpha, x = iit_funcs.apply_iit(db_session,
                                                                                                      iit_combo_query[
                                                                                                          inst_ind],
                                                                                                      lbcc_image,
                                                                                                      use_ind,
                                                                                                      los_list[
                                                                                                          inst_ind],
                                                                                                      R0=R0)
                # iit_method = {'meth_name': ("IIT", "IIT"), 'meth_description': ["IIT Fit Method"] * 2, 'var_name': (
                # "alpha", "x"), 'var_description': ("IIT correction coefficient: alpha", "IIT correction coefficient:
                # x"), 'var_val': (alpha, x)}
                # add methods to dataframe
                ipp_method = {'meth_name': ("LBCC", "IIT"), 'meth_description': ["LBCC Theoretic Fit Method",
                                                                                 "IIT Fit Method"],
                              'var_name': ("LBCC", "IIT"), 'var_description': (" ", " ")}
                methods_list[index] = methods_list[index].append(pd.DataFrame(data=ipp_method), sort=False)
                # methods_list[inst_ind] = pd.DataFrame(data=ipp_method)
        end = time.time()
        print("Image Pre-Processing Corrections (Limb-Brightening and Inter-Instrument Transformation) have been "
              "applied "
              " in", end - start, "seconds.")

    return date_pd, los_list, iit_list, use_indices, methods_list, ref_alpha, ref_x


def apply_ipp_2(db_session, center_date, query_pd, inst_list, hdf_data_dir,
                n_intensity_bins=200, R0=1.01):
    """
    Function to apply image pre-processing (limb-brightening, inter-instrument transformation) corrections
    to EUV images for creation of maps. Three major differences from original function:
        1. Expects query_pd to contain all images that should be corrected. No temporal
           selection of images in this function.
        2. Queries the DB directly for previous/next IIT and LBC values. No need to input
           combo-query results.
        3. methods_list is generated internally and outputted to be appended to an existing
           list as appropriate.
    @param db_session: database session from which to query correction variable values
    @param center_date: date for querying
    @param query_pd: pandas dataframe of euv_images
    @param inst_list: instrument list
    @param hdf_data_dir: directory of hdf5 files
    @param n_intensity_bins: number of intensity bins
    @param R0: radius
    @return: image dataframe (identical to input 'query_pd', but retained for backward compatibility),
             list of los images,
             list of iit images,
             indices used for correction,
             methods list,
             ref alpha, ref x
    """
    start = time.time()
    # create image lists
    n_images = query_pd.shape[0]
    los_list = [None] * n_images
    iit_list = [None] * n_images
    methods_list = db_funcs.generate_methdf(query_pd)
    use_indices = [np.full((2048, 2048), True, dtype=bool)] * len(inst_list)
    # convert date to correct format
    print("\nStarting corrections for", center_date, "images:")
    date_time = np.datetime64(center_date).astype(datetime.datetime)
    # alpha, x for threshold
    euvia_iit = db_funcs.get_correction_pars(db_session, meth_name="IIT",
                                             date_obs=date_time, instrument='EUVI-A')
    ref_alpha = euvia_iit[0]
    ref_x = euvia_iit[1]

    # create dataframe for date
    date_pd = query_pd
    if len(date_pd) == 0:
        print("No Images to Process for this date.")
    else:
        for index in range(date_pd.shape[0]):
            # get image row
            image_row = date_pd.iloc[index]
            print("Processing image number", image_row.data_id, "for LBC and IIT Corrections.")
            # apply LBC
            los_list[index], lbcc_image, mu_indices, use_ind, theoretic_query = \
                lbcc_funcs.apply_lbc_2(db_session, hdf_data_dir, image_row=image_row,
                                       n_intensity_bins=n_intensity_bins, R0=R0)
            # update method with LBCC parameter values? Would need to associate each LBCC
            #   parameter set with an image # and store in DB. For now, simply record method
            #   without values. Same for IIT below.
            # apply IIT
            lbcc_image, iit_list[index], use_indices[index], alpha, x = \
                iit_funcs.apply_iit_2(db_session, lbcc_image, use_ind,
                                      los_list[index], R0=R0)
            # set unused points to no_data_val
            # if los_list[index].no_data_val is None:
            #     no_data_val = -9999.0
            #     iit_list[index].no_data_val = no_data_val
            # else:
            #     no_data_val = iit_list[index].no_data_val
            # iit_list[index].iit_data[~use_indices[index]] = no_data_val
            # JT - this should be handled in minimum intensity merge, not here

            # add methods to dataframe
            ipp_method = {'meth_name': ("LBCC", "IIT"), 'meth_description': ["LBCC Theoretic Fit Method",
                                                                             "IIT Fit Method"],
                          'var_name': ("LBCC", "IIT"), 'var_description': (" ", " ")}
            methods_list[index] = methods_list[index].append(pd.DataFrame(data=ipp_method), sort=False)
            # methods_list[inst_ind] = pd.DataFrame(data=ipp_method)
        end = time.time()
        print("Image Pre-Processing Corrections (Limb-Brightening and Inter-Instrument Transformation) have been "
              "applied "
              " in", end - start, "seconds.")

    return date_pd, los_list, iit_list, use_indices, methods_list, ref_alpha, ref_x


#### STEP THREE: CORONAL HOLE DETECTION ####
def chd(iit_list, los_list, use_indices, inst_list, thresh1, thresh2, ref_alpha, ref_x, nc, iters):
    """
    function to create CHD Images from IIT Images
    @param iit_list: list of iit images
    @param los_list: list of los images
    @param use_indices: viable indices for detection
    @param inst_list: instrument list
    @param thresh1: lower threshold - seed placement
    @param thresh2: upper threshold - stopping criteria
    @param ref_alpha: reference IIT scale factor to calculate threshold parameters
    @param ref_x: reference IIT offset to calculate threshold parameters
    @param nc: pixel connectivity parameter
    @param iters: maximum number of iterations
    @return: list of chd images
    """
    start = time.time()
    chd_image_list = [datatypes.CHDImage()]*len(inst_list)
    for inst_ind, instrument in enumerate(inst_list):
        if iit_list[inst_ind] is not None:
            # define CHD parameters
            image_data = iit_list[inst_ind].iit_data
            use_chd = use_indices[inst_ind].astype(int)
            use_chd = np.where(use_chd == 1, use_chd, -9999)
            nx = iit_list[inst_ind].x.size
            ny = iit_list[inst_ind].y.size
            # calculate new threshold parameters based off reference (AIA) instrument
            t1 = thresh1 * ref_alpha + ref_x
            t2 = thresh2 * ref_alpha + ref_x

            # we will only use 'use_chd' pixels, but to avoid warnings on np.log10(image_data)
            image_data[image_data <= 0.] = 1e-8

            # fortran CHD algorithm. use keyword-assignment for all variables because
            #             # f2py will sometimes identify inputs as optional and reorder them.
            ezseg_output, iters_used = ezsegwrapper.ezseg(img=np.log10(image_data), seg=use_chd,
                                                          nt=ny, np=nx, thresh1=t1, thresh2=t2,
                                                          nc=nc, iters=iters)
            chd_result = np.logical_and(ezseg_output == 0, use_chd == 1)
            chd_result = chd_result.astype(int)

            # create CHD image
            chd_image_list[inst_ind] = datatypes.create_chd_image(los_list[inst_ind], chd_result)
            chd_image_list[inst_ind].get_coordinates()
    end = time.time()
    print("Coronal Hole Detection algorithm implemented in", end - start, "seconds.")

    return chd_image_list


def chd_2(iit_list, los_list, use_indices, thresh1, thresh2, ref_alpha, ref_x, nc, iters):
    """
    Function to create CHD Images from IIT Images.

    New in version 2: will do a coronal hole detection for all images in iit_list, rather
    than assuming that each list entry corresponds to an instrument.
    @param iit_list: list of iit images
    @param los_list: list of los images
    @param use_indices: viable indices for detection
    @param thresh1: lower threshold - seed placement
    @param thresh2: upper threshold - stopping criteria
    @param ref_alpha: reference IIT scale factor to calculate threshold parameters
    @param ref_x: reference IIT offset to calculate threshold parameters
    @param nc: pixel connectivity parameter
    @param iters: maximum number of iterations
    @return: list of chd images
    """
    start = time.time()
    chd_image_list = [datatypes.CHDImage()]*len(iit_list)
    for iit_ind in range(iit_list.__len__()):
        if iit_list[iit_ind] is not None:
            # define CHD parameters
            image_data = iit_list[iit_ind].iit_data
            use_chd = use_indices[iit_ind].astype(int)
            use_chd = np.where(use_chd == 1, use_chd, -9999)
            nx = iit_list[iit_ind].x.size
            ny = iit_list[iit_ind].y.size
            # calculate new threshold parameters based off reference (AIA) instrument
            t1 = thresh1 * ref_alpha + ref_x
            t2 = thresh2 * ref_alpha + ref_x

            # we will only use 'use_chd' pixels, but to avoid warnings on np.log10(image_data)
            image_data[image_data <= 0.] = 1e-8

            # fortran CHD algorithm. use keyword-assignment for all variables because
            # f2py will sometimes identify inputs as optional and reorder them.
            ezseg_output, iters_used = ezsegwrapper.ezseg(
                img=np.log10(image_data), seg=use_chd, nt=ny, np=nx, thresh1=t1,
                thresh2=t2, nc=nc, iters=iters)
            chd_result = np.logical_and(ezseg_output == 0, use_chd == 1)
            chd_result = chd_result.astype(int)
            chd_result[~use_indices[iit_ind]] = round(iit_list[iit_ind].no_data_val)

            # create CHD image
            chd_image_list[iit_ind] = datatypes.create_chd_image(los_list[iit_ind], chd_result)
            chd_image_list[iit_ind].get_coordinates()
    end = time.time()
    print("Coronal Hole Detection algorithm implemented in", end - start, "seconds.")

    return chd_image_list


#### STEP FOUR: CONVERT TO MAPS ####
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
                                                                  image_num=image_row.data_id)
            # CHD map
            chd_map_list[iit_ind] = chd_image_list[iit_ind].interp_to_map(R0=R0, map_x=map_x, map_y=map_y,
                                                                            image_num=image_row.data_id)
            # record image and map info
            chd_map_list[iit_ind].append_data_info(image_row)
            map_list[iit_ind].append_data_info(image_row)
            data_info.append(image_row)
            map_info.append(map_list[iit_ind].map_info)

            # generate a record of the method and variable values used for interpolation
            interp_method = {'meth_name': ("Im2Map_Lin_Interp_1",), 'meth_description':
                ["Use SciPy.RegularGridInterpolator() to linearly interpolate from an Image to a Map"] * 1,
                             'var_name': ("R0",), 'var_description': ("Solar radii",), 'var_val': (R0,)}
            # add to the methods dataframe for this map
            methods_list[iit_ind] = methods_list[iit_ind].append(pd.DataFrame(data=interp_method), sort=False)

            # also record a method for map grid size
            grid_method = {'meth_name': ("GridSize_sinLat", "GridSize_sinLat"), 'meth_description':
                           ["Map number of grid points: phi x sin(lat)"] * 2, 'var_name': ("n_phi", "n_SinLat"),
                           'var_description': ("Number of grid points in phi", "Number of grid points in sin(lat)"),
                           'var_val': (len(map_x), len(map_y))}
            methods_list[iit_ind] = methods_list[iit_ind].append(pd.DataFrame(data=grid_method), sort=False)

            # incorporate the methods dataframe into the map object
            map_list[iit_ind].append_method_info(methods_list[iit_ind])
            chd_map_list[iit_ind].append_method_info(methods_list[iit_ind])

    end = time.time()
    print("Images interpolated to maps in", end - start, "seconds.")
    return map_list, chd_map_list, methods_list, data_info, map_info


#### STEP FIVE: CREATE COMBINED MAPS AND SAVE TO DB ####
def create_combined_maps(db_session, map_data_dir, map_list, chd_map_list, methods_list,
                         data_info, map_info, mu_merge_cutoff=None, del_mu=None, mu_cutoff=0.0):
    """
    function to create combined EUV and CHD maps and save to database with associated method information
    @param db_session: database session to save maps to
    @param map_data_dir: directory to save map files
    @param map_list: list of EUV maps
    @param chd_map_list: list of CHD maps
    @param methods_list: methods list
    @param data_info: image info list
    @param map_info: map info list
    @param mu_merge_cutoff: cutoff mu value for overlap areas
    @param del_mu: maximum mu threshold value
    @param mu_cutoff: lower mu value
    @return: combined euv map, combined chd map
    """
    # start time
    start = time.time()
    # create combined maps
    euv_maps = []
    chd_maps = []
    for euv_map in map_list:
        if len(euv_map.data) != 0:
            euv_maps.append(euv_map)
    for chd_map in chd_map_list:
        if len(chd_map.data) != 0:
            chd_maps.append(chd_map)
    if del_mu is not None:
        euv_combined, chd_combined = combine_maps(euv_maps, chd_maps, del_mu=del_mu, mu_cutoff=mu_cutoff)
        combined_method = {'meth_name': ("Min-Int-Merge-del_mu", "Min-Int-Merge-del_mu"), 'meth_description':
            ["Minimum intensity merge for synchronic map: using del mu"] * 2,
                           'var_name': ("mu_cutoff", "del_mu"), 'var_description': ("lower mu cutoff value",
                                                                                    "max acceptable mu range"),
                           'var_val': (mu_cutoff, del_mu)}
    else:
        euv_combined, chd_combined = combine_maps(euv_maps, chd_maps, mu_merge_cutoff=mu_merge_cutoff, mu_cutoff=mu_cutoff)
        combined_method = {'meth_name': ("Min-Int-Merge-mu_merge", "Min-Int-Merge-mu_merge"), 'meth_description':
            ["Minimum intensity merge for synchronic map: based on Caplan et. al."] * 2,
                           'var_name': ("mu_cutoff", "mu_merge_cutoff"), 'var_description': ("lower mu cutoff value",
                                                                                         "mu cutoff value in areas of "
                                                                                         "overlap"),
                           'var_val': (mu_cutoff, mu_merge_cutoff)}

    # generate a record of the method and variable values used for interpolation
    euv_combined.append_method_info(methods_list)
    euv_combined.append_method_info(pd.DataFrame(data=combined_method))
    euv_combined.append_data_info(data_info)
    euv_combined.append_map_info(map_info)
    chd_combined.append_method_info(methods_list)
    chd_combined.append_method_info(pd.DataFrame(data=combined_method))
    chd_combined.append_data_info(data_info)
    chd_combined.append_map_info(map_info)

    # plot maps
    Plotting.PlotMap(euv_combined, nfig="EUV Combined Map for: " + str(euv_combined.data_info.date_obs[0]),
                     title="Minimum Intensity Merge EUV Map\nDate: " + str(euv_combined.data_info.date_obs[0]),
                     map_type='EUV')
    Plotting.PlotMap(chd_combined, nfig="CHD Combined Map for: " + str(chd_combined.data_info.date_obs[0]),
                     title="Minimum Intensity Merge CHD Map\nDate: " + str(chd_combined.data_info.date_obs[0]),
                     map_type='CHD')
    #     Plotting.PlotMap(chd_combined, nfig="CHD Contour Map for: " + str(chd_combined.data_info.date_obs[0]),
    #                      title="Minimum Intensity Merge CHD Contour Map\nDate: " + str(chd_combined.data_info.date_obs[0]),
    #                      map_type='Contour')

    # save EUV and CHD maps to database
    # euv_combined.write_to_file(map_data_dir, map_type='synchronic_euv', filename=None, db_session=db_session)
    # chd_combined.write_to_file(map_data_dir, map_type='synchronic_chd', filename=None, db_session=db_session)

    # end time
    end = time.time()
    print("Combined EUV and CHD Maps created and saved to the database in", end - start, "seconds.")
    return euv_combined, chd_combined


def create_combined_maps_2(map_list, mu_merge_cutoff=None, del_mu=None, mu_cutoff=0.0,
                           EUV_CHD_sep=False, low_int_filt=1.):
    """
    Function to create combined EUV and CHD maps.

    New in version 2: - function does not try to save to the database.
                      - assumes that EUV and CHD are in the same map object
                      - do some filtering on low-intensity pixels (some images are
                        cropped or do not include the sun. We do not want these to
                        dominate the minimum intensity merge.
    @param map_list: list of EUV maps
    @param mu_merge_cutoff: float
                            cutoff mu value for overlap areas
    @param del_mu: float
                   maximum mu threshold value
    @param mu_cutoff: float
                      lower mu value
    @param low_int_filt: float
                         low intensity filter. pixels with image intensity less than
                         1.0 are generally bad data. Data < low_int_filt are recast
                         as no_data_vals prior to minimum intensity merge
    @return: PsiMap object
             Minimum Intensity Merge (MIM) euv map, MIM chd map, and merged
             map meta data.
    """
    # start time
    start = time.time()
    n_maps = len(map_list)
    # remove excessively low image data (these are almost always some form of
    # bad data and we don't want them to 'win' the minimum intensity merge)
    for ii in range(n_maps):
        map_list[ii].data[map_list[ii].data < low_int_filt] = map_list[ii].no_data_val
    if EUV_CHD_sep:
        # create chd dummy-maps (for separate minimum-intensity merge)
        chd_maps = []
        for ii in range(n_maps):
            chd_maps.append(map_list[ii].__copy__())
            chd_maps[ii].data = chd_maps[ii].chd.astype(DTypes.MAP_DATA)
            chd_maps[ii].data[chd_maps[ii].data <= chd_maps[ii].no_data_val] = \
                chd_maps[ii].no_data_val

        if del_mu is not None:
            euv_combined = combine_maps(map_list, del_mu=del_mu, mu_cutoff=mu_cutoff)
            chd_combined = combine_maps(chd_maps, del_mu=del_mu, mu_cutoff=mu_cutoff)
            combined_method = {'meth_name': ["Min-Int-Merge-del_mu"] * 3, 'meth_description':
                ["Minimum intensity merge for synchronic map: using del mu"] * 3,
                 'var_name': ("mu_cutoff", "del_mu", "mu_merge_cutoff"),
                 'var_description': ("lower mu cutoff value", "max acceptable mu range",
                                     "mu cutoff value in areas of overlap"),
                 'var_val': (mu_cutoff, del_mu, None)}
        else:
            euv_combined = combine_maps(map_list, mu_merge_cutoff=mu_merge_cutoff,
                                        mu_cutoff=mu_cutoff)
            chd_combined = combine_maps(chd_maps, mu_merge_cutoff=mu_merge_cutoff,
                                        mu_cutoff=mu_cutoff)
            combined_method = {'meth_name': ["Min-Int-Merge-mu_merge"] * 3, 'meth_description':
                ["Minimum intensity merge for synchronic map: based on Caplan et. al."] * 3,
                 'var_name': ("mu_cutoff", "del_mu", "mu_merge_cutoff"),
                 'var_description': ("lower mu cutoff value", "max acceptable mu range",
                                     "mu cutoff value in areas of overlap"),
                               'var_val': (mu_cutoff, None, mu_merge_cutoff)}

        euv_combined.chd = chd_combined.data.astype(DTypes.MAP_CHD)
    else:
        # Use indexing from EUV min-merge to select CHD merge values
        if del_mu is not None:
            euv_combined = combine_maps(map_list, del_mu=del_mu, mu_cutoff=mu_cutoff)
            combined_method = {'meth_name': ["MIDM-Comb-del_mu"]*3, 'meth_description':
                ["Minimum intensity merge for (combined EUV/CHD) synchronic map: using del mu"]*3,
                               'var_name': ("mu_cutoff", "del_mu", "mu_merge_cutoff"),
                               'var_description': ("lower mu cutoff value", "max acceptable mu range",
                                                   "mu cutoff value in areas of overlap"),
                               'var_val': (mu_cutoff, del_mu, None)}
        else:
            euv_combined = combine_maps(map_list, mu_merge_cutoff=mu_merge_cutoff,
                                        mu_cutoff=mu_cutoff)
            combined_method = {'meth_name': ["MIDM-Comb-mu_merge"]*3, 'meth_description':
                ["Minimum intensity merge for (combined EUV/CHD) synchronic map: based on Caplan et. al."]*3,
                               'var_name': ("mu_cutoff", "del_mu", "mu_merge_cutoff"),
                               'var_description': ("lower mu cutoff value", "max acceptable mu range",
                                                   "mu cutoff value in areas of overlap"),
                               'var_val': (mu_cutoff, None, mu_merge_cutoff)}

    # merge map meta data
    for ii in range(len(map_list)):
        euv_combined.append_method_info(map_list[ii].method_info)
        euv_combined.append_data_info(map_list[ii].data_info)
        # euv_combined.append_map_info(map_list[ii].map_info)
    euv_combined.append_method_info(pd.DataFrame(data=combined_method))
    # combining maps produces duplicate methods. drop duplicates
    euv_combined.method_info = euv_combined.method_info.drop_duplicates()
    # for completeness, also drop duplicate data/image info
    euv_combined.data_info = euv_combined.data_info.drop_duplicates()
    # record compute time
    euv_combined.append_map_info(pd.DataFrame({'time_of_compute':
                                               [datetime.datetime.now(), ]}))

    # end time
    end = time.time()
    print("Minimum Intensity EUV and CHD Maps created in", end - start, "seconds.")
    return euv_combined


def select_synchronic_images(center_time, del_interval, image_pd, inst_list):
    """
    Select PSI-database images for synchronic map generation, consistent with the way we select
    images for download.

    Parameters
    ----------
    center_time - numpy.datetime64
    del_interval - numpy.timedelta64
    image_pd - pandas.DataFrame
    inst_list - list

    Returns
    -------
    synch_images pandas.DataFrame

    """
    # define a method to attach to synchronic maps
    map_method_dict = {'meth_name': ("Synch_Im_Sel",), 'meth_description': [
        "Synchronic image selection", ],
                       'var_name': ("clust_meth",), 'var_description': ("Clustering method",),
                       'var_val': (1,)}
    map_method = pd.DataFrame(data=map_method_dict)

    jd0 = pd.DatetimeIndex([center_time, ]).to_julian_date().item()
    # choose which images to use at this datetime
    interval_max = center_time + del_interval
    interval_min = center_time - del_interval
    f_list = []
    image_list = []
    for instrument in inst_list:
        # find instrument images in interval
        inst_images_index = image_pd.date_obs.between(interval_min, interval_max) & \
                            image_pd.instrument.eq(instrument)
        inst_images = image_pd[inst_images_index]
        if inst_images.__len__() > 0:
            f_list_pd = pd.DataFrame({'date_obs': inst_images.date_obs,
                                      'jd': pd.DatetimeIndex(inst_images.date_obs).to_julian_date(),
                                      'instrument': inst_images.instrument})
            f_list.append(f_list_pd)
            image_list.append(inst_images)

    if f_list.__len__() == 0:
        print("No instrument images in time range around ", center_time, ".\n")
        # return None
        return None, map_method

    # Now loop over all the image pairs to select the "perfect" group of images.
    cluster_index = cluster_meth_1(f_list=f_list, jd0=jd0)
    # combine selected image-rows into a dataframe
    synch_images = image_list[0].iloc[cluster_index[0]]
    if cluster_index.__len__() > 1:
        for ii in range(1, cluster_index.__len__()):
            synch_images = synch_images.append(image_list[ii].iloc[cluster_index[ii]])

    return synch_images, map_method
