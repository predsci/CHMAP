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

import modules.Plotting as Plotting
import ezseg.ezsegwrapper as ezsegwrapper
import modules.DB_funs as db_funcs
from modules.map_manip import combine_maps
import modules.datatypes as datatypes
import analysis.lbcc_analysis.LBCC_theoretic_funcs as lbcc_funcs
import analysis.iit_analysis.IIT_pipeline_funcs as iit_funcs


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
                print("Processing image number", image_row.image_id, "for LBC and IIT Corrections.")
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
    chd_image_list = [datatypes.CHDImage()] * len(inst_list)
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

            # fortran CHD algorithm
            ezseg_output, iters_used = ezsegwrapper.ezseg(np.log10(image_data), use_chd, nx, ny, t1, t2, nc, iters)
            chd_result = np.logical_and(ezseg_output == 0, use_chd == 1)
            chd_result = chd_result.astype(int)

            # create CHD image
            chd_image_list[inst_ind] = datatypes.create_chd_image(los_list[inst_ind], chd_result)
            chd_image_list[inst_ind].get_coordinates()
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
    image_info = []
    map_info = []
    map_list = [datatypes.PsiMap()] * len(inst_list)
    chd_map_list = [datatypes.PsiMap()] * len(inst_list)

    for inst_ind, instrument in enumerate(inst_list):
        if iit_list[inst_ind] is not None:
            # query correct image combos
            index = np.where(date_pd['instrument'] == instrument)[0][0]
            inst_image = date_pd[date_pd['instrument'] == instrument]
            image_row = inst_image.iloc[0]
            # EUV map
            map_list[inst_ind] = iit_list[inst_ind].interp_to_map(R0=R0, map_x=map_x, map_y=map_y,
                                                                  image_num=image_row.image_id)
            # CHD map
            chd_map_list[inst_ind] = chd_image_list[inst_ind].interp_to_map(R0=R0, map_x=map_x, map_y=map_y,
                                                                            image_num=image_row.image_id)
            # record image and map info
            chd_map_list[inst_ind].append_image_info(image_row)
            map_list[inst_ind].append_image_info(image_row)
            image_info.append(image_row)
            map_info.append(map_list[inst_ind].map_info)

            # generate a record of the method and variable values used for interpolation
            interp_method = {'meth_name': ("Im2Map_Lin_Interp_1",), 'meth_description':
                ["Use SciPy.RegularGridInterpolator() to linearly interpolate from an Image to a Map"] * 1,
                             'var_name': ("R0",), 'var_description': ("Solar radii",), 'var_val': (R0,)}
            # add to the methods dataframe for this map
            methods_list[index] = methods_list[index].append(pd.DataFrame(data=interp_method), sort=False)

            # incorporate the methods dataframe into the map object
            map_list[inst_ind].append_method_info(methods_list[index])
            chd_map_list[inst_ind].append_method_info(methods_list[index])

    end = time.time()
    print("Images interpolated to maps in", end - start, "seconds.")
    return map_list, chd_map_list, methods_list, image_info, map_info


#### STEP FIVE: CREATE COMBINED MAPS AND SAVE TO DB ####
def create_combined_maps(db_session, map_data_dir, map_list, chd_map_list, methods_list,
                         image_info, map_info, mu_merge_cutoff=None, del_mu=None, mu_cutoff=0.0):
    """
    function to create combined EUV and CHD maps and save to database with associated method information
    @param db_session: database session to save maps to
    @param map_data_dir: directory to save map files
    @param map_list: list of EUV maps
    @param chd_map_list: list of CHD maps
    @param methods_list: methods list
    @param image_info: image info list
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
    euv_combined.append_image_info(image_info)
    euv_combined.append_map_info(map_info)
    chd_combined.append_method_info(methods_list)
    chd_combined.append_method_info(pd.DataFrame(data=combined_method))
    chd_combined.append_image_info(image_info)
    chd_combined.append_map_info(map_info)

    # plot maps
    Plotting.PlotMap(euv_combined, nfig="EUV Combined Map for: " + str(euv_combined.image_info.date_obs[0]),
                     title="Minimum Intensity Merge EUV Map\nDate: " + str(euv_combined.image_info.date_obs[0]))
    Plotting.PlotMap(chd_combined, nfig="CHD Combined Map for: " + str(chd_combined.image_info.date_obs[0]),
                     title="Minimum Intensity Merge CHD Map\nDate: " + str(chd_combined.image_info.date_obs[0]),
                     map_type='CHD')
    Plotting.PlotMap(chd_combined, nfig="CHD Contour Map for: " + str(chd_combined.image_info.date_obs[0]),
                     title="Minimum Intensity Merge CHD Contour Map\nDate: " + str(chd_combined.image_info.date_obs[0]),
                     map_type='Contour')

    # save EUV and CHD maps to database
    # euv_combined.write_to_file(map_data_dir, map_type='synchronic_euv', filename=None, db_session=db_session)
    # chd_combined.write_to_file(map_data_dir, map_type='synchronic_chd', filename=None, db_session=db_session)

    # end time
    end = time.time()
    print("Combined EUV and CHD Maps created and saved to the database in", end - start, "seconds.")
    return euv_combined, chd_combined
