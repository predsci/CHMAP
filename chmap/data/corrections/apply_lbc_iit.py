import datetime
import time

import numpy as np
import pandas as pd

from chmap.data.corrections.iit import IIT_pipeline_funcs as iit_funcs
from chmap.data.corrections.lbcc import LBCC_theoretic_funcs as lbcc_funcs
from database import db_funs as db_funcs


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