"""
overarching pipeline to correct images
input: dataframe of images
output: array of corrected images
This function takes a lot of memory and is therefore slow:(
"""

import time
import datetime
import database.db_funs as db_funcs
import analysis.lbcc_analysis.LBCC_theoretic_funcs as lbcc_funcs
import analysis.iit_analysis.IIT_pipeline_funcs as iit_funcs


####### ------ FULL CORRECTION FUNCTION  ------ #########
def correct_euv_images(db_session, query_time_min, query_time_max, image_pd, inst_list, hdf_data_dir, n_intensity_bins,
                       R0):
    """
    function to take in dataframe of LOS images and return corrected images list
    """
    # start time
    start_time = time.time()
    corrected_images = []
    ##### QUERY CORRECTION COMBOS ######
    for inst_index, instrument in enumerate(inst_list):
        print("Starting corrections for", instrument, "images.")
        # query correct image combos
        time_min = query_time_min - datetime.timedelta(days=7)
        time_max = query_time_max + datetime.timedelta(days=7)
        lbc_combo_query = db_funcs.query_inst_combo(db_session, time_min, time_max,
                                                    meth_name='LBCC Theoretic', instrument=instrument)
        iit_combo_query = db_funcs.query_inst_combo(db_session, time_min, time_max, meth_name='IIT',
                                                    instrument=instrument)

        # create dataframe for instrument
        hist_inst = image_pd['instrument']
        instrument_pd = image_pd[hist_inst == instrument]

        for index, row in instrument_pd.iterrows():
            # apply LBC
            original_los, lbcc_image, mu_indices, use_indices, theoretic_query = lbcc_funcs.apply_lbc(db_session, hdf_data_dir,
                                                                                     lbc_combo_query, image_row=row,
                                                                                     n_intensity_bins=n_intensity_bins,
                                                                                     R0=R0)
            # apply IIT
            lbcc_image, iit_image, use_indices, alpha, x = iit_funcs.apply_iit(db_session, hdf_data_dir, iit_combo_query,
                                                                     lbcc_image, use_indices, image_row=row, R0=R0)
            corrected_images.append(iit_image)

    end_time = time.time()
    print("Total time to query and correct images:", end_time - start_time, "seconds.")

    return corrected_images
