"""
overarching pipeline to correct images
input: dataframe of images
output: array of corrected images
"""

import time
import modules.DB_funs as db_funcs
import analysis.lbcc_analysis.LBCC_theoretic_funcs as lbcc_funcs
import analysis.iit_analysis.IIT_pipeline_funcs as iit_funcs


####### ------ CORRECTION FUNCTION BELOW ------ #########
def correct_euv_images(db_session, query_time_min, query_time_max, image_pd, inst_list, hdf_data_dir, n_intensity_bins, R0):
    """
    function to take in dataframe of LOS images and return corrected images list
    @param db_session:
    @param query_time_min:
    @param query_time_max:
    @param inst_list:
    @param hdf_data_dir:
    @param n_intensity_bins:
    @param R0:
    @return:
    """
    # start time
    start_time = time.time()
    corrected_images = []
    ##### QUERY IMAGES ######
    for inst_index, instrument in enumerate(inst_list):
        # query correct image combos
        lbc_combo_query = db_funcs.query_inst_combo(db_session, query_time_min, query_time_max,
                                                    meth_name='LBCC Theoretic', instrument=instrument)
        iit_combo_query = db_funcs.query_inst_combo(db_session, query_time_min, query_time_max, meth_name='IIT',
                                                    instrument=instrument)

        # create dataframe for instrument
        hist_inst = image_pd['instrument']
        instrument_pd = image_pd[hist_inst == instrument]

        for index, row in instrument_pd.iterrows():
            # apply LBC
            original_los, lbcc_image, mu_indices, use_indices = lbcc_funcs.apply_lbc(db_session, hdf_data_dir,
                                                                                     lbc_combo_query, image_row=row,
                                                                                     n_intensity_bins=n_intensity_bins,
                                                                                     R0=R0)
            # apply IIT
            lbcc_image, iit_image, use_indices = iit_funcs.apply_iit(db_session, hdf_data_dir, iit_combo_query,
                                                                     lbcc_image, use_indices, image_row=row, R0=R0)
            corrected_images.append(iit_image)

    end_time = time.time()
    print("Total time to query and correct images:", end_time - start_time, "seconds.")

    return corrected_images
