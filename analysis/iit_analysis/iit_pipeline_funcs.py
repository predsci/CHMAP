"""
Functions to Generate Inter-Instrument Correction
"""

import os
import numpy as np
import time

import modules.DB_funs as db_funcs
import modules.datatypes as psi_d_types
import modules.lbcc_funs as lbcc


##### PRE STEP: APPLY LBC TO IMAGES ######
def apply_lbc_correction(db_session, hdf_data_dir, inst_list, lbc_query_time_min, lbc_query_time_max, n_mu_bins=18,
                         R0=1.01):
    """
    function to apply limb-brightening correction to use for IIT
    @param db_session: connected database session to query theoretic fit parameters from
    @param hdf_data_dir: directory of processed images to plot original images
    @param inst_list: list of instruments
    @param lbc_query_time_min: minimum query time for applying lbc fit
    @param lbc_query_time_max: maximum query time for applying lbc fit
    @param n_mu_bins: number of mu bins
    @param R0: radius
    @return:
    """
    # start time
    start_time_tot = time.time()

    meth_name = "LBCC Theoretic"
    mu_bin_edges = np.array(range(n_mu_bins + 1), dtype="float") * 0.05 + 0.1
    mu_bin_centers = (mu_bin_edges[1:] + mu_bin_edges[:-1]) / 2

    ##### QUERY IMAGES ######
    for inst_index, instrument in enumerate(inst_list):

        query_instrument = [instrument, ]
        image_pd = db_funcs.query_euv_images(db_session=db_session, time_min=lbc_query_time_min,
                                             time_max=lbc_query_time_max, instrument=query_instrument)
        # TODO: these 2048 numbers may need to be adjusted
        # TODO: should be original_los.mu however that is in the loop - deal with this later
        corrected_lbcc_data = np.ndarray(shape=(len(inst_list), len(image_pd), 2048, 2048))
        corrected_los_data = np.zeros((len(image_pd), 2048, 2048))

        ###### GET LOS IMAGES COORDINATES (DATA) #####
        for index, row in image_pd.iterrows():
            print("Processing image number", row.image_id, ".")
            if row.fname_hdf == "":
                print("Warning: Image # " + str(row.image_id) + " does not have an associated hdf file. Skipping")
                continue
            hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
            original_los = psi_d_types.read_los_image(hdf_path)
            original_los.get_coordinates(R0=R0)
            theoretic_query = db_funcs.query_var_val(db_session, meth_name, date_obs=original_los.info['date_string'],
                                                     instrument=instrument)


            # get beta and y from theoretic fit
            beta, y = lbcc.get_beta_y_theoretic_interp(theoretic_query, mu_array_2d=original_los.mu,
                                                       mu_array_1d=mu_bin_centers)

            ###### APPLY LBC CORRECTION ######
            corrected_data = beta * original_los.data + y
            corrected_los_data[index, :, :] = corrected_data
        # corrected_lbcc_data[inst_index, :, :, :] = corrected_los_data[:, :, :]
    # end time
    end_time_tot = time.time()
    print("LBC has been applied.")
    print("Total elapsed time to apply correction: " + str(round(end_time_tot - start_time_tot, 3))
          + " seconds.")

    return corrected_los_data
