"""
generate LBC for images using theoretic fit
"""

import os
import datetime
import numpy as np
import time
import scipy.optimize as optim

import modules.DB_funs as db_funcs
import modules.datatypes as psi_d_types
import modules.lbcc_funs as lbcc
import modules.plotting as Plotting


####### STEP ONE: CREATE AND SAVE HISTOGRAMS #######
def save_histograms(db_session, hdf_data_dir, inst_list, query_time_min, query_time_max, n_mu_bins=18,
                    n_intensity_bins=200,lat_band=[-np.pi/64., np.pi/64.], log10=True, R0=1.01):

    # creates mu bin & intensity bin arrays
    mu_bin_edges = np.array(range(n_mu_bins + 1), dtype="float") * 0.05 + 0.1
    image_intensity_bin_edges = np.linspace(0, 5, num=n_intensity_bins + 1, dtype='float')

    # loop over instrument
    for instrument in inst_list:

        # query EUV images
        query_instrument = [instrument, ]
        query_pd = db_funcs.query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max,
                                             instrument=query_instrument)

        for index, row in query_pd.iterrows():
            print("Processing image number", row.image_id, ".")
            if row.fname_hdf == "":
                print("Warning: Image # " + str(row.image_id) + " does not have an associated hdf file. Skipping")
                continue
            hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
            los_temp = psi_d_types.read_los_image(hdf_path)
            # add coordinates to los object
            los_temp.get_coordinates(R0=R0)
            # perform 2D histogram on mu and image intensity
            temp_hist = los_temp.mu_hist(image_intensity_bin_edges, mu_bin_edges, lat_band=lat_band, log10=log10)
            hist_lbcc = psi_d_types.create_hist(hdf_path, row.image_id, mu_bin_edges, image_intensity_bin_edges,
                                                lat_band, temp_hist)

            # add this histogram and meta data to database
            db_funcs.add_lbcc_hist(hist_lbcc, db_session)

    db_session.close()
    return None


###### STEP TWO: CALCULATE AND SAVE THEORETIC FIT PARAMETERS #######
def calc_theoretic_fit(db_session, inst_list, query_time_min, number_of_weeks, number_of_days, n_mu_bins=18,
                       n_intensity_bins=200, lat_band=[-np.pi / 64., np.pi / 64.], create=False):

    # returns array of moving averages center dates, based off start date and number of weeks
    moving_avg_centers = np.array(
        [np.datetime64(str(query_time_min)) + ii * np.timedelta64(1, 'W') for ii in range(number_of_weeks)])

    # returns moving width based of number of days
    moving_width = np.timedelta64(number_of_days, 'D')

    optim_vals_theo = ["a1", "a2", "b1", "b2", "n", "log_alpha", "SSE", "optim_time", "optim_status"]
    results_theo = np.zeros((len(moving_avg_centers), len(inst_list), len(optim_vals_theo)))

    for date_index, center_date in enumerate(moving_avg_centers):
        print("Begin date " + str(center_date))

        # determine time range based off moving average centers
        start_time_tot = time.time()
        min_date = center_date - moving_width / 2
        max_date = center_date + moving_width / 2

        for inst_index, instrument in enumerate(inst_list):
            print("\nStarting calculations for " + instrument + "\n")

            # query the histograms for time range based off moving average centers
            query_instrument = [instrument, ]
            pd_hist = db_funcs.query_hist(db_session=db_session, n_mu_bins=n_mu_bins, n_intensity_bins=n_intensity_bins,
                                          lat_band=np.array(lat_band).tobytes(),
                                          time_min=np.datetime64(min_date).astype(datetime.datetime),
                                          time_max=np.datetime64(max_date).astype(datetime.datetime),
                                          instrument=query_instrument)

            # convert the binary types back to arrays
            lat_band, mu_bin_array, intensity_bin_array, full_hist = psi_d_types.binary_to_hist(pd_hist, n_mu_bins,
                                                                                                n_intensity_bins)

            # create list of observed dates in time frame
            date_obs_npDT64 = pd_hist['date_obs']

            # creates array of mu bin centers
            mu_bin_centers = (mu_bin_array[1:] + mu_bin_array[:-1]) / 2

            # determine appropriate date range
            date_ind = (date_obs_npDT64 >= min_date) & (date_obs_npDT64 <= max_date)

            # sum the appropriate histograms
            summed_hist = full_hist[:, :, date_ind].sum(axis=2)

            # normalize in mu
            norm_hist = np.full(summed_hist.shape, 0.)
            row_sums = summed_hist.sum(axis=1, keepdims=True)
            # but do not divide by zero
            zero_row_index = np.where(row_sums != 0)
            norm_hist[zero_row_index[0]] = summed_hist[zero_row_index[0]] / row_sums[zero_row_index[0]]

            # separate the reference bin from the fitted bins
            hist_ref = norm_hist[-1,]
            hist_mat = norm_hist[:-1, ]
            mu_vec = mu_bin_centers[:-1]

            ##### OPTIMIZATION METHOD: THEORETICAL ######

            model = 3
            init_pars = np.array([-0.05, -0.3, -.01, 0.4, -1., 6.])
            method = "BFGS"

            start3 = time.time()
            optim_out3 = optim.minimize(lbcc.get_functional_sse, init_pars,
                                        args=(hist_ref, hist_mat, mu_vec, intensity_bin_array, model),
                                        method=method)

            end3 = time.time()

            results_theo[date_index, inst_index, 0:6] = optim_out3.x
            results_theo[date_index, inst_index, 6] = optim_out3.fun
            results_theo[date_index, inst_index, 7] = round(end3 - start3, 3)
            results_theo[date_index, inst_index, 8] = optim_out3.status

            meth_name = 'LBCC Theoretic'
            meth_desc = 'LBCC Theoretic Fit Method'
            var_name = "TheoVar"
            var_desc = "Theoretic fit parameter at index "
            db_funcs.store_lbcc_values(db_session, pd_hist, meth_name, meth_desc, var_name, var_desc, date_index,
                                       inst_index, optim_vals=optim_vals_theo[0:6], results=results_theo, create=create)

            end_time_tot = time.time()
            print("Total elapsed time: " + str(round(end_time_tot - start_time_tot, 3)) + " seconds.")

    return None


###### STEP THREE: APPLY CORRECTION AND PLOT IMAGES #######
def apply_lbc_correction(db_session, hdf_data_dir, inst_list, query_time_min, query_time_max, n_mu_bins=18, R0=1.01):

    meth_name = "LBCC Theoretic"
    mu_bin_edges = np.array(range(n_mu_bins + 1), dtype="float") * 0.05 + 0.1
    mu_bin_centers = (mu_bin_edges[1:] + mu_bin_edges[:-1]) / 2

    ##### QUERY IMAGES ######
    for inst_index, instrument in enumerate(inst_list):

        query_instrument = [instrument, ]
        image_pd = db_funcs.query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max,
                                             instrument=query_instrument)

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
            corrected_los_data = beta * original_los.data + y

            ###### PLOT IMAGES #####
            Plotting.PlotImage(original_los, nfig=100 + inst_index, title="Original LOS Image for " + instrument)
            Plotting.PlotLBCCImage(lbcc_data=corrected_los_data, los_image=original_los, nfig=200 + inst_index,
                                   title="Corrected LBCC Image for " + instrument)
            Plotting.PlotLBCCImage(lbcc_data=original_los.data - corrected_los_data, los_image=original_los,
                                   nfig=300 + inst_index, title="Difference Plot for " + instrument)

    return None
