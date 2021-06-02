"""
Functions to Generate Inter-Instrument Correction
"""

import numpy as np
import time
import datetime

import database.db_funs as db_funcs
import utilities.datatypes.datatypes as psi_d_types
import chmap.data.corrections.lbcc.lbcc_utils as lbcc
import chmap.data.corrections.iit.iit_utils as iit
import utilities.plotting.psi_plotting as Plotting
import chmap.data.corrections.iit.IIT_pipeline_funcs as iit_funcs
import chmap.data.corrections.lbcc.LBCC_theoretic_funcs as lbcc_funcs


##### STEP ONE: CREATE 1D HISTOGRAMS AND SAVE TO DATABASE ######
def create_histograms(db_session, inst_list, lbc_query_time_min, lbc_query_time_max, hdf_data_dir,
                      n_intensity_bins=200, lat_band=[-np.pi / 2.4, np.pi / 2.4], log10=True, R0=1.01,
                      wavelengths=None):
    """
    create and save (to database) IIT-Histograms from LBC Data
    @param db_session: connected db session for querying EUV images and saving histograms
    @param inst_list: list of instruments
    @param lbc_query_time_min: minimum query time for applying lbc fit
    @param lbc_query_time_max: maximum query time for applying lbc fit
    @param hdf_data_dir: directory of processed images to plot original images
    @param n_intensity_bins: number of intensity bins
    @param lat_band: latitude band
    @param log10: boolean value
    @param R0: radius
    @return: None, saves histograms to database
    """
    # start time
    start_time = time.time()

    # create IIT method
    meth_name = "IIT"
    meth_desc = "IIT Fit Method"
    method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=True)

    for instrument in inst_list:
        # query EUV images
        query_instrument = [instrument, ]
        image_pd = db_funcs.query_euv_images(db_session=db_session, time_min=lbc_query_time_min,
                                             time_max=lbc_query_time_max, instrument=query_instrument,
                                             wavelength=wavelengths)
        # query correct image combos
        combo_query = db_funcs.query_inst_combo(db_session, lbc_query_time_min, lbc_query_time_max,
                                                meth_name="LBCC", instrument=instrument)
        # apply LBC
        for index, row in image_pd.iterrows():
            original_los, lbcc_image, mu_indices, use_indices, theoretic_query = lbcc_funcs.apply_lbc(
                db_session, hdf_data_dir, combo_query, image_row=row,
                n_intensity_bins=n_intensity_bins, R0=R0)
            # calculate IIT histogram from LBC
            hist = psi_d_types.LBCCImage.iit_hist(lbcc_image, lat_band, log10)

            # create IIT histogram datatype
            iit_hist = psi_d_types.create_iit_hist(lbcc_image, method_id[1], lat_band, hist)

            # add IIT histogram and meta data to database
            db_funcs.add_hist(db_session, iit_hist)

    db_session.close()

    end_time = time.time()
    print("Inter-instrument transformation histograms have been created and saved to the database.")
    print(
        "Total elapsed time for histogram creation: " + str(round(end_time - start_time, 3)) + " seconds.")

    return None


##### STEP TWO: CALCULATE INTER-INSTRUMENT TRANSFORMATION COEFFICIENTS AND SAVE TO DATABASE ######
def calc_iit_coefficients(db_session, inst_list, ref_inst, calc_query_time_min, calc_query_time_max, weekday=0,
                          number_of_days=180, n_intensity_bins=200, lat_band=[-np.pi / 2.4, np.pi / 2.4], create=False,
                          wavelengths=None):
    # start time
    start_time = time.time()

    # create IIT method
    meth_name = "IIT"
    meth_desc = "IIT Fit Method"
    method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=False)

    #### GET REFERENCE INFO FOR LATER USE ####
    # get index number of reference instrument
    ref_index = inst_list.index(ref_inst)
    # query euv images to get carrington rotation range
    ref_instrument = [ref_inst, ]
    euv_images = db_funcs.query_euv_images(db_session, time_min=calc_query_time_min, time_max=calc_query_time_max,
                                           instrument=ref_instrument, wavelength=wavelengths)
    # get min and max carrington rotation
    rot_max = euv_images.cr_rot.max()
    rot_min = euv_images.cr_rot.min()

    # query histograms
    ref_hist_pd = db_funcs.query_hist(db_session=db_session, meth_id=method_id[1],
                                      n_intensity_bins=n_intensity_bins, lat_band=lat_band,
                                      time_min=calc_query_time_min - datetime.timedelta(days=number_of_days),
                                      time_max=calc_query_time_max + datetime.timedelta(days=number_of_days),
                                      instrument=ref_instrument, wavelength=wavelengths)

    # convert binary to histogram data
    mu_bin_edges, intensity_bin_edges, ref_full_hist = psi_d_types.binary_to_hist(hist_binary=ref_hist_pd,
                                                                                  n_mu_bins=None,
                                                                                  n_intensity_bins=n_intensity_bins)

    for inst_index, instrument in enumerate(inst_list):
        # check if this is the reference instrument
        if inst_index == ref_index:
            # calculate the moving average centers
            moving_avg_centers, moving_width = lbcc.moving_averages(calc_query_time_min, calc_query_time_max, weekday,
                                                                    number_of_days)
            # loop through moving average centers
            for date_index, center_date in enumerate(moving_avg_centers):
                print("Starting calculations for", instrument, ":", center_date)
                # determine time range based off moving average centers
                min_date = center_date - moving_width / 2
                max_date = center_date + moving_width / 2
                # get the correct date range to use for image combos
                ref_pd_use = ref_hist_pd[(ref_hist_pd['date_obs'] >= str(min_date)) & (
                        ref_hist_pd['date_obs'] <= str(max_date))]

                # save alpha/x as [1, 0] for reference instrument
                alpha = 1
                x = 0
                db_funcs.store_iit_values(db_session, ref_pd_use, meth_name, meth_desc, [alpha, x], create)
        else:
            # query euv_images for correct carrington rotation
            query_instrument = [instrument, ]
            rot_images = db_funcs.query_euv_images_rot(db_session, rot_min=rot_min, rot_max=rot_max,
                                                       instrument=query_instrument, wavelength=wavelengths)
            # get time minimum and maximum for instrument
            inst_time_min = rot_images.date_obs.min()
            inst_time_max = rot_images.date_obs.max()
            moving_avg_centers, moving_width = lbcc.moving_averages(inst_time_min, inst_time_max, weekday,
                                                                    number_of_days)
            inst_hist_pd = db_funcs.query_hist(db_session=db_session, meth_id=method_id[1],
                                               n_intensity_bins=n_intensity_bins, lat_band=lat_band,
                                               time_min=inst_time_min - datetime.timedelta(days=number_of_days),
                                               time_max=inst_time_max + datetime.timedelta(days=number_of_days),
                                               instrument=query_instrument, wavelength=wavelengths)
            # convert binary to histogram data
            mu_bin_edges, intensity_bin_edges, inst_full_hist = psi_d_types.binary_to_hist(
                hist_binary=inst_hist_pd, n_mu_bins=None, n_intensity_bins=n_intensity_bins)
            # loops through moving average centers
            for date_index, center_date in enumerate(moving_avg_centers):
                print("Starting calculations for", instrument, ":", center_date)
                # determine time range based off moving average centers
                min_date = center_date - moving_width / 2
                max_date = center_date + moving_width / 2
                # get indices for calculation of reference histogram
                ref_hist_ind = np.where(
                    (ref_hist_pd['date_obs'] >= str(min_date)) & (ref_hist_pd['date_obs'] <= str(max_date)))
                ref_ind_min = np.min(ref_hist_ind)
                ref_ind_max = np.max(ref_hist_ind)
                ref_hist_use = ref_full_hist[:, ref_ind_min:ref_ind_max]

                # get the correct date range to use for the instrument histogram
                inst_pd_use = inst_hist_pd[
                    (inst_hist_pd['date_obs'] >= str(min_date)) & (inst_hist_pd['date_obs'] <= str(max_date))]
                # get indices and histogram for calculation
                inst_hist_ind = np.where(
                    (inst_hist_pd['date_obs'] >= str(min_date)) & (inst_hist_pd['date_obs'] <= str(max_date)))
                inst_ind_min = np.min(inst_hist_ind)
                inst_ind_max = np.max(inst_hist_ind)
                inst_hist_use = inst_full_hist[:, inst_ind_min:inst_ind_max]

                # sum histograms
                hist_fit = inst_hist_use.sum(axis=1)
                hist_ref = ref_hist_use.sum(axis=1)

                # normalize fit histogram
                fit_sums = hist_fit.sum(axis=0, keepdims=True)
                norm_hist_fit = hist_fit / fit_sums

                # normalize reference histogram
                ref_sums = hist_ref.sum(axis=0, keepdims=True)
                norm_hist_ref = hist_ref / ref_sums

                # get reference/fit peaks
                ref_peak_index = np.argmax(norm_hist_ref)  # index of max value of hist_ref
                ref_peak_val = norm_hist_ref[ref_peak_index]  # max value of hist_ref
                fit_peak_index = np.argmax(norm_hist_fit)  # index of max value of hist_fit
                fit_peak_val = norm_hist_fit[fit_peak_index]  # max value of hist_fit
                # estimate correction coefficients that match fit_peak to ref_peak
                alpha_est = fit_peak_val / ref_peak_val
                x_est = intensity_bin_edges[ref_peak_index] - alpha_est * intensity_bin_edges[fit_peak_index]
                init_pars = np.asarray([alpha_est, x_est], dtype=np.float64)

                # calculate alpha and x
                alpha_x_parameters = iit.optim_iit_linear(norm_hist_ref, norm_hist_fit, intensity_bin_edges,
                                                          init_pars=init_pars)
                # save alpha and x to database
                db_funcs.store_iit_values(db_session, inst_pd_use, meth_name, meth_desc,
                                          alpha_x_parameters.x, create)

    end_time = time.time()
    tot_time = end_time - start_time
    time_tot = str(datetime.timedelta(minutes=tot_time))

    print("Inter-instrument transformation fit parameters have been calculated and saved to the database.")
    print("Total elapsed time for IIT fit parameter calculation: " + time_tot)

    return None


##### STEP THREE: APPLY TRANSFORMATION AND PLOT NEW IMAGES ######
def apply_iit_correction(db_session, hdf_data_dir, iit_query_time_min, iit_query_time_max, inst_list, ref_inst,
                         n_intensity_bins=200, R0=1.01, n_images_plot=1, plot=False):
    # start time
    start_time = time.time()

    #### GET REFERENCE INFO FOR LATER USE ####
    # query euv images to get carrington rotation range
    ref_instrument = [ref_inst, ]
    euv_images = db_funcs.query_euv_images(db_session, time_min=iit_query_time_min, time_max=iit_query_time_max,
                                           instrument=ref_instrument)
    # get min and max carrington rotation
    rot_max = euv_images.cr_rot.max()
    rot_min = euv_images.cr_rot.min()

    for inst_index, instrument in enumerate(inst_list):
        #### QUERY IMAGES ####
        query_instrument = [instrument, ]
        rot_images = db_funcs.query_euv_images_rot(db_session, rot_min=rot_min, rot_max=rot_max,
                                                   instrument=query_instrument)
        image_pd = rot_images.sort_values(by=['cr_rot'])
        # get time minimum and maximum for instrument
        inst_time_min = rot_images.date_obs.min()
        inst_time_max = rot_images.date_obs.max()
        # query correct image combos
        lbc_meth_name = "LBCC"
        combo_query_lbc = db_funcs.query_inst_combo(db_session, inst_time_min, inst_time_max, lbc_meth_name,
                                                    instrument)
        iit_meth_name = "IIT"
        combo_query_iit = db_funcs.query_inst_combo(db_session, inst_time_min, inst_time_max, iit_meth_name,
                                                    instrument)
        # apply LBC
        for index in range(n_images_plot):
            row = image_pd.iloc[index]
            print("Processing image number", row.data_id, "for IIT Correction.")
            #### APPLY LBC CORRECTION #####
            original_los, lbcc_image, mu_indices, use_indices, theoretic_query = lbcc_funcs.apply_lbc(db_session,
                                                                                                      hdf_data_dir,
                                                                                                      combo_query_lbc,
                                                                                                      image_row=row,
                                                                                                      n_intensity_bins=n_intensity_bins,
                                                                                                      R0=R0)
            #### APPLY IIT CORRECTION ####
            lbcc_image, iit_image, use_indices, alpha, x = apply_iit(db_session, combo_query_iit,
                                                                     lbcc_image, use_indices, original_los, R0=R0)

            if plot:
                lbcc_data = lbcc_image.lbcc_data
                corrected_iit_data = iit_image.iit_data
                # plot LBC image
                Plotting.PlotCorrectedImage(lbcc_data, los_image=original_los, nfig=100 + inst_index * 10 + index,
                                            title="Corrected LBCC Image for " + instrument)
                # plot IIT image
                Plotting.PlotCorrectedImage(corrected_iit_data, los_image=original_los,
                                            nfig=200 + inst_index * 10 + index,
                                            title="Corrected IIT Image for " + instrument)
                # plot difference
                Plotting.PlotCorrectedImage(corrected_iit_data - lbcc_data, los_image=original_los,
                                            nfig=300 + inst_index * 10 + index,
                                            title="Difference Plot for " + instrument)

    # end time
    end_time = time.time()
    print("ITT has been applied and specified images plotted.")
    print("Total elapsed time to apply correction and plot: " + str(round(end_time - start_time, 3))
          + " seconds.")

    return None


###### APPLY IIT ######
def apply_iit(db_session, inst_combo_query, lbcc_image, use_indices, los_image, R0=1.01):
    ###### GET VARIABLE VALUES #####
    meth_name = "IIT"
    method_id_info = db_funcs.get_method_id(db_session, meth_name, meth_desc=None, var_names=None,
                                            var_descs=None, create=False)

    alpha_x_parameters = db_funcs.query_var_val(db_session, meth_name, date_obs=lbcc_image.date_obs,
                                                inst_combo_query=inst_combo_query)
    alpha, x = alpha_x_parameters

    ##### APPLY IIT TRANSFORMATION ######
    lbcc_data = lbcc_image.lbcc_data
    corrected_iit_data = np.copy(lbcc_data)
    corrected_iit_data[use_indices] = 10 ** (alpha * np.log10(lbcc_data[use_indices]) + x)
    # create IIT datatype
    iit_image = psi_d_types.create_iit_image(los_image, lbcc_image, corrected_iit_data, method_id_info[0])
    psi_d_types.LosImage.get_coordinates(iit_image, R0=R0)

    return lbcc_image, iit_image, use_indices, alpha, x


def apply_iit_2(db_session, lbcc_image, use_indices, los_image, R0=1.01):
    """
    Different from apply_iit() because it does not require pre-queried iit_combos.
    This function finds the previous and next IIT coefs based on lbcc_image.date_obs
    Parameters
    ----------
    db_session
    lbcc_image
    use_indices
    los_image
    R0

    Returns
    -------

    """
    ###### GET VARIABLE VALUES #####
    meth_name = "IIT"
    method_id_info = db_funcs.get_method_id(db_session, meth_name, meth_desc=None, var_names=None,
                                            var_descs=None, create=False)

    theoretic_query = db_funcs.get_correction_pars(db_session, meth_name, date_obs=lbcc_image.date_obs,
                                                   instrument=lbcc_image.instrument)
    # separate alpha and x
    alpha = theoretic_query[0]
    x = theoretic_query[1]

    ##### APPLY IIT TRANSFORMATION ######
    lbcc_data = lbcc_image.lbcc_data
    corrected_iit_data = np.copy(lbcc_data)
    corrected_iit_data[use_indices] = 10 ** (alpha * np.log10(lbcc_data[use_indices]) + x)
    # create IIT datatype
    iit_image = psi_d_types.create_iit_image(los_image, lbcc_image, corrected_iit_data, method_id_info[0])
    psi_d_types.LosImage.get_coordinates(iit_image, R0=R0)

    return lbcc_image, iit_image, use_indices, alpha, x


###### STEP FOUR: GENERATE NEW HISTOGRAM PLOTS ######
def plot_iit_histograms(db_session, hdf_data_dir, hist_query_time_min, hist_query_time_max, inst_list, ref_inst,
                        n_intensity_bins=200, lat_band=[-np.pi / 2.4, np.pi / 2.4], R0=1.01, log10=True):
    # start time
    start_time = time.time()

    #### GET REFERENCE INFO FOR LATER USE ####
    # get index number of reference instrument
    ref_index = inst_list.index(ref_inst)
    # query euv images to get carrington rotation range
    ref_instrument = [ref_inst, ]
    euv_images = db_funcs.query_euv_images(db_session, time_min=hist_query_time_min, time_max=hist_query_time_max,
                                           instrument=ref_instrument)
    # get min and max carrington rotation
    rot_max = euv_images.cr_rot.max()
    rot_min = euv_images.cr_rot.min()

    # method information
    meth_name = "IIT"
    method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc=None, var_names=None, var_descs=None,
                                       create=False)

    # query for IIT histograms
    pd_lbc_hist = db_funcs.query_hist(db_session=db_session, meth_id=method_id[1],
                                      n_intensity_bins=n_intensity_bins,
                                      lat_band=lat_band,
                                      time_min=hist_query_time_min,
                                      time_max=hist_query_time_max)
    pd_lbc_hist_srt = pd_lbc_hist.sort_values(by=['image_id'])
    # convert the binary types back to arrays
    mu_bin_edges, intensity_bin_edges, full_lbc_hist = psi_d_types.binary_to_hist(pd_lbc_hist_srt,
                                                                                  n_mu_bins=None,
                                                                                  n_intensity_bins=
                                                                                  n_intensity_bins)
    # create corrected/original histograms
    original_hist_list = np.full(full_lbc_hist.shape, 0, dtype=np.int64)
    corrected_hist_list = np.full(full_lbc_hist.shape, 0, dtype=np.int64)
    for inst_index, instrument in enumerate(inst_list):
        print("Applying corrections for", instrument)
        #### QUERY IMAGES ####
        query_instrument = [instrument, ]
        rot_images = db_funcs.query_euv_images_rot(db_session, rot_min=rot_min, rot_max=rot_max,
                                                   instrument=query_instrument)
        image_pd = rot_images.sort_values(by=['cr_rot'])
        # get time minimum and maximum for instrument
        inst_time_min = rot_images.date_obs.min()
        inst_time_max = rot_images.date_obs.max()
        # query correct image combos
        lbc_meth_name = "LBCC"
        combo_query_lbc = db_funcs.query_inst_combo(db_session, inst_time_min, inst_time_max, lbc_meth_name,
                                                    instrument)
        iit_meth_name = "IIT"
        combo_query_iit = db_funcs.query_inst_combo(db_session, inst_time_min, inst_time_max, iit_meth_name,
                                                    instrument)
        # query correct image combos
        combo_query_lbc = db_funcs.query_inst_combo(db_session, hist_query_time_min, hist_query_time_max,
                                                    meth_name="LBCC", instrument=instrument)
        # query correct image combos
        combo_query_iit = db_funcs.query_inst_combo(db_session, hist_query_time_min, hist_query_time_max,
                                                    meth_name="IIT",
                                                    instrument=instrument)
        for index, row in image_pd.iterrows():
            # apply LBC
            original_los, lbcc_image, mu_indices, use_indices, theoretic_query = lbcc_funcs.apply_lbc(db_session,
                                                                                                      hdf_data_dir,
                                                                                                      combo_query_lbc,
                                                                                                      image_row=row,
                                                                                                      n_intensity_bins=n_intensity_bins,
                                                                                                      R0=R0)

            #### ORIGINAL LOS DATA ####
            # calculate IIT histogram from original data
            original_los_hist = psi_d_types.LosImage.iit_hist(original_los, intensity_bin_edges, lat_band, log10)
            # add 1D histogram to array
            original_hist_list[:, index] = original_los_hist

            #### CORRECTED DATA ####
            # apply IIT correction
            lbcc_image, iit_image, use_indices, alpha, x = iit_funcs.apply_iit(db_session, combo_query_iit, lbcc_image,
                                                                               use_indices, original_los, R0=R0)

            #### CREATE CORRECTED IIT HISTOGRAM #####
            # calculate IIT histogram from LBC
            hist_iit = psi_d_types.IITImage.iit_hist(iit_image, lat_band, log10)
            # create IIT histogram datatype
            corrected_hist = psi_d_types.create_iit_hist(iit_image, method_id[1], lat_band, hist_iit)
            corrected_hist_list[:, index] = corrected_hist.hist

    # plotting definitions
    color_list = ['red', 'blue', 'black']
    linestyle_list = ['solid', 'dashed', 'dashdot']

    #### CREATE NEW HISTOGRAM ####
    for inst_index, instrument in enumerate(inst_list):
        print("Plotting Histograms for", instrument)
        #### GET INDICES TO USE ####
        # get index of instrument in histogram dataframe
        hist_inst = pd_lbc_hist_srt['instrument']
        pd_inst_index = hist_inst[hist_inst == instrument].index

        #### ORIGINAL HISTOGRAM #####
        # define histogram
        original_hist = original_hist_list[:, pd_inst_index].sum(axis=1)
        # normalize histogram
        row_sums = original_hist.sum(axis=0, keepdims=True)
        norm_original_hist = original_hist / row_sums

        # plot original
        Plotting.Plot1d_Hist(norm_original_hist, instrument, inst_index, intensity_bin_edges, color_list,
                             linestyle_list,
                             figure=100, xlabel="Intensity (log10)", ylabel="H(I)",
                             title="Histogram: Original LOS Data")

        #### LBCC HISTOGRAM #####
        # define histogram
        lbc_hist = full_lbc_hist[:, pd_inst_index].sum(axis=1)
        # normalize histogram
        lbc_sums = lbc_hist.sum(axis=0, keepdims=True)
        norm_lbc_hist = lbc_hist / lbc_sums

        # plot lbcc
        Plotting.Plot1d_Hist(norm_lbc_hist, instrument, inst_index, intensity_bin_edges, color_list, linestyle_list,
                             figure=200, xlabel="Intensity (log10)", ylabel="H(I)", title="Histogram: Post LBCC")

        #### CORRECTED HISTOGRAM ####
        # define histogram
        corrected_hist = corrected_hist_list[:, pd_inst_index].sum(axis=1)
        # normalize histogram
        iit_sums = corrected_hist.sum(axis=0, keepdims=True)
        norm_corrected_hist = corrected_hist / iit_sums

        # plot corrected
        Plotting.Plot1d_Hist(norm_corrected_hist, instrument, inst_index, intensity_bin_edges, color_list,
                             linestyle_list,
                             figure=300, xlabel="Intensity (log10)", ylabel="H(I)", title="Histogram: Post IIT")

    # end time
    end_time = time.time()
    print("ITT has been applied and original/resulting histograms plotted.")
    print("Total elapsed time to apply correction and plot histograms: " + str(round(end_time - start_time, 3))
          + " seconds.")

    return None
