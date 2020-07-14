"""
Functions to Generate Inter-Instrument Correction
"""

import os
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt

import modules.DB_funs as db_funcs
import modules.datatypes as psi_d_types
import modules.lbcc_funs as lbcc
import modules.iit_funs as iit
import modules.Plotting as Plotting


##### PRE STEP: APPLY LBC TO IMAGES ######
def apply_lbc_correction(db_session, hdf_data_dir, instrument, image_row, n_intensity_bins=200, R0=1.01):
    """
    function to apply limb-brightening correction to use for IIT
    @param db_session: connected database session to query theoretic fit parameters from
    @param hdf_data_dir: directory of processed images to plot original images
    @param instrument: instrument
    @param image_row: row in image query
    @param n_intensity_bins: number of intensity bins
    @param R0: radius
    @return:
    """

    meth_name = "LBCC Theoretic"
    db_sesh, meth_id, var_ids = db_funcs.get_method_id(db_session, meth_name, meth_desc=None, var_names=None,
                                                       var_descs=None, create=False)
    intensity_bin_edges = np.linspace(0, 5, num=n_intensity_bins + 1, dtype='float')

    ###### GET LOS IMAGES COORDINATES (DATA) #####
    print("Processing image number", image_row.image_id, ".")
    if image_row.fname_hdf == "":
        print("Warning: Image # " + str(image_row.image_id) + " does not have an associated hdf file. Skipping")
        pass
    hdf_path = os.path.join(hdf_data_dir, image_row.fname_hdf)
    original_los = psi_d_types.read_los_image(hdf_path)
    original_los.get_coordinates(R0=R0)
    theoretic_query = db_funcs.query_var_val(db_session, meth_name, date_obs=original_los.info['date_string'],
                                             instrument=instrument)

    # get beta and y from theoretic fit
    ###### DETERMINE LBC CORRECTION (for valid mu values) ######
    beta1d, y1d, mu_indices, use_indices = lbcc.get_beta_y_theoretic_continuous_1d_indices(theoretic_query,
                                                                                           los_image=original_los)

    ###### APPLY LBC CORRECTION (log10 space) ######
    corrected_lbc_data = np.copy(original_los.data)
    corrected_lbc_data[use_indices] = 10 ** (beta1d * np.log10(original_los.data[use_indices]) + y1d)

    ###### CREATE LBCC DATATYPE ######
    lbcc_image = psi_d_types.create_lbcc_image(hdf_path, corrected_lbc_data, image_id=image_row.image_id,
                                              meth_id=meth_id, intensity_bin_edges=intensity_bin_edges)
    psi_d_types.LosImage.get_coordinates(lbcc_image, R0=R0)

    return original_los, lbcc_image, mu_indices, use_indices


##### STEP ONE: CREATE 1D HISTOGRAMS AND SAVE TO DATABASE ######
def create_histograms(db_session, inst_list, lbc_query_time_min, lbc_query_time_max, hdf_data_dir,
                      n_intensity_bins=200, lat_band=[-np.pi / 2.4, np.pi / 2.4], log10=True, R0=1.01):
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
    start_time_tot = time.time()

    # create IIT method
    meth_name = "IIT"
    meth_desc = "IIT Fit Method"
    method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=True)

    for instrument in inst_list:
        # query EUV images
        query_instrument = [instrument, ]
        image_pd = db_funcs.query_euv_images(db_session=db_session, time_min=lbc_query_time_min,
                                             time_max=lbc_query_time_max, instrument=query_instrument)
        # apply LBC
        for index, row in image_pd.iterrows():
            original_los, lbcc_image, mu_indices, use_indices = apply_lbc_correction(db_session, hdf_data_dir, instrument,
                                                                      image_row=row, n_intensity_bins=n_intensity_bins, R0=R0)

            # calculate IIT histogram from LBC
            hist = psi_d_types.LBCCImage.iit_hist(lbcc_image, lat_band, log10)

            # create IIT histogram datatype
            iit_hist = psi_d_types.create_iit_hist(lbcc_image, method_id[1], lat_band, hist)

            # add IIT histogram and meta data to database
            db_funcs.add_hist(db_session, iit_hist)

    db_session.close()

    end_time_tot = time.time()
    print("Inter-instrument transformation histograms have been created and saved to the database.")
    print(
        "Total elapsed time for histogram creation: " + str(round(end_time_tot - start_time_tot, 3)) + " seconds.")
    return None


##### STEP TWO: CALCULATE INTER-INSTRUMENT TRANSFORMATION COEFFICIENTS AND SAVE TO DATABASE ######
def calc_iit_coefficients(db_session, inst_list, ref_inst, calc_query_time_min, calc_query_time_max, weekday=0,
                          number_of_days=180, n_intensity_bins=200, lat_band=[-np.pi / 2.4, np.pi / 2.4], create=False):
    # start time
    start_time_tot = time.time()

    # returns array of moving averages center dates, based off start date and number of weeks
    moving_avg_centers, moving_width = lbcc.moving_averages(calc_query_time_min, calc_query_time_max, weekday,
                                                            number_of_days)

    # create IIT method
    meth_name = "IIT"
    meth_desc = "IIT Fit Method"
    method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=False)

    # get index number of reference instrument
    ref_index = inst_list.index(ref_inst)

    for date_index, center_date in enumerate(moving_avg_centers):
        print("Begin date " + str(center_date))

        # determine time range based off moving average centers
        min_date = center_date - moving_width / 2
        max_date = center_date + moving_width / 2

        # create arrays for summed histograms and intensity bins
        hist_array = np.zeros((len(inst_list), n_intensity_bins))
        intensity_bin_array = np.zeros((len(inst_list), n_intensity_bins + 1))

        # query IIT histograms
        pd_hist = db_funcs.query_hist(db_session=db_session, meth_id=method_id[1], n_intensity_bins=n_intensity_bins,
                                      lat_band=np.array(lat_band).tobytes(),
                                      time_min=np.datetime64(min_date).astype(datetime.datetime),
                                      time_max=np.datetime64(max_date).astype(datetime.datetime))

        # convert the binary types back to arrays
        lat_band, mu_bin_edges, intensity_bin_edges, full_hist = psi_d_types.binary_to_hist(hist_binary=pd_hist,
                                                                                            n_mu_bins=None,
                                                                                            n_intensity_bins=
                                                                                            n_intensity_bins)

        for inst_index, instrument in enumerate(inst_list):
            # get index of instrument
            pd_inst_index = pd_hist['instrument'] == instrument
            # sum the appropriate histograms
            hist = full_hist[:, pd_inst_index]
            summed_hist = hist.sum(axis=1)
            hist_array[inst_index, :] = summed_hist
            intensity_bin_array[inst_index, :] = intensity_bin_edges

        # use lbcc function to calculate alpha and x
        for inst_index, instrument in enumerate(inst_list):
            # get reference and fit histograms
            hist_fit = hist_array[inst_index, :]
            hist_ref = hist_array[ref_index, :]
            intensity_bin_edges = intensity_bin_array[inst_index, :]

            # get reference/fit peaks
            ref_peak_index = np.argmax(hist_ref)  # index of max value of hist_ref
            ref_peak_val = hist_ref[ref_peak_index]  # max value of hist_ref
            fit_peak_index = np.argmax(hist_fit)  # index of max value of hist_fit
            fit_peak_val = hist_fit[fit_peak_index]  # max value of hist_fit
            # estimate correction coefficients that match fit_peak to ref_peak
            alpha_est = fit_peak_val / ref_peak_val
            x_est = intensity_bin_edges[ref_peak_index] - alpha_est * intensity_bin_edges[fit_peak_index]
            init_pars = np.asarray([alpha_est, x_est], dtype=np.float32)

            # calculate alpha and x
            alpha_x_parameters = iit.optim_iit_linear(hist_ref, hist_fit, intensity_bin_edges,
                                                      init_pars=init_pars)
            # save alpha and x to database
            pd_inst_index = pd_hist['instrument'] == instrument
            db_funcs.store_iit_values(db_session, pd_hist[pd_inst_index], meth_name, meth_desc, alpha_x_parameters.x,
                                      create)

    end_time_tot = time.time()
    tot_time = end_time_tot - start_time_tot
    time_tot = str(datetime.timedelta(minutes=tot_time))

    print("Inter-instrument transformation fit parameters have been calculated and saved to the database.")
    print("Total elapsed time for IIT fit parameter calculation: " + time_tot)

    return None


##### STEP THREE: APPLY TRANSFORMATION AND PLOT NEW IMAGES ######
def apply_iit_correction(db_session, hdf_data_dir, iit_query_time_min, iit_query_time_max, inst_list,
                         n_intensity_bins=200, R0=1.01, plot=False):
    # start time
    start_time_tot = time.time()

    meth_name = "IIT"

    for inst_index, instrument in enumerate(inst_list):

        #### QUERY IMAGES ####
        query_instrument = [instrument, ]
        image_pd = db_funcs.query_euv_images(db_session=db_session, time_min=iit_query_time_min,
                                             time_max=iit_query_time_max, instrument=query_instrument)
        # apply LBC
        for index, row in image_pd.iterrows():

            #### APPLY LBC CORRECTION #####
            original_los, lbcc_image, mu_indices, use_indices = apply_lbc_correction(db_session, hdf_data_dir,
                                                                                               instrument,
                                                                                               image_row=row,
                                                                                               n_intensity_bins=n_intensity_bins,
                                                                                               R0=R0)

            ###### GET VARIABLE VALUES #####
            method_id_info = db_funcs.get_method_id(db_session, meth_name, meth_desc=None, var_names=None,
                                                    var_descs=None,
                                                    create=False)

            alpha_x_parameters = db_funcs.query_var_val(db_session, meth_name, date_obs=lbcc_image.date_obs,
                                                        instrument=instrument)
            alpha, x = alpha_x_parameters

            ##### APPLY IIT TRANSFORMATION ######
            lbcc_data = lbcc_image.lbcc_data
            corrected_iit_data = np.copy(lbcc_data)
            corrected_iit_data[use_indices] = 10 ** (alpha * np.log10(lbcc_data[use_indices]) + x)
            # create IIT datatype
            hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
            iit_image = psi_d_types.create_iit_image(lbcc_image, corrected_iit_data, method_id_info[1], hdf_path)
            psi_d_types.LosImage.get_coordinates(iit_image, R0=R0)

            if plot:
                # plot LBC image
                Plotting.PlotCorrectedImage(lbcc_data, los_image=original_los, nfig=100 + inst_index * 10 + index,
                                            title="Corrected LBCC Image for " + instrument)
                # plot IIT image
                Plotting.PlotCorrectedImage(corrected_iit_data, los_image=original_los,
                                            nfig=200 + inst_index * 10 + index,
                                            title="Corrected IIT Image for " + instrument)
                # plot difference
                Plotting.PlotCorrectedImage(lbcc_data - corrected_iit_data, los_image=original_los,
                                            nfig=300 + inst_index * 10 + index,
                                            title="Difference Plot for " + instrument)

    # end time
    end_time_tot = time.time()
    print("ITT has been applied and specified images plotted.")
    print("Total elapsed time to apply correction and plot: " + str(round(end_time_tot - start_time_tot, 3))
          + " seconds.")
    return None


###### STEP FOUR: GENERATE NEW HISTOGRAM PLOTS ######
def generate_iit_histograms(db_session, hist_query_time_min, hist_query_time_max, inst_list, ref_inst,
                            n_intensity_bins=200, lat_band=[-np.pi / 2.4, np.pi / 2.4]):
    # start time
    start_time_tot = time.time()

    meth_name = "IIT"
    ref_index = inst_list.index(ref_inst)
    method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc=None, var_names=None, var_descs=None,
                                       create=False)

    # query for IIT histograms
    for inst_index, instrument in enumerate(inst_list):
        # query for IIT histograms
        query_instrument = [instrument, ]
        pd_hist = db_funcs.query_hist(db_session=db_session, meth_id=method_id[1],
                                      n_intensity_bins=n_intensity_bins,
                                      lat_band=np.array(lat_band).tobytes(),
                                      time_min=hist_query_time_min,
                                      time_max=hist_query_time_max,
                                      instrument=query_instrument)

        # convert the binary types back to arrays
        lat_band, mu_bin_edges, intensity_bin_edges, full_hist = psi_d_types.binary_to_hist(pd_hist, n_mu_bins=None,
                                                                                            n_intensity_bins=
                                                                                            n_intensity_bins)
        # create corrected histograms
        corrected_hist_list = np.full((full_hist.shape), 0, dtype=np.int64)
        for index, row in pd_hist.iterrows():
            #### QUERY IIT CORRECTION COEFFICIENTS ####
            alpha, x = db_funcs.query_var_val(db_session, meth_name, date_obs=pd_hist.date_obs[index],
                                              instrument=instrument)
            #### CORRECTED DATA ####
            original_hist = full_hist[:, index]
            indices = np.where(original_hist > 0)
            corrected = np.copy(original_hist)
            corrected[indices] = 10 ** (alpha * np.log10(original_hist[indices]) + x)
            corrected_hist_list[:, index] = corrected

        # plotting definitions
        color_list = ['red', 'blue', 'black']
        linestyle_list = ['solid', 'dashed', 'dashdot']

    #### CREATE NEW HISTOGRAM ####
    for inst_index, instrument in enumerate(inst_list):
        Plotting.Plot_IIT_Hists(pd_hist, corrected_hist_list, full_hist, instrument, ref_inst, inst_index, ref_index,
                                intensity_bin_edges, color_list, linestyle_list)
    # end time
    end_time_tot = time.time()
    print("ITT has been applied and original/resulting histograms plotted.")
    print("Total elapsed time to apply correction and plot histograms: " + str(round(end_time_tot - start_time_tot, 3))
          + " seconds.")
    return None
