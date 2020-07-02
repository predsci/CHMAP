"""
Functions to Generate Inter-Instrument Correction
"""
import sys
sys.path.append('/Users/tamarervin/Dropbox/work/CHD')

import os
import numpy as np
import time
import datetime

import modules.DB_funs as db_funcs
import modules.datatypes as psi_d_types
import modules.lbcc_funs as lbcc
import modules.iit_funs as iit
import modules.Plotting as Plotting


##### PRE STEP: APPLY LBC TO IMAGES ######
# TODO: we need to have this be an array of all instruments
# TODO: save to database??
def apply_lbc_correction(db_session, hdf_data_dir, inst_list, lbc_query_time_min, lbc_query_time_max, n_mu_bins=18,
                         n_intensity_bins=200, R0=1.01):
    """
    function to apply limb-brightening correction to use for IIT
    @param db_session: connected database session to query theoretic fit parameters from
    @param hdf_data_dir: directory of processed images to plot original images
    @param inst_list: list of instruments
    @param lbc_query_time_min: minimum query time for applying lbc fit
    @param lbc_query_time_max: maximum query time for applying lbc fit
    @param n_mu_bins: number of mu bins
    @param n_intensity_bins: number of intensity bins
    @param R0: radius
    @return:
    """
    # start time
    start_time_tot = time.time()

    meth_name = "LBCC Theoretic"
    mu_bin_edges = np.array(range(n_mu_bins + 1), dtype="float") * 0.05 + 0.1
    mu_bin_centers = (mu_bin_edges[1:] + mu_bin_edges[:-1]) / 2
    intensity_bin_edges = np.linspace(0, 5, num=n_intensity_bins + 1, dtype='float')

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
            # TODO: do we need to do this by instrument?? - figure out index
            lbcc_data = psi_d_types.create_lbcc_image(corrected_los_data, los_image=original_los,
                                                      intensity_bin_edges=intensity_bin_edges)


    # end time
    end_time_tot = time.time()
    print("LBC has been applied.")
    print("Total elapsed time to apply correction: " + str(round(end_time_tot - start_time_tot, 3))
          + " seconds.")

    return None


##### STEP ONE: CREATE 1D HISTOGRAMS AND SAVE TO DATABASE ######
def create_histograms(db_session, inst_list, lbc_query_time_min, lbc_query_time_max, hdf_data_dir, n_mu_bins=18,
                      n_intensity_bins=200, lat_band=[-np.pi / 64., np.pi / 64.],
                      log10=True, R0=1.01):
    """
    create and save (to database) IIT-Histograms from LBC Data
    @param db_session: connected db session for querying EUV images and saving histograms
    @param inst_list: list of instruments
    @param lbc_query_time_min: minimum query time for applying lbc fit
    @param lbc_query_time_max: maximum query time for applying lbc fit
    @param hdf_data_dir: directory of processed images to plot original images
    @param n_mu_bins: number of mu bins
    @param n_intensity_bins: number of intensity bins
    @param lat_band: latitude band
    @param log10: boolean value
    @param R0: radius
    @return: None, saves histograms to database
    """
    # start time
    start_time_tot = time.time()

    # creates intensity bin arrays
    intensity_bin_edges = np.linspace(0, 5, num=n_intensity_bins + 1, dtype='float')

    # use function to apply LBC
    apply_lbc_correction(db_session, hdf_data_dir, inst_list, lbc_query_time_min, lbc_query_time_max,
                                     n_mu_bins, n_intensity_bins, R0)
    # loop over instrument
    # TODO: figure out how this should deal with instruments
    for instrument in inst_list:

        # query LBCC Images
        lbcc_data = db_funcs.query_lbcc_images(db_session, lbc_query_time_min, lbc_query_time_max, instrument,
                                               wavelength=None) # TODO: check this output

        # calculate IIT histogram from LBC
        # TODO: check this instrument thing
        hist, use_data = psi_d_types.LBCCImage.iit_hist(lbcc_data[:, :], lat_band, log10)

        # create IIT histogram datatype
        iit_hist = psi_d_types.create_iit_hist(lbcc_data[:, :], intensity_bin_edges, lat_band, hist)

        # add IIT histogram and meta data to database
        # TODO: check that this saves correctly in db since some columns are empty
        db_funcs.add_iit_hist(db_session, iit_hist)

    db_session.close()

    end_time_tot = time.time()
    print("Inter-instrument transformation histograms have been created and saved to the database.")
    print(
        "Total elapsed time for histogram creation: " + str(round(end_time_tot - start_time_tot, 3)) + " seconds.")
    return None


##### STEP TWO: CALCULATE TRANSFORMATION COEFFICIENTS ######
def calc_iit_coefficients(db_session, inst_list, ref_inst, calc_query_time_min, number_of_weeks=27, number_of_days=180,
                          n_intensity_bins=200, lat_band=[-np.pi / 64., np.pi / 64.], create=False):
    # start time
    start_time_tot = time.time()

    # returns array of moving averages center dates, based off start date and number of weeks
    moving_avg_centers = np.array(
        [np.datetime64(str(calc_query_time_min)) + ii * np.timedelta64(1, 'W') for ii in range(number_of_weeks)])

    # returns moving width based of number of days
    moving_width = np.timedelta64(number_of_days, 'D')

    # create definitions to save alpha and x to database
    meth_name = "IIT"
    meth_desc = "IIT Fit Method"

    # get index number of reference instrument
    ref_index = inst_list.index(ref_inst)

    for date_index, center_date in enumerate(moving_avg_centers):
        print("Begin date " + str(center_date))

        # determine time range based off moving average centers
        min_date = center_date - moving_width / 2
        max_date = center_date + moving_width / 2

        # create arrays for summed histograms and intensity bins
        hist_array = np.zeros((len(inst_list), 2048*2048)) # TODO: check these dimensions
        intensity_bin_array = np.zeroes((len(inst_list), n_intensity_bins))
        for inst_index, instrument in enumerate(inst_list):

            # query for IIT histograms
            # query the histograms for time range based off moving average centers
            query_instrument = [instrument, ]
            pd_hist = db_funcs.query_hist(db_session=db_session, n_intensity_bins=n_intensity_bins,
                                          lat_band=np.array(lat_band).tobytes(),
                                          time_min=np.datetime64(min_date).astype(datetime.datetime),
                                          time_max=np.datetime64(max_date).astype(datetime.datetime),
                                          instrument=query_instrument)
            # convert the binary types back to arrays
            lat_band, intensity_bin_edges, mu_bin_edges, full_hist = psi_d_types.binary_to_hist(pd_hist, n_mu_bins=None,
                                                                                                n_intensity_bins=
                                                                                                n_intensity_bins)

            # create list of observed dates in time frame
            date_obs_npDT64 = pd_hist['date_obs']

            # determine appropriate date range
            date_ind = (date_obs_npDT64 >= min_date) & (date_obs_npDT64 <= max_date)

            # sum the appropriate histograms
            summed_hist = full_hist[:, date_ind].sum(axis=1)
            hist_array[inst_index, :] = summed_hist
            intensity_bin_array[inst_index, :] = intensity_bin_edges

        # use lbcc function to calculate alpha and x
        # TODO: going to have to do some loop or something to deal with the different instruments
        for instrument, inst_index in enumerate(inst_list):
            hist_ref = hist_array[inst_index, :]
            hist_fit = hist_array[ref_index, :]
            intensity_bin_edges = intensity_bin_array[inst_index, :]
            alpha_x_parameters = iit.optim_iit_linear(hist_ref, hist_fit, intensity_bin_edges, init_pars=np.asarray([1., 0.]))
            db_funcs.store_iit_values(db_session, pd_hist, meth_name, meth_desc, alpha_x_parameters, create)

    end_time_tot = time.time()
    print("Inter-instrument transformation fit parameters have been calculated and saved to the database.")
    print("Total elapsed time for IIT fit parameter calculation: " +
          str(round(end_time_tot - start_time_tot, 3)) + " seconds.")
    return None


##### STEP THREE: APPLY TRANSFORMATION AND PLOT NEW IMAGES ######
def apply_iit_correction(db_session, hdf_data_dir, iit_query_time_min, iit_query_time_max, inst_list, plot=False):

    # start time
    start_time_tot = time.time()

    # definitions
    meth_name = 'IIT'

    ##### QUERY IMAGES ######
    for inst_index, instrument in enumerate(inst_list):

        #### QUERY IMAGES ####
        query_instrument = [instrument, ]
        lbcc_image_pd = db_funcs.query_corrected_images(db_session=db_session, time_min=iit_query_time_min,
                                                   time_max=iit_query_time_max, instrument=query_instrument)
        # transformation binary lbcc data back to array lbcc data
        mu_array, lat_array, lbcc_data = psi_d_types.binary_to_lbcc(lbcc_image_pd)

        ###### GET VARIABLE VALUES #####
        for index, row in lbcc_image_pd.iterrows():
            alpha, x = db_funcs.query_var_val(db_session, meth_name, date_obs=row.date_obs, instrument=instrument)

            ##### APPLY IIT TRANSFORMATION ######
            corrected_iit_data = alpha*lbcc_data[index] + x

            ##### ADD CORRECTED IIT DATA TO DATABASE ######
            db_funcs.add_corrected_image(db_session, corrected_image=corrected_iit_data)

        ##### PLOT IMAGES ######
        # TODO: check this plotting - using the original LOS images
        # TODO: if this works, make PlotCorrectedImage function
        if plot:
            image_pd = db_funcs.query_EUV_images(db_session=db_session, time_min=iit_query_time_min,
                                                 time_max=iit_query_time_min+datetime.timedelta(hours=3),
                                                 instrument=query_instrument)
            ###### GET LOS IMAGES COORDINATES (DATA) #####
            for index, row in image_pd.iterrows():
                print("Processing image number", row.image_id, ".")
                if row.fname_hdf == "":
                    print("Warning: Image # " + str(row.image_id) + " does not have an associated hdf file. Skipping")
                    continue
                hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
                original_los = psi_d_types.read_los_image(hdf_path)
                original_los.get_coordinates(R0=1.01)

            Plotting.PlotImage(lbcc_data, nfig=100 + inst_index, title="Original LBCC Image for " + instrument)
            Plotting.PlotLBCCImage(iit_data=corrected_iit_data, los_image=original_los, nfig=200 + inst_index,
                                   title="Corrected IIT Image for " + instrument)
            Plotting.PlotLBCCImage(lbcc_data=lbcc_data - corrected_iit_data, los_image=original_los,
                                   nfig=300 + inst_index, title="Difference Plot for " + instrument)

    # end time
    end_time_tot = time.time()
    print("ITT has been applied and specified images plotted.")
    print("Total elapsed time to apply correction and plot: " + str(round(end_time_tot - start_time_tot, 3))
          + " seconds.")
    return None


###### STEP FOUR: GENERATE NEW HISTOGRAM PLOTS ######
def generate_iit_histograms(db_session, hist_query_min, hist_query_max, inst_list, n_intensity_bins=200,
                            lat_band=[-np.pi / 64., np.pi / 64.]):

    ##### QUERY ORIGINAL HISTOGRAMS #####
    for inst_index, instrument in enumerate(inst_list):
        # query for IIT histograms
        # TODO: check this because might query all histograms and not just IIT...
        # TODO: should we connect a method id to this???
        query_instrument = [instrument, ]
        pd_hist = db_funcs.query_hist(db_session=db_session, n_intensity_bins=n_intensity_bins,
                                      lat_band=np.array(lat_band).tobytes(),
                                      time_min=hist_query_min,
                                      time_max=hist_query_max,
                                      instrument=query_instrument)
        # convert the binary types back to arrays
        lat_band, intensity_bin_edges, mu_bin_edges, full_hist = psi_d_types.binary_to_hist(pd_hist, n_mu_bins=None,
                                                                                            n_intensity_bins=
                                                                                            n_intensity_bins)

    #### QUERY IIT CORRECTION COEFFICIENTS ####

    #### CREATE NEW HISTOGRAM AND PLOT ####
    return None
