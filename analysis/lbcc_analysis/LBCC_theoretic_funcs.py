"""
functions to generate LBC for images using theoretic fit
"""

import os
import datetime
import numpy as np
import time
import scipy.optimize as optim
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

import modules.DB_funs as db_funcs
import modules.datatypes as psi_d_types
import modules.lbcc_funs as lbcc
import modules.Plotting as Plotting


####### STEP ONE: CREATE AND SAVE HISTOGRAMS #######
def save_histograms(db_session, hdf_data_dir, inst_list, hist_query_time_min, hist_query_time_max, n_mu_bins=18,
                    n_intensity_bins=200, lat_band=[-np.pi / 64., np.pi / 64.], log10=True, R0=1.01):
    """
    create and save (to database) mu-histograms from EUV images
    @param db_session: connected db session for querying EUV images and saving histograms
    @param hdf_data_dir: directory of processed hdf images
    @param inst_list: list of instruments
    @param hist_query_time_min: minimum time for histogram creation
    @param hist_query_time_max: maximum time for histogram creation
    @param n_mu_bins: number of mu bins
    @param n_intensity_bins: number of intensity bins
    @param lat_band: latitude band
    @param log10: boolean value
    @param R0: radius
    @return: None, saves histograms to database
    """
    # start time
    start_time_tot = time.time()

    # creates mu bin & intensity bin arrays
    mu_bin_edges = np.linspace(0.1, 1.0, n_mu_bins + 1, dtype='float')
    image_intensity_bin_edges = np.linspace(0, 5, num=n_intensity_bins + 1, dtype='float')

    # create LBC method
    meth_name = 'LBCC Theoretic'
    meth_desc = 'LBCC Theoretic Fit Method'
    method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=True)

    # loop over instrument
    for instrument in inst_list:

        # query EUV images
        query_instrument = [instrument, ]
        query_pd = db_funcs.query_euv_images(db_session=db_session, time_min=hist_query_time_min,
                                             time_max=hist_query_time_max, instrument=query_instrument)

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
            hist_lbcc = psi_d_types.create_lbcc_hist(hdf_path, row.image_id, method_id[1], mu_bin_edges,
                                                     image_intensity_bin_edges, lat_band, temp_hist)

            # add this histogram and meta data to database
            db_funcs.add_hist(db_session, hist_lbcc)

    db_session.close()

    end_time_tot = time.time()
    print("Histograms have been created and saved to the database.")
    print("Total elapsed time for histogram creation: " + str(round(end_time_tot - start_time_tot, 3)) + " seconds.")
    return None


###### STEP TWO: CALCULATE AND SAVE THEORETIC FIT PARAMETERS #######
def calc_theoretic_fit(db_session, inst_list, calc_query_time_min, calc_query_time_max, weekday=0, number_of_days=180,
                       n_mu_bins=18, n_intensity_bins=200, lat_band=[-np.pi / 64., np.pi / 64.], create=False):
    """
    calculate and save (to database) theoretic LBC fit parameters
    @param db_session: connected database session to query histograms from and save fit parameters
    @param inst_list: list of instruments
    @param calc_query_time_min: start time for creation of moving average centers
    @param calc_query_time_max: end time for creation of moving average centers
    @param weekday: weekday for the moving average
    @param number_of_days: number of days for creation of moving width
    @param n_mu_bins: number mu bins
    @param n_intensity_bins: number intensity bins
    @param lat_band: latitude band
    @param create: boolean, whether to create new variable values in database (True if create new)
    @return: None, saves theoretic fit parameters to database
    """
    # start time
    start_time_tot = time.time()
    # calculate moving averages
    moving_avg_centers, moving_width = lbcc.moving_averages(calc_query_time_min, calc_query_time_max, weekday,
                                                            number_of_days)
    optim_vals_theo = ["a1", "a2", "b1", "b2", "n", "log_alpha", "SSE", "optim_time", "optim_status"]
    results_theo = np.zeros((len(moving_avg_centers), len(inst_list), len(optim_vals_theo)))

    # get method id
    meth_name = 'LBCC Theoretic'
    meth_desc = 'LBCC Theoretic Fit Method'
    method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=False)

    for date_index, center_date in enumerate(moving_avg_centers):
        print("Begin date " + str(center_date))

        # determine time range based off moving average centers
        min_date = center_date - moving_width / 2
        max_date = center_date + moving_width / 2

        for inst_index, instrument in enumerate(inst_list):
            print("\nStarting calculations for " + instrument)

            # query the histograms for time range based off moving average centers
            query_instrument = [instrument, ]
            pd_hist = db_funcs.query_hist(db_session=db_session, meth_id=method_id[1], n_mu_bins=n_mu_bins,
                                          n_intensity_bins=n_intensity_bins,
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

            start_time = time.time()
            optim_out_theo = optim.minimize(lbcc.get_functional_sse, init_pars,
                                            args=(hist_ref, hist_mat, mu_vec, intensity_bin_array, model),
                                            method=method)

            end_time = time.time()

            results_theo[date_index, inst_index, 0:6] = optim_out_theo.x
            results_theo[date_index, inst_index, 6] = optim_out_theo.fun
            results_theo[date_index, inst_index, 7] = round(end_time - start_time, 3)
            results_theo[date_index, inst_index, 8] = optim_out_theo.status

            meth_name = 'LBCC Theoretic'
            meth_desc = 'LBCC Theoretic Fit Method'
            var_name = "TheoVar"
            var_desc = "Theoretic fit parameter at index "
            db_funcs.store_lbcc_values(db_session, pd_hist, meth_name, meth_desc, var_name, var_desc, date_index,
                                       inst_index, optim_vals=optim_vals_theo[0:6], results=results_theo, create=create)

    end_time_tot = time.time()
    print("Theoretical fit parameters have been calculated and saved to the database.")
    print("Total elapsed time for theoretical fit parameter calculation: " +
          str(round(end_time_tot - start_time_tot, 3)) + " seconds.")

    return None


###### STEP THREE: APPLY CORRECTION AND PLOT IMAGES #######
def apply_lbc_correction(db_session, hdf_data_dir, inst_list, lbc_query_time_min, lbc_query_time_max,
                         n_intensity_bins=200, R0=1.01, n_images_plot=1, plot=False):
    """
    function to apply limb-brightening correction and plot images within a certain time frame
    @param db_session: connected database session to query theoretic fit parameters from
    @param hdf_data_dir: directory of processed images to plot original images
    @param inst_list: list of instruments
    @param lbc_query_time_min: minimum query time for applying lbc fit
    @param lbc_query_time_max: maximum query time for applying lbc fit
    @param n_intensity_bins: number of intensity bins
    @param R0: radius
    @param plot: whether or not to plot images
    @return:
    """
    # start time
    start_time_tot = time.time()

    # method information
    meth_name = "LBCC Theoretic"

    ##### QUERY IMAGES ######
    for inst_index, instrument in enumerate(inst_list):
        query_instrument = [instrument, ]
        image_pd = db_funcs.query_euv_images(db_session=db_session, time_min=lbc_query_time_min,
                                             time_max=lbc_query_time_max, instrument=query_instrument)
        # query correct image combos
        combo_query = db_funcs.query_inst_combo(db_session, lbc_query_time_min, lbc_query_time_max, meth_name,
                                                instrument)
        ###### GET LOS IMAGES COORDINATES (DATA) #####
        for index in range(n_images_plot):
            row = image_pd.iloc[index]
            print("Processing image number", row.image_id, "for LB Correction.")
            original_los, lbcc_image, mu_indices, use_indices = apply_lbc(db_session, hdf_data_dir,
                                                                          combo_query, image_row=row,
                                                                          n_intensity_bins=n_intensity_bins, R0=R0)
            ##### PLOTTING ######
            if plot:
                Plotting.PlotImage(original_los, nfig=100 + inst_index * 10 + index, title="Original LOS Image for " +
                                                                                           instrument)
                Plotting.PlotCorrectedImage(corrected_data=lbcc_image.lbcc_data, los_image=original_los,
                                            nfig=200 + inst_index * 10 + index, title="Corrected LBCC Image for " +
                                                                                      instrument)
                Plotting.PlotCorrectedImage(corrected_data=original_los.data - lbcc_image.lbcc_data,
                                            los_image=original_los, nfig=300 + inst_index * 10 + index,
                                            title="Difference Plot for " + instrument)
    # end time
    end_time_tot = time.time()
    print("LBC has been applied and specified images plotted.")
    print("Total elapsed time to apply correction and plot: " + str(round(end_time_tot - start_time_tot, 3))
          + " seconds.")

    return None


### FUNCTION TO APPLY LBC TO IMAGE ###
def apply_lbc(db_session, hdf_data_dir, inst_combo_query, image_row, n_intensity_bins=200, R0=1.01):
    """
    function to apply limb-brightening correction to use for IIT
    @param db_session: connected database session to query theoretic fit parameters from
    @param hdf_data_dir: directory of processed images to plot original images
    @param inst_combo_query: query results of combo ids corresponding with instrument
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
    if image_row.fname_hdf == "":
        print("Warning: Image # " + str(image_row.image_id) + " does not have an associated hdf file. Skipping")
        pass
    hdf_path = os.path.join(hdf_data_dir, image_row.fname_hdf)
    original_los = psi_d_types.read_los_image(hdf_path)
    original_los.get_coordinates(R0=R0)
    theoretic_query = db_funcs.query_var_val(db_session, meth_name, date_obs=original_los.info['date_string'],
                                             inst_combo_query=inst_combo_query)

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


###### STEP FOUR: GENERATE PLOTS OF BETA AND Y ######
def generate_theoretic_plots(db_session, inst_list, plot_query_time_min, plot_query_time_max, weekday, image_out_path,
                             year='2011', time_period='6 Month', plot_week=0, n_mu_bins=18):
    """
    function to generate plot of theoretical beta and y over time and beta/y v. mu
    @param db_session: connected database session to query theoretic fit parameters from
    @param inst_list: list of instruments
    @param plot_query_time_min: minimum time to query fit parameters and create moving average centers
    @param plot_query_time_max: maximum time to query fit parameters and create moving average centers
    @param weekday: day of the week to plot
    @param image_out_path: path to save plots to
    @param year: year
    @param time_period: time period of query
    @param plot_week: specific index of week you want to plot for beta/y v. mu
    @param n_mu_bins: number of mu bins
    @return:
    """
    # start time
    start_time_tot = time.time()

    # create mu bin array
    mu_bin_array = np.linspace(0.1, 1.0, n_mu_bins + 1, dtype='float')
    mu_bin_centers = (mu_bin_array[1:] + mu_bin_array[:-1]) / 2

    # time arrays
    # returns array of moving averages center dates, based off start and end date
    moving_avg_centers, moving_width = lbcc.moving_averages(plot_query_time_min, plot_query_time_max, weekday)

    # calc beta and y for a few sample mu-values
    sample_mu = [0.125, 0.325, 0.575, 0.875]

    # sample mu colors
    v_cmap = cm.get_cmap('viridis')
    n_mu = len(sample_mu)
    color_dist = np.linspace(0., 1., n_mu)

    linestyles = ['dashed']
    marker_types = ['None']
    meth_name = 'LBCC Theoretic'

    for inst_index, instrument in enumerate(inst_list):
        print("Generating plots for " + instrument + ".")
        # query correct image combos
        combo_query = db_funcs.query_inst_combo(db_session, plot_query_time_min, plot_query_time_max, meth_name,
                                                instrument)
        plot_beta = np.zeros((sample_mu.__len__(), moving_avg_centers.__len__()))
        plot_y = np.zeros((sample_mu.__len__(), moving_avg_centers.__len__()))
        theoretic_query = np.zeros((len(moving_avg_centers), 6))

        for mu_index, mu in enumerate(sample_mu):
            for date_index, center_date in enumerate(moving_avg_centers):
                # query for variable value
                theoretic_query[date_index, :] = db_funcs.query_var_val(db_session, meth_name,
                                                                        date_obs=np.datetime64(center_date).astype(
                                                                            datetime.datetime),
                                                                        inst_combo_query=combo_query)
                plot_beta[mu_index, date_index], plot_y[mu_index, date_index] = lbcc.get_beta_y_theoretic_based(
                    theoretic_query[date_index, :], mu)
        #### BETA AND Y AS FUNCTION OF TIME ####
        # plot beta for the different models as a function of time
        plt.figure(10 + inst_index)

        mu_lines = []
        for mu_index, mu in enumerate(sample_mu):
            plt.plot(moving_avg_centers, plot_beta[mu_index, :], ls=linestyles[0],
                     c=v_cmap(color_dist[mu_index]), marker=marker_types[0])
        plt.ylabel(r"$\beta$ " + instrument)
        plt.xlabel("Center Date")
        ax = plt.gca()
        model_lines = [Line2D([0], [0], color="black", linestyle=linestyles[0], lw=2,
                              marker=marker_types[0])]
        legend1 = plt.legend(mu_lines, [str(round(x, 3)) for x in sample_mu], loc='upper left', bbox_to_anchor=(1., 1.),
                             title=r"$\mu$ value")
        ax.legend(model_lines, ["theoretic"], loc='upper left',
                  bbox_to_anchor=(1., 0.65), title="model")
        plt.gca().add_artist(legend1)
        # adjust margin to incorporate legend
        plt.subplots_adjust(right=0.8)
        plt.grid()

        plot_fname = image_out_path + instrument + '_beta_' + year + "-" + time_period.replace(" ", "") + '.pdf'
        plt.savefig(plot_fname)

        plt.close(10 + inst_index)

        # plot y for the different models as a function of time
        plt.figure(20 + inst_index)

        mu_lines = []
        for mu_index, mu in enumerate(sample_mu):
            mu_lines.append(Line2D([0], [0], color=v_cmap(color_dist[mu_index]), lw=2))
            plt.plot(moving_avg_centers, plot_y[mu_index, :], ls=linestyles[0],
                     c=v_cmap(color_dist[mu_index]), marker=marker_types[0])
        plt.ylabel(r"$y$ " + instrument)
        plt.xlabel("Center Date")
        ax = plt.gca()
        model_lines = [Line2D([0], [0], color="black", linestyle=linestyles[0], lw=2,
                              marker=marker_types[0])]
        legend1 = plt.legend(mu_lines, [str(round(x, 3)) for x in sample_mu], loc='upper left', bbox_to_anchor=(1., 1.),
                             title=r"$\mu$ value")
        ax.legend(model_lines, ["theoretic"], loc='upper left',
                  bbox_to_anchor=(1., 0.65),
                  title="model")
        plt.gca().add_artist(legend1)
        # adjust margin to incorporate legend
        plt.subplots_adjust(right=0.8)
        plt.grid()

        plot_fname = image_out_path + instrument + '_y_' + year + "-" + time_period.replace(" ", "") + '.pdf'
        plt.savefig(plot_fname)

        plt.close(20 + inst_index)

        #### BETA AND Y v. MU FOR SPECIFIED WEEK #####

        plt.figure(100 + inst_index)

        beta_y_v_mu = np.zeros((mu_bin_centers.shape[0], 2))

        for index, mu in enumerate(mu_bin_centers):
            beta_y_v_mu[index, :] = lbcc.get_beta_y_theoretic_based(theoretic_query[plot_week, :], mu)

        plt.plot(mu_bin_centers, beta_y_v_mu[:, 0], ls=linestyles[0],
                 c=v_cmap(color_dist[0 - 3]), marker=marker_types[0])

        plt.ylabel(r"$\beta$ " + instrument)
        plt.xlabel(r"$\mu$")
        plt.title(instrument + " " + time_period + " average " + str(moving_avg_centers[plot_week]))
        ax = plt.gca()

        ax.legend(["theoretic"], loc='upper right',
                  bbox_to_anchor=(1., 1.),
                  title="model")
        plt.grid()

        plot_fname = image_out_path + instrument + '_beta_v_mu_' + year + "-" + time_period.replace(" ", "") + '.pdf'
        plt.savefig(plot_fname)

        plt.close(100 + inst_index)

        # repeat for y
        plt.figure(200 + inst_index)

        plt.plot(mu_bin_centers, beta_y_v_mu[:, 1], ls=linestyles[0],
                 c=v_cmap(color_dist[0 - 3]), marker=marker_types[0])

        plt.ylabel(r"$y$ " + instrument)
        plt.xlabel(r"$\mu$")
        plt.title(instrument + " " + time_period + " average " + str(moving_avg_centers[plot_week]))
        ax = plt.gca()

        ax.legend(["theoretic"], loc='lower right',
                  bbox_to_anchor=(1., 0.),
                  title="model")
        plt.grid()

        plot_fname = image_out_path + instrument + '_y_v_mu_' + year + "-" + time_period.replace(" ", "") + '.pdf'
        plt.savefig(plot_fname)

        plt.close(200 + inst_index)

    end_time_tot = time.time()
    print("Theoretical plots of beta and y over time hvae been generated and saved.")
    print("Total elapsed time for plot creation: " + str(round(end_time_tot - start_time_tot, 3)) + " seconds.")

    return None


###### STEP FIVE: GENERATE HISTOGRAM PLOTS ######
def generate_histogram_plots(db_session, hdf_data_dir, inst_list, hist_plot_query_time_min, hist_plot_query_time_max,
                             n_hist_plots=1, n_mu_bins=18, n_intensity_bins=200, lat_band=[-np.pi / 64., np.pi / 64.],
                             log10=True, R0=1.01):
    # start time
    start_time_tot = time.time()

    meth_name = 'LBCC Theoretic'
    method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc=None, var_names=None, var_descs=None,
                                       create=False)

    # mu bin edges and intensity bin edges
    mu_bin_edges = np.linspace(0.1, 1.0, n_mu_bins + 1, dtype='float')
    intensity_bin_edges = np.linspace(0, 5, num=n_intensity_bins + 1, dtype='float')

    ### PLOT HISTOGRAMS ###
    # query histograms
    for instrument in inst_list:
        query_instrument = [instrument, ]
        pd_hist = db_funcs.query_hist(db_session=db_session, meth_id=method_id[1], n_mu_bins=n_mu_bins,
                                      n_intensity_bins=n_intensity_bins,
                                      lat_band=np.array(lat_band).tobytes(),
                                      time_min=hist_plot_query_time_min,
                                      time_max=hist_plot_query_time_max,
                                      instrument=query_instrument)
        # convert from binary to usable histogram type
        lat_band, mu_bin_array, intensity_bin_array, full_hist = psi_d_types.binary_to_hist(pd_hist, n_mu_bins,
                                                                                            n_intensity_bins)
        # query correct image combos
        combo_query = db_funcs.query_inst_combo(db_session, hist_plot_query_time_min, hist_plot_query_time_max,
                                                meth_name, instrument)
        #### PLOT ORIGINAL HISTOGRAMS ####
        for plot_index in range(n_hist_plots):
            # definitions
            plot_hist = full_hist[:, :, plot_index]
            date_obs = pd_hist.date_obs[plot_index]
            figure = "Original Histogram Plot: "
            # plot histogram
            Plotting.Plot_LBCC_Hists(plot_hist, date_obs, instrument, intensity_bin_edges, mu_bin_edges, figure,
                                     plot_index)

            #### APPLY LBC CORRECTION ####
            # query EUV images
            query_instrument = [instrument, ]
            image_pd = db_funcs.query_euv_images(db_session=db_session, time_min=hist_plot_query_time_min,
                                                 time_max=hist_plot_query_time_max, instrument=query_instrument)
            for index, row in image_pd.iterrows():
                # apply LBC
                original_los, lbcc_image, mu_indices, use_indices = apply_lbc(db_session, hdf_data_dir, combo_query, row,
                                                                             n_intensity_bins=n_intensity_bins, R0=R0)
                #### CREATE NEW HISTOGRAMS ####
                # perform 2D histogram on mu and image intensity
                hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
                temp_hist = psi_d_types.LosImage.mu_hist(lbcc_image, intensity_bin_edges, mu_bin_edges,
                                                         lat_band=lat_band,
                                                         log10=log10)
                hist_lbcc = psi_d_types.create_lbcc_hist(hdf_path, row.image_id, method_id[1], mu_bin_edges,
                                                         intensity_bin_edges, lat_band, temp_hist)
                #### PLOT NEW HISTOGRAMS ####
                # definitions
                date_obs = hist_lbcc.date_obs
                plot_hist = hist_lbcc.hist
                figure = "LBCC Histogram Plot: "
                # plot histogram
                Plotting.Plot_LBCC_Hists(plot_hist, date_obs, instrument, intensity_bin_edges, mu_bin_edges, figure,
                                         plot_index)
    end_time_tot = time.time()
    print("Histogram plots of have been generated.")
    print("Total elapsed time for plot creation: " + str(round(end_time_tot - start_time_tot, 3)) + " seconds.")
    return None
