

"""
Track beta-y functional fits as moving average goes through time
"""

import sys
# location of modules/settings folders for import
sys.path.append('/Users/tamarervin/work/chd')
import numpy as np
import pickle
import time
import pandas as pd
import scipy.optimize as optim
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import modules.lbcc_funs as lbcc
from settings.app import App

# PARAMETERS TO UPDATE
bin_n = 400
start_date = "2011-01-04"
number_of_weeks = 1
number_of_days = 3
year = "2011" # used for naming plot file
time_period = "1week" # used for naming plot file
title_time_period = "1 Week" # used for plot titles
plot_week = 0 # index of week you want to plot

# IMAGE PATHS
# locate histogram files
data_path = os.path.join(App.DATABASE_HOME, "data_files")
# path to save plots to
image_out_path = os.path.join(App.APP_HOME, "test_data", "analysis/lbcc_functionals/")

# EVERYTHING BELOW IS GENERIC
# whether you want to generate plots
gen_plots = True

instruments = ['AIA', "EUVI-A", "EUVI-B"]
optim_vals = ["Beta", "y", "SSE", "optim_time", "optim_status"]
optim_vals1 = ["a1", "a2", "a3", "b1", "b2", "b3", "SSE", "optim_time", "optim_status"]
optim_vals2 = ["a1", "a2", "b1", "SSE", "optim_time", "optim_status"]
optim_vals3 = ["a1", "a2", "b1", "b2", "n", "log_alpha", "SSE", "optim_time", "optim_status"]

# bin number - must match bins for data you use
int_bin_n = [bin_n, ]
temp_results = np.zeros((17, 1, len(optim_vals)))

# moving_avg_centers = [datetime.datetime(2011, 4, 1, 0, 0, 0, 0) + ii*datetime.timedelta(7, 0, 0, 0) for ii in range(27)]
# returns center date, based off start date and number of weeks
moving_avg_centers = np.array([np.datetime64(start_date) + ii*np.timedelta64(number_of_weeks, 'W') for ii in range(number_of_weeks)])

# moving_width = datetime.timedelta(180)
# number of days
moving_width = np.timedelta64(number_of_days, 'D')

results = np.zeros((len(moving_avg_centers), len(instruments), 17, len(optim_vals)))
results1 = np.zeros((len(moving_avg_centers), len(instruments), len(optim_vals1)))
results2 = np.zeros((len(moving_avg_centers), len(instruments), len(optim_vals2)))
results3 = np.zeros((len(moving_avg_centers), len(instruments), len(optim_vals3)))

for inst_index, instrument in enumerate(instruments):
    print("\nStarting calcs for " + instrument + "\n")

    # change this depending on bin size you use
    Inst_fname = os.path.join(data_path, "mu-hists-" + year + "_" + str(bin_n) + "_" + instrument + ".pkl")
    # start with the 400 bin file
    f = open(Inst_fname, 'rb')
    Inst = pickle.load(f)
    f.close()

    # generate 1-yr window histogram
    full_year_Inst = Inst['all_hists']
    date_obs_npDT64 = Inst['date_obs']
    date_obs_pdTS = [pd.Timestamp(x) for x in date_obs_npDT64]
    date_obs = [x.to_pydatetime() for x in date_obs_pdTS]

    image_intensity_bin_edges = Inst['intensity_bin_edges']
    intensity_centers = (image_intensity_bin_edges[:-1] + image_intensity_bin_edges[1:])/2

    mu_bin_edges = Inst['mu_bin_edges']
    mu_bin_centers = (mu_bin_edges[1:] + mu_bin_edges[:-1])/2 #creates array of mu bin centers

    for date_index, center_date in enumerate(moving_avg_centers):
        print("Begin date " + str(center_date))
        start_time_tot = time.time()
        min_date = center_date - moving_width/2
        max_date = center_date + moving_width/2
        # determine appropriate date range
        date_ind = (date_obs_npDT64 >= min_date) & (date_obs_npDT64 <= max_date)
        # sum the appropriate histograms
        summed_hist = full_year_Inst[:, :, date_ind].sum(axis=2)

        # normalize in mu
        norm_hist = np.full(summed_hist.shape, 0.)
        row_sums = summed_hist.sum(axis=1, keepdims=True)
        # but do not divide by zero
        zero_row_index = np.where(row_sums != 0)
        norm_hist[zero_row_index[0]] = summed_hist[zero_row_index[0]]/row_sums[zero_row_index[0]]

        # separate the reference bin from the fitted bins
        hist_ref = norm_hist[-1, ]
        hist_mat = norm_hist[:-1, ]

        mu_vec = mu_bin_centers[:-1]
        int_bin_edges = image_intensity_bin_edges


        # -- fit the theoretical functional -----------
        model = 3
        init_pars = np.array([-0.05, -0.3, -.01, 0.4, -1., 6.])
        method = "BFGS"

        start3 = time.time()
        optim_out3 = optim.minimize(lbcc.get_functional_sse, init_pars,
                                    args=(hist_ref, hist_mat, mu_vec, image_intensity_bin_edges, model),
                                    method=method)

        end3 = time.time()
        # print("Optimization time for theoretic functional: " + str(round(end3 - start3, 3)) + " seconds.")
        # resulting_pars3 = pd.DataFrame(data={'beta': mu_bin_centers, 'y': mu_bin_centers})
        # for index, mu in enumerate(mu_bin_centers):
        #     resulting_pars3.beta[index], resulting_pars3.y[index] = lbcc.get_beta_y_theoretic_based(optim_out3.x, mu)

        results3[date_index, inst_index, 0:6] = optim_out3.x
        results3[date_index, inst_index, 6] = optim_out3.fun
        results3[date_index, inst_index, 7] = round(end3 - start3, 3)
        results3[date_index, inst_index, 8] = optim_out3.status


        # -- fit the power/log functionals -------------
        model = 2
        init_pars = np.array([.93, -0.13, 0.6])
        method = "BFGS"
        gtol = 1e-4

        start2 = time.time()
        optim_out2 = optim.minimize(lbcc.get_functional_sse, init_pars,
                                    args=(hist_ref, hist_mat, mu_vec, image_intensity_bin_edges, model),
                                    method=method, jac='2-point', options={'gtol': gtol})
        end2 = time.time()
        # print("Optimization time for power/log functional: " + str(round(end2 - start2, 3)) + " seconds.")
        # resulting_pars2 = pd.DataFrame(data={'beta': mu_bin_centers, 'y': mu_bin_centers})
        # for index, mu in enumerate(mu_bin_centers):
        #     resulting_pars2.beta[index], resulting_pars2.y[index] = lbcc.get_beta_y_power_log(optim_out2.x, mu)

        results2[date_index, inst_index, 0:3] = optim_out2.x
        results2[date_index, inst_index, 3] = optim_out2.fun
        results2[date_index, inst_index, 4] = round(end2 - start2, 3)
        results2[date_index, inst_index, 5] = optim_out2.status

        # -- fit constrained cubic functionals -------
        model = 1
        init_pars = np.array([-1., 1.1, -0.4, 4.7, -5., 2.1])
        method = "SLSQP"

        # cubic constraints for first derivative
        lb = np.zeros(4)
        ub = np.zeros(4)
        lb[[0, 1]] = -np.inf
        ub[[2, 3]] = np.inf
        A = np.zeros((4, 6))
        A[0, 0] = 1
        A[1, 0:3] = [1, 2, 3]
        A[2, 3] = 1
        A[3, 3:6] = [1, 2, 3]

        lin_constraint = optim.LinearConstraint(A, lb, ub)

        # unconstrained optimization using Nelder-Meade
        # optim_out1 = optim.minimize(get_functional_sse, init_pars, args=(hist_ref, hist_mat, mu_vec, image_intensity_bin_edges, model),
        #                                method=method)

        start1 = time.time()
        # constrained optimization using SLSQP with numeric Jacobian
        optim_out1 = optim.minimize(lbcc.get_functional_sse, init_pars,
                                    args=(hist_ref, hist_mat, mu_vec, image_intensity_bin_edges, model),
                                    method=method, jac="2-point", constraints=lin_constraint)
        end1 = time.time()
        # print("Optimization time for constrained cubic functional: " + str(round(end1 - start1, 3)) + " seconds.")

        # resulting_pars1 = pd.DataFrame(data={'beta': mu_bin_centers, 'y': mu_bin_centers})
        # for index, mu in enumerate(mu_bin_centers):
        #     resulting_pars1.beta[index], resulting_pars1.y[index] = lbcc.get_beta_y_cubic(optim_out1.x, mu)

        results1[date_index, inst_index, 0:6] = optim_out1.x
        results1[date_index, inst_index, 6] = optim_out1.fun
        results1[date_index, inst_index, 7] = round(end1-start1, 3)
        results1[date_index, inst_index, 8] = optim_out1.status


        # -- Do mu-bin direct calcs of beta and y -----
        ref_peak_index = np.argmax(hist_ref)
        ref_peak_val = hist_ref[ref_peak_index]

        for ii in range(mu_bin_centers.__len__() - 1):
            hist_fit = norm_hist[ii, ]


            # estimate correction coefs that match fit_peak to ref_peak
            fit_peak_index = np.argmax(hist_fit) #index of max value of hist_fit
            fit_peak_val = hist_fit[fit_peak_index] #max value of hist_fit
            beta_est = fit_peak_val/ref_peak_val

            #convert everything from type float64 to float
            fit_peak_val = np.float32(fit_peak_val)
            ref_peak_val = np.float32(ref_peak_val)
            beta_est = np.float32(beta_est)

            y_est = image_intensity_bin_edges[ref_peak_index] - beta_est*image_intensity_bin_edges[fit_peak_index]
            y_est = np.float32(y_est)
            init_pars = np.asarray([beta_est, y_est], dtype=np.float32)
            hist_ref.astype(np.float32)

            # optimize correction coefs
            start_time = time.time()

            # doesn't like the data types in the argument - hist_ref, hist_fit, image_intensity... - says float64 not callable (works now apparently)
            optim_result = lbcc.optim_lbcc_linear(hist_ref, hist_fit, image_intensity_bin_edges, init_pars)
            end_time = time.time()
            # record results
            results[date_index, inst_index, ii, 0] = optim_result.x[0]
            results[date_index, inst_index, ii, 1] = optim_result.x[1]
            results[date_index, inst_index, ii, 2] = optim_result.fun
            results[date_index, inst_index, ii, 3] = round(end_time-start_time, 3)
            results[date_index, inst_index, ii, 4] = optim_result.status

        end_time_tot = time.time()
        print("Total elapsed time: " + str(round(end_time_tot - start_time_tot, 3)) + " seconds.")

# save results
np.save(image_out_path + "/cubic_" + time_period, results1)
np.save(image_out_path + "/power-log_" + time_period, results2)
np.save(image_out_path + "/theoretic_" + time_period, results3)
np.save(image_out_path + "/mu-bins_" + time_period, results)


# load results
# results1 = np.load("/Users/turtle/Dropbox/MyNACD/analysis/lbcc_functionals/cubic_6month.npy")
# results2 = np.load("/Users/turtle/Dropbox/MyNACD/analysis/lbcc_functionals/power-log_6month.npy")
# results3 = np.load("/Users/turtle/Dropbox/MyNACD/analysis/lbcc_functionals/theoretic_6month.npy")
# results = np.load("/Users/turtle/Dropbox/MyNACD/analysis/lbcc_functionals/mu-bins_6month.npy")

if gen_plots:

    # generate some plots to compare methods
    sse_index1 = np.array([x == "SSE" for x in optim_vals1])
    npar1 = np.where(sse_index1)[0][0]
    sse_index2 = np.array([x == "SSE" for x in optim_vals2])
    npar2 = np.where(sse_index2)[0][0]
    sse_index3 = np.array([x == "SSE" for x in optim_vals3])
    npar3 = np.where(sse_index3)[0][0]

    # calc beta and y for a few sample mu-values
    results_mu = mu_bin_centers[0:-1]
    sample_mu = [0.125, 0.325, 0.575, 0.875]

    mu_results_index = np.nonzero(np.in1d(results_mu, sample_mu))[0]

    # sample mu colors
    v_cmap = cm.get_cmap('viridis')
    n_mu = len(sample_mu)
    color_dist = np.linspace(0., 1., n_mu)

    linestyles = ['solid', 'dashed', 'dashdot', 'None']
    marker_types = ['None', 'None', 'None', 'x']

    for inst_index, instrument in enumerate(instruments):
        mu_bins_SSE_tots = results[:, inst_index, :, 2].sum(axis=1)
        # plot SSEs for each instrument
        plt.figure(0+inst_index)

        plt.plot(moving_avg_centers, results1[:, inst_index, sse_index1], c="blue", label="cubic")
        plt.plot(moving_avg_centers, results2[:, inst_index, sse_index2], c="red", label="power-log")
        plt.plot(moving_avg_centers, results3[:, inst_index, sse_index3], c="green", label="theoretic")
        plt.plot(moving_avg_centers, mu_bins_SSE_tots, c="black", marker='x', linestyle="None", label="mu-bins")

        # !!!!!!!!!! Stopped Here !!!!!!!!!!!!!!!!!!!!!!!
        # Add mu-bin fits to all plots/legends

        plt.ylabel(str(time_period) + " SSE " + instrument)
        plt.xlabel("Center Date")
        ax = plt.gca()
        ax.legend(loc='upper right', bbox_to_anchor=(1., 1.), title="Model")
        plt.grid()

        plot_fname = image_out_path + instrument + '_SSE_' + year + "-" + time_period + '.pdf'
        plt.savefig(plot_fname)
        plt.close(0+inst_index)

        plot_beta = np.zeros((sample_mu.__len__(), moving_avg_centers.__len__(), 4))
        plot_y = np.zeros((sample_mu.__len__(), moving_avg_centers.__len__(), 4))
        for mu_index, mu in enumerate(sample_mu):
            for date_index, center_date in enumerate(moving_avg_centers):
                plot_beta[mu_index, date_index, 0], plot_y[mu_index, date_index, 0] = \
                    lbcc.get_beta_y_cubic(results1[date_index, inst_index, 0:npar1], mu)
                plot_beta[mu_index, date_index, 1], plot_y[mu_index, date_index, 1] = \
                    lbcc.get_beta_y_power_log(results2[date_index, inst_index, 0:npar2], mu)
                plot_beta[mu_index, date_index, 2], plot_y[mu_index, date_index, 2] = \
                    lbcc.get_beta_y_theoretic_based(results3[date_index, inst_index, 0:npar3], mu)
                plot_beta[mu_index, date_index, 3] = results[date_index, inst_index, mu_results_index[mu_index], 0]
                plot_y[mu_index, date_index, 3] = results[date_index, inst_index, mu_results_index[mu_index], 1]


        # plot beta for the different models as a function of time
        plt.figure(10+inst_index)

        mu_lines = []
        for mu_index, mu in enumerate(sample_mu):
            mu_lines.append(Line2D([0], [0], color=v_cmap(color_dist[mu_index]), lw=2))
            for model_index in range(linestyles.__len__()):
                plt.plot(moving_avg_centers, plot_beta[mu_index, :, model_index], ls=linestyles[model_index],
                         c=v_cmap(color_dist[mu_index]), marker=marker_types[model_index])
        plt.ylabel(r"$\beta$ " + instrument)
        plt.xlabel("Center Date")
        ax = plt.gca()
        model_lines = []
        for model_index in range(linestyles.__len__()):
            model_lines.append(Line2D([0], [0], color="black", linestyle=linestyles[model_index], lw=2,
                                      marker=marker_types[model_index]))
        legend1 = plt.legend(mu_lines, [str(round(x, 3)) for x in sample_mu], loc='upper left', bbox_to_anchor=(1., 1.),
                  title=r"$\mu$ value")
        ax.legend(model_lines, ["cubic", "power/log", "theoretic", r"$\mu$-bins"], loc='upper left',
                  bbox_to_anchor=(1., 0.65), title="model")
        plt.gca().add_artist(legend1)
        # adjust margin to incorporate legend
        plt.subplots_adjust(right=0.8)
        plt.grid()

        plot_fname = image_out_path + instrument + '_beta_' + year + "-" +  time_period + '.pdf'
        plt.savefig(plot_fname)

        plt.close(10+inst_index)


        # plot y for the different models as a function of time
        plt.figure(20 + inst_index)

        mu_lines = []
        for mu_index, mu in enumerate(sample_mu):
            mu_lines.append(Line2D([0], [0], color=v_cmap(color_dist[mu_index]), lw=2))
            for model_index in range(linestyles.__len__()):
                plt.plot(moving_avg_centers, plot_y[mu_index, :, model_index], ls=linestyles[model_index],
                         c=v_cmap(color_dist[mu_index]), marker=marker_types[model_index])
        plt.ylabel(r"$y$ " + instrument)
        plt.xlabel("Center Date")
        ax = plt.gca()
        model_lines = []
        for model_index in range(linestyles.__len__()):
            model_lines.append(Line2D([0], [0], color="black", linestyle=linestyles[model_index], lw=2,
                                      marker=marker_types[model_index]))
        legend1 = plt.legend(mu_lines, [str(round(x, 3)) for x in sample_mu], loc='upper left', bbox_to_anchor=(1., 1.),
                             title=r"$\mu$ value")
        ax.legend(model_lines, ["cubic", "power/log", "theoretic", r"$\mu$-bins"], loc='upper left', bbox_to_anchor=(1., 0.65),
                  title="model")
        plt.gca().add_artist(legend1)
        # adjust margin to incorporate legend
        plt.subplots_adjust(right=0.8)
        plt.grid()

        plot_fname = image_out_path + instrument + '_y_' + year + "-" + time_period + '.pdf'
        plt.savefig(plot_fname)

        plt.close(20 + inst_index)


        # plot some sample beta and y v mu curves

        plt.figure(30 + inst_index)

        beta_y_v_mu = np.zeros((mu_bin_centers.shape[0], 2, 4))
        for index, mu in enumerate(mu_bin_centers):
            beta_y_v_mu[index, :, 0] = lbcc.get_beta_y_cubic(results1[plot_week, inst_index, 0:npar1], mu)
            beta_y_v_mu[index, :, 1] = lbcc.get_beta_y_power_log(results2[plot_week, inst_index, 0:npar2], mu)
            beta_y_v_mu[index, :, 2] = lbcc.get_beta_y_theoretic_based(results3[plot_week, inst_index, 0:npar3], mu)
        beta_y_v_mu[:-1, :, 3] = results[plot_week, inst_index, :, 0:2]

        for model_index in range(linestyles.__len__()):
            if model_index != 3:
                plt.plot(mu_bin_centers, beta_y_v_mu[:, 0, model_index], ls=linestyles[model_index],
                         c=v_cmap(color_dist[model_index-3]), marker=marker_types[model_index])
            else:
                plt.plot(mu_bin_centers[:-1], beta_y_v_mu[:-1, 0, model_index], ls=linestyles[model_index],
                         c=v_cmap(color_dist[model_index-3]), marker=marker_types[model_index])

        plt.ylabel(r"$\beta$ " + instrument)
        plt.xlabel(r"$\mu$")
        plt.title(instrument + " " + title_time_period + " average " + str(moving_avg_centers[plot_week]))
        ax = plt.gca()

        ax.legend(["cubic", "power/log", "theoretic", r"$\mu$-bins"], loc='upper right',
                  bbox_to_anchor=(1., 1.),
                  title="model")
        plt.grid()

        plot_fname = image_out_path + instrument + '_beta_v_mu_' + year + "-" + time_period + '.pdf'
        plt.savefig(plot_fname)

        plt.close(30 + inst_index)


        # repeat for y
        plt.figure(40 + inst_index)

        for model_index in range(linestyles.__len__()):
            if model_index != 3:
                plt.plot(mu_bin_centers, beta_y_v_mu[:, 1, model_index], ls=linestyles[model_index],
                         c=v_cmap(color_dist[model_index - 3]), marker=marker_types[model_index])
            else:
                plt.plot(mu_bin_centers[:-1], beta_y_v_mu[:-1, 1, model_index], ls=linestyles[model_index],
                         c=v_cmap(color_dist[model_index - 3]), marker=marker_types[model_index])

        plt.ylabel(r"$y$ " + instrument)
        plt.xlabel(r"$\mu$")
        plt.title(instrument + " " + title_time_period + " average " + str(moving_avg_centers[plot_week]))
        ax = plt.gca()

        ax.legend(["cubic", "power/log", "theoretic", r"$\mu$-bins"], loc='lower right',
                  bbox_to_anchor=(1., 0.),
                  title="model")
        plt.grid()

        plot_fname = image_out_path + instrument + '_y_v_mu_' + year + "-" + time_period + '.pdf'
        plt.savefig(plot_fname)

        plt.close(40 + inst_index)

        # Finally, take the I_0 distribution and convert it to an estimation of log(Temp)
        # !!!!!!! Stopped working here !!!!!!!!!!!!!!!





