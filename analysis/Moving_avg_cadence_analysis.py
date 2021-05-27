
"""
Track 2011 histogram changes as moving-average cadence is changed
"""

import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm

import modules.lbcc_utils as lbcc


image_out_path = "/Users/turtle/Dropbox/MyNACD/analysis/move-avg_cadence/"
data_path = "/Users/turtle/GitReps/CHD/test_data/"

optim_vals = ["Beta", "y", "SSE"]

v_cmap = cm.get_cmap('viridis')

# try evaluating the moving average at 2 hour cadence. Then 4, 6, ..., 24 hours; then 1.5, 2, 3, 4, 5, 6, 7 days
skip_steps = list(range(1, 13)) + [18, 24, 36, 48, 60, 72, 84]
hour_cadence = [x * 2 for x in skip_steps]

for instrument in ['AIA', "EUVI-A", "EUVI-B"]:
    print("Calculating LBCC fit parameters for " + instrument + " 2011 as the moving-average cadence varies.")
    # use only the 400 bin files
    hists_400_fname = data_path + "mu-hists-2011_400_" + instrument + ".pkl"

    # start with the 400 bin file
    f = open(hists_400_fname, 'rb')
    hists_400 = pickle.load(f)
    f.close()

    # extract histograms
    full_year_hists = hists_400['all_hists']
    # extract intensity bins
    image_intensity_bin_edges = hists_400['intensity_bin_edges']
    intensity_centers = (image_intensity_bin_edges[:-1] + image_intensity_bin_edges[1:])/2
    # extract mu bins
    mu_bin_edges = hists_400['mu_bin_edges']
    mu_bin_centers = (mu_bin_edges[1:] + mu_bin_edges[:-1])/2
    # clean-up
    del hists_400

    results = np.zeros((len(mu_bin_centers) - 1, len(skip_steps), len(optim_vals)))

    for skip_index, skip in enumerate(skip_steps):
        # using every iith image, sum over the year.
        # Note: this is not perfect. Some images are missing, so we are not getting exactly 2, 4, 6, ... hour steps
        full_year = full_year_hists[:, :, ::skip].sum(axis=2)
        # normalize in mu
        norm_hist = np.full(full_year.shape, 0.)
        row_sums = full_year.sum(axis=1, keepdims=True)
        # but do not divide by zero
        zero_row_index = np.where(row_sums != 0)
        norm_hist[zero_row_index[0]] = full_year[zero_row_index[0]]/row_sums[zero_row_index[0]]

        # determine peak bin and value for reference histogram (mu=1.)
        hist_ref = norm_hist[-1, ]
        ref_peak_index = np.argmax(hist_ref)
        ref_peak_val = hist_ref[ref_peak_index]
        # fit coefficients for each mu_bin
        for ii in range(mu_bin_centers.__len__() - 1):
            hist_fit = norm_hist[ii, ]

            # estimate correction coefs that match fit_peak to ref_peak
            fit_peak_index = np.argmax(hist_fit)
            fit_peak_val = hist_fit[fit_peak_index]
            beta_est = fit_peak_val/ref_peak_val
            y_est = image_intensity_bin_edges[ref_peak_index] - beta_est*image_intensity_bin_edges[fit_peak_index]
            init_pars = np.asarray([beta_est, y_est])

            # optimize correction coefs
            optim_result = lbcc.optim_lbcc_linear(hist_ref, hist_fit, image_intensity_bin_edges, init_pars)
            # record results
            results[ii, skip_index, 0] = optim_result.x[0]
            results[ii, skip_index, 1] = optim_result.x[1]
            results[ii, skip_index, 2] = optim_result.fun

    # save intsrument specific results
    save_object = {'mu_bins': mu_bin_edges, 'hour_cadence': hour_cadence, 'pars': optim_vals, 'result_array': results}
    file_path = image_out_path + 'lbcc-vals_2011_' + instrument + '.pkl'
    print('\nSaving LBCC coefficients to ' + file_path + '\n')
    f = open(file_path, 'wb')
    pickle.dump(save_object, f)
    f.close()


    # add plotting routines here
    n_mu = len(mu_bin_centers)
    color_dist = np.linspace(0., 1., n_mu - 1)

    par_list = ["Beta", "y"]

    for par_name in par_list:
        plt.figure(0)
        par_index = optim_vals.index(par_name)

        # first plot lbcc coeffs v image cadence
        # add lines/markers
        for ii in range(n_mu - 1):
            plt.plot(hour_cadence, results[ii, :, par_index], c=v_cmap(color_dist[ii]), marker='x',
                     label=str(mu_bin_centers[ii].round(decimals=3)))

        plt.xscale('log')
        plt.ylabel(par_name + "(mu)")
        plt.xlabel("Moving AVG Image Interval (hours)")
        plt.title("2011 " + instrument + " Image Interval Analysis")
        ax = plt.gca()
        ax.legend(loc='upper left', bbox_to_anchor=(1., 0., 0.2, 1.0), title="mu bin")
        # adjust margin to incorporate legend
        plt.subplots_adjust(right=0.8)
        plt.grid()

        plot_fname = image_out_path + 'lbcc-' + par_name + '_v_mu_' + instrument + '.pdf'
        plt.savefig(plot_fname)
        plt.close(0)


        # now plot 'convergence rate' of coeffs
        plt.figure(0)
        par_index = optim_vals.index(par_name)
        for ii in range(n_mu - 1):
            par_change = np.divide(np.abs(np.diff(results[ii, :, par_index])), np.divide(hour_cadence[:-1],
                                                                                         hour_cadence[1:]))
            plt.plot(hour_cadence[0:-1], par_change, c=v_cmap(color_dist[ii]), marker='x',
                     label=str(mu_bin_centers[ii].round(decimals=3)))

        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel("Relative Error in " + par_name + "(mu)")
        plt.xlabel("Moving AVG Image Interval (hours)")
        plt.title("2011 " + instrument + " Image Interval Analysis")
        ax = plt.gca()
        ax.legend(loc='upper left', bbox_to_anchor=(1., 0., 0.2, 1.0), title="mu bin")
        # adjust margin to incorporate legend
        plt.subplots_adjust(right=0.8)
        plt.grid()

        plot_fname = image_out_path + 'lbcc-rel-error' + par_name + '_v_mu_' + instrument + '.pdf'
        plt.savefig(plot_fname)
        plt.close(0)



