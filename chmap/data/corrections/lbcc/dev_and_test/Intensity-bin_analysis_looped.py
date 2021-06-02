
"""
Track 2011 histogram changes as bin size is increased.
"""

import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib import cm

import chmap.data.corrections.lbcc.lbcc_utils as lbcc


image_out_path = "/Users/turtle/Dropbox/MyNACD/analysis/intensity_bins/"
data_path = "/Users/turtle/GitReps/CHD/test_data/"

int_bin_n = [1000, 500, 400, 300, 250, 200, 150, 125, 100, 75, 50, 25]
optim_vals = ["Beta", "y", "SSE"]

v_cmap = cm.get_cmap('viridis')

for instrument in ['AIA', "EUVI-A", "EUVI-B"]:

    SterA_1000_fname = data_path + "mu-hists-2011_1000_" + instrument + ".pkl"
    SterA_400_fname = data_path + "mu-hists-2011_400_" + instrument + ".pkl"
    SterA_300_fname = data_path + "mu-hists-2011_300_" + instrument + ".pkl"

    # start with the 400 bin file
    f = open(SterA_400_fname, 'rb')
    SterA_400 = pickle.load(f)
    f.close()

    # generate 1-yr window histogram
    full_year_SterA = SterA_400['all_hists'].sum(axis=2)
    # normalize in mu
    norm_hist = np.full(full_year_SterA.shape, 0.)
    row_sums = full_year_SterA.sum(axis=1, keepdims=True)
    # but do not divide by zero
    zero_row_index = np.where(row_sums != 0)
    norm_hist[zero_row_index[0]] = full_year_SterA[zero_row_index[0]]/row_sums[zero_row_index[0]]

    image_intensity_bin_edges = SterA_400['intensity_bin_edges']
    intensity_centers = (image_intensity_bin_edges[:-1] + image_intensity_bin_edges[1:])/2

    mu_bin_edges = SterA_400['mu_bin_edges']
    mu_bin_centers = (mu_bin_edges[1:] + mu_bin_edges[:-1])/2


    results = np.zeros((len(mu_bin_centers)-1, len(int_bin_n), len(optim_vals)))

    del SterA_400

    results = lbcc.eval_lbcc_4reduced_bins(norm_hist, image_intensity_bin_edges, int_bin_n, mu_bin_centers, results, 5)


    # repeat for 300 bin set
    f = open(SterA_300_fname, 'rb')
    SterA_300 = pickle.load(f)
    f.close()
    # generate 1-yr window histogram
    full_year_SterA = SterA_300['all_hists'].sum(axis=2)
    # normalize in mu
    norm_hist = np.full(full_year_SterA.shape, 0.)
    row_sums = full_year_SterA.sum(axis=1, keepdims=True)
    # but do not divide by zero
    zero_row_index = np.where(row_sums != 0)
    norm_hist[zero_row_index[0]] = full_year_SterA[zero_row_index[0]]/row_sums[zero_row_index[0]]

    image_intensity_bin_edges = SterA_300['intensity_bin_edges']

    del SterA_300

    results = lbcc.eval_lbcc_4reduced_bins(norm_hist, image_intensity_bin_edges, int_bin_n, mu_bin_centers, results, 3)


    # repeat for 1000 bin set
    f = open(SterA_1000_fname, 'rb')
    SterA_1000 = pickle.load(f)
    f.close()
    # generate 1-yr window histogram
    full_year_SterA = SterA_1000['all_hists'].sum(axis=2)
    # normalize in mu
    norm_hist = np.full(full_year_SterA.shape, 0.)
    row_sums = full_year_SterA.sum(axis=1, keepdims=True)
    # but do not divide by zero
    zero_row_index = np.where(row_sums != 0)
    norm_hist[zero_row_index[0]] = full_year_SterA[zero_row_index[0]]/row_sums[zero_row_index[0]]

    image_intensity_bin_edges = SterA_1000['intensity_bin_edges']

    del SterA_1000

    for step in range(1, 5):
        if step == 1:
            hist_ref = norm_hist[-1, ]
            new_bins = image_intensity_bin_edges
        else:
            new_bins = image_intensity_bin_edges[::2**(step-1)]
            hist_ref = lbcc.hist_integration(norm_hist[-1, ], image_intensity_bin_edges, new_bins)

        n_bins = len(new_bins) - 1
        n_bin_index = np.where(np.equal(n_bins, int_bin_n))
        if len(n_bin_index) == 0:
            raise("Number of bins does not match index.")
        ref_peak_index = np.argmax(hist_ref)
        ref_peak_val = hist_ref[ref_peak_index]

        for ii in range(mu_bin_centers.__len__()-1):
            if step == 1:
                hist_fit = norm_hist[ii, ]
            else:
                hist_fit = lbcc.hist_integration(norm_hist[ii, ], image_intensity_bin_edges, new_bins)

            # estimate correction coefs that match fit_peak to ref_peak
            fit_peak_index = np.argmax(hist_fit)
            fit_peak_val = hist_fit[fit_peak_index]
            beta_est = fit_peak_val/ref_peak_val
            y_est = new_bins[ref_peak_index] - beta_est*new_bins[fit_peak_index]
            init_pars = np.asarray([beta_est, y_est])

            # optimize correction coefs
            optim_result = lbcc.optim_lbcc_linear(hist_ref, hist_fit, new_bins, init_pars)
            # record results
            results[ii, n_bin_index, 0] = optim_result.x[0]
            results[ii, n_bin_index, 1] = optim_result.x[1]
            results[ii, n_bin_index, 2] = optim_result.fun

    # save results to a file
    save_ob = {'mu_bins': mu_bin_edges, 'n_int_bins': int_bin_n, 'pars': optim_vals, 'result_array': results}
    file_path = '/Users/turtle/GitReps/CHD/test_data/lbcc-vals_2011_' + instrument + '.pkl'
    print('\nSaving LBCC coefficients to ' + file_path + '\n')
    f = open(file_path, 'wb')
    pickle.dump(save_ob, f)
    f.close()

    # simple plot of normed histogram
    # plt.figure(1)
    #
    # plt.imshow(norm_hist, aspect="auto", interpolation='nearest', origin='low',
    #            extent=[image_intensity_bin_edges[0], image_intensity_bin_edges[-1], mu_bin_edges[0], mu_bin_edges[-1]])
    # plt.xlabel("Pixel log10 intensities")
    # plt.ylabel("mu")
    # plt.title("2D Histogram Data Normalized by mu Bin")

    # test coefficient fitting
    # plt.figure(2)
    # plt.plot(intensity_centers, norm_hist[-1, ], "b")
    # plt.plot(intensity_centers, norm_hist[-3, ], "r")

    # plt.figure(3)
    # plt.imshow(results[:, :, 0], origin='low')

    # To load results rather than calculate:
    # file = open('/Users/turtle/GitReps/CHD/test_data/lbcc-vals_2011_SterA.pkl', 'rb')
    # lbcc_results = pickle.load(file)
    # file.close()
    #
    # results = lbcc_results['result_array']

    n_mu = len(mu_bin_centers)
    color_dist = np.linspace(0., 1., n_mu-1)

    plt.figure(4)
    par_name = "Beta"
    par_index = optim_vals.index(par_name)
    bins_array = np.tile(np.asarray(int_bin_n, order='F'), [n_mu-1, 1])

    # add lines/markers
    for ii in range(n_mu - 1):
        plt.plot(bins_array[ii], results[ii, :, par_index], c=v_cmap(color_dist[ii]), marker='x',
                 label=str(mu_bin_centers[ii].round(decimals=3)))

    plt.ylabel(par_name + "(mu)")
    plt.xlabel("Number of Intensity Bins")
    ax = plt.gca()
    ax.legend(loc='upper left', bbox_to_anchor=(1., 0., 0.2, 1.0), title="mu bin")
    # adjust margin to incorporate legend
    plt.subplots_adjust(right=0.8)
    plt.grid()

    plot_fname = image_out_path + 'lbcc-' + par_name + '_v_mu_' + instrument + '.pdf'
    plt.savefig(plot_fname)
    plt.close(4)


    plt.figure(5)
    par_name = "y"
    par_index = optim_vals.index(par_name)
    bins_array = np.tile(np.asarray(int_bin_n, order='F'), [n_mu-1, 1])

    # add lines/markers
    for ii in range(n_mu-1):
        plt.plot(bins_array[ii], results[ii, :, par_index], c=v_cmap(color_dist[ii]), marker='x',
                 label=str(mu_bin_centers[ii].round(decimals=3)))

    plt.ylabel(par_name + "(mu)")
    plt.xlabel("Number of Intensity Bins")
    ax = plt.gca()
    ax.legend(loc='upper left', bbox_to_anchor=(1., 0., 0.2, 1.0), title="mu bin")
    # adjust margin to incorporate legend
    plt.subplots_adjust(right=0.8)
    plt.grid()

    plot_fname = image_out_path + 'lbcc-' + par_name + '_v_mu_' + instrument + '.pdf'
    plt.savefig(plot_fname)
    plt.close(5)


    plt.figure(6)
    par_name = "Beta"
    par_index = optim_vals.index(par_name)
    for ii in range(n_mu-1):
        par_change = np.divide(np.abs(np.diff(results[ii, :, par_index])), np.divide(int_bin_n[:-1], int_bin_n[1:]))
        plt.plot(int_bin_n[0:-1], par_change, c=v_cmap(color_dist[ii]), marker='x',
                    label=str(mu_bin_centers[ii].round(decimals=3)))

    plt.yscale('log')
    plt.ylabel("Relative Error in " + par_name + "(mu)")
    plt.xlabel("Number of Intensity Bins")
    ax = plt.gca()
    ax.legend(loc='upper left', bbox_to_anchor=(1., 0., 0.2, 1.0), title="mu bin")
    # adjust margin to incorporate legend
    plt.subplots_adjust(right=0.8)
    plt.grid()

    plot_fname = image_out_path + 'lbcc-rel-error' + par_name + '_v_mu_' + instrument + '.pdf'
    plt.savefig(plot_fname)
    plt.close(6)


    plt.figure(7)
    par_name = "y"
    par_index = optim_vals.index(par_name)
    for ii in range(n_mu-1):
        par_change = np.divide(np.abs(np.diff(results[ii, :, par_index])), np.divide(int_bin_n[:-1], int_bin_n[1:]))
        plt.plot(int_bin_n[0:-1], par_change, c=v_cmap(color_dist[ii]), marker='x',
                    label=str(mu_bin_centers[ii].round(decimals=3)))

    plt.yscale('log')
    plt.ylabel("Relative Error in " + par_name + "(mu)")
    plt.xlabel("Number of Intensity Bins")
    ax = plt.gca()
    ax.legend(loc='upper left', bbox_to_anchor=(1., 0., 0.2, 1.0), title="mu bin")
    # adjust margin to incorporate legend
    plt.subplots_adjust(right=0.8)
    plt.grid()

    plot_fname = image_out_path + 'lbcc-rel-error' + par_name + '_v_mu_' + instrument + '.pdf'
    plt.savefig(plot_fname)
    plt.close(7)
