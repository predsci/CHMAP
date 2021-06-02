
"""
Track 2011 histogram changes as bin size is increased.
    - Functional forms for Beta and y
"""

import numpy as np
import pickle

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

    results = lbcc.eval_lbcc_4reduced_bins(norm_hist, image_intensity_bin_edges, int_bin_n, mu_bin_centers, results, 4)


    # save results to a file
    save_ob = {'mu_bins': mu_bin_edges, 'n_int_bins': int_bin_n, 'pars': optim_vals, 'result_array': results}
    file_path = '/Users/turtle/GitReps/CHD/test_data/lbcc-FunForm_vals_2011_' + instrument + '.pkl'
    print('\nSaving LBCC function parameters to ' + file_path + '\n')
    f = open(file_path, 'wb')
    pickle.dump(save_ob, f)
    f.close()


