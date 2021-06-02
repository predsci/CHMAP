

import numpy as np
import pickle
import pandas as pd

import matplotlib.pyplot as plt
from scipy import optimize

# load data from 2011 EUVI-A
file = open('/Users/turtle/GitReps/CHD/test_data/lbcc-vals_2011_AIA.pkl', 'rb')
lbcc_results = pickle.load(file)
file.close()

results = lbcc_results['result_array']
mu_bin_edges = lbcc_results['mu_bins']
n_int_bins = lbcc_results['n_int_bins']
optim_bins = lbcc_results['pars']
mu_bin_centers = (mu_bin_edges[1:] + mu_bin_edges[:-1])/2

# save to csv for Eureka curve fitting
out_file2 = "/Users/turtle/Dropbox/MyNACD/analysis/theoretic_lbcc/applied_lbcc_beta_y.csv"
output_csv = pd.DataFrame(data={'mu': mu_bin_centers[:-1], 'beta': results[:, 0, 0], 'y': results[:, 0, 1]})
output_csv.to_csv(out_file2)

if plt.fignum_exists(0):
    plt.close(0)
plt.figure(0)
plt.plot(mu_bin_centers[:-1], results[:, 0, 0], marker='x')
plt.plot(mu_bin_centers[:-1], np.power(results[:, 0, 0], 1/2), 'r', marker='x')
plt.plot(mu_bin_centers[:-1], np.power(results[:, 0, 0], 1/3), 'g', marker='x')
plt.plot(mu_bin_centers[:-1], np.power(results[:, 0, 0], 1/4), 'k', marker='x')
plt.plot(mu_bin_centers[:-1], np.exp(-results[:, 0, 0]), 'y', marker='x')
plt.plot(mu_bin_centers[:-1], np.power(10., -results[:, 0, 0]), 'm', marker='x')
plt.grid()


plt.figure(1)
plt.plot(mu_bin_centers[:-1], results[:, 0, 1], marker='x')

beta = results[:, 0, 0]
mu   = mu_bin_centers[:-1]


def poly_simple(x, a, b, c):
    out = a + b*np.power(x, c)
    return out

def poly_simple_pinned(x, b, c):
    out = 1. + b*(np.power(x, c) - 1.)
    return out

popt, pcov = optimize.curve_fit(poly_simple, mu, beta, bounds=([-100., -2., -4.], [1., 100., 4.]))
test_beta = poly_simple(mu, popt[0], popt[1], popt[2])
plt.plot(mu, test_beta, 'r', marker='x')

popt4, pcov4 = optimize.curve_fit(poly_simple_pinned, mu, beta, bounds=([-2., -4.], [100., 4.]))
test_beta = poly_simple_pinned(mu, popt4[0], popt4[1])
plt.plot(mu, test_beta, 'm', marker='x')


def power_simple(x, a, b, c):
    out = a + b*np.power(c, x)
    return out


def power_simple_pinned(x, b, c):
    out = 1. - b*c + b*np.power(c, x)
    return out

popt2, pcov2 = optimize.curve_fit(power_simple, mu, beta, bounds=([-100., -2., 0.], [1., 100., 20.]))
test_beta2 = power_simple(mu, popt2[0], popt2[1], popt2[2])
plt.plot(mu, test_beta2, 'g', marker='x')


popt3, pcov3 = optimize.curve_fit(power_simple_pinned, mu, beta, bounds=([-2., 0.], [100., 20.]))
test_beta3 = power_simple_pinned(mu, popt3[0], popt3[1])
plt.plot(mu, test_beta3, 'r', marker='x')

