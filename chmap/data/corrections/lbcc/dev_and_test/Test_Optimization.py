import pickle
import pandas as pd
import time

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

from chmap.data.corrections.lbcc.lbcc_utils import *


image_out_path = "/Users/turtle/Dropbox/MyNACD/analysis/theoretic_lbcc/"
data_path = "/Users/turtle/GitReps/CHD/test_data/"

int_bin_n = [1000, 500, 400, 300, 250, 200, 150, 125, 100, 75, 50, 25]
optim_vals = ["Beta", "y", "SSE"]

v_cmap = cm.get_cmap('viridis')

instrument = "AIA"

SterA_400_fname = data_path + "mu-hists-2011_400_" + instrument + ".pkl"

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

hist_ref = norm_hist[-1, ]
hist_mat = norm_hist[:-1, ]

mu_vec = mu_bin_centers[:-1]
int_bin_edges = image_intensity_bin_edges


model = 3
init_pars = [-0.05, -0.3, -.01, 0.4, -1., 6.]
init_pars = np.array([-0.03346808, -0.20866697, -0.00685885,  0.35009559, -0.98211564, 5.81369148])
method = "BFGS"

start3 = time.time()
optim_out3 = optim.minimize(get_functional_sse, init_pars, args=(hist_ref, hist_mat, mu_vec, int_bin_edges, model),
                               method=method, jac='2-point')
# optim_out3_test_bfgs = optim.minimize(get_functional_sse, init_pars, args=(hist_ref, hist_mat, mu_vec, int_bin_edges,
#                                                                            model),
#                                       method="CG", jac='2-point')
end3 = time.time()
print("Optimization time for theoretic functional: " + str(round(end3-start3, 3)) + " seconds.")
resulting_pars3 = pd.DataFrame(data={'beta': mu_bin_centers, 'y': mu_bin_centers})
for index, mu in enumerate(mu_bin_centers):
    resulting_pars3.beta[index], resulting_pars3.y[index] = get_beta_y_theoretic_based(optim_out3.x, mu)

model = 2
init_pars = [.93, -0.13, 0.6]
method = "Nelder-Mead"

start2 = time.time()
# optim_out2 = optim.minimize(get_functional_sse, init_pars, args=(hist_ref, hist_mat, mu_vec, int_bin_edges, model),
#                                method=method)
optim_out2_test_bfgs = optim.minimize(get_functional_sse, init_pars, args=(hist_ref, hist_mat, mu_vec, int_bin_edges,
                                                                           model),
                               method="BFGS", jac='2-point', options={'gtol': 5e-3})
# optim_out2_test_cg = optim.minimize(get_functional_sse, init_pars, args=(hist_ref, hist_mat, mu_vec, int_bin_edges,
#                                                                            model),
#                                method="CG", jac='2-point')
end2 = time.time()
print("Optimization time for power/log functional: " + str(round(end2-start2, 3)) + " seconds.")
resulting_pars2 = pd.DataFrame(data={'beta': mu_bin_centers, 'y': mu_bin_centers})
for index, mu in enumerate(mu_bin_centers):
    resulting_pars2.beta[index], resulting_pars2.y[index] = get_beta_y_power_log(optim_out2.x, mu)

model = 1
init_pars = [-1., 1.1, -0.4, 4.7, -5., 2.1]
init_pars = [-2, 2.2, -0.8, 9., -10., 3.9]
method = "Nelder-Mead"

# cubic constraints for first and second derivative
# lb = np.zeros(8)
# ub = np.zeros(8)
# lb[[0, 3, 6, 7]] = -np.inf
# ub[[1, 2, 4, 5]] = np.inf
# A = np.zeros((8, 6))
# A[0, 0] = 1
# A[1, 1] = 1
# A[2, 1:3] = [2, 6]
# A[3, 0:3] = [1, 2, 3]
# A[4, 3] = 1
# A[6, 4] = 1
# A[7, 4:6] = [2, 6]
# A[5, 3:6] = [1, 2, 3]

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
# optim_out1 = optim.minimize(get_functional_sse, init_pars, args=(hist_ref, hist_mat, mu_vec, int_bin_edges, model),
#                                method=method)

start1 = time.time()
# constrained optimization using SLSQP with numeric Jacobian
optim_out1 = optim.minimize(get_functional_sse, init_pars, args=(hist_ref, hist_mat, mu_vec, int_bin_edges, model),
                            method="SLSQP", jac="2-point", constraints=lin_constraint)
end1 = time.time()
print("Optimization time for constrained cubic functional: " + str(round(end1-start1, 3)) + " seconds.")

# constrained optimization using COBYLA
# optim1_cobyla = optim.minimize(get_functional_sse, init_pars, args=(hist_ref, hist_mat, mu_vec, int_bin_edges, model),
#                             method="COBYLA", constraints=lin_constraint)

resulting_pars1 = pd.DataFrame(data={'beta': mu_bin_centers, 'y': mu_bin_centers})
for index, mu in enumerate(mu_bin_centers):
    resulting_pars1.beta[index], resulting_pars1.y[index] = get_beta_y_cubic(optim_out1.x, mu)

start_global1 = time.time()
optim_global1 = optim.basinhopping(get_functional_sse, init_pars, minimizer_kwargs={'args': (hist_ref, hist_mat, mu_vec,
                                                                                    int_bin_edges, model),
                            'method': "SLSQP", 'jac': "2-point", 'constraints': lin_constraint}, niter=50)
end_global1 = time.time()
print("Optimization time for basin-hopping constrained cubic functional: " + str(round(end_global1-start_global1, 3)) +
      " seconds.")


# load data from 2011 EUVI-A
file = open('/Users/turtle/GitReps/CHD/test_data/lbcc-vals_2011_AIA.pkl', 'rb')
lbcc_results = pickle.load(file)
file.close()

results = lbcc_results['result_array']
# mu_bin_edges = lbcc_results['mu_bins']
n_int_bins = lbcc_results['n_int_bins']
optim_bins = lbcc_results['pars']
# mu_bin_centers = (mu_bin_edges[1:] + mu_bin_edges[:-1])/2


# plot the different fitting methods
plt.figure(0)
rc('text', usetex=True)
par_name = "Beta"
par_index = optim_vals.index(par_name)

# add lines/markers
plt.plot(mu_bin_centers[:-1], resulting_pars1.beta[:-1], c="red", label="Cubic/Cubic")
plt.plot(mu_bin_centers[:-1], resulting_pars2.beta[:-1], c="purple", label="Power/Log10")
plt.plot(mu_bin_centers[:-1], resulting_pars3.beta[:-1], c="green", label="Theor/Theor")
plt.plot(mu_bin_centers[:-1], results[:, 0, par_index], c="blue", marker='x',
         linestyle='', label="Bin fits")

plt.ylabel(par_name + "($\mu$)")
plt.xlabel("$\mu$")
ax = plt.gca()
ax.legend(loc='upper right', bbox_to_anchor=(1., 0., 0.0, 1.0))
# adjust margin to incorporate legend
plt.subplots_adjust(right=0.8)
plt.grid()

plot_fname = image_out_path + 'TheoreticAndApplied_lbcc-' + par_name + '_v_mu_' + instrument + '.pdf'
plt.savefig(plot_fname)
plt.close(0)

# plot the different fitting methods
plt.figure(1)
par_name = "y"
par_index = optim_vals.index(par_name)

# add lines/markers
plt.plot(mu_bin_centers[:-1], resulting_pars1.y[:-1], c="red", label="Cubic/Cubic")
plt.plot(mu_bin_centers[:-1], resulting_pars2.y[:-1], c="purple", label="Power/Log10")
plt.plot(mu_bin_centers[:-1], resulting_pars3.y[:-1], c="green", label="Theor/Theor")
plt.plot(mu_bin_centers[:-1], results[:, 0, par_index], c="blue", marker='x',
         linestyle='', label="Bin fits")

plt.ylabel(par_name + "($\mu$)")
plt.xlabel("$\mu$")
ax = plt.gca()
ax.legend(loc='upper left', bbox_to_anchor=(0., 1.))
# adjust margin to incorporate legend
plt.subplots_adjust(right=0.8)
plt.grid()

plot_fname = image_out_path + 'TheoreticAndApplied_lbcc-' + par_name + '_v_mu_' + instrument + '.pdf'
plt.savefig(plot_fname)
plt.close(1)

