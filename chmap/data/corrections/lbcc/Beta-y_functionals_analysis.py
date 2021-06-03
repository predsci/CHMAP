"""
Track beta-y functional fits as moving average goes through time
"""

import numpy as np
import datetime
import time
import pandas as pd
import scipy.optimize as optim
import os
import chmap.data.corrections.lbcc.lbcc_utils as lbcc
from chmap.settings.app import App
from chmap.database.db_funs import init_db_conn, query_hist, store_lbcc_values, store_mu_values, store_beta_y_values
import utilities.datatypes.datatypes as psi_d_types
import chmap.database.db_classes as db_class


# HISTOGRAM PARAMETERS TO UPDATE
n_mu_bins = 18 # number of mu bins
n_intensity_bins = 200 # number of intensity bins
lat_band = [- np.pi / 64., np.pi / 64.]

# TIME FRAME TO QUERY HISTOGRAMS
query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
query_time_max = datetime.datetime(2011, 4, 8, 0, 0, 0)
number_of_weeks = 1
number_of_days = 7

# DATABASE PATHS
save_all_to_db = False
store_sse = True
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME
# initialize database connection
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

# ------------ NO NEED TO UPDATE ANYTHING BELOW  ------------- #

instruments = ['AIA', "EUVI-A", "EUVI-B"]
optim_vals_mu = ["Beta", "y", "SSE", "optim_time", "optim_status"]
optim_vals_cubic = ["a1", "a2", "a3", "b1", "b2", "b3", "SSE", "optim_time", "optim_status"]
optim_vals_power = ["a1", "a2", "b1", "SSE", "optim_time", "optim_status"]
optim_vals_theo = ["a1", "a2", "b1", "b2", "n", "log_alpha", "SSE", "optim_time", "optim_status"]

# returns array of moving averages center dates, based off start date and number of weeks
moving_avg_centers = np.array([np.datetime64(str(query_time_min)) + ii*np.timedelta64(1, 'W') for ii in range(number_of_weeks)])

# returns moving width based of number of days
moving_width = np.timedelta64(number_of_days, 'D')

results_mu = np.zeros((len(moving_avg_centers), len(instruments), 17, len(optim_vals_mu)))
results_cubic = np.zeros((len(moving_avg_centers), len(instruments), len(optim_vals_cubic)))
results_power = np.zeros((len(moving_avg_centers), len(instruments), len(optim_vals_power)))
results_theo = np.zeros((len(moving_avg_centers), len(instruments), len(optim_vals_theo)))

for date_index, center_date in enumerate(moving_avg_centers):
    print("Begin date " + str(center_date))

    # determine time range based off moving average centers
    start_time_tot = time.time()
    min_date = center_date - moving_width / 2
    max_date = center_date + moving_width / 2

    for inst_index, instrument in enumerate(instruments):
        print("\nStarting calcs for " + instrument + "\n")

        # query the histograms for time range based off moving average centers
        query_instrument = [instrument, ]
        pd_hist = query_hist(db_session=db_session, n_mu_bins = n_mu_bins, n_intensity_bins = n_intensity_bins, lat_band = np.array(lat_band).tobytes(),
                             time_min = np.datetime64(min_date).astype(datetime.datetime), time_max = np.datetime64(max_date).astype(datetime.datetime),
                             instrument=query_instrument)

        # convert the binary types back to arrays
        lat_band, mu_bin_array, intensity_bin_array, full_hist = psi_d_types.binary_to_hist(pd_hist, n_mu_bins, n_intensity_bins)

        # create list of observed dates in time frame
        date_obs_npDT64 = pd_hist['date_obs']
        date_obs_pdTS = [pd.Timestamp(x) for x in date_obs_npDT64]
        date_obs = [x.to_pydatetime() for x in date_obs_pdTS]

        # creates array of mu bin centers
        mu_bin_centers = (mu_bin_array[1:] + mu_bin_array[:-1])/2

        # creates array of intensity bin centers
        intensity_centers = (intensity_bin_array[:-1] + intensity_bin_array[1:])/2

        # determine appropriate date range
        date_ind = (date_obs_npDT64 >= min_date) & (date_obs_npDT64 <= max_date)

        # sum the appropriate histograms
        summed_hist = full_hist[:, :, date_ind].sum(axis=2)

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

        ##### OPTIMIZATION METHODS ######

        # -- fit the THEORETICAL functional -----------
        model = 3
        init_pars = np.array([-0.05, -0.3, -.01, 0.4, -1., 6.])
        method = "BFGS"

        start3 = time.time()
        optim_out3 = optim.minimize(lbcc.get_functional_sse, init_pars,
                                    args=(hist_ref, hist_mat, mu_vec, intensity_bin_array, model),
                                    method=method)

        end3 = time.time()

        results_theo[date_index, inst_index, 0:6] = optim_out3.x
        results_theo[date_index, inst_index, 6] = optim_out3.fun
        results_theo[date_index, inst_index, 7] = round(end3 - start3, 3)
        results_theo[date_index, inst_index, 8] = optim_out3.status

        ###### STORE RESULT PARAMETERS IN DATABASE #######
        if save_all_to_db:
            meth_name = 'LBCC Theoretic'
            meth_desc = 'LBCC Theoretic Fit Method'
            var_name = "TheoVar"
            var_desc = "Theoretic fit parameter at index "
            store_lbcc_values(db_session, pd_hist, meth_name, meth_desc, var_name, var_desc, date_index,
                              inst_index, optim_vals = optim_vals_theo,  results = results_theo, create=True)


        # -- fit the POWER/LOG functionals -------------
        model = 2
        init_pars = np.array([.93, -0.13, 0.6])
        method = "BFGS"
        gtol = 1e-4

        start2 = time.time()
        optim_out2 = optim.minimize(lbcc.get_functional_sse, init_pars,
                                    args=(hist_ref, hist_mat, mu_vec, intensity_bin_array, model),
                                    method=method, jac='2-point', options={'gtol': gtol})
        end2 = time.time()

        results_power[date_index, inst_index, 0:3] = optim_out2.x
        results_power[date_index, inst_index, 3] = optim_out2.fun
        results_power[date_index, inst_index, 4] = round(end2 - start2, 3)
        results_power[date_index, inst_index, 5] = optim_out2.status

        ###### STORE RESULT PARAMETERS IN DATABASE #######
        if save_all_to_db:
            meth_name = 'LBCC Power'
            meth_desc = 'LBCC Power-Log Fit Method'
            var_name = "PowerVar"
            var_desc = "Power fit parameter at index "
            store_lbcc_values(db_session, pd_hist, meth_name, meth_desc, var_name, var_desc, date_index,
                              inst_index, optim_vals = optim_vals_power,  results = results_power, create=True)


        # -- fit constrained CUBIC functionals -------
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

        start1 = time.time()

        # constrained optimization using SLSQP with numeric Jacobian
        optim_out1 = optim.minimize(lbcc.get_functional_sse, init_pars,
                                    args=(hist_ref, hist_mat, mu_vec, intensity_bin_array, model),
                                    method=method, jac="2-point", constraints=lin_constraint)

        end1 = time.time()

        results_cubic[date_index, inst_index, 0:6] = optim_out1.x
        results_cubic[date_index, inst_index, 6] = optim_out1.fun
        results_cubic[date_index, inst_index, 7] = round(end1-start1, 3)
        results_cubic[date_index, inst_index, 8] = optim_out1.status

        ###### STORE RESULT PARAMETERS IN DATABASE #######
        if save_all_to_db:
            meth_name = 'LBCC Cubic'
            meth_desc = 'LBCC Cubic Fit Method'
            var_name = "CubicVar"
            var_desc = "Cubic fit parameter at index "
            store_lbcc_values(db_session, pd_hist, meth_name, meth_desc, var_name, var_desc, date_index,
                              inst_index, optim_vals = optim_vals_cubic,  results = results_cubic, create=True)


        # -- Do MU-BIN direct calcs of beta and y -----
        ref_peak_index = np.argmax(hist_ref)
        ref_peak_val = hist_ref[ref_peak_index]

        for ii in range(mu_bin_centers.__len__() - 1):
            hist_fit = norm_hist[ii, ]

            # estimate correction coefs that match fit_peak to ref_peak
            fit_peak_index = np.argmax(hist_fit) # index of max value of hist_fit
            fit_peak_val = hist_fit[fit_peak_index] # max value of hist_fit
            beta_est = fit_peak_val/ref_peak_val

            #convert everything from type float64 to float
            fit_peak_val = np.float32(fit_peak_val)
            ref_peak_val = np.float32(ref_peak_val)
            beta_est = np.float32(beta_est)

            y_est = intensity_bin_array[ref_peak_index] - beta_est*intensity_bin_array[fit_peak_index]
            y_est = np.float32(y_est)
            init_pars = np.asarray([beta_est, y_est], dtype=np.float32)
            hist_ref.astype(np.float32)

            # optimize correction coefs
            start_time = time.time()
            optim_result = lbcc.optim_lbcc_linear(hist_ref, hist_fit, intensity_bin_array, init_pars)
            end_time = time.time()

            # record results
            results_mu[date_index, inst_index, ii, 0] = optim_result.x[0]
            results_mu[date_index, inst_index, ii, 1] = optim_result.x[1]
            results_mu[date_index, inst_index, ii, 2] = optim_result.fun
            results_mu[date_index, inst_index, ii, 3] = round(end_time-start_time, 3)
            results_mu[date_index, inst_index, ii, 4] = optim_result.status

            ###### STORE RESULT PARAMETERS IN DATABASE ########
            if save_all_to_db:
                meth_name = 'LBCC Mu Bin'
                meth_desc = 'LBCC Mu Bin Fit Method'
                var_name = "MuBinVar"
                var_desc = "Mu Bins fit parameter at index "
                store_mu_values(db_session, pd_hist, meth_name, meth_desc, var_name, var_desc, date_index,
                                  inst_index, ii, optim_vals = optim_vals_mu,  results = results_mu, create=True)

        ##### SAVE BETA AND Y VALUES TO DATABASE ######
        sse_index1 = np.array([x == "SSE" for x in optim_vals_cubic])
        npar1 = np.where(sse_index1)[0][0]
        sse_index2 = np.array([x == "SSE" for x in optim_vals_power])
        npar2 = np.where(sse_index2)[0][0]
        sse_index3 = np.array([x == "SSE" for x in optim_vals_theo])
        npar3 = np.where(sse_index3)[0][0]
        
        for mu_index, mu in enumerate(mu_bin_centers):
            # cubic parameters
            beta_cubic, y_cubic = lbcc.get_beta_y_cubic(results_cubic[date_index, inst_index, 0:npar1], mu)
            if store_sse:
                sse_cubic = np.float(results_cubic[date_index, inst_index, sse_index1])
                beta_y_cubic = [beta_cubic, y_cubic, sse_cubic]
            else:
                beta_y_cubic = [beta_cubic, y_cubic]
            meth_name = 'LBCC Cubic'
            meth_desc = 'LBCC Cubic Fit Method'
            var_name = "Cubic_"
            var_desc = "Cubic parameter: "
            store_beta_y_values(db_session, pd_hist, meth_name, meth_desc, var_name, var_desc,
                                beta_y_parameters = beta_y_cubic, create=True)

            # power-log parameters
            beta_power_log, y_power_log = lbcc.get_beta_y_power_log(results_power[date_index, inst_index, 0:npar2], mu)
            if store_sse:
                sse_power = np.float(results_power[date_index, inst_index, sse_index2])
                beta_y_power_log = [beta_power_log, y_power_log, sse_power]
            else:
                beta_y_power_log = [beta_power_log, y_power_log]
            meth_name = 'LBCC Power-Log'
            meth_desc = 'LBCC Power-Log Fit Method'
            var_name = "Power-Log_"
            var_desc = "Power-Log Parameter: "
            store_beta_y_values(db_session, pd_hist, meth_name, meth_desc, var_name, var_desc,
                                beta_y_parameters=beta_y_power_log, create=True)

            # theoretic parameters
            beta_theoretic, y_theoretic = lbcc.get_beta_y_theoretic_based(results_theo[date_index, inst_index, 0:npar3], mu)
            if store_sse:
                sse_theoretic = np.float(results_theo[date_index, inst_index, sse_index3])
                beta_y_theoretic = [beta_theoretic, y_theoretic, sse_theoretic]
            else:
                beta_y_theoretic = [beta_theoretic, y_theoretic]
            meth_name = 'LBCC Theoretic'
            meth_desc = 'LBCC Theoretic Fit Method'
            var_name = "Theoretic_"
            var_desc = "Theoretic Parameter: "
            store_beta_y_values(db_session, pd_hist, meth_name, meth_desc, var_name, var_desc,
                                beta_y_parameters=beta_y_theoretic, create=True)

            # mu bin parameters
            for ii in range(mu_bin_centers.__len__() - 1):
                beta_mu = np.float(results_mu[date_index, inst_index, ii, 0])
                y_mu = np.float(results_mu[date_index, inst_index, ii, 1])
                if store_sse:
                    mu_sse = np.float(results_mu[date_index, inst_index, ii, 2])
                    beta_y_mu = [beta_mu, y_mu, mu_sse]
                else:
                    beta_y_mu = [beta_mu, y_mu]
                meth_name = 'LBCC Mu Bin'
                meth_desc = 'LBCC Mu Bin Fit Method'
                var_name = "Mu_"
                var_desc = "Mu parameter: "
                store_beta_y_values(db_session, pd_hist, meth_name, meth_desc, var_name, var_desc,
                                    beta_y_parameters=beta_y_mu, create=True)

        end_time_tot = time.time()
        print("Total elapsed time: " + str(round(end_time_tot - start_time_tot, 3)) + " seconds.")
