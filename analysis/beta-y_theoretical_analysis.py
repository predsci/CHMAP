"""
Track beta-y functional fits
for theoretical fit.
"""

import numpy as np
import datetime
import time
import pandas as pd
import scipy.optimize as optim
import os
import modules.lbcc_funs as lbcc
from settings.app import App
from modules.DB_funs import init_db_conn, query_hist, get_combo_id, add_combo_image_assoc, get_method_id, get_var_id, \
    get_var_val, store_lbcc_values, store_mu_values, store_beta_y_values
import modules.datatypes as psi_d_types
import modules.DB_classes as db_class


# HISTOGRAM PARAMETERS TO UPDATE
n_mu_bins = 18  # number of mu bins
n_intensity_bins = 200  # number of intensity bins
lat_band = [- np.pi / 64., np.pi / 64.]

# TIME FRAME TO QUERY HISTOGRAMS
query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
query_time_max = datetime.datetime(2011, 4, 3, 0, 0, 0)
number_of_weeks = 1
number_of_days = 2

# DATABASE PATHS
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME
# initialize database connection
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

# ------------ NO NEED TO UPDATE ANYTHING BELOW  ------------- #

instruments = ['AIA', "EUVI-A", "EUVI-B"]
optim_vals_theo = ["a1", "a2", "b1", "b2", "n", "log_alpha", "SSE", "optim_time", "optim_status"]

# returns array of moving averages center dates, based off start date and number of weeks
moving_avg_centers = np.array(
    [np.datetime64(str(query_time_min)) + ii * np.timedelta64(1, 'W') for ii in range(number_of_weeks)])

# returns moving width based of number of days
moving_width = np.timedelta64(number_of_days, 'D')

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
        pd_hist = query_hist(db_session=db_session, n_mu_bins=n_mu_bins, n_intensity_bins=n_intensity_bins,
                             lat_band=np.array(lat_band).tobytes(),
                             time_min=np.datetime64(min_date).astype(datetime.datetime),
                             time_max=np.datetime64(max_date).astype(datetime.datetime),
                             instrument=query_instrument)

        # convert the binary types back to arrays
        lat_band, mu_bin_array, intensity_bin_array, full_hist = psi_d_types.binary_to_hist(pd_hist, n_mu_bins,
                                                                                            n_intensity_bins)

        # create list of observed dates in time frame
        date_obs_npDT64 = pd_hist['date_obs']
        date_obs_pdTS = [pd.Timestamp(x) for x in date_obs_npDT64]
        date_obs = [x.to_pydatetime() for x in date_obs_pdTS]

        # creates array of mu bin centers
        mu_bin_centers = (mu_bin_array[1:] + mu_bin_array[:-1]) / 2

        # creates array of intensity bin centers
        intensity_centers = (intensity_bin_array[:-1] + intensity_bin_array[1:]) / 2

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

        # theoretic parameters
        sse_index_theo = np.array([x == "SSE" for x in optim_vals_theo])
        npar_theo = np.where(sse_index_theo)[0][0]

        # get beta and y from theoretic fit
        beta_y_theoretic = lbcc.get_beta_y_theoretic_matrix(results_theo[date_index, inst_index, 0:npar_theo],
                                                                      mu_bin_centers)

        # save to database
        meth_name = 'LBCC Theoretic'
        meth_desc = 'LBCC Theoretic Fit Method'
        var_name = "Theoretic_"
        var_desc = "Theoretic Parameter: "
        for var_index, var_value in enumerate(beta_y_theoretic):
            store_beta_y_values(db_session, pd_hist, meth_name, meth_desc, var_name, var_desc, var_index, mu_value = mu_bin_centers[var_index],
                          beta_y_parameters=var_value, create=True)


        end_time_tot = time.time()
        print("Total elapsed time: " + str(round(end_time_tot - start_time_tot, 3)) + " seconds.")

