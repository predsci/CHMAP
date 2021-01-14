"""
Track beta-y functional fits
for theoretical fit.
"""
import os
import numpy as np
import datetime
import time
import scipy.optimize as optim

import modules.lbcc_funs as lbcc
from settings.app import App
from modules.DB_funs import init_db_conn, query_hist, get_method_id, store_lbcc_values
import modules.datatypes as psi_d_types
import modules.DB_classes as db_class

# INSTRUMENT LIST
inst_list = ["AIA", "EUVI-A", "EUVI-B"]

# HISTOGRAM PARAMETERS TO UPDATE
n_mu_bins = 18  # number of mu bins
n_intensity_bins = 200  # number of intensity bins
lat_band = [- np.pi / 64., np.pi / 64.]
R0 = 1.01
# define directory paths
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME

# TIME FRAME TO QUERY HISTOGRAMS
query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
query_time_max = datetime.datetime(2011, 4, 2, 0, 0, 0)
weekday = 0
number_of_days = 180

# DATABASE PATHS
create = True  # true if save to database
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME
# designate which database to connect to
use_db = "mysql-Q"       # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.


# ------------ NO NEED TO UPDATE ANYTHING BELOW  ------------- #
start_time_tot = time.time()

# connect to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
elif use_db == 'mysql-Q':
    # setup database connection to MySQL database on Q
    db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user, password=password)

# calculate moving averages
moving_avg_centers, moving_width = lbcc.moving_averages(query_time_min, query_time_max, weekday,
                                                        number_of_days)
optim_vals_theo = ["a1", "a2", "b1", "b2", "n", "log_alpha", "SSE", "optim_time", "optim_status"]
results_theo = np.zeros((len(moving_avg_centers), len(inst_list), len(optim_vals_theo)))

# get method id
meth_name = 'LBCC'
meth_desc = 'LBCC Theoretic Fit Method'
method_id = get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=False)

for date_index, center_date in enumerate(moving_avg_centers):
    print("Begin date " + str(center_date))

    # determine time range based off moving average centers
    min_date = center_date - moving_width / 2
    max_date = center_date + moving_width / 2

    for inst_index, instrument in enumerate(inst_list):
        print("\nStarting calculations for " + instrument + "\n")

        # query the histograms for time range based off moving average centers
        query_instrument = [instrument, ]
        pd_hist = query_hist(db_session=db_session, meth_id=method_id[1], n_mu_bins=n_mu_bins,
                                      n_intensity_bins=n_intensity_bins,
                                      lat_band=lat_band,
                                      time_min=np.datetime64(min_date).astype(datetime.datetime),
                                      time_max=np.datetime64(max_date).astype(datetime.datetime),
                                      instrument=query_instrument)

        # convert the binary types back to arrays
        mu_bin_array, intensity_bin_array, full_hist = psi_d_types.binary_to_hist(pd_hist, n_mu_bins,
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
        hist_ref = norm_hist[-1, ]
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

        meth_name = 'LBCC'
        meth_desc = 'LBCC Theoretic Fit Method'
        var_name = "TheoVar"
        var_desc = "Theoretic fit parameter at index "
        store_lbcc_values(db_session, pd_hist, meth_name, meth_desc, var_name, var_desc, date_index,
                                   inst_index, optim_vals=optim_vals_theo[0:6], results=results_theo, create=create)

        end_time_tot = time.time()
        print("Total elapsed time: " + str(round(end_time_tot - start_time_tot, 3)) + " seconds.")