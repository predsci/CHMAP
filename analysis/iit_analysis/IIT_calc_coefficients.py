"""
code to calculate IIT correction coefficients and save to database
"""

import os
import time
import datetime
import numpy as np
from settings.app import App
import modules.DB_funs as db_funcs
import modules.iit_funs as iit
import modules.DB_classes as db_class
import modules.datatypes as psi_d_types
import modules.lbcc_funs as lbcc

####### -------- updateable parameters ------ #######

# TIME RANGE FOR FIT PARAMETER CALCULATION
calc_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
calc_query_time_max = datetime.datetime(2011, 10, 1, 0, 0, 0)
weekday = 0  # start at 0 for Monday
number_of_days = 180

# define instruments
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
ref_inst = "AIA"

# declare map and binning parameters
n_mu_bins = 18
n_intensity_bins = 200
R0 = 1.01
log10 = True
lat_band = [-np.pi / 2.4, np.pi / 2.4]

# define database paths
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME

# setup database connection
create = True  # true if you want to add to database
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = db_funcs.init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

###### ------- nothing to update below -------- #######
# start time
start_time = time.time()

# create IIT method
meth_name = "IIT"
meth_desc = "IIT Fit Method"
method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=False)

#### GET REFERENCE INFO FOR LATER USE ####
# get index number of reference instrument
ref_index = inst_list.index(ref_inst)
# query euv images to get carrington rotation range
ref_instrument = [ref_inst, ]
euv_images = db_funcs.query_euv_images(db_session, time_min=calc_query_time_min, time_max=calc_query_time_max,
                                       instrument=ref_instrument)
# get min and max carrington rotation
rot_max = euv_images.cr_rot.max()
rot_min = euv_images.cr_rot.min()

# query histograms
ref_hist_pd = db_funcs.query_hist(db_session=db_session, meth_id=method_id[1],
                                  n_intensity_bins=n_intensity_bins,
                                  lat_band=np.array(lat_band).tobytes(),
                                  time_min=calc_query_time_min - datetime.timedelta(days=number_of_days),
                                  time_max=calc_query_time_max + datetime.timedelta(days=number_of_days),
                                  instrument=ref_instrument)

# convert binary to histogram data
lat_band, mu_bin_edges, intensity_bin_edges, ref_full_hist = psi_d_types.binary_to_hist(hist_binary=ref_hist_pd,
                                                                                        n_mu_bins=None,
                                                                                        n_intensity_bins=
                                                                                        n_intensity_bins)

for inst_index, instrument in enumerate(inst_list):
    # check if this is the reference instrument
    if inst_index == ref_index:
        # calculate the moving average centers
        moving_avg_centers, moving_width = lbcc.moving_averages(calc_query_time_min, calc_query_time_max, weekday,
                                                                number_of_days)
        # loop through moving average centers
        for date_index, center_date in enumerate(moving_avg_centers):
            print("Starting calculations for", instrument, ":", center_date)
            # determine time range based off moving average centers
            min_date = center_date - moving_width / 2
            max_date = center_date + moving_width / 2
            # get the correct date range to use for image combos
            ref_pd_use = ref_hist_pd[(ref_hist_pd['date_obs'] >= str(min_date)) & (
                    ref_hist_pd['date_obs'] <= str(max_date))]

            # save alpha/x as [1, 0] for reference instrument
            alpha = 1
            x = 0
            db_funcs.store_iit_values(db_session, ref_pd_use, meth_name, meth_desc, [alpha, x], create)
    else:
        # query euv_images for correct carrington rotation
        query_instrument = [instrument, ]
        rot_images = db_funcs.query_euv_images_rot(db_session, rot_min=rot_min, rot_max=rot_max,
                                                   instrument=query_instrument)
        # get time minimum and maximum for instrument
        inst_time_min = rot_images.date_obs.min()
        inst_time_max = rot_images.date_obs.max()
        moving_avg_centers, moving_width = lbcc.moving_averages(inst_time_min, inst_time_max, weekday,
                                                                number_of_days)
        inst_hist_pd = db_funcs.query_hist(db_session=db_session, meth_id=method_id[1],
                                           n_intensity_bins=n_intensity_bins,
                                           lat_band=np.array(lat_band).tobytes(),
                                           time_min=inst_time_min,
                                           time_max=inst_time_max,
                                           instrument=query_instrument)
        # convert binary to histogram data
        lat_band, mu_bin_edges, intensity_bin_edges, inst_full_hist = psi_d_types.binary_to_hist(
            hist_binary=inst_hist_pd,
            n_mu_bins=None,
            n_intensity_bins=
            n_intensity_bins)
        # loops through moving average centers
        for date_index, center_date in enumerate(moving_avg_centers):
            print("Starting calculations for", instrument, ":", center_date)
            # determine time range based off moving average centers
            min_date = center_date - moving_width / 2
            max_date = center_date + moving_width / 2
            # get indices for calculation of reference histogram
            ref_hist_ind = np.where(
                (ref_hist_pd['date_obs'] >= str(min_date)) & (ref_hist_pd['date_obs'] <= str(max_date)))
            ref_ind_min = np.min(ref_hist_ind)
            ref_ind_max = np.max(ref_hist_ind)
            ref_hist_use = ref_full_hist[:, ref_ind_min:ref_ind_max]

            # get the correct date range to use for the instrument histogram
            inst_pd_use = inst_hist_pd[
                (inst_hist_pd['date_obs'] >= str(min_date)) & (inst_hist_pd['date_obs'] <= str(max_date))]
            # get indices and histogram for calculation
            inst_hist_ind = np.where(
                (inst_hist_pd['date_obs'] >= str(min_date)) & (inst_hist_pd['date_obs'] <= str(max_date)))
            inst_ind_min = np.min(inst_hist_ind)
            inst_ind_max = np.max(inst_hist_ind)
            inst_hist_use = inst_full_hist[:, inst_ind_min:inst_ind_max]

            # sum histograms
            hist_fit = inst_hist_use.sum(axis=1)
            hist_ref = ref_hist_use.sum(axis=1)

            # get reference/fit peaks
            ref_peak_index = np.argmax(hist_ref)  # index of max value of hist_ref
            ref_peak_val = hist_ref[ref_peak_index]  # max value of hist_ref
            fit_peak_index = np.argmax(hist_fit)  # index of max value of hist_fit
            fit_peak_val = hist_fit[fit_peak_index]  # max value of hist_fit
            # estimate correction coefficients that match fit_peak to ref_peak
            alpha_est = fit_peak_val / ref_peak_val
            x_est = intensity_bin_edges[ref_peak_index] - alpha_est * intensity_bin_edges[fit_peak_index]
            init_pars = np.asarray([alpha_est, x_est], dtype=np.float32)

            # calculate alpha and x
            alpha_x_parameters = iit.optim_iit_linear(hist_ref, hist_fit, intensity_bin_edges,
                                                      init_pars=init_pars)
            # save alpha and x to database
            db_funcs.store_iit_values(db_session, inst_pd_use, meth_name, meth_desc,
                                      alpha_x_parameters.x, create)

end_time = time.time()
tot_time = end_time - start_time
time_tot = str(datetime.timedelta(minutes=tot_time))

print("Inter-instrument transformation fit parameters have been calculated and saved to the database.")
print("Total elapsed time for IIT fit parameter calculation: " + time_tot)
