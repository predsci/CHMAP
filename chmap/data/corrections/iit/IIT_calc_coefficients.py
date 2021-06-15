
"""
code to calculate IIT correction coefficients and save to database
    - Default behavior pre-AIA is to use first 6 months of AIA as the reference
        for Stereo A and B.  A separate script IIT_coefficients_preAIA.py
        deletes all pre-AIA coefficients and inserts pseudo-AIA parameters for
        Stereo A and then fits Stereo B to the adjusted Stereo A.

"""

import os
import time
import datetime
import numpy as np
from sqlalchemy import func

from chmap.settings.app import App
import chmap.database.db_funs as db_funcs
import chmap.data.corrections.iit.iit_utils as iit
import chmap.database.db_classes as db_class
import chmap.utilities.datatypes.datatypes as psi_d_types
import chmap.data.corrections.lbcc.lbcc_utils as lbcc
import chmap.maps.synchronic.synch_utils as synch_utils

####### -------- updateable parameters ------ #######

# TIME RANGE FOR FIT PARAMETER CALCULATION
calc_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
calc_query_time_max = datetime.datetime(2012, 9, 1, 0, 0, 0)

weekday = 0  # start at 0 for Monday
number_of_days = 180
# image window params
image_freq = 2
image_del = np.timedelta64(30, 'm')

# define instruments
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
ref_inst = "AIA"
wavelengths = [193, 195]

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

# setup database parameters
create = True  # true if you want to add to database
# designate which database to connect to
# use_db = "mysql-Q_test"
use_db = "mysql-Q"       # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn_old() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

###### ------- nothing to update below -------- #######
# start time
start_time = time.time()

# connect to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = db_funcs.init_db_conn_old(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
elif use_db in ['mysql-Q', 'mysql-Q_test']:
    # setup database connection to MySQL database on Q
    db_session = db_funcs.init_db_conn_old(db_name=use_db, chd_base=db_class.Base, user=user, password=password)

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
                                       instrument=ref_instrument, wavelength=wavelengths)
# get min and max carrington rotation
rot_max = euv_images.cr_rot.max()
rot_min = euv_images.cr_rot.min()

# calculate the parameter moving average centers
moving_avg_centers, moving_width = lbcc.moving_averages(calc_query_time_min, calc_query_time_max, weekday,
                                                        number_of_days)

# calculate image cadence centers
range_min_date = moving_avg_centers[0] - moving_width/2
range_max_date = moving_avg_centers[-1] + moving_width/2
image_centers = synch_utils.get_dates(
    time_min=range_min_date.astype(datetime.datetime),
    time_max=range_max_date.astype(datetime.datetime), map_freq=image_freq)

# query histograms
ref_hist_pd = db_funcs.query_hist(db_session=db_session, meth_id=method_id[1],
                                  n_intensity_bins=n_intensity_bins, lat_band=lat_band,
                                  time_min=calc_query_time_min - datetime.timedelta(days=number_of_days),
                                  time_max=calc_query_time_max + datetime.timedelta(days=number_of_days),
                                  instrument=ref_instrument, wavelength=wavelengths)
# keep only one observation-histogram per image_center window
keep_ind = lbcc.cadence_choose(ref_hist_pd.date_obs, image_centers, image_del)
ref_hist_pd = ref_hist_pd.iloc[keep_ind]

# convert binary to histogram data
mu_bin_edges, intensity_bin_edges, ref_full_hist = psi_d_types.binary_to_hist(hist_binary=ref_hist_pd,
                                                                              n_mu_bins=None,
                                                                              n_intensity_bins=n_intensity_bins)

# determine date of first AIA image
min_ref_time = db_session.query(func.min(db_class.EUV_Images.date_obs)).filter(
    db_class.EUV_Images.instrument == ref_inst
).all()
base_ref_min    = min_ref_time[0][0]
base_ref_center = base_ref_min + datetime.timedelta(days=number_of_days)/2
base_ref_max    = base_ref_center + datetime.timedelta(days=number_of_days)/2
if (calc_query_time_min - datetime.timedelta(days=7)) < base_ref_center:
    # generate histogram for first year of reference instrument
    ref_base_hist = ref_full_hist[:, (ref_hist_pd['date_obs'] >= str(base_ref_min)) & (
            ref_hist_pd['date_obs'] <= str(base_ref_max))]
else:
    ref_base_hist = None

for inst_index, instrument in enumerate(inst_list):
    # check if this is the reference instrument
    if inst_index == ref_index:
        # calculate the moving average centers
        moving_avg_centers, moving_width = lbcc.moving_averages(calc_query_time_min, calc_query_time_max, weekday,
                                                                number_of_days)
        # loop through moving average centers
        for date_index, center_date in enumerate(moving_avg_centers):
            print("Starting calculations for", instrument, ":", center_date)

            if center_date > ref_hist_pd.date_obs.max() or center_date < ref_hist_pd.date_obs.min():
                print("Date is out of instrument range, skipping.")
                continue

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
                                                   instrument=query_instrument, wavelength=wavelengths)
        if rot_images.shape[0] == 0:
            print("No images in timeframe for ", instrument, ". Skipping")
            continue
        # get time minimum and maximum for instrument
        inst_time_min = rot_images.date_obs.min()
        inst_time_max = rot_images.date_obs.max()
        # if Stereo A or B has images before AIA, calc IIT for those weeks
        if inst_time_min > calc_query_time_min:
            all_images = db_funcs.query_euv_images(db_session, time_min=calc_query_time_min,
                                                   time_max=calc_query_time_max, instrument=query_instrument,
                                                   wavelength=wavelengths)
            if all_images.date_obs.min() < inst_time_min:
                inst_time_min = all_images.date_obs.min()

        moving_avg_centers, moving_width = lbcc.moving_averages(inst_time_min, inst_time_max, weekday,
                                                                number_of_days)
        inst_hist_pd = db_funcs.query_hist(db_session=db_session, meth_id=method_id[1],
                                           n_intensity_bins=n_intensity_bins, lat_band=lat_band,
                                           time_min=inst_time_min - datetime.timedelta(days=number_of_days),
                                           time_max=inst_time_max + datetime.timedelta(days=number_of_days),
                                           instrument=query_instrument, wavelength=wavelengths)
        # keep only one observation-histogram per image_center window
        keep_ind = lbcc.cadence_choose(inst_hist_pd.date_obs, image_centers, image_del)
        inst_hist_pd = inst_hist_pd.iloc[keep_ind]

        # convert binary to histogram data
        mu_bin_edges, intensity_bin_edges, inst_full_hist = psi_d_types.binary_to_hist(
            hist_binary=inst_hist_pd, n_mu_bins=None, n_intensity_bins=n_intensity_bins)
        # loops through moving average centers
        for date_index, center_date in enumerate(moving_avg_centers):
            print("Starting calculations for", instrument, ":", center_date)

            if center_date > inst_hist_pd.date_obs.max() or center_date < inst_hist_pd.date_obs.min():
                print("Date is out of instrument range, skipping.")
                continue

            # determine time range based off moving average centers
            min_date = center_date - moving_width / 2
            max_date = center_date + moving_width / 2
            # get proper time-range of reference histograms
            if center_date <= base_ref_center:
                # if date is earlier than reference (AIA) first year, use reference (AIA) first year
                ref_hist_use = ref_base_hist
            else:
                # get indices for calculation of reference histogram
                ref_hist_ind = (ref_hist_pd['date_obs'] >= str(min_date)) & (ref_hist_pd['date_obs'] <= str(max_date))
                ref_hist_use = ref_full_hist[:, ref_hist_ind]


            # get the correct date range to use for the instrument histogram
            inst_hist_ind = (inst_hist_pd['date_obs'] >= str(min_date)) & (inst_hist_pd['date_obs'] <= str(max_date))
            inst_pd_use = inst_hist_pd[inst_hist_ind]
            # get indices and histogram for calculation
            inst_hist_use = inst_full_hist[:, inst_hist_ind]

            # sum histograms
            hist_fit = inst_hist_use.sum(axis=1)
            hist_ref = ref_hist_use.sum(axis=1)

            # normalize fit histogram
            fit_sum = hist_fit.sum()
            norm_hist_fit = hist_fit / fit_sum

            # normalize reference histogram
            ref_sum = hist_ref.sum()
            norm_hist_ref = hist_ref / ref_sum

            # get reference/fit peaks
            ref_peak_index = np.argmax(norm_hist_ref)  # index of max value of hist_ref
            ref_peak_val = norm_hist_ref[ref_peak_index]  # max value of hist_ref
            fit_peak_index = np.argmax(norm_hist_fit)  # index of max value of hist_fit
            fit_peak_val = norm_hist_fit[fit_peak_index]  # max value of hist_fit
            # estimate correction coefficients that match fit_peak to ref_peak
            alpha_est = fit_peak_val / ref_peak_val
            x_est = intensity_bin_edges[ref_peak_index] - alpha_est * intensity_bin_edges[fit_peak_index]
            init_pars = np.asarray([alpha_est, x_est], dtype=np.float64)

            # calculate alpha and x
            alpha_x_parameters = iit.optim_iit_linear(norm_hist_ref, norm_hist_fit, intensity_bin_edges,
                                                      init_pars=init_pars)
            # save alpha and x to database
            db_funcs.store_iit_values(db_session, inst_pd_use, meth_name, meth_desc,
                                      alpha_x_parameters.x, create)

end_time = time.time()
tot_time = end_time - start_time
time_tot = str(datetime.timedelta(minutes=tot_time))

print("Inter-instrument transformation fit parameters have been calculated and saved to the database.")
print("Total elapsed time for IIT fit parameter calculation: " + time_tot)
