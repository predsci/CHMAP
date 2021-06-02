
"""
code to calculate IIT correction coefficients for EUVI-A and EUVI-B prior to AIA coming online
"""

import os
import time
import datetime
import numpy as np
import pandas as pd
from sqlalchemy import func
import pickle

from settings.app import App
import database.db_funs as db_funcs
import chmap.data.corrections.iit.iit_utils as iit
import database.db_classes as db_class
import utilities.datatypes.datatypes as psi_d_types
import chmap.data.corrections.lbcc.lbcc_utils as lbcc

####### -------- updateable parameters ------ #######

# TIME RANGE FOR FIT PARAMETER CALCULATION
# Determined automatically here
# calc_query_time_min = datetime.datetime(2007, 4, 1, 0, 0, 0)
# calc_query_time_max = datetime.datetime(2020, 8, 1, 0, 0, 0)

weekday = 0  # start at 0 for Monday
number_of_days = 180

# define instruments
inst_list = ["EUVI-A", "EUVI-B"]
ref_inst = "EUVI-A"

# path to alpha and x values that transform EUVI-A to pseudo AIA
IIT_pars_file2 = '/Users/turtle/Dropbox/MyNACD/analysis/iit/IIT_pseudo-AIA_pars.pkl'

# choose if Stereo A alpha and x are shifted to earliest AIA fit point
pseudo_shifted = True

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
use_db = "mysql-Q"       # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

###### ------- nothing to update below -------- #######
# start time
start_time = time.time()

# connect to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = db_funcs.init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
elif use_db == 'mysql-Q':
    # setup database connection to MySQL database on Q
    db_session = db_funcs.init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user, password=password)

# determine first AIA LBC combo
aia_combo_query = db_session.query(func.min(db_class.Data_Combos.date_mean),
                                   func.max(db_class.Data_Combos.date_mean))\
    .filter(db_class.Data_Combos.instrument == "AIA", db_class.Data_Combos.meth_id == 1)
aia_combo_date = pd.read_sql(aia_combo_query.statement, db_session.bind)

# determine first EUVI-A image
sterA_query = db_session.query(func.min(db_class.EUV_Images.date_obs))\
    .filter(db_class.EUV_Images.instrument == "EUVI-A")
sterA_min_date = pd.read_sql(sterA_query.statement, db_session.bind)

# load pseudo-AIA IIT pars for EUVI-A
file = open(IIT_pars_file2, 'rb')
iit_dict = pickle.load(file)
file.close()
# determine dates to evaluate
par_dates = iit_dict['moving_avg_centers']
par_dates_index = (par_dates >= sterA_min_date.iloc[0, 0]) &\
                  (par_dates <= aia_combo_date.iloc[0, 0])
moving_avg_centers = par_dates[par_dates_index]
moving_width = datetime.timedelta(days=number_of_days)
# extract pre-calculated alpha and x values
if pseudo_shifted:
    sterA_alpha = iit_dict['pseudo_alpha_shift']
    sterA_x = iit_dict['pseudo_x_shift']
else:
    sterA_alpha = iit_dict['pseudo_alpha_shift']
    sterA_x = iit_dict['pseudo_x_shift']
sterA_alpha = sterA_alpha[par_dates_index]
sterA_x = sterA_x[par_dates_index]

calc_query_time_min = sterA_min_date.iloc[0, 0]
calc_query_time_max = aia_combo_date.iloc[0, 0]

# create IIT method
meth_name = "IIT"
meth_desc = "IIT Fit Method"
method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=False)

# delete any existing IIT parameters in this range and in inst_list
print("Deleting existing IIT parameters for pre-AIA timeframe.")
combo_query = db_session.query(db_class.Data_Combos).filter(
    db_class.Data_Combos.meth_id == method_id[1],
    db_class.Data_Combos.date_mean.between(calc_query_time_min - datetime.timedelta(days=7),
                                            calc_query_time_max),
    db_class.Data_Combos.instrument.in_(inst_list)
)
# combo_query.count()
del_combos = pd.read_sql(combo_query.statement, db_session.bind)
# first, delete variables in Var_Vals table
del_par_query = db_session.query(db_class.Var_Vals).filter(
    db_class.Var_Vals.combo_id.in_(del_combos.combo_id))
num_pars = del_par_query.delete(synchronize_session=False)
db_session.commit()
# second, delete image-combo associations
del_query = db_session.query(db_class.Data_Combo_Assoc).filter(
    db_class.Data_Combo_Assoc.combo_id.in_(del_combos.combo_id))
num_assoc = del_query.delete(synchronize_session=False)
db_session.commit()
# finally, delete the combos
del_combo_query = db_session.query(db_class.Data_Combos).filter(
    db_class.Data_Combos.combo_id.in_(del_combos.combo_id))
num_del = del_combo_query.delete(synchronize_session=False)
db_session.commit()

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
                                  lat_band=lat_band,
                                  time_min=calc_query_time_min - datetime.timedelta(days=number_of_days),
                                  time_max=calc_query_time_max + datetime.timedelta(days=number_of_days),
                                  instrument=ref_instrument)

# convert binary to histogram data
mu_bin_edges, intensity_bin_edges, ref_full_hist = psi_d_types.binary_to_hist(hist_binary=ref_hist_pd,
                                                                              n_mu_bins=None,
                                                                              n_intensity_bins=n_intensity_bins)
# calculate the moving average centers
# moving_avg_centers, moving_width = lbcc.moving_averages(calc_query_time_min, calc_query_time_max, weekday,
#                                                         number_of_days)
# # determine date of first AIA image
# min_ref_time = db_session.query(func.min(db_class.EUV_Images.date_obs)).filter(
#     db_class.EUV_Images.instrument == ref_inst
# ).all()
# base_ref_min    = min_ref_time[0][0]
# base_ref_center = base_ref_min + datetime.timedelta(days=number_of_days)/2
# base_ref_max    = base_ref_center + datetime.timedelta(days=number_of_days)/2
# if calc_query_time_min < base_ref_center:
#     # generate histogram for first year of reference instrument
#     ref_base_hist = ref_full_hist[:, (ref_hist_pd['date_obs'] >= str(base_ref_min)) & (
#             ref_hist_pd['date_obs'] <= str(base_ref_max))]
# else:
#     ref_base_hist = None

for inst_index, instrument in enumerate(inst_list):
    # check if this is the reference instrument
    if inst_index == ref_index:
        # calculate the moving average centers
        # moving_avg_centers, moving_width = lbcc.moving_averages(calc_query_time_min, calc_query_time_max, weekday,
        #                                                         number_of_days)
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
            alpha = sterA_alpha[date_index]
            x = sterA_x[date_index]
            db_funcs.store_iit_values(db_session, ref_pd_use, meth_name, meth_desc, [alpha, x], create)
    else:
        # query euv_images for correct carrington rotation
        query_instrument = [instrument, ]

        rot_images = db_funcs.query_euv_images_rot(db_session, rot_min=rot_min, rot_max=rot_max,
                                                   instrument=query_instrument)
        if rot_images.shape[0] == 0:
            print("No images in timeframe for ", instrument, ". Skipping")
            continue
        # get time minimum and maximum for instrument
        inst_time_min = rot_images.date_obs.min()
        inst_time_max = rot_images.date_obs.max()
        # if Stereo A or B has images before AIA, calc IIT for those weeks
        # if inst_time_min > calc_query_time_min:
        #     all_images = db_funcs.query_euv_images(db_session, time_min=calc_query_time_min,
        #                                            time_max=calc_query_time_max, instrument=query_instrument)
        #     if all_images.date_obs.min() < inst_time_min:
        #         inst_time_min = all_images.date_obs.min()
        #
        # moving_avg_centers, moving_width = lbcc.moving_averages(inst_time_min, inst_time_max, weekday,
        #                                                         number_of_days)
        inst_hist_pd = db_funcs.query_hist(db_session=db_session, meth_id=method_id[1],
                                           n_intensity_bins=n_intensity_bins,
                                           lat_band=lat_band,
                                           time_min=inst_time_min - datetime.timedelta(days=number_of_days),
                                           time_max=inst_time_max + datetime.timedelta(days=number_of_days),
                                           instrument=query_instrument)
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
            # get indices for calculation of reference histogram
            ref_hist_ind = (ref_hist_pd['date_obs'] >= str(min_date)) & (ref_hist_pd['date_obs'] <= str(max_date))
            ref_hist_use = ref_full_hist[:, ref_hist_ind]
            # recover pseudo-AIA alpha and x
            ref_alpha = sterA_alpha[date_index]
            ref_x = sterA_x[date_index]

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
            # use pseudo alpha and x to transform ref_hist
            pseudo_hist_ref = lbcc.LinTrans_1Dhist(norm_hist_ref, intensity_bin_edges, ref_alpha, ref_x)

            # get reference/fit peaks
            ref_peak_index = np.argmax(pseudo_hist_ref)  # index of max value of hist_ref
            ref_peak_val = pseudo_hist_ref[ref_peak_index]  # max value of hist_ref
            fit_peak_index = np.argmax(norm_hist_fit)  # index of max value of hist_fit
            fit_peak_val = norm_hist_fit[fit_peak_index]  # max value of hist_fit
            # estimate correction coefficients that match fit_peak to ref_peak
            alpha_est = fit_peak_val / ref_peak_val
            x_est = intensity_bin_edges[ref_peak_index] - alpha_est * intensity_bin_edges[fit_peak_index]
            init_pars = np.asarray([alpha_est, x_est], dtype=np.float64)

            # calculate alpha and x
            alpha_x_parameters = iit.optim_iit_linear(pseudo_hist_ref, norm_hist_fit, intensity_bin_edges,
                                                      init_pars=init_pars)
            # save alpha and x to database
            db_funcs.store_iit_values(db_session, inst_pd_use, meth_name, meth_desc,
                                      alpha_x_parameters.x, create)

end_time = time.time()
tot_time = end_time - start_time
time_tot = str(datetime.timedelta(minutes=tot_time))

print("Inter-instrument transformation fit parameters have been calculated and saved to the database.")
print("Total elapsed time for IIT fit parameter calculation: " + time_tot)

db_session.close()
