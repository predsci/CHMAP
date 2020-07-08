"""
Use the IIT pipeline functions to calculate the correction
"""

import os
import datetime
import numpy as np
from settings.app import App
from modules.DB_funs import init_db_conn
import modules.DB_funs as db_funcs
import modules.iit_funs as iit
import modules.DB_classes as db_class
import modules.datatypes as psi_d_types
import modules.lbcc_funs as lbcc
import analysis.iit_analysis.IIT_pipeline_funcs as iit_funcs

# TIME RANGE FOR LBC CORRECTION AND IMAGE PLOTTING
lbc_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
lbc_query_time_max = datetime.datetime(2011, 4, 1, 6, 0, 0)

# TIME RANGE FOR FIT PARAMETER CALCULATION
calc_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
calc_query_time_max = datetime.datetime(2011, 10, 1, 0, 0, 0)
weekday = 0 # start at 0 for Monday
number_of_days = 3

# define instruments
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
ref_inst = "AIA"

# declare map and binning parameters
n_mu_bins = 18
n_intensity_bins = 200
R0 = 1.01
log10 = True
lat_band = [- np.pi / 64., np.pi / 64.]

# define database paths
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME

# setup database connection
create = True  # true if you want to add to database
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

###### TESTING FUNCTION ######
moving_avg_centers, moving_width = lbcc.moving_averages(calc_query_time_min, calc_query_time_max, weekday, number_of_days)

# create IIT method
meth_name = "IIT"
meth_desc = "IIT Fit Method"
method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=False)

# get index number of reference instrument
ref_index = inst_list.index(ref_inst)

for date_index, center_date in enumerate(moving_avg_centers):
    print("Begin date " + str(center_date))

    # determine time range based off moving average centers
    min_date = center_date - moving_width / 2
    max_date = center_date + moving_width / 2

    # create arrays for summed histograms and intensity bins
    hist_array = np.zeros((len(inst_list), 200))
    intensity_bin_array = np.zeros((len(inst_list), n_intensity_bins))

    for inst_index, instrument in enumerate(inst_list):
        # query for IIT histograms
        # query the histograms for time range based off moving average centers
        query_instrument = [instrument, ]
        pd_hist = db_funcs.query_hist(db_session=db_session, meth_id=method_id[1], n_intensity_bins=n_intensity_bins,
                                      lat_band=np.array(lat_band).tobytes(),
                                      time_min=np.datetime64(min_date).astype(datetime.datetime),
                                      time_max=np.datetime64(max_date).astype(datetime.datetime),
                                      instrument=query_instrument)
        # convert the binary types back to arrays
        lat_band, intensity_bin_edges, mu_bin_edges, full_hist = psi_d_types.binary_to_hist(hist_binary=pd_hist,
                                                                                            n_mu_bins=None,
                                                                                            n_intensity_bins=
                                                                                            n_intensity_bins)

        # create list of observed dates in time frame
        date_obs_npDT64 = pd_hist['date_obs']

        # determine appropriate date range
        date_ind = (date_obs_npDT64 >= min_date) & (date_obs_npDT64 <= max_date)

        # sum the appropriate histograms
        summed_hist = full_hist[:, date_ind].sum(axis=1)
        hist_array[inst_index, :] = summed_hist
        intensity_bin_array[inst_index, :] = intensity_bin_edges

    # use lbcc function to calculate alpha and x
    # TODO: going to have to do some loop or something to deal with the different instruments
    for instrument, inst_index in enumerate(inst_list):
        hist_ref = hist_array[inst_index, :]
        hist_fit = hist_array[ref_index, :]
        intensity_bin_edges = intensity_bin_array[inst_index, :]
        alpha_x_parameters = iit.optim_iit_linear(hist_ref, hist_fit, intensity_bin_edges,
                                                  init_pars=np.asarray([1., 0.]))
        db_funcs.store_iit_values(db_session, pd_hist, meth_name, meth_desc, alpha_x_parameters, create)




##### STEP ONE: CREATE 1D HISTOGRAMS AND SAVE TO DATABASE ######
# iit_funcs.create_histograms(db_session, inst_list, lbc_query_time_min, lbc_query_time_max, hdf_data_dir,
#                              n_mu_bins=n_mu_bins, n_intensity_bins=n_intensity_bins, lat_band=lat_band,
#                              log10=log10, R0=R0)



# ##### STEP TWO: CALCULATE INTER-INSTRUMENT TRANSFORMATION COEFFICIENTS ######
# iit_funcs.calc_iit_coefficients(db_session, inst_list, ref_inst, calc_query_time_min, number_of_weeks=number_of_weeks,
#                                 number_of_days=number_of_days, n_intensity_bins=n_intensity_bins, lat_band=lat_band,
#                                 create=create)
