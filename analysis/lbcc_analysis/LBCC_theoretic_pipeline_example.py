"""
Example of LBC theoretic pipeline
Calls functions from analysis/lbcc_analysis/lbcc_theoretic_funcs.py
"""

import os
import time
import datetime
import numpy as np
from settings.app import App
import modules.DB_classes as db_class
from modules.DB_funs import init_db_conn
import analysis.lbcc_analysis.LBCC_theoretic_funcs as lbcc_funcs

start_time_tot = time.time()

###### ------ PARAMETERS TO UPDATE -------- ########

# TIME RANGE FOR HISTOGRAM CALCULATION
hist_query_time_min = datetime.datetime(2012, 1, 1, 0, 0, 0)
hist_query_time_max = datetime.datetime(2013, 1, 1, 0, 0, 0)

# TIME RANGE FOR FIT PARAMETER CALCULATION
calc_query_time_min = datetime.datetime(2011, 10, 1, 0, 0, 0)
calc_query_time_max = datetime.datetime(2012, 4, 1, 0, 0, 0)
weekday_calc = 0  # start at 0 for Monday
days = 180  # days for moving average

# TIME RANGE FOR LBC CORRECTION AND IMAGE PLOTTING
lbc_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
lbc_query_time_max = datetime.datetime(2011, 4, 10, 0, 0, 0)
n_images_plot = 1
plot = True  # true if you want images plotted

# TIME RANGE FOR BETA AND Y PLOT GENERATION
plot_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
plot_query_time_max = datetime.datetime(2011, 10, 1, 0, 0, 0)
weekday_plot = 0  # start at 0 for Monday

# TIME RANGE FOR HISTOGRAM PLOTTING
hist_plot_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
hist_plot_query_time_max = datetime.datetime(2011, 4, 1, 3, 0, 0)
n_hist_plots = 1  # number of histograms to plot

# INSTRUMENTS
inst_list = ["AIA", "EUVI-A", "EUVI-B"]

# MAP AND BINNING PARAMETERS
n_mu_bins = 18
n_intensity_bins = 200
R0 = 1.01
log10 = True
lat_band = [- np.pi / 64., np.pi / 64.]

# DATABASE PATHS
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME

# DATABASE CONNECTION
create = True  # true if you want to add to database
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

# STORAGE PATHS AND TITLES FOR BETA/Y PLOTS
year = "2011"  # used for naming plot file
time_period = "6 Month"  # used for plot file and title
plot_week = 5  # index of week you want to plot
image_out_path = os.path.join(App.APP_HOME, "test_data", "analysis/lbcc_functionals/")  # path to save plots to

###### --------- LIMB BRIGHTENING CORRECTIONS FUNCTIONS ------------ ######

####### STEP ONE: CREATE AND SAVE HISTOGRAMS #######
lbcc_funcs.save_histograms(db_session, hdf_data_dir, inst_list, hist_query_time_min, hist_query_time_max, n_mu_bins=18,
                           n_intensity_bins=n_intensity_bins, lat_band=lat_band, log10=log10, R0=R0)

###### STEP TWO: CALCULATE AND SAVE THEORETIC FIT PARAMETERS #######
lbcc_funcs.calc_theoretic_fit(db_session, inst_list, calc_query_time_min, calc_query_time_max, weekday=weekday_calc,
                              number_of_days=days, n_mu_bins=n_mu_bins, n_intensity_bins=n_intensity_bins,
                              lat_band=lat_band, create=create)

end_time_tot = time.time()
tot_time = end_time_tot - start_time_tot
time_test = str(datetime.timedelta(minutes=tot_time))
print("Total elapsed time for Limb-Brightening: " + time_test + " seconds.")

###### STEP THREE: APPLY CORRECTION AND PLOT IMAGES #######
lbcc_funcs.apply_lbc_correction(db_session, hdf_data_dir, inst_list, lbc_query_time_min, lbc_query_time_max,
                                n_intensity_bins=n_intensity_bins, R0=R0, n_images_plot=n_images_plot, plot=plot)

###### STEP FOUR: GENERATE PLOTS OF BETA AND Y ######
lbcc_funcs.generate_theoretic_plots(db_session, inst_list, plot_query_time_min, plot_query_time_max,
                                    weekday=weekday_plot, image_out_path=image_out_path,
                                    year=year, time_period=time_period, plot_week=plot_week, n_mu_bins=n_mu_bins)

###### STEP FIVE: GENERATE HISTOGRAM PLOTS ######
lbcc_funcs.generate_histogram_plots(db_session, hdf_data_dir, inst_list, hist_plot_query_time_min,
                                    hist_plot_query_time_max, n_hist_plots=n_hist_plots, n_mu_bins=n_mu_bins,
                                    n_intensity_bins=n_intensity_bins, lat_band=lat_band, log10=log10, R0=R0)
end_time_tot = time.time()
tot_time = end_time_tot - start_time_tot
time_test = str(datetime.timedelta(minutes=tot_time))
print("Total elapsed time for Limb-Brightening: " + time_test + " seconds.")
