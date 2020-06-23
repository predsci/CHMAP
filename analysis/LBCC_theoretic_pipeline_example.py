"""
example of LBC theoretic pipeline
calls functions from analysis/lbcc_theoretic_pipeline.py
"""
import os
import datetime
import numpy as np
from settings.app import App
from modules.DB_funs import init_db_conn
import modules.DB_classes as db_class
import analysis.LBCC_theoretic_pipeline as lbcc_funcs

###### ------ PARAMETERS TO UPDATE -------- ########

# TIME RANGE
query_time_min = datetime.datetime(2011, 1, 1, 0, 0, 0)
query_time_max = datetime.datetime(2012, 1, 1, 0, 0, 0)
number_of_weeks = 27
number_of_days = 180

# define instruments
inst_list = ["AIA", "EUVI-A", "EUVI-B"]

# define number of bins
n_mu_bins = 18
n_intensity_bins = 200

# declare map and binning parameters
R0 = 1.01
mu_bin_edges = np.array(range(n_mu_bins + 1), dtype="float") * 0.05 + 0.1
image_intensity_bin_edges = np.linspace(0, 5, num=n_intensity_bins + 1, dtype='float')
log10 = True
lat_band = [- np.pi / 64., np.pi / 64.]

# define database paths
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME

# setup database connection
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

####### STEP ONE: CREATE AND SAVE HISTOGRAMS #######
lbcc_funcs.save_histograms(db_session, hdf_data_dir, inst_list, query_time_min, query_time_max, n_mu_bins=n_mu_bins,
                           n_intensity_bins=n_intensity_bins, lat_band=[-np.pi / 64., np.pi / 64.], log10=True, R0=1.01)

###### STEP TWO: CALCULATE AND SAVE THEORETIC FIT PARAMETERS #######
lbcc_funcs.calc_theoretic_fit(db_session, inst_list, query_time_min, number_of_weeks, number_of_days,
                              n_mu_bins=n_mu_bins,
                              n_intensity_bins=n_intensity_bins, lat_band=[-np.pi / 64., np.pi / 64.], create=True)

###### STEP THREE: APPLY CORRECTION AND PLOT IMAGES #######
lbcc_funcs.apply_lbc_correction(db_session, hdf_data_dir, inst_list, query_time_min, query_time_max,
                                n_mu_bins=n_mu_bins,
                                R0=1.01)
