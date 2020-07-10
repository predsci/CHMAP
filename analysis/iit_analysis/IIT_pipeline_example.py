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

####### ------ UPDATABLE PARAMETERS ------ #########
# TIME RANGE FOR LBC CORRECTION AND IMAGE PLOTTING
lbc_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
lbc_query_time_max = datetime.datetime(2011, 4, 30, 0, 0, 0)

# TIME RANGE FOR FIT PARAMETER CALCULATION
calc_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
calc_query_time_max = datetime.datetime(2011, 10, 1, 0, 0, 0)
weekday = 0  # start at 0 for Monday
number_of_days = 3 # days for moving average

# TIME RANGE FOR IIT CORRECTION AND IMAGE PLOTTING
iit_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
iit_query_time_max = datetime.datetime(2011, 4, 1, 6, 0, 0)
plot = True # true if you want to plot resulting images

# INSTRUMENTS
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
ref_inst = "AIA" # reference instrument to fit histograms to

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
db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)


##### ------ INTER INSTRUMENT TRANSFORMATION FUNCTIONS BELOW ------- ########

##### STEP ONE: CREATE 1D HISTOGRAMS AND SAVE TO DATABASE ######
iit_funcs.create_histograms(db_session, inst_list, lbc_query_time_min, lbc_query_time_max, hdf_data_dir,
                            n_mu_bins=n_mu_bins, n_intensity_bins=n_intensity_bins, lat_band=lat_band,
                            log10=log10, R0=R0)

# ##### STEP TWO: CALCULATE INTER-INSTRUMENT TRANSFORMATION COEFFICIENTS ######
iit_funcs.calc_iit_coefficients(db_session, inst_list, ref_inst, calc_query_time_min, calc_query_time_max, weekday,
                                number_of_days=number_of_days, n_intensity_bins=n_intensity_bins, lat_band=lat_band,
                                create=create)
