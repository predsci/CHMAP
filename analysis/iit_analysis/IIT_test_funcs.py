"""
Use the IIT pipeline functions to calculate the correction
"""

import sys

sys.path.append('/Users/tamarervin/Dropbox/work/CHD')
import os
import time
import datetime
import numpy as np
from settings.app import App
from modules.DB_funs import init_db_conn
import modules.DB_classes as db_class
import analysis.lbcc_analysis.LBCC_theoretic_funcs as lbcc_funcs
import analysis.iit_analysis.iit_pipeline_funcs as iit_funcs


# TIME RANGE FOR LBC CORRECTION AND IMAGE PLOTTING
lbc_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
lbc_query_time_max = datetime.datetime(2011, 4, 1, 3, 0, 0)

# define instruments
inst_list = ["AIA", "EUVI-A", "EUVI-B"]

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
create = True # true if you want to add to database
use_db = "sqlite"

sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

iit_funcs.create_histograms(db_session, inst_list, lbc_query_time_min, lbc_query_time_max, hdf_data_dir, n_mu_bins=18,
                            n_intensity_bins=n_intensity_bins, lat_band=[-np.pi / 64., np.pi / 64.],
                            log10=True, R0=1.01)