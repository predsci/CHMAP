"""
Use the IIT pipeline functions to calculate the correction
"""

import os
import datetime
import numpy as np
from settings.app import App
from database.db_funs import init_db_conn
import database.db_classes as db_class
import chmap.data.corrections.iit.IIT_pipeline_funcs as iit_funcs

####### ------ UPDATABLE PARAMETERS ------ #########
# TIME RANGE FOR LBC CORRECTION AND IIT HISTOGRAM CREATION
lbc_query_time_min = datetime.datetime(2011, 1, 1, 0, 0, 0)
lbc_query_time_max = datetime.datetime(2012, 1, 1, 0, 0, 0)

# TIME RANGE FOR FIT PARAMETER CALCULATION
calc_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
calc_query_time_max = datetime.datetime(2011, 10, 1, 0, 0, 0)
weekday = 0  # start at 0 for Monday
number_of_days = 180  # days for moving average

# TIME RANGE FOR IIT CORRECTION AND IMAGE PLOTTING
iit_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
iit_query_time_max = datetime.datetime(2011, 10, 1, 0, 0, 0)
plot = True  # true if you want to plot resulting images
n_images_plot = 1  # number of images to plot

# TIME RANGE FOR HISTOGRAM CREATION
hist_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
hist_query_time_max = datetime.datetime(2011, 10, 1, 0, 0, 0)

# INSTRUMENTS
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
ref_inst = "AIA"  # reference instrument to fit histograms to
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

# setup database connection
create = True  # true if you want to add to database
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

##### ------ INTER INSTRUMENT TRANSFORMATION FUNCTIONS BELOW ------- ########

##### STEP ONE: CREATE 1D HISTOGRAMS AND SAVE TO DATABASE ######
iit_funcs.create_histograms(db_session, inst_list, lbc_query_time_min, lbc_query_time_max, hdf_data_dir,
                            n_intensity_bins=n_intensity_bins, lat_band=lat_band, log10=log10, R0=R0,
                            wavelengths=wavelengths)

##### STEP TWO: CALCULATE INTER-INSTRUMENT TRANSFORMATION COEFFICIENTS AND SAVE TO DATABASE ######
iit_funcs.calc_iit_coefficients(db_session, inst_list, ref_inst, calc_query_time_min, calc_query_time_max,
                                weekday=weekday, number_of_days=number_of_days, n_intensity_bins=n_intensity_bins,
                                lat_band=lat_band, create=create, wavelengths=wavelengths)

##### STEP THREE: APPLY TRANSFORMATION AND PLOT NEW IMAGES ######
iit_funcs.apply_iit_correction(db_session, hdf_data_dir, iit_query_time_min, iit_query_time_max, inst_list, ref_inst,
                               n_intensity_bins=n_intensity_bins, R0=R0, n_images_plot=n_images_plot, plot=plot)

###### STEP FOUR: GENERATE NEW HISTOGRAM PLOTS ######
iit_funcs.plot_iit_histograms(db_session, hdf_data_dir, hist_query_time_min, hist_query_time_max, inst_list, ref_inst,
                              n_intensity_bins=n_intensity_bins, lat_band=lat_band, R0=R0, log10=log10)
