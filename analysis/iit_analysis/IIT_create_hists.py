"""
code for creation of IIT histograms and saving to database
"""
import os
import time
import datetime
import numpy as np
from settings.app import App
from modules.DB_funs import init_db_conn
import modules.DB_funs as db_funcs
import modules.DB_classes as db_class
import modules.datatypes as psi_d_types
import analysis.iit_analysis.IIT_pipeline_funcs as iit_funcs

###### ------ UPDATEABLE PARAMETERS ------- #######
# TIME RANGE FOR LBC CORRECTION AND HISTOGRAM CREATION
lbc_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
lbc_query_time_max = datetime.datetime(2011, 4, 1, 3, 0, 0)

# define instruments
inst_list = ["AIA", "EUVI-A", "EUVI-B"]

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

###### ------- NOTHING TO UPDATE BELOW ------- #######
start_time_tot = time.time()

# create IIT method
meth_name = "IIT"
meth_desc = "IIT Fit Method"
method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=True)

for instrument in inst_list:
    # query EUV images
    query_instrument = [instrument, ]
    image_pd = db_funcs.query_euv_images(db_session=db_session, time_min=lbc_query_time_min,
                                         time_max=lbc_query_time_max, instrument=query_instrument)
    # apply LBC
    for index, row in image_pd.iterrows():
        lbcc_data, mu_indices = iit_funcs.apply_lbc_correction(db_session, hdf_data_dir, instrument, image_row=row,
                                                               n_intensity_bins=n_intensity_bins, R0=R0)

        # calculate IIT histogram from LBC
        hist = psi_d_types.LBCCImage.iit_hist(lbcc_data, lat_band, log10)

        # create IIT histogram datatype
        iit_hist = psi_d_types.create_iit_hist(lbcc_data, method_id[1], lat_band, hist)

        # add IIT histogram and meta data to database
        db_funcs.add_hist(db_session, iit_hist)

db_session.close()

end_time_tot = time.time()
print("Inter-instrument transformation histograms have been created and saved to the database.")
print(
    "Total elapsed time for histogram creation: " + str(round(end_time_tot - start_time_tot, 3)) + " seconds.")
