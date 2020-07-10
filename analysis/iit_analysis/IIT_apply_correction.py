"""
code to apply IIT correction and plot resulting images
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
import modules.Plotting as Plotting

##### ------ updateable parameters ------- #######

# TIME RANGE FOR IIT CORRECTION AND IMAGE PLOTTING
iit_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
iit_query_time_max = datetime.datetime(2011, 4, 1, 6, 0, 0)
plot = True

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

##### ------- nothing to update below --------- #####
meth_name = "IIT"

for inst_index, instrument in enumerate(inst_list):

    #### QUERY IMAGES ####
    query_instrument = [instrument, ]
    image_pd = db_funcs.query_euv_images(db_session=db_session, time_min=iit_query_time_min,
                                         time_max=iit_query_time_max, instrument=query_instrument)
    # apply LBC
    for index, row in image_pd.iterrows():
        lbcc_data = iit_funcs.apply_lbc_correction(db_session, hdf_data_dir, instrument, image_row=row,
                                                   n_mu_bins=n_mu_bins, n_intensity_bins=n_intensity_bins, R0=R0)

        ###### GET VARIABLE VALUES #####
        method_id_info = db_funcs.get_method_id(db_session, meth_name, meth_desc=None, var_names=None, var_descs=None,
                                                create=False)

        alpha_x_parameters = db_funcs.query_var_val(db_session, meth_name, date_obs=lbcc_data.date_obs,
                                                    instrument=instrument)
        alpha, x = alpha_x_parameters

        ##### APPLY IIT TRANSFORMATION ######
        corrected_iit_data = alpha * lbcc_data.lbcc_data + x

        print("Processing image number", row.image_id, ".")
        if row.fname_hdf == "":
            print("Warning: Image # " + str(row.image_id) + " does not have an associated hdf file. Skipping")
            continue
        hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
        original_los = psi_d_types.read_los_image(hdf_path)
        original_los.get_coordinates(R0=1.01)

        if plot:
            Plotting.PlotLBCCImage(lbcc_data.lbcc_data, los_image=original_los, nfig=100 + inst_index * 10 + index,
                                   title="Corrected LBCC Image for " + instrument)
            Plotting.PlotLBCCImage(corrected_iit_data, los_image=original_los, nfig=200 + inst_index * 10 + index,
                                   title="Corrected IIT Image for " + instrument)
            Plotting.PlotLBCCImage(lbcc_data.lbcc_data - corrected_iit_data, los_image=original_los,
                                   nfig=300 + inst_index * 10 + index, title="Difference Plot for " + instrument)
