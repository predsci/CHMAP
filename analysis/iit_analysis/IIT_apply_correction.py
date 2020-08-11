"""
code to apply IIT correction and plot resulting images
"""

import os
import datetime
import time
import numpy as np
from settings.app import App
from modules.DB_funs import init_db_conn
import modules.DB_funs as db_funcs
import modules.DB_classes as db_class
import analysis.lbcc_analysis.LBCC_theoretic_funcs as lbcc_funcs
import analysis.iit_analysis.IIT_pipeline_funcs as iit_funcs
import modules.Plotting as Plotting

##### ------ updateable parameters ------- #######

# TIME RANGE FOR IIT CORRECTION AND IMAGE PLOTTING
iit_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
iit_query_time_max = datetime.datetime(2011, 10, 1, 0, 0, 0)
plot = True
n_images_plot = 1

# define instruments
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
ref_inst = "AIA"

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
# start time
start_time = time.time()

#### GET REFERENCE INFO FOR LATER USE ####
# get index number of reference instrument
ref_index = inst_list.index(ref_inst)
# query euv images to get carrington rotation range
ref_instrument = [ref_inst, ]
euv_images = db_funcs.query_euv_images(db_session, time_min=iit_query_time_min, time_max=iit_query_time_max,
                                       instrument=ref_instrument)
# get min and max carrington rotation
rot_max = euv_images.cr_rot.max()
rot_min = euv_images.cr_rot.min()

for inst_index, instrument in enumerate(inst_list):
    #### QUERY IMAGES ####
    query_instrument = [instrument, ]
    rot_images = db_funcs.query_euv_images_rot(db_session, rot_min=rot_min, rot_max=rot_max,
                                               instrument=query_instrument)
    image_pd = rot_images.sort_values(by=['cr_rot'])
    # get time minimum and maximum for instrument
    inst_time_min = rot_images.date_obs.min()
    inst_time_max = rot_images.date_obs.max()
    # query correct image combos
    lbc_meth_name = "LBCC"
    combo_query_lbc = db_funcs.query_inst_combo(db_session, inst_time_min, inst_time_max, lbc_meth_name,
                                                instrument)
    iit_meth_name = "IIT"
    combo_query_iit = db_funcs.query_inst_combo(db_session, inst_time_min, inst_time_max, iit_meth_name,
                                                instrument)
    # apply LBC
    for index in range(n_images_plot):
        row = image_pd.iloc[index]
        #### APPLY LBC CORRECTION #####
        original_los, lbcc_image, mu_indices, use_indices,theoretic_query = lbcc_funcs.apply_lbc(db_session, hdf_data_dir,
                                                                                 combo_query_lbc,
                                                                                 image_row=row,
                                                                                 n_intensity_bins=n_intensity_bins,
                                                                                 R0=R0)
        #### APPLY IIT CORRECTION ####
        lbcc_image, iit_image, use_indices, alpha, x = iit_funcs.apply_iit(db_session, hdf_data_dir, combo_query_iit,
                                                                 lbcc_image, use_indices, image_row=row, R0=R0)

        if plot:
            lbcc_data = lbcc_image.lbcc_data
            corrected_iit_data = iit_image.iit_data
            # plot LBC image
            Plotting.PlotCorrectedImage(lbcc_data, los_image=original_los, nfig=100 + inst_index * 10 + index,
                                        title="Corrected LBCC Image for " + instrument)
            # plot IIT image
            Plotting.PlotCorrectedImage(corrected_iit_data, los_image=original_los,
                                        nfig=200 + inst_index * 10 + index,
                                        title="Corrected IIT Image for " + instrument)
            # plot difference
            Plotting.PlotCorrectedImage(lbcc_data - corrected_iit_data, los_image=original_los,
                                        nfig=300 + inst_index * 10 + index,
                                        title="Difference Plot for " + instrument)

# end time
end_time = time.time()
print("ITT has been applied and specified images plotted.")
print("Total elapsed time to apply correction and plot: " + str(round(end_time - start_time, 3))
      + " seconds.")