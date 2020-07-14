"""
code to plot histograms before and after IIT correction
"""

import os
import datetime
import numpy as np
import time
import modules.Plotting as Plotting
from settings.app import App
from modules.DB_funs import init_db_conn
import modules.DB_funs as db_funcs
import analysis.iit_analysis.IIT_pipeline_funcs as iit_funcs
import modules.DB_classes as db_class
import modules.datatypes as psi_d_types

####### ------ UPDATABLE PARAMETERS ------ #########
# TIME RANGE FOR HISTOGRAM CREATION
hist_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
hist_query_time_max = datetime.datetime(2011, 4, 1, 3, 0, 0)

# INSTRUMENTS
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
ref_inst = 'AIA'

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

#### ----- GENERATE HISTOGRAMS ---- ####
# start time
start_time_tot = time.time()

# method information
meth_name = "IIT"
ref_index = inst_list.index(ref_inst)
method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc=None, var_names=None, var_descs=None, create=False)

# query for IIT histograms
pd_hist = db_funcs.query_hist(db_session=db_session, meth_id=method_id[1],
                              n_intensity_bins=n_intensity_bins,
                              lat_band=np.array(lat_band).tobytes(),
                              time_min=hist_query_time_min,
                              time_max=hist_query_time_max)

# convert the binary types back to arrays
lat_band, mu_bin_edges, intensity_bin_edges, full_hist = psi_d_types.binary_to_hist(pd_hist, n_mu_bins=None,
                                                                                    n_intensity_bins=
                                                                                    n_intensity_bins)
# create corrected histograms
corrected_hist_list = np.full(full_hist.shape, 0, dtype=np.int64)
for inst_index, instrument in enumerate(inst_list):
    # query EUV images
    query_instrument = [instrument, ]
    image_pd = db_funcs.query_euv_images(db_session=db_session, time_min=hist_query_time_min,
                                         time_max=hist_query_time_max, instrument=query_instrument)
    for index, row in image_pd.iterrows():
        # apply LBC
        start = time.time()
        original_los, lbcc_image, mu_indices, use_indices = iit_funcs.apply_lbc_correction(db_session, hdf_data_dir,
                                                                                           instrument,
                                                                                           image_row=row,
                                                                                           n_intensity_bins=n_intensity_bins,
                                                                                           R0=R0)
        end = time.time()
        print("apply LBC:", end - start)

        #### QUERY IIT CORRECTION COEFFICIENTS ####
        start = time.time()
        alpha, x = db_funcs.query_var_val(db_session, meth_name, date_obs=lbcc_image.date_obs,
                                          instrument=instrument)
        end = time.time()
        print("query var val:", end - start)

        #### CORRECTED DATA ####
        # apply IIT correction
        start = time.time()
        lbcc_data = lbcc_image.lbcc_data
        corrected_iit_data = np.copy(lbcc_data)
        corrected_iit_data[use_indices] = 10 ** (alpha * np.log10(lbcc_data[use_indices]) + x)
        # create IIT datatype
        hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
        iit_image = psi_d_types.create_iit_image(lbcc_image, corrected_iit_data, method_id[1], hdf_path)
        psi_d_types.LosImage.get_coordinates(iit_image, R0=R0)
        end = time.time()
        print("apply IIT:", end - start)

        #### CREATE CORRECTED IIT HISTOGRAM #####
        start = time.time()
        # calculate IIT histogram from LBC
        hist = psi_d_types.IITImage.iit_hist(iit_image, lat_band, log10)
        # create IIT histogram datatype
        corrected_hist = psi_d_types.create_iit_hist(iit_image, method_id[1], lat_band, hist)
        corrected_hist_list[:, inst_index] = corrected_hist.hist
        end = time.time()
        print("calc IIT hist:", end - start)

# plotting definitions
color_list = ['red', 'blue', 'black']
linestyle_list = ['solid', 'dashed', 'dashdot']

#### CREATE NEW HISTOGRAM ####
for inst_index, instrument in enumerate(inst_list):
    Plotting.Plot_IIT_Hists(pd_hist, corrected_hist_list, full_hist, instrument, ref_inst, inst_index, ref_index,
                            intensity_bin_edges, color_list, linestyle_list)
# end time
end_time_tot = time.time()
print("ITT has been applied and original/resulting histograms plotted.")
print("Total elapsed time to apply correction and plot histograms: " + str(round(end_time_tot - start_time_tot, 3))
      + " seconds.")