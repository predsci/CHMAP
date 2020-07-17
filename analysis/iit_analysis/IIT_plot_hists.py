"""
 plot histograms before and after IIT correction
"""

import os
import datetime
import numpy as np
import time
from settings.app import App
from modules.DB_funs import init_db_conn
import modules.DB_funs as db_funcs
import modules.DB_classes as db_class
import modules.datatypes as psi_d_types
import modules.Plotting as Plotting
import analysis.lbcc_analysis.LBCC_theoretic_funcs as lbcc_funcs
import analysis.iit_analysis.IIT_pipeline_funcs as iit_funcs

####### ------ UPDATABLE PARAMETERS ------ #########
# TIME RANGE FOR HISTOGRAM CREATION
hist_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
hist_query_time_max = datetime.datetime(2011, 10, 1, 0, 0, 0)

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
start_time = time.time()

# method information
meth_name = "IIT"
ref_index = inst_list.index(ref_inst)
method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc=None, var_names=None, var_descs=None, create=False)

# query for IIT histograms
pd_lbc_hist = db_funcs.query_hist(db_session=db_session, meth_id=method_id[1],
                              n_intensity_bins=n_intensity_bins,
                              lat_band=np.array(lat_band).tobytes(),
                              time_min=hist_query_time_min,
                              time_max=hist_query_time_max)
pd_lbc_hist_srt = pd_lbc_hist.sort_values(by=['image_id'])
# convert the binary types back to arrays
lat_band, mu_bin_edges, intensity_bin_edges, full_lbc_hist = psi_d_types.binary_to_hist(pd_lbc_hist_srt, n_mu_bins=None,
                                                                                    n_intensity_bins=
                                                                                    n_intensity_bins)
# create corrected/original histograms
original_hist_list = np.full(full_lbc_hist.shape, 0, dtype=np.int64)
corrected_hist_list = np.full(full_lbc_hist.shape, 0, dtype=np.int64)
for inst_index, instrument in enumerate(inst_list):
    print("Applying corrections for", instrument)
    # query EUV images
    query_instrument = [instrument, ]
    image_pd = db_funcs.query_euv_images(db_session=db_session, time_min=hist_query_time_min,
                                         time_max=hist_query_time_max, instrument=query_instrument)
    # query correct image combos
    combo_query_lbc = db_funcs.query_inst_combo(db_session, hist_query_time_min, hist_query_time_max,
                                                meth_name="LBCC Theoretic", instrument=instrument)
    # query correct image combos
    combo_query_iit = db_funcs.query_inst_combo(db_session, hist_query_time_min, hist_query_time_max, meth_name="IIT",
                                                instrument=instrument)
    for index, row in image_pd.iterrows():
        # apply LBC
        original_los, lbcc_image, mu_indices, use_indices = lbcc_funcs.apply_lbc(db_session, hdf_data_dir,
                                                                                 combo_query_lbc,
                                                                                 image_row=row,
                                                                                 n_intensity_bins=n_intensity_bins,
                                                                                 R0=R0)

        #### ORIGINAL LOS DATA ####
        # calculate IIT histogram from original data
        original_los_hist = psi_d_types.LosImage.iit_hist(original_los, intensity_bin_edges, lat_band, log10)
        # add 1D histogram to array
        original_hist_list[:, index] = original_los_hist

        #### CORRECTED DATA ####
        # apply IIT correction
        lbcc_image, iit_image, use_indices = iit_funcs.apply_iit(db_session, hdf_data_dir, combo_query_iit, lbcc_image,
                                                       use_indices, image_row=row, R0=R0)

        #### CREATE CORRECTED IIT HISTOGRAM #####
        # calculate IIT histogram from LBC
        hist_iit = psi_d_types.IITImage.iit_hist(iit_image, lat_band, log10)
        # create IIT histogram datatype
        corrected_hist = psi_d_types.create_iit_hist(iit_image, method_id[1], lat_band, hist_iit)
        corrected_hist_list[:, index] = corrected_hist.hist

# plotting definitions
color_list = ['red', 'blue', 'black']
linestyle_list = ['solid', 'dashed', 'dashdot']

#### CREATE NEW HISTOGRAM ####
for inst_index, instrument in enumerate(inst_list):
    print("Plotting Histograms for", instrument)
    #### GET INDICES TO USE ####
    # get index of instrument in histogram dataframe
    hist_inst = pd_lbc_hist_srt['instrument']
    pd_inst_index = hist_inst[hist_inst == instrument].index
    # get index of reference instrument in histogram dataframe
    pd_ref_index = hist_inst[hist_inst == ref_inst].index

    #### ORIGINAL HISTOGRAM #####
    # define histograms
    original_hist = original_hist_list[:, pd_inst_index].sum(axis=1)
    ref_hist = original_hist_list[:, pd_ref_index].sum(axis=1)
    # normalize histogram
    norm_original_hist = original_hist / np.max(ref_hist)

    # plot original
    Plotting.Plot1d_Hist(norm_original_hist, instrument, inst_index, intensity_bin_edges, color_list, linestyle_list, figure=100,
                xlabel="Intensity (log10)", ylabel="H(I)", title="Histogram: Original Image")

    #### LBCC HISTOGRAM #####
    # define histograms
    lbc_hist = full_lbc_hist[:, pd_inst_index].sum(axis=1)
    ref_lbc_hist = full_lbc_hist[:, pd_ref_index].sum(axis=1)
    # normalize histogram
    norm_lbc_hist = lbc_hist / np.max(ref_lbc_hist)

    # plot lbcc
    Plotting.Plot1d_Hist(norm_lbc_hist, instrument, inst_index, intensity_bin_edges, color_list, linestyle_list,
                         figure=200, xlabel="Intensity (log10)", ylabel="H(I)", title="Histogram: Post LBCC")

    #### CORRECTED HISTOGRAM ####
    # define histograms
    corrected_hist = corrected_hist_list[:, pd_inst_index].sum(axis=1)
    ref_corrected_hist = corrected_hist_list[:, pd_ref_index].sum(axis=1)
    # normalize histogram
    norm_corrected_hist = corrected_hist / np.max(ref_corrected_hist)

    # plot corrected
    Plotting.Plot1d_Hist(norm_corrected_hist, instrument, inst_index, intensity_bin_edges, color_list, linestyle_list,
                         figure=300, xlabel="Intensity (log10)", ylabel="H(I)", title="Histogram: Post IIT")

# end time
end_time = time.time()
print("ITT has been applied and original/resulting histograms plotted.")
print("Total elapsed time to apply correction and plot histograms: " + str(round(end_time - start_time, 3))
      + " seconds.")
