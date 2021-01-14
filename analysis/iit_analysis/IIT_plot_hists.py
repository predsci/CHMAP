#!/usr/bin/env python
"""
 plot histograms before and after IIT correction
"""
import sys

sys.path.append("/Users/tamarervin/CH_Project/CHD")
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
hist_query_time_min = datetime.datetime(2011, 9, 1, 0, 0, 0)
hist_query_time_max = datetime.datetime(2011, 11, 1, 0, 0, 0)

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
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
# use_db = "mysql-Q"
# user = "tervin"
# password = ""
# sqlite_path = os.path.join(database_dir, sqlite_filename)
# db_session = db_funcs.init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user, password=password)

#### ----- GENERATE HISTOGRAMS ---- ####
# start time
start_time = time.time()

#### GET REFERENCE INFO FOR LATER USE ####
#### GET REFERENCE INFO FOR LATER USE ####
# get index number of reference instrument
ref_index = inst_list.index(ref_inst)
# query euv images to get carrington rotation range
ref_instrument = [ref_inst, ]
euv_images = db_funcs.query_euv_images(db_session, time_min=hist_query_time_min, time_max=hist_query_time_max,
                                       instrument=ref_instrument)
# get min and max carrington rotation
rot_max = euv_images.cr_rot.max()
rot_min = euv_images.cr_rot.min()

# method information
meth_name = "IIT"
method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc=None, var_names=None, var_descs=None,
                                   create=False)

# query for IIT histograms
pd_lbc_hist = db_funcs.query_hist(db_session=db_session, meth_id=method_id[1],
                                  n_intensity_bins=n_intensity_bins,
                                  lat_band=lat_band,
                                  time_min=hist_query_time_min,
                                  time_max=hist_query_time_max)
pd_lbc_hist_srt = pd_lbc_hist.sort_values(by=['image_id'])
# convert the binary types back to arrays
mu_bin_edges, intensity_bin_edges, full_lbc_hist = psi_d_types.binary_to_hist(pd_lbc_hist_srt,
                                                                              n_mu_bins=None,
                                                                              n_intensity_bins=
                                                                              n_intensity_bins)
# create corrected/original histograms
original_hist_list = np.full(full_lbc_hist.shape, 0, dtype=np.int64)
corrected_hist_list = np.full(full_lbc_hist.shape, 0, dtype=np.int64)
for inst_index, instrument in enumerate(inst_list):
    print("Applying corrections for", instrument)
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
    # query correct image combos
    combo_query_lbc = db_funcs.query_inst_combo(db_session, hist_query_time_min, hist_query_time_max,
                                                meth_name="LBCC", instrument=instrument)
    # query correct image combos
    combo_query_iit = db_funcs.query_inst_combo(db_session, hist_query_time_min, hist_query_time_max,
                                                meth_name="IIT",
                                                instrument=instrument)
    for index, row in image_pd.iterrows():
        # apply LBC
        print("Processing image number", row.image_id, "for Histogram Creation.")
        original_los, lbcc_image, mu_indices, use_indices, theoretic_query = lbcc_funcs.apply_lbc(db_session,
                                                                                                  hdf_data_dir,
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
        lbcc_image, iit_image, use_indices, alpha, x = iit_funcs.apply_iit(db_session, combo_query_iit, lbcc_image,
                                                                           use_indices, original_los, R0=R0)

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

    #### ORIGINAL HISTOGRAM #####
    #     # define histogram
    #     original_hist = original_hist_list[:, pd_inst_index].sum(axis=1)
    #     # normalize histogram
    #     row_sums = original_hist.sum(axis=0, keepdims=True)
    #     norm_original_hist = original_hist / row_sums
    #
    #     # plot original
    #     Plotting.Plot1d_Hist(norm_original_hist, instrument, inst_index, intensity_bin_edges, color_list,
    #                          linestyle_list,
    #                          figure=100, xlabel="Intensity (log10)", ylabel="H(I)",
    #                          title="Histogram: Original LOS Data")

    #### LBCC HISTOGRAM #####
    # define histogram
    lbc_hist = full_lbc_hist[:, pd_inst_index].sum(axis=1)
    # normalize histogram
    lbc_sums = lbc_hist.sum(axis=0, keepdims=True)
    norm_lbc_hist = lbc_hist / lbc_sums

    # plot lbcc
    Plotting.Plot1d_Hist(norm_lbc_hist, instrument, inst_index, intensity_bin_edges, color_list, linestyle_list,
                         figure=200, xlabel="Intensity (log10)", ylabel="H(I)", title="Histogram: Post LBCC",
                         save="post_lbc_local")

    #### CORRECTED HISTOGRAM ####
    # define histogram
    corrected_hist = corrected_hist_list[:, pd_inst_index].sum(axis=1)
    # normalize histogram
    iit_sums = corrected_hist.sum(axis=0, keepdims=True)
    norm_corrected_hist = corrected_hist / iit_sums

    # plot corrected
    Plotting.Plot1d_Hist(norm_corrected_hist, instrument, inst_index, intensity_bin_edges, color_list,
                         linestyle_list,
                         figure=300, xlabel="Intensity (log10)", ylabel="H(I)", title="Histogram: Post IIT",
                         save="post_iit_local")
# end time
end_time = time.time()
print("ITT has been applied and original/resulting histograms plotted.")
print("Total elapsed time to apply correction and plot histograms: " + str(round(end_time - start_time, 3))
      + " seconds.")
