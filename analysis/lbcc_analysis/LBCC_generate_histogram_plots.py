"""
Generate plots of histograms based off histograms grabbed from database
Allows you to plot one histogram or multiple starting from a specific index
"""

import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from settings.app import App
import modules.DB_classes as db_class
import modules.Plotting as Plotting
from modules.DB_funs import init_db_conn, query_hist, get_method_id, query_euv_images, query_inst_combo
import analysis.iit_analysis.IIT_pipeline_funcs as iit_funcs
import modules.datatypes as psi_d_types

# PARAMETERS TO UPDATE

# time frame to query histograms
hist_plot_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
hist_plot_query_time_max = datetime.datetime(2011, 4, 1, 3, 0, 0)
# define instruments
inst_list = ["AIA", "EUVI-A", "EUVI-B"]

# number of histograms to plot
n_hist_plots = 1

# define number of bins
n_mu_bins = 18
n_intensity_bins = 200
log10 = True
R0 = 1.01
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

# ------------ NO NEED TO UPDATE ANYTHING BELOW  ------------- #
# start time
start_time_tot = time.time()

meth_name = 'LBCC'
method_id = get_method_id(db_session, meth_name, meth_desc=None, var_names=None, var_descs=None, create=False)

# mu bin edges and intensity bin edges
mu_bin_edges = np.linspace(0.1, 1.0, n_mu_bins + 1, dtype='float')
intensity_bin_edges = np.linspace(0, 5, num=n_intensity_bins + 1, dtype='float')

### PLOT HISTOGRAMS ###
# query histograms
for instrument in inst_list:
    query_instrument = [instrument, ]
    pd_hist = query_hist(db_session=db_session, meth_id=method_id[1], n_mu_bins=n_mu_bins,
                         n_intensity_bins=n_intensity_bins,
                         lat_band=np.array(lat_band).tobytes(),
                         time_min=hist_plot_query_time_min,
                         time_max=hist_plot_query_time_max,
                         instrument=query_instrument)
    # convert from binary to usable histogram type
    mu_bin_array, intensity_bin_array, full_hist = psi_d_types.binary_to_hist(pd_hist, n_mu_bins,
                                                                                        n_intensity_bins)
    # query correct image combos
    combo_query = query_inst_combo(db_session, hist_plot_query_time_min, hist_plot_query_time_max, meth_name, instrument)
    #### PLOT ORIGINAL HISTOGRAMS ####
    for plot_index in range(n_hist_plots):
        # definitions
        plot_hist = full_hist[:, :, plot_index]
        date_obs = pd_hist.date_obs[plot_index]
        figure = "Original Histogram Plot: "
        # plot histogram
        Plotting.Plot_LBCC_Hists(plot_hist, date_obs, instrument, intensity_bin_edges, mu_bin_edges, figure, plot_index)

        #### APPLY LBC CORRECTION ####
        # query EUV images
        query_instrument = [instrument, ]
        image_pd = query_euv_images(db_session=db_session, time_min=hist_plot_query_time_min,
                                    time_max=hist_plot_query_time_max, instrument=query_instrument)
        for index, row in image_pd.iterrows():
            # apply LBC
            original_los, lbcc_image, mu_indices, use_indices = iit_funcs.apply_lbc_correction(db_session, hdf_data_dir,
                                                                                               combo_query, row,
                                                                                               n_intensity_bins=n_intensity_bins, R0=R0)
            #### CREATE NEW HISTOGRAMS ####
            # perform 2D histogram on mu and image intensity
            hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
            temp_hist = psi_d_types.LosImage.mu_hist(lbcc_image, intensity_bin_edges, mu_bin_edges, lat_band=lat_band,
                                                     log10=log10)
            hist_lbcc = psi_d_types.create_lbcc_hist(hdf_path, row.image_id, method_id[1], mu_bin_edges,
                                                     intensity_bin_edges, lat_band, temp_hist)
            #### PLOT NEW HISTOGRAMS ####
            # definitions
            date_obs = hist_lbcc.date_obs
            plot_hist = hist_lbcc.hist
            figure = "LBCC Histogram Plot: "
            # plot histogram
            Plotting.Plot_LBCC_Hists(plot_hist, date_obs, instrument, intensity_bin_edges, mu_bin_edges, figure, plot_index)


end_time_tot = time.time()
print("Histogram plots of have been generated.")
print("Total elapsed time for plot creation: " + str(round(end_time_tot - start_time_tot, 3)) + " seconds.")
