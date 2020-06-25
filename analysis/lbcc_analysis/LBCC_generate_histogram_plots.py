"""
Generate plots of histograms based off histograms grabbed from database
Allows you to plot one histogram or multiple starting from a specific index
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from settings.app import App
import modules.DB_classes as db_class
from modules.DB_funs import init_db_conn, query_hist
import modules.datatypes as psi_d_types

# PARAMETERS TO UPDATE

# time frame to query histograms
query_time_min = datetime.datetime(2011, 1, 4, 0, 0, 0)
query_time_max = datetime.datetime(2011, 1, 7, 0, 0, 0)
# define instruments
inst_list = ["AIA", "EUVI-A", "EUVI-B"]

# number of histograms to plot
# if plotting one histogram choose index to plot
plot_index = 0  # 0 will plot first histogram in time frame
# true if want to plot more than one histogram
plot_plus = True
# starts looping histograms from index zero
n_hist_plots = 2

# define number of bins
n_mu_bins = 18
n_intensity_bins = 200
lat_band = [- np.pi / 64., np.pi / 64.]
# DATABASE PATHS
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME

# setup database connection
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

# declare map and binning parameters
R0 = 1.01
mu_bin_edges = np.array(range(n_mu_bins + 1), dtype="float") * 0.05 + 0.1
image_intensity_bin_edges = np.linspace(0, 5, num=n_intensity_bins + 1, dtype='float')

# ------------ NO NEED TO UPDATE ANYTHING BELOW  ------------- #

### PLOT HISTOGRAMS ###
# query histograms
for instrument in inst_list:
    query_instrument = [instrument, ]
    pd_hist = query_hist(db_session=db_session, n_mu_bins=n_mu_bins, n_intensity_bins=n_intensity_bins,
                         lat_band=np.array(lat_band).tobytes(), time_min=query_time_min, time_max=query_time_max,
                         instrument=query_instrument)
    # convert from binary to usable histogram type
    lat_band, mu_bin_array, intensity_bin_array, full_hist = psi_d_types.binary_to_hist(pd_hist, n_mu_bins,
                                                                                        n_intensity_bins)

    if plot_plus:
        for plot_index in range(n_hist_plots):
            plot_hist = full_hist[:, :, plot_index]
            date_obs = pd_hist.date_obs[plot_index]

            # # simple plot of raw histogram
            plt.figure(instrument + " Plot: " + str(1 + 2 * plot_index))
            # this will make the plot show up
            plt.imshow(plot_hist, aspect="auto", interpolation='nearest', origin='low',
                       extent=[image_intensity_bin_edges[0], image_intensity_bin_edges[-2] + 1., mu_bin_edges[0],
                               mu_bin_edges[-1]])
            plt.xlabel("Pixel intensities")
            plt.ylabel("mu")
            plt.title("Raw 2D Histogram Data for Histogram: \n" + "Instrument: " + instrument + " \n " + str(date_obs))

            # # Normalize each mu bin
            norm_hist = np.full(plot_hist.shape, 0.)
            row_sums = plot_hist.sum(axis=1, keepdims=True)
            # but do not divide by zero
            zero_row_index = np.where(row_sums != 0)
            norm_hist[zero_row_index[0]] = plot_hist[zero_row_index[0]] / row_sums[zero_row_index[0]]

            # # simple plot of normed histogram
            plt.figure(instrument + " Plot: " + str(2 + 2 * plot_index))
            # this will make the plot show up
            plt.imshow(norm_hist, aspect="auto", interpolation='nearest', origin='low',
                       extent=[image_intensity_bin_edges[0], image_intensity_bin_edges[-1], mu_bin_edges[0],
                               mu_bin_edges[-1]])
            plt.xlabel("Pixel intensities")
            plt.ylabel("mu")
            plt.title(
                "2D Histogram Data Normalized by mu Bin: \n" + "Instrument: " + instrument + " \n " + str(date_obs))

    else:
        # plot histogram at specific index
        plot_hist = full_hist[:, :, plot_index]
        date_obs = pd_hist.date_obs[plot_index]

        # # simple plot of raw histogram
        plt.figure(instrument + " Plot: " + str(1 + 2 * plot_index))
        plt.imshow(plot_hist, aspect="auto", interpolation='nearest', origin='low',
                   extent=[image_intensity_bin_edges[0], image_intensity_bin_edges[-2] + 1., mu_bin_edges[0],
                           mu_bin_edges[-1]])
        plt.xlabel("Pixel intensities")
        plt.ylabel("mu")
        plt.title("Raw 2D Histogram Data for Histogram: \n" + "Instrument: " + instrument + " \n " + str(date_obs))

        # # Normalize each mu bin
        norm_hist = np.full(plot_hist.shape, 0.)
        row_sums = plot_hist.sum(axis=1, keepdims=True)
        # but do not divide by zero
        zero_row_index = np.where(row_sums != 0)
        norm_hist[zero_row_index[0]] = plot_hist[zero_row_index[0]] / row_sums[zero_row_index[0]]

        # # simple plot of normed histogram
        plt.figure(instrument + " Plot: " + str(2 + 2 * plot_index))
        plt.imshow(norm_hist, aspect="auto", interpolation='nearest', origin='low',
                   extent=[image_intensity_bin_edges[0], image_intensity_bin_edges[-1], mu_bin_edges[0],
                           mu_bin_edges[-1]])
        plt.xlabel("Pixel intensities")
        plt.ylabel("mu")
        plt.title("2D Histogram Data Normalized by mu Bin: \n" + "Instrument: " + instrument + " \n " + str(date_obs))
