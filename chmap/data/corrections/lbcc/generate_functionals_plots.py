"""
Generate plots of lbcc functionals methods
Grabs parameter values from database
"""
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

from chmap.settings.app import App
from chmap.database.db_funs import init_db_conn, query_var_val
import chmap.database.db_classes as db_class

# PLOT PARAMETERS
n_mu_bins = 18
n_intensity_bins = 200
lat_band = [-np.pi / 64., np.pi / 64.]
year = "2011" # used for naming plot file
time_period = "Test" # used for naming plot file
title_time_period = "Test" # used for plot titles
plot_week = 1 # index of week you want to plot
# path to save plots to
image_out_path = os.path.join(App.APP_HOME, "test_data", "analysis/lbcc_functionals/")

# TIME FRAME TO QUERY HISTOGRAMS
query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
query_time_max = datetime.datetime(2011, 4, 8, 0, 0, 0)
number_of_weeks = 1
number_of_days = 7

# INITIALIZE DATABASE CONNECTION
# DATABASE PATHS
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME
# initialize database connection
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

# ------------ NO NEED TO UPDATE ANYTHING BELOW  ------------- #

# basic info
instruments = ['AIA', "EUVI-A", "EUVI-B"]
# create mu bin array
mu_bin_array = np.array(range(n_mu_bins + 1), dtype="float") * 0.05 + 0.1
mu_bin_centers = (mu_bin_array[1:] + mu_bin_array[:-1])/2

# optimization values
optim_vals_mu = ["Beta", "y", "SSE", "optim_time", "optim_status"]
optim_vals_cubic = ["a1", "a2", "a3", "b1", "b2", "b3", "SSE", "optim_time", "optim_status"]
optim_vals_power = ["a1", "a2", "b1", "SSE", "optim_time", "optim_status"]
optim_vals_theo = ["a1", "a2", "b1", "b2", "n", "log_alpha", "SSE", "optim_time", "optim_status"]

# time arrays
# returns array of moving averages center dates, based off start date and number of weeks
moving_avg_centers = np.array([np.datetime64(str(query_time_min)) + ii*np.timedelta64(1, 'W') for ii in range(number_of_weeks)])
# returns moving width based of number of days
moving_width = np.timedelta64(number_of_days, 'D')

# generate some plots to compare methods
sse_index1 = np.array([x == "SSE" for x in optim_vals_cubic])
npar1 = np.where(sse_index1)[0][0]
sse_index2 = np.array([x == "SSE" for x in optim_vals_power])
npar2 = np.where(sse_index2)[0][0]
sse_index3 = np.array([x == "SSE" for x in optim_vals_theo])
npar3 = np.where(sse_index3)[0][0]

# calc beta and y for a few sample mu-values
mu_results = mu_bin_centers[0:-1]
sample_mu = [0.125, 0.325, 0.575, 0.875]

mu_results_index = np.nonzero(np.in1d(mu_results, sample_mu))[0]

# sample mu colors
v_cmap = cm.get_cmap('viridis')
n_mu = len(sample_mu)
color_dist = np.linspace(0., 1., n_mu)

linestyles = ['solid', 'dashed', 'dashdot', 'None']
marker_types = ['None', 'None', 'None', 'x']

for inst_index, instrument in enumerate(instruments):

    # get variable values for each image combination
    for date_ind, center_date in enumerate(moving_avg_centers):
        cubic_query = query_var_val(db_session, n_mu_bins, n_intensity_bins, lat_band, center_date, moving_width, meth_name = "LBCC Cubic", instrument = instrument)
        power_log_query = query_var_val(db_session, n_mu_bins, n_intensity_bins, lat_band, center_date, moving_width, meth_name = "LBCC Power-Log", instrument = instrument)
        theoretic_query = query_var_val(db_session, n_mu_bins, n_intensity_bins, lat_band, center_date, moving_width, meth_name = "LBCC Theoretic", instrument = instrument)
        mu_bin_query = query_var_val(db_session, n_mu_bins, n_intensity_bins, lat_band, center_date, moving_width, meth_name = "LBCC Mu Bin", instrument = instrument)

    # plot SSEs for each instrument
    plt.figure(0 + inst_index)

    plt.plot(moving_avg_centers, cubic_query[:, inst_index, sse_index1], c="blue", label="cubic")
    plt.plot(moving_avg_centers, power_log_query[:, inst_index, sse_index2], c="red", label="power-log")
    plt.plot(moving_avg_centers, theoretic_query[:, inst_index, sse_index3], c="green", label="theoretic")
    plt.plot(moving_avg_centers, mu_bin_query[:, inst_index, :, 2].sum(axis=1), c="black", marker='x', linestyle="None", label="mu-bins")


    # Add mu-bin fits to all plots/legends

    plt.ylabel(str(time_period) + " SSE " + instrument)
    plt.xlabel("Center Date")
    ax = plt.gca()
    ax.legend(loc='upper right', bbox_to_anchor=(1., 1.), title="Model")
    plt.grid()

    plot_fname = image_out_path + instrument + '_SSE_' + year + "-" + time_period + '.pdf'
    plt.savefig(plot_fname)
    plt.close(0+inst_index)

    plot_beta = np.zeros((sample_mu.__len__(), moving_avg_centers.__len__(), 4))
    plot_y = np.zeros((sample_mu.__len__(), moving_avg_centers.__len__(), 4))
    for mu_index, mu in enumerate(sample_mu):
        for date_index, center_date in enumerate(moving_avg_centers):
            plot_beta[mu_index, date_index, 0], plot_y[mu_index, date_index, 0] = cubic_query
            plot_beta[mu_index, date_index, 1], plot_y[mu_index, date_index, 1] = power_log_query
            plot_beta[mu_index, date_index, 2], plot_y[mu_index, date_index, 2] = theoretic_query
            plot_beta[mu_index, date_index, 3], plot_y[mu_index, date_index, 3] = mu_bin_query

    # plot beta for the different models as a function of time
    plt.figure(10+inst_index)

    mu_lines = []
    for mu_index, mu in enumerate(sample_mu):
        mu_lines.append(Line2D([0], [0], color=v_cmap(color_dist[mu_index]), lw=2))
        for model_index in range(linestyles.__len__()):
            plt.plot(moving_avg_centers, plot_beta[mu_index, :, model_index], ls=linestyles[model_index],
                     c=v_cmap(color_dist[mu_index]), marker=marker_types[model_index])
    plt.ylabel(r"$\beta$ " + instrument)
    plt.xlabel("Center Date")
    ax = plt.gca()
    model_lines = []
    for model_index in range(linestyles.__len__()):
        model_lines.append(Line2D([0], [0], color="black", linestyle=linestyles[model_index], lw=2,
                                  marker=marker_types[model_index]))
    legend1 = plt.legend(mu_lines, [str(round(x, 3)) for x in sample_mu], loc='upper left', bbox_to_anchor=(1., 1.),
              title=r"$\mu$ value")
    ax.legend(model_lines, ["cubic", "power/log", "theoretic", r"$\mu$-bins"], loc='upper left',
              bbox_to_anchor=(1., 0.65), title="model")
    plt.gca().add_artist(legend1)
    # adjust margin to incorporate legend
    plt.subplots_adjust(right=0.8)
    plt.grid()

    plot_fname = image_out_path + instrument + '_beta_' + year + "-" + time_period + '.pdf'
    plt.savefig(plot_fname)

    plt.close(10+inst_index)


    # plot y for the different models as a function of time
    plt.figure(20 + inst_index)

    mu_lines = []
    for mu_index, mu in enumerate(sample_mu):
        mu_lines.append(Line2D([0], [0], color=v_cmap(color_dist[mu_index]), lw=2))
        for model_index in range(linestyles.__len__()):
            plt.plot(moving_avg_centers, plot_y[mu_index, :, model_index], ls=linestyles[model_index],
                     c=v_cmap(color_dist[mu_index]), marker=marker_types[model_index])
    plt.ylabel(r"$y$ " + instrument)
    plt.xlabel("Center Date")
    ax = plt.gca()
    model_lines = []
    for model_index in range(linestyles.__len__()):
        model_lines.append(Line2D([0], [0], color="black", linestyle=linestyles[model_index], lw=2,
                                  marker=marker_types[model_index]))
    legend1 = plt.legend(mu_lines, [str(round(x, 3)) for x in sample_mu], loc='upper left', bbox_to_anchor=(1., 1.),
                         title=r"$\mu$ value")
    ax.legend(model_lines, ["cubic", "power/log", "theoretic", r"$\mu$-bins"], loc='upper left', bbox_to_anchor=(1., 0.65),
              title="model")
    plt.gca().add_artist(legend1)
    # adjust margin to incorporate legend
    plt.subplots_adjust(right=0.8)
    plt.grid()

    plot_fname = image_out_path + instrument + '_y_' + year + "-" + time_period + '.pdf'
    plt.savefig(plot_fname)

    plt.close(20 + inst_index)


    # plot some sample beta and y v mu curves

    plt.figure(30 + inst_index)

    beta_y_v_mu = np.zeros((mu_bin_centers.shape[0], 2, 4))
    for index, mu in enumerate(mu_bin_centers):
        beta_y_v_mu[index, :, 0] = cubic_query[plot_week]
        beta_y_v_mu[index, :, 1] = power_log_query[plot_week]
        beta_y_v_mu[index, :, 2] = theoretic_query[plot_week]
    beta_y_v_mu[:-1, :, 3] = mu_bin_query[plot_week]

    for model_index in range(linestyles.__len__()):
        if model_index != 3:
            plt.plot(mu_bin_centers, beta_y_v_mu[:, 0, model_index], ls=linestyles[model_index],
                     c=v_cmap(color_dist[model_index-3]), marker=marker_types[model_index])
        else:
            plt.plot(mu_bin_centers[:-1], beta_y_v_mu[:-1, 0, model_index], ls=linestyles[model_index],
                     c=v_cmap(color_dist[model_index-3]), marker=marker_types[model_index])

    plt.ylabel(r"$\beta$ " + instrument)
    plt.xlabel(r"$\mu$")
    plt.title(instrument + " " + title_time_period + " average " + str(moving_avg_centers[plot_week]))
    ax = plt.gca()

    ax.legend(["cubic", "power/log", "theoretic", r"$\mu$-bins"], loc='upper right',
              bbox_to_anchor=(1., 1.),
              title="model")
    plt.grid()

    plot_fname = image_out_path + instrument + '_beta_v_mu_' + year + "-" + time_period + '.pdf'
    plt.savefig(plot_fname)

    plt.close(30 + inst_index)


    # repeat for y
    plt.figure(40 + inst_index)

    for model_index in range(linestyles.__len__()):
        if model_index != 3:
            plt.plot(mu_bin_centers, beta_y_v_mu[:, 1, model_index], ls=linestyles[model_index],
                     c=v_cmap(color_dist[model_index - 3]), marker=marker_types[model_index])
        else:
            plt.plot(mu_bin_centers[:-1], beta_y_v_mu[:-1, 1, model_index], ls=linestyles[model_index],
                     c=v_cmap(color_dist[model_index - 3]), marker=marker_types[model_index])

    plt.ylabel(r"$y$ " + instrument)
    plt.xlabel(r"$\mu$")
    plt.title(instrument + " " + title_time_period + " average " + str(moving_avg_centers[plot_week]))
    ax = plt.gca()

    ax.legend(["cubic", "power/log", "theoretic", r"$\mu$-bins"], loc='lower right',
              bbox_to_anchor=(1., 0.),
              title="model")
    plt.grid()

    plot_fname = image_out_path + instrument + '_y_v_mu_' + year + "-" + time_period + '.pdf'
    plt.savefig(plot_fname)

    plt.close(40 + inst_index)