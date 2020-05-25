"""
Generate plots of lbcc theoretic methods
Grabs parameter values from database
"""

import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

from settings.app import App
import modules.lbcc_funs as lbcc
from modules.DB_funs import init_db_conn, query_var_vals
import modules.datatypes as psi_d_types
import modules.DB_classes as db_class

# PLOT PARAMETERS
n_mu_bins = 18
year = "2011" # used for naming plot file
time_period = "6Month" # used for naming plot file
title_time_period = "6 Month" # used for plot titles
plot_week = 5 # index of week you want to plot
# path to save plots to
image_out_path = os.path.join(App.APP_HOME, "test_data", "analysis/lbcc_functionals/")

# TIME FRAME TO QUERY PARAMETERS
query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
query_time_max = datetime.datetime(2011, 10, 1, 0, 0, 0)
number_of_weeks = 27
number_of_days = 180

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
# returns array of moving averages center dates, based off start date and number of weeks
moving_avg_centers = np.array([np.datetime64(str(query_time_min)) + ii*np.timedelta64(1, 'W') for ii in range(number_of_weeks)])
# returns moving width based of number of days
moving_width = np.timedelta64(number_of_days, 'D')
# create mu bin array
mu_bin_array = np.array(range(n_mu_bins + 1), dtype="float") * 0.05 + 0.1
mu_bin_centers = (mu_bin_array[1:] + mu_bin_array[:-1])/2
# optimization values
optim_vals3 = ["a1", "a2", "b1", "b2", "n", "log_alpha", "SSE", "optim_time", "optim_status"]

# generate plots of theoretic fit
sse_index3 = np.array([x == "SSE" for x in optim_vals3])
npar3 = np.where(sse_index3)[0][0]

# calc beta and y for a few sample mu-values
results_mu = mu_bin_centers[0:-1]
sample_mu = [0.125, 0.325, 0.575, 0.875]

mu_results_index = np.nonzero(np.in1d(results_mu, sample_mu))[0]

# sample mu colors
v_cmap = cm.get_cmap('viridis')
n_mu = len(sample_mu)
color_dist = np.linspace(0., 1., n_mu)

linestyles = ['solid', 'dashed', 'dashdot', 'None']
marker_types = ['None', 'None', 'None', 'x']

for inst_index, instrument in enumerate(instruments):
    # query lbcc fit parameters
    # TODO: write function to query parameters
    meth_name = "LBCC"
    var_val_query = query_var_vals(db_session, moving_avg_centers, moving_width, meth_name, instrument)

    # plot SSEs for each instrument
    plt.figure(0+inst_index)

    plt.plot(moving_avg_centers, results3[:, inst_index, sse_index3], c="green", label="theoretic")


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
            plot_beta[mu_index, date_index, 2], plot_y[mu_index, date_index, 2] = \
                lbcc.get_beta_y_theoretic_based(results3[date_index, inst_index, 0:npar3], mu)



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

    plot_fname = image_out_path + instrument + '_beta_' + year + "-" +  time_period + '.pdf'
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
        beta_y_v_mu[index, :, 2] = lbcc.get_beta_y_theoretic_based(results3[plot_week, inst_index, 0:npar3], mu)


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