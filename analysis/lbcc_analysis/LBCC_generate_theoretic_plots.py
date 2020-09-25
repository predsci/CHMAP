"""
Generate beta and y plots of lbcc theoretic method for each instrument
- beta and y over time
- beta and y v. mu
Queries parameter values from database
"""

import os
import time
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

from settings.app import App
from modules.DB_funs import init_db_conn, query_var_val, query_inst_combo
import modules.DB_classes as db_class
import modules.lbcc_funs as lbcc

# INSTRUMENTS
inst_list = ['AIA', "EUVI-A", "EUVI-B"]

# PLOT PARAMETERS
n_mu_bins = 18
year = "2011"  # used for naming plot file
time_period = "Theoretic-6Month(2)"  # used for naming plot file
title_time_period = "6 Month"  # used for plot titles
plot_week = 5  # index of week you want to plot
# path to save plots to
image_out_path = os.path.join(App.APP_HOME, "test_data", "analysis/lbcc_functionals/")

# TIME FRAME TO QUERY HISTOGRAMS
query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
query_time_max = datetime.datetime(2012, 10, 1, 0, 0, 0)
weekday = 0
number_of_days = 180

# INITIALIZE DATABASE CONNECTION
# DATABASE PATHS
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
# initialize database connection
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

# ------------ NO NEED TO UPDATE ANYTHING BELOW  ------------- #

# start time
start_time_tot = time.time()

# create mu bin array
mu_bin_array = np.array(range(n_mu_bins + 1), dtype="float") * 0.05 + 0.1
mu_bin_centers = (mu_bin_array[1:] + mu_bin_array[:-1]) / 2

# time arrays
# returns array of moving averages center dates, based off start and end date
moving_avg_centers, moving_width = lbcc.moving_averages(query_time_min, query_time_max, weekday,
                                                        days=None)

# calc beta and y for a few sample mu-values
sample_mu = [0.125, 0.325, 0.575, 0.875]

# sample mu colors
v_cmap = cm.get_cmap('viridis')
n_mu = len(sample_mu)
color_dist = np.linspace(0., 1., n_mu)

linestyles = ['dashed']
marker_types = ['None']
meth_name = 'LBCC'

for inst_index, instrument in enumerate(inst_list):
    print("Generating plots for " + instrument + ".")
    # query theoretic parameters
    theoretic_query = np.zeros((len(moving_avg_centers), 6))
    # query correct image combos
    combo_query = query_inst_combo(db_session, query_time_min, query_time_max, meth_name,
                                   instrument)
    plot_beta = np.zeros((sample_mu.__len__(), moving_avg_centers.__len__()))
    plot_y = np.zeros((sample_mu.__len__(), moving_avg_centers.__len__()))
    for mu_index, mu in enumerate(sample_mu):
        for date_index, center_date in enumerate(moving_avg_centers):
            # query for variable value
            theoretic_query[date_index, :] = query_var_val(db_session, meth_name,
                                                           date_obs=np.datetime64(center_date).astype(
                                                               datetime.datetime),
                                                           inst_combo_query=combo_query)
            plot_beta[mu_index, date_index], plot_y[mu_index, date_index] = lbcc.get_beta_y_theoretic_based(
                theoretic_query[date_index, :], mu)
    #### BETA AND Y AS FUNCTION OF TIME ####
    # plot beta for the different models as a function of time
    plt.figure(10 + inst_index)

    mu_lines = []
    for mu_index, mu in enumerate(sample_mu):
        plt.plot(moving_avg_centers, plot_beta[mu_index, :], ls=linestyles[0],
                 c=v_cmap(color_dist[mu_index]), marker=marker_types[0])
    plt.ylabel(r"$\beta$ " + instrument)
    plt.xlabel("Center Date")
    ax = plt.gca()
    model_lines = [Line2D([0], [0], color="black", linestyle=linestyles[0], lw=2,
                          marker=marker_types[0])]
    legend1 = plt.legend(mu_lines, [str(round(x, 3)) for x in sample_mu], loc='upper left', bbox_to_anchor=(1., 1.),
                         title=r"$\mu$ value")
    ax.legend(model_lines, ["theoretic"], loc='upper left',
              bbox_to_anchor=(1., 0.65), title="model")
    plt.gca().add_artist(legend1)
    # adjust margin to incorporate legend
    plt.subplots_adjust(right=0.8)
    plt.grid()

    plot_fname = image_out_path + instrument + '_beta_' + year + "-" + time_period.replace(" ", "") + '.pdf'
    plt.savefig(plot_fname)

    plt.close(10 + inst_index)

    # plot y for the different models as a function of time
    plt.figure(20 + inst_index)

    mu_lines = []
    for mu_index, mu in enumerate(sample_mu):
        mu_lines.append(Line2D([0], [0], color=v_cmap(color_dist[mu_index]), lw=2))
        plt.plot(moving_avg_centers, plot_y[mu_index, :], ls=linestyles[0],
                 c=v_cmap(color_dist[mu_index]), marker=marker_types[0])
    plt.ylabel(r"$y$ " + instrument)
    plt.xlabel("Center Date")
    ax = plt.gca()
    model_lines = [Line2D([0], [0], color="black", linestyle=linestyles[0], lw=2,
                          marker=marker_types[0])]
    legend1 = plt.legend(mu_lines, [str(round(x, 3)) for x in sample_mu], loc='upper left', bbox_to_anchor=(1., 1.),
                         title=r"$\mu$ value")
    ax.legend(model_lines, ["theoretic"], loc='upper left',
              bbox_to_anchor=(1., 0.65),
              title="model")
    plt.gca().add_artist(legend1)
    # adjust margin to incorporate legend
    plt.subplots_adjust(right=0.8)
    plt.grid()

    plot_fname = image_out_path + instrument + '_y_' + year + "-" + time_period.replace(" ", "") + '.pdf'
    plt.savefig(plot_fname)

    plt.close(20 + inst_index)

    #### BETA AND Y v. MU FOR SPECIFIED WEEK #####

    plt.figure(100 + inst_index)

    beta_y_v_mu = np.zeros((mu_bin_centers.shape[0], 2))

    for index, mu in enumerate(mu_bin_centers):
        beta_y_v_mu[index, :] = lbcc.get_beta_y_theoretic_based(theoretic_query[plot_week, :], mu)

    plt.plot(mu_bin_centers, beta_y_v_mu[:, 0], ls=linestyles[0],
             c=v_cmap(color_dist[0 - 3]), marker=marker_types[0])

    plt.ylabel(r"$\beta$ " + instrument)
    plt.xlabel(r"$\mu$")
    plt.title(instrument + " " + time_period + " average " + str(moving_avg_centers[plot_week]))
    ax = plt.gca()

    ax.legend(["theoretic"], loc='upper right',
              bbox_to_anchor=(1., 1.),
              title="model")
    plt.grid()

    plot_fname = image_out_path + instrument + '_beta_v_mu_' + year + "-" + time_period.replace(" ", "") + '.pdf'
    plt.savefig(plot_fname)

    plt.close(100 + inst_index)

    # repeat for y
    plt.figure(200 + inst_index)

    plt.plot(mu_bin_centers, beta_y_v_mu[:, 1], ls=linestyles[0],
             c=v_cmap(color_dist[0 - 3]), marker=marker_types[0])

    plt.ylabel(r"$y$ " + instrument)
    plt.xlabel(r"$\mu$")
    plt.title(instrument + " " + time_period + " average " + str(moving_avg_centers[plot_week]))
    ax = plt.gca()

    ax.legend(["theoretic"], loc='lower right',
              bbox_to_anchor=(1., 0.),
              title="model")
    plt.grid()

    plot_fname = image_out_path + instrument + '_y_v_mu_' + year + "-" + time_period.replace(" ", "") + '.pdf'
    plt.savefig(plot_fname)

    plt.close(200 + inst_index)

end_time_tot = time.time()
print("Theoretical plots of beta and y over time have been generated and saved.")
print("Total elapsed time for plot creation: " + str(round(end_time_tot - start_time_tot, 3)) + " seconds.")
