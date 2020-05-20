"""
Generate plots to compare optimization methods
This will be updated to grab parameters from the database - not useful right now:)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import modules.lbcc_funs as lbcc

# PLOT PARAMETERS

year = "2011" # used for naming plot file
time_period = "3Day" # used for naming plot file
title_time_period = "3 Day" # used for plot titles
plot_week = 0 # index of week you want to plot
# path to save plots to
image_out_path = os.path.join(App.APP_HOME, "test_data", "analysis/lbcc_functionals/")

# ------------ NO NEED TO UPDATE ANYTHING BELOW  ------------- #

# generate some plots to compare methods
sse_index1 = np.array([x == "SSE" for x in optim_vals1])
npar1 = np.where(sse_index1)[0][0]
sse_index2 = np.array([x == "SSE" for x in optim_vals2])
npar2 = np.where(sse_index2)[0][0]
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
    mu_bins_SSE_tots = results[:, inst_index, :, 2].sum(axis=1)
    # plot SSEs for each instrument
    plt.figure(0+inst_index)

    plt.plot(moving_avg_centers, results1[:, inst_index, sse_index1], c="blue", label="cubic")
    plt.plot(moving_avg_centers, results2[:, inst_index, sse_index2], c="red", label="power-log")
    plt.plot(moving_avg_centers, results3[:, inst_index, sse_index3], c="green", label="theoretic")
    plt.plot(moving_avg_centers, mu_bins_SSE_tots, c="black", marker='x', linestyle="None", label="mu-bins")

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
            plot_beta[mu_index, date_index, 0], plot_y[mu_index, date_index, 0] = \
                lbcc.get_beta_y_cubic(results1[date_index, inst_index, 0:npar1], mu)
            plot_beta[mu_index, date_index, 1], plot_y[mu_index, date_index, 1] = \
                lbcc.get_beta_y_power_log(results2[date_index, inst_index, 0:npar2], mu)
            plot_beta[mu_index, date_index, 2], plot_y[mu_index, date_index, 2] = \
                lbcc.get_beta_y_theoretic_based(results3[date_index, inst_index, 0:npar3], mu)
            plot_beta[mu_index, date_index, 3] = results[date_index, inst_index, mu_results_index[mu_index], 0]
            plot_y[mu_index, date_index, 3] = results[date_index, inst_index, mu_results_index[mu_index], 1]


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
        beta_y_v_mu[index, :, 0] = lbcc.get_beta_y_cubic(results1[plot_week, inst_index, 0:npar1], mu)
        beta_y_v_mu[index, :, 1] = lbcc.get_beta_y_power_log(results2[plot_week, inst_index, 0:npar2], mu)
        beta_y_v_mu[index, :, 2] = lbcc.get_beta_y_theoretic_based(results3[plot_week, inst_index, 0:npar3], mu)
    beta_y_v_mu[:-1, :, 3] = results[plot_week, inst_index, :, 0:2]

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