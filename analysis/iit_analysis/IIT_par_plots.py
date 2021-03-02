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
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

from settings.app import App
import modules.DB_funs as db_funs
import modules.DB_classes as db_class
import modules.lbcc_funs as lbcc
import modules.datatypes as psi_d_types

# INSTRUMENTS
inst_list = ["AIA", "EUVI-A", "EUVI-B"]

# PLOT PARAMETERS
n_mu_bins = 18
year = "AllYears"  # used for naming plot file
time_period = "6-month"  # used for naming plot file
title_time_period = "6 Month"  # used for plot titles

# Color palette
v_cmap = cm.get_cmap('viridis')
n_inst = inst_list.__len__()
color_dist = np.linspace(0., 1., n_inst)

test_pal = ["#1b9e77", "#d95f02", "#7570b3"]

# path to save plots to
image_out_path = os.path.join(App.APP_HOME, "test_data", "analysis/iit_analysis/")
image_out_path = "/Users/turtle/Dropbox/MyNACD/analysis/iit_analysis/"
# TIME FRAME TO QUERY HISTOGRAMS
query_time_min = datetime.datetime(2007, 4, 1, 0, 0, 0)
query_time_max = datetime.datetime(2021, 1, 1, 0, 0, 0)
weekday = 0
# Histogram characteristics
number_of_days = 180
n_mu_bins = 18
n_intensity_bins = 200
R0 = 1.01
log10 = True
lat_band = [-np.pi / 2.4, np.pi / 2.4]

# INITIALIZE DATABASE CONNECTION
# DATABASE PATHS
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME

# DATABASE PATHS
create = True  # true if save to database
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME

# designate which database to connect to
use_db = "mysql-Q"       # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.


# ------------ NO NEED TO UPDATE ANYTHING BELOW  ------------- #
start_time_tot = time.time()

# connect to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = db_funs.init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
elif use_db == 'mysql-Q':
    # setup database connection to MySQL database on Q
    db_session = db_funs.init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user, password=password)

# create mu bin array
mu_bin_array = np.array(range(n_mu_bins + 1), dtype="float") * 0.05 + 0.1
mu_bin_centers = (mu_bin_array[1:] + mu_bin_array[:-1]) / 2

# time arrays
# returns array of moving averages center dates, based off start and end date
moving_avg_centers, moving_width = lbcc.moving_averages(query_time_min, query_time_max, weekday,
                                                        days=number_of_days)

# retrieve beta and y for all instruments
# plot all three instruments beta plot, then y plot
linestyles = ['dashed']
marker_types = ['None']

meth_name = 'IIT'
method_id = db_funs.get_method_id(db_session, meth_name, create=False)

plot_alpha = np.zeros((moving_avg_centers.__len__(), inst_list.__len__()))
plot_x     = np.zeros((moving_avg_centers.__len__(), inst_list.__len__()))
plot_mean  = np.zeros((moving_avg_centers.__len__(), inst_list.__len__()))
plot_std   = np.zeros((moving_avg_centers.__len__(), inst_list.__len__()))
plot_act_mean = np.zeros((moving_avg_centers.__len__(), inst_list.__len__()))
plot_mean_act = np.zeros((moving_avg_centers.__len__(), inst_list.__len__()))

for inst_index, instrument in enumerate(inst_list):
    print("Interpolating IIT parameters for " + instrument + ".")

    # get all instrument histograms
    inst_hist_pd = db_funs.query_hist(db_session=db_session, meth_id=method_id[1],
                                       n_intensity_bins=n_intensity_bins,
                                       lat_band=lat_band,
                                       time_min=query_time_min - datetime.timedelta(days=number_of_days),
                                       time_max=query_time_max + datetime.timedelta(days=number_of_days),
                                       instrument=[instrument, ])
    # convert binary to histogram data
    mu_bin_edges, intensity_bin_edges, inst_full_hist = psi_d_types.binary_to_hist(
        hist_binary=inst_hist_pd, n_mu_bins=None, n_intensity_bins=n_intensity_bins)
    intensity_bin_centers = (intensity_bin_edges[1:] + intensity_bin_edges[0:-1])/2

    for date_index, center_date in enumerate(moving_avg_centers):
        date_str = str(center_date)
        datetime_date = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        # query IIT parameters
        iit_query = db_funs.get_correction_pars(db_session, meth_name, date_obs=datetime_date,
                                                instrument=instrument)
        plot_alpha[date_index, inst_index] = iit_query[0]
        plot_x[date_index, inst_index] = iit_query[1]

        # calculate 6-month histogram
        min_date = center_date - moving_width/2
        max_date = center_date + moving_width/2
        hist_index = ((inst_hist_pd['date_obs'] >= str(min_date)) &
                      (inst_hist_pd['date_obs'] <= str(max_date)))
        inst_pd_use = inst_hist_pd[hist_index]
        # inst_pd_use = inst_hist_pd[(inst_hist_pd['date_obs'] >= str(min_date)) &
        #                            (inst_hist_pd['date_obs'] <= str(max_date))]
        if inst_pd_use.shape[0] == 0:
            # set outputs to zero
            plot_mean[date_index, inst_index]     = 0
            plot_std[date_index, inst_index]      = 0
            plot_act_mean[date_index, inst_index] = 0
            plot_mean_act[date_index, inst_index] = 0
        else:
            # inst_hist_ind = np.where(
            #     (inst_hist_pd['date_obs'] >= str(min_date)) & (inst_hist_pd['date_obs'] <= str(max_date)))
            # inst_ind_min = np.min(inst_hist_ind)
            # inst_ind_max = np.max(inst_hist_ind)
            # inst_hist_use = inst_full_hist[:, inst_ind_min:inst_ind_max]
            inst_hist_use = inst_full_hist[:, hist_index]
            # sum histograms
            hist_fit = inst_hist_use.sum(axis=1)

            # normalize fit histogram
            fit_sums = hist_fit.sum()
            norm_hist_fit = hist_fit/fit_sums

            # mean and std
            hist_mean = (intensity_bin_centers*norm_hist_fit).sum()
            hist_std  = np.sqrt(((intensity_bin_centers - hist_mean)**2 * norm_hist_fit).sum())
            # action on the mean
            new_mean = hist_mean*iit_query[0] + iit_query[1]
            act_mean = new_mean - hist_mean
            # mean action ...(?)
            mean_act = ((intensity_bin_centers*iit_query[0] + iit_query[1]
                         - intensity_bin_centers)*norm_hist_fit).sum()

            # record plotting variables
            plot_mean[date_index, inst_index] = hist_mean
            plot_std[date_index, inst_index] = hist_std
            plot_act_mean[date_index, inst_index] = act_mean
            plot_mean_act[date_index, inst_index] = mean_act


#### Alpha AND X AS FUNCTION OF TIME ####
# plot Alpha for the different models as a function of time
fig = plt.figure(11)

inst_lines = []
for ii in range(n_inst):
    inst_lines.append(Line2D([0], [0], color=test_pal[ii], lw=2))
    # plt.plot(moving_avg_centers, plot_alpha[:, ii], ls=linestyles[0],
    #          c=v_cmap(color_dist[ii]), marker=marker_types[0])
    plt.plot(moving_avg_centers, plot_alpha[:, ii], ls=linestyles[0],
             c=test_pal[ii], marker=marker_types[0])
plt.ylabel(r"$\alpha$")
plt.xlabel("Date (weekly interpolation)")
fig.autofmt_xdate()
ax = plt.gca()

legend1 = plt.legend(inst_lines, inst_list, loc='upper left', bbox_to_anchor=(1., 1.),
                     title="Instrument")
plt.gca().add_artist(legend1)
# adjust margin to incorporate legend
plt.subplots_adjust(right=0.8)
plt.grid()

plot_fname = image_out_path + 'IIT_alpha_' + year + "-" + time_period.replace(" ", "") + '.pdf'
plt.savefig(plot_fname)

plt.close(11)

# plot X for the different models as a function of time
fig = plt.figure(12)

inst_lines = []
for ii in range(n_inst):
    inst_lines.append(Line2D([0], [0], color=test_pal[ii], lw=2))
    plt.plot(moving_avg_centers, plot_x[:, ii], ls=linestyles[0],
             c=test_pal[ii], marker=marker_types[0])
plt.ylabel(r"$x$ ")
plt.xlabel("Date (weekly interpolation)")
fig.autofmt_xdate()
ax = plt.gca()

legend1 = plt.legend(inst_lines, inst_list, loc='upper left', bbox_to_anchor=(1., 1.),
                     title="Instrument")
plt.gca().add_artist(legend1)
# adjust margin to incorporate legend
plt.subplots_adjust(right=0.8)
plt.grid()

plot_fname = image_out_path + 'IIT_x_' + year + "-" + time_period.replace(" ", "") + '.pdf'
plt.savefig(plot_fname)

plt.close(12)

end_time_tot = time.time()
print("Total elapsed time for plot creation: " + str(round(end_time_tot - start_time_tot, 3)) + " seconds.")


#### Histogram means as a function of time ####
fig = plt.figure(13)

inst_lines = []
for ii in range(n_inst):
    inst_lines.append(Line2D([0], [0], color=test_pal[ii], lw=2))
    # plt.plot(moving_avg_centers, plot_alpha[:, ii], ls=linestyles[0],
    #          c=v_cmap(color_dist[ii]), marker=marker_types[0])
    plt.plot(moving_avg_centers, plot_mean[:, ii], ls=linestyles[0],
             c=test_pal[ii], marker=marker_types[0])
plt.ylabel("Mean of 6-month LBCC histogram")
plt.xlabel("Date (weekly)")
fig.autofmt_xdate()
ax = plt.gca()

legend1 = plt.legend(inst_lines, inst_list, loc='upper left', bbox_to_anchor=(1., 1.),
                     title="Instrument")
plt.gca().add_artist(legend1)
# adjust margin to incorporate legend
plt.subplots_adjust(right=0.8)
plt.grid()

plot_fname = image_out_path + 'LBCC-hist-mean_' + year + "-" + time_period.replace(" ", "") + '.pdf'
plt.savefig(plot_fname)

plt.close(13)


#### Histogram standard deviation as a function of time ####
fig = plt.figure(14)

inst_lines = []
for ii in range(n_inst):
    inst_lines.append(Line2D([0], [0], color=test_pal[ii], lw=2))
    # plt.plot(moving_avg_centers, plot_alpha[:, ii], ls=linestyles[0],
    #          c=v_cmap(color_dist[ii]), marker=marker_types[0])
    plt.plot(moving_avg_centers, plot_std[:, ii], ls=linestyles[0],
             c=test_pal[ii], marker=marker_types[0])
plt.ylabel("Std. Dev. of 6-month LBCC histogram")
plt.xlabel("Date (weekly)")
fig.autofmt_xdate()
ax = plt.gca()

legend1 = plt.legend(inst_lines, inst_list, loc='upper left', bbox_to_anchor=(1., 1.),
                     title="Instrument")
plt.gca().add_artist(legend1)
# adjust margin to incorporate legend
plt.subplots_adjust(right=0.8)
plt.grid()

plot_fname = image_out_path + 'LBCC-hist-std_' + year + "-" + time_period.replace(" ", "") + '.pdf'
plt.savefig(plot_fname)

plt.close(14)


#### IIT action on 6-month histogram mean ####
fig = plt.figure(15)

inst_lines = []
for ii in range(n_inst):
    inst_lines.append(Line2D([0], [0], color=test_pal[ii], lw=2))
    # plt.plot(moving_avg_centers, plot_alpha[:, ii], ls=linestyles[0],
    #          c=v_cmap(color_dist[ii]), marker=marker_types[0])
    plt.plot(moving_avg_centers, plot_act_mean[:, ii], ls=linestyles[0],
             c=test_pal[ii], marker=marker_types[0])
plt.ylabel("IIT action on mean of 6-month LBCC histogram")
plt.xlabel("Date (weekly)")
fig.autofmt_xdate()
ax = plt.gca()

legend1 = plt.legend(inst_lines, inst_list, loc='upper left', bbox_to_anchor=(1., 1.),
                     title="Instrument")
plt.gca().add_artist(legend1)
# adjust margin to incorporate legend
plt.subplots_adjust(right=0.8)
plt.grid()

plot_fname = image_out_path + 'IIT-hist-act-mean_' + year + "-" + time_period.replace(" ", "") + '.pdf'
plt.savefig(plot_fname)

plt.close(15)


## --- REPEAT FOR RAW-HISTS AND LBCC ------------
meth_name = 'LBCC'
method_id = db_funs.get_method_id(db_session, meth_name, create=False)
lat_band = [- np.pi / 64., np.pi / 64.]

plot_mean2  = np.zeros((moving_avg_centers.__len__(), inst_list.__len__()))
plot_std2   = np.zeros((moving_avg_centers.__len__(), inst_list.__len__()))

for inst_index, instrument in enumerate(inst_list):
    print("Calculating mean and variance of processed histograms for " + instrument + ".")

    # get all instrument histograms
    inst_hist_pd = db_funs.query_hist(db_session=db_session, meth_id=method_id[1],
                                       n_intensity_bins=n_intensity_bins,
                                       lat_band=lat_band, n_mu_bins=n_mu_bins,
                                       time_min=query_time_min - datetime.timedelta(days=number_of_days),
                                       time_max=query_time_max + datetime.timedelta(days=number_of_days),
                                       instrument=[instrument, ])
    # convert binary to histogram data
    mu_bin_edges, intensity_bin_edges, inst_full_hist = psi_d_types.binary_to_hist(
        hist_binary=inst_hist_pd, n_mu_bins=n_mu_bins, n_intensity_bins=n_intensity_bins)
    intensity_bin_centers = (intensity_bin_edges[1:] + intensity_bin_edges[0:-1])/2

    for date_index, center_date in enumerate(moving_avg_centers):
        date_str = str(center_date)

        # calculate 6-month histogram
        min_date = center_date - moving_width/2
        max_date = center_date + moving_width/2
        hist_index = ((inst_hist_pd['date_obs'] >= str(min_date)) &
                      (inst_hist_pd['date_obs'] <= str(max_date)))
        inst_pd_use = inst_hist_pd[hist_index]
        if inst_pd_use.shape[0] == 0:
            # set outputs to zero
            plot_mean2[date_index, inst_index]     = 0
            plot_std2[date_index, inst_index]      = 0

        else:
            # inst_hist_ind = np.where(
            #     (inst_hist_pd['date_obs'] >= str(min_date)) & (inst_hist_pd['date_obs'] <= str(max_date)))
            # inst_ind_min = np.min(inst_hist_ind)
            # inst_ind_max = np.max(inst_hist_ind)
            # inst_hist_use = inst_full_hist[:, inst_ind_min:inst_ind_max]
            inst_hist_use = inst_full_hist[:, :, hist_index]
            # sum histograms along mu and time
            hist_fit = inst_hist_use.sum(axis=(0, 2))

            # normalize fit histogram
            fit_sum = hist_fit.sum()
            norm_hist_fit = hist_fit/fit_sum

            # mean and std
            hist_mean = (intensity_bin_centers*norm_hist_fit).sum()
            hist_std  = np.sqrt(((intensity_bin_centers - hist_mean)**2 * norm_hist_fit).sum())

            # record plotting variables
            plot_mean2[date_index, inst_index] = hist_mean
            plot_std2[date_index, inst_index] = hist_std


#### Histogram means as a function of time ####
fig = plt.figure(23)

inst_lines = []
for ii in range(n_inst):
    inst_lines.append(Line2D([0], [0], color=test_pal[ii], lw=2))
    # plt.plot(moving_avg_centers, plot_alpha[:, ii], ls=linestyles[0],
    #          c=v_cmap(color_dist[ii]), marker=marker_types[0])
    plt.plot(moving_avg_centers, plot_mean2[:, ii], ls=linestyles[0],
             c=test_pal[ii], marker=marker_types[0])
plt.ylabel("Mean of 6-month processed histogram")
plt.xlabel("Date (weekly)")
fig.autofmt_xdate()
ax = plt.gca()

legend1 = plt.legend(inst_lines, inst_list, loc='upper left', bbox_to_anchor=(1., 1.),
                     title="Instrument")
plt.gca().add_artist(legend1)
# adjust margin to incorporate legend
plt.subplots_adjust(right=0.8)
plt.grid()

plot_fname = image_out_path + 'proc-hist-mean_' + year + "-" + time_period.replace(" ", "") + '.pdf'
plt.savefig(plot_fname)

plt.close(23)


#### Histogram means as a function of time (time-colored markers) ####
inst_markers = ["o", "^", "s"]
n_time = plot_mean2.shape[0]
color_dist = np.linspace(0., 1., n_time)

fig = plt.figure(123)
inst_lines = []
for ii in range(n_inst):
    inst_lines.append(Line2D([0], [0], color="black", marker=inst_markers[ii]))
    # plt.plot(moving_avg_centers, plot_alpha[:, ii], ls=linestyles[0],
    #          c=v_cmap(color_dist[ii]), marker=marker_types[0])
    plt.scatter(moving_avg_centers, plot_mean2[:, ii], c=v_cmap(color_dist),
                marker=inst_markers[ii], s=6)
plt.ylabel("Mean of 6-month processed histogram")
plt.xlabel("Date (weekly)")
fig.autofmt_xdate()
ax = plt.gca()

legend1 = plt.legend(inst_lines, inst_list, loc='upper left', bbox_to_anchor=(1., 1.),
                     title="Instrument")
plt.gca().add_artist(legend1)
# adjust margin to incorporate legend
plt.subplots_adjust(right=0.8)
plt.grid()

plot_fname = image_out_path + 'proc-hist-mean-col_' + year + "-" + time_period.replace(" ", "") + '.pdf'
plt.savefig(plot_fname)

plt.close(123)


#### Histogram standard deviation as a function of time ####
fig = plt.figure(24)

inst_lines = []
for ii in range(n_inst):
    inst_lines.append(Line2D([0], [0], color=test_pal[ii], lw=2))
    # plt.plot(moving_avg_centers, plot_alpha[:, ii], ls=linestyles[0],
    #          c=v_cmap(color_dist[ii]), marker=marker_types[0])
    plt.plot(moving_avg_centers, plot_std2[:, ii], ls=linestyles[0],
             c=test_pal[ii], marker=marker_types[0])
plt.ylabel("Std. Dev. of 6-month processed histogram")
plt.xlabel("Date (weekly)")
fig.autofmt_xdate()
ax = plt.gca()

legend1 = plt.legend(inst_lines, inst_list, loc='upper left', bbox_to_anchor=(1., 1.),
                     title="Instrument")
plt.gca().add_artist(legend1)
# adjust margin to incorporate legend
plt.subplots_adjust(right=0.8)
plt.grid()

plot_fname = image_out_path + 'proc-hist-std_' + year + "-" + time_period.replace(" ", "") + '.pdf'
plt.savefig(plot_fname)

plt.close(24)


#### Scatter EUVI-A alpha vs EUVI-A mean ####
fig = plt.figure(25)

plt.scatter(plot_mean2[:, 1], plot_alpha[:, 1], c=v_cmap(color_dist))
plt.ylabel("EUVI-A IIT-alpha")
plt.xlabel("EUVI-A intensity mean")

plt.grid()

plot_fname = image_out_path + 'EUVI-A_scatter_alpha' + year + "-" + time_period.replace(" ", "") + '.pdf'
plt.savefig(plot_fname)

plt.close(25)


#### Scatter EUVI-A alpha vs EUVI-A mean ####
fig = plt.figure(26)

plt.scatter(plot_mean2[:, 1], plot_x[:, 1], c=v_cmap(color_dist))
plt.ylabel("EUVI-A IIT-x")
plt.xlabel("EUVI-A intensity mean")

plt.grid()

plot_fname = image_out_path + 'EUVI-A_scatter_x' + year + "-" + time_period.replace(" ", "") + '.pdf'
plt.savefig(plot_fname)

plt.close(26)


db_session.close()
# save to file
save_dict = {"instruments": inst_list, "moving_avg_centers": moving_avg_centers, "raw_hist_mean": plot_mean2, "raw_hist_std": plot_std2,
             "lbc_hist_mean": plot_mean, "lbc_hist_std": plot_std, "IIT_alpha": plot_alpha,
             "IIT_x": plot_x}
pickle_fname = image_out_path + "IIT_pars-and-hists.pkl"
f = open(pickle_fname, "wb")
pickle.dump(save_dict, f)
f.close()

# test file open
# file = open(pickle_fname, 'rb')
# object_file = pickle.load(file)
# file.close()
