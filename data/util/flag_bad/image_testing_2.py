#!/usr/bin/env python
"""
use existing LBC histograms from database
to determine "bad" images and flag in the database
"""
import os
import time
import datetime
import numpy as np
import pandas as pd

from settings.app import App
import database.db_classes as db_class
import database.db_funs as db_funs
import modules.datatypes as psi_d_types
import matplotlib.pyplot as plt
import matplotlib as mpl
import modules.Plotting as EasyPlot

###### ------ PARAMETERS TO UPDATE -------- ########

view_bad_images = False

# directory to save plots
plot_dir = "/Users/turtle/Dropbox/MyNACD/analysis/flag_bad"

# TIME RANGE
min_year = 2007
max_year = 2020
# query_time_min = datetime.datetime(2012, 1, 1, 0, 0, 0)
# query_time_max = datetime.datetime(2013, 1, 1, 0, 0, 0)

# define instruments
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
wavelengths = [193, 195]

# define bad image cutoffs
hist_sum_cutoffs = [120000, 75000, 55000]
thresh_inst = ["EUVI-A", "EUVI-A", "EUVI-B", "EUVI-B"]
thresh_year = [2009, 2100, 2009, 2100]
thresh_mean = [2.2, 2.4, 2.05, 2.3]

# define number of bins
n_mu_bins = 18
n_intensity_bins = 200

# declare map and binning parameters
R0 = 1.01
log10 = True
lat_band = [- np.pi / 64., np.pi / 64.]

# recover database paths
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME

# designate which database to connect to
use_db = "mysql-Q"      # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
                        # 'mysql-Q_test' Use the development database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

start_time = time.time()

# Establish connection to database
db_session = db_funs.init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user,
                                  password=password)

#### FUNCTIONS
# remove image from "bad data" list
def remove_image(data_ids, bad_data):
    for data_id in data_ids:
        index = np.where(bad_data.data_id == int(data_id))
        bad_data = bad_data.drop(index[0][0])
    return bad_data


# create list of image ids to flag
def flag_image(data_ids, bad_data):
    bad_images = pd.DataFrame()
    for data_id in data_ids:
        index = np.where(bad_data.data_id == int(data_id))
        print(index)
        bad_images = bad_images.append(bad_data.iloc[index[0][0]])
    return bad_images


# find outliers
def find_anomalies(data):
    anomalies = []
    # Set upper and lower limit to 3 standard deviation
    random_data_std = np.std(data)
    random_data_mean = np.mean(data)
    anomaly_cut_off = random_data_std * 3

    lower_limit = random_data_mean - anomaly_cut_off
    upper_limit = random_data_mean + anomaly_cut_off
    # Generate outliers
    for index, outlier in enumerate(data):
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(index)
    return anomalies


# plot bad images
def plot_bad_images(hdf_dir, data):
    for im_ind, im_row in data.iterrows():
        print("Plotting image number", im_row.data_id, ".")
        if im_row.fname_hdf == "":
            print("Warning: Image # " + str(im_row.data_id) + " does not have an associated hdf file. Skipping")
            continue
        hdf = os.path.join(hdf_dir, im_row.fname_hdf)
        los_image = psi_d_types.read_los_image(hdf)
        # add coordinates to los object
        los_image.get_coordinates(R0=R0)

        #### plot image
        # set color palette and normalization (improve by using Ron's colormap setup)
        norm = mpl.colors.LogNorm(vmin=1.0, vmax=np.nanmax(los_image.data))
        # norm = mpl.colors.LogNorm()
        im_cmap = plt.get_cmap('sohoeit195')

        # remove extremely small values from data so that log color scale treats them as black
        # rather than white
        plot_arr = los_image.data
        plot_arr[plot_arr < .001] = .001

        # plot the initial image
        plt.figure(im_row.data_id)
        plt.imshow(plot_arr, extent=[los_image.x.min(), los_image.x.max(), los_image.y.min(), los_image.y.max()],
                   origin="lower", cmap=im_cmap, aspect="equal", norm=norm)
        plt.xlabel("x (solar radii)")
        plt.ylabel("y (solar radii)")
        plt.title(im_row.data_id)
    plt.show()


# ------------ NO NEED TO UPDATE ANYTHING BELOW  ------------- #
# get method id
meth_name = 'LBCC'
meth_desc = 'LBCC Theoretic Fit Method'
method_id = db_funs.get_method_id(db_session, meth_name, meth_desc,
                                  var_names=None, var_descs=None, create=False)

bad_image_lists = []
# loop over instrument
for inst_index, instrument in enumerate(inst_list):
    start_time = time.time()
    # initialize image metrics
    bad_images = []
    metric_pd = pd.DataFrame([], columns=['image_id', 'date_obs', 'hist_mean',
                                          'hist_sum', 'hist_median', 'c70_width',
                                          'z_score'])
    # loop over years
    for year in range(min_year, max_year+1):
        print("Starting on year: ", year)
        year_min_time = datetime.datetime(year, 1, 1, 0, 0, 0)
        year_max_time = datetime.datetime(year+1, 1, 1, 0, 0, 0)

        pd_hist = db_funs.query_hist(db_session=db_session, meth_id=method_id[1],
                                     n_mu_bins=n_mu_bins, n_intensity_bins=n_intensity_bins,
                                     lat_band=lat_band, time_min=year_min_time,
                                     time_max=year_max_time, instrument=[instrument, ],
                                     wavelength=wavelengths)
        # check for no data
        if pd_hist.shape[0] == 0:
            # skip this year
            continue
        # convert the binary types back to arrays
        mu_bin_array, intensity_bin_array, full_hist = psi_d_types.binary_to_hist(
            pd_hist, n_mu_bins, n_intensity_bins)

        # end_time = time.time()
        # array_size = full_hist.size * full_hist.itemsize
        # print(end_time - start_time, " seconds elapsed querying and processing histograms.")
        # print(array_size/1e6, " MB histogram array size.")

        # don't need mu-bins, so sum
        int_hist = full_hist.sum(axis=0)
        int_bin_centers = (intensity_bin_array[1:] + intensity_bin_array[:-1])/2
        # promote to 2D column vector
        int_bin_centers = int_bin_centers[:, np.newaxis]

        # calc image sums
        hist_sums = np.sum(int_hist, axis=0)
        non_zero = hist_sums != 0
        # mark zero-col images as bad and remove
        remove_images = pd_hist.image_id[~non_zero]
        bad_images = bad_images + list(remove_images)
        pd_hist = pd_hist.iloc[non_zero]
        int_hist = int_hist[:, non_zero]
        hist_sums = hist_sums[non_zero]
        # calc mean
        inst_mean = np.sum(int_hist*int_bin_centers, axis=0)
        inst_mean = inst_mean/hist_sums
        # calc bin quantiles and median
        hist_cumsum = np.cumsum(int_hist, axis=0)
        hist_quants = hist_cumsum/hist_sums
        median_index = np.argmax(hist_quants >= 0.5, axis=0)
        medians = int_bin_centers[median_index].flatten()
        # calc central 70% as a substitute for standard deviation
        q15_index = median_index = np.argmax(hist_quants >= 0.15, axis=0)
        q85_index = median_index = np.argmax(hist_quants >= 0.85, axis=0)
        q15 = int_bin_centers[q15_index]
        q85 = int_bin_centers[q85_index]
        c70_width = (q85 - q15).flatten()
        # z_score
        median_mean = np.median(inst_mean)
        z_score = (inst_mean - median_mean)/c70_width

        # generate a dataframe
        temp_metric_pd = pd.DataFrame({
            'image_id': pd_hist.image_id, 'date_obs': pd_hist.date_obs,
            'hist_mean': inst_mean, 'hist_sum': hist_sums, 'hist_median': medians,
            'c70_width': c70_width, 'z_score': z_score})
        if metric_pd.shape[0] == 0:
            metric_pd = temp_metric_pd.copy()
        else:
            metric_pd = metric_pd.append(temp_metric_pd)

    metric_pd = metric_pd.reset_index()
    end_time = time.time()
    print(end_time - start_time, "seconds elapsed compiling ", instrument)

    # plot mean
    plt.figure("Mean " + instrument)
    plt.scatter(metric_pd.date_obs, metric_pd.hist_mean)
    plt.xlabel("Time")
    plt.ylabel("Mean")
    plt.title(instrument + " Mean")
    filename = instrument + '_mean.png'
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

    # plot standard deviation
    plt.figure("Central 70% " + instrument)
    plt.scatter(metric_pd.date_obs, metric_pd.c70_width)
    plt.xlabel("Time")
    plt.ylabel("Central 70% width")
    plt.title(instrument + " Distribution Width")
    filename = instrument + '_c70width.png'
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

    # plot mean v. std
    plt.figure("Mean v. STD " + instrument)
    plt.scatter(metric_pd.hist_mean, metric_pd.c70_width)
    plt.xlabel("Mean")
    plt.ylabel("Central 70% width")
    plt.title(instrument + " Width v Mean")
    filename = instrument + '_mean-v-width.png'
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

    # z-score
    plt.figure("Z-Score " + instrument)
    plt.scatter(metric_pd.date_obs, metric_pd.z_score)
    plt.ylim([-2, 2])
    plt.xlabel("Time")
    plt.ylabel("Z-Score")
    plt.title(instrument + " Z-Score")
    filename = instrument + '_z-score.png'
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

    # hist sum
    plt.figure("Hist-Sum " + instrument)
    plt.scatter(metric_pd.date_obs, metric_pd.hist_sum)
    plt.xlabel("Time")
    plt.ylabel("Hist sum")
    plt.title(instrument + " Histogram Sum")
    filename = instrument + '_hist-sum.png'
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

    # mark everything with a sum less than 70k as 'bad'
    cmap = mpl.colors.ListedColormap(['tab:blue', 'red'])
    color_list = metric_pd.hist_sum < hist_sum_cutoffs[inst_index]
    instrument_index = np.where([inst == instrument for inst in thresh_inst])
    if len(instrument_index[0]) > 0:
        for ii in instrument_index[0]:
            color_list = color_list | ((metric_pd.hist_mean > thresh_mean[ii]) &
                         (metric_pd.date_obs < datetime.datetime(thresh_year[ii], 1, 1)))

    # hist sum
    plt.figure("Hist-Sum " + instrument)
    plt.scatter(metric_pd.date_obs, metric_pd.hist_sum, c=color_list, cmap=cmap)
    plt.xlabel("Time")
    plt.ylabel("Hist sum")
    plt.title(instrument + " Histogram Sum")
    filename = instrument + '_hist-sum_filtered.png'
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

    # plot mean
    plt.figure("Mean " + instrument)
    plt.scatter(metric_pd.date_obs, metric_pd.hist_mean, c=color_list, cmap=cmap)
    plt.xlabel("Time")
    plt.ylabel("Mean")
    plt.title(instrument + " Mean")
    filename = instrument + '_mean_filtered.png'
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

    # plot standard deviation
    plt.figure("Central 70% " + instrument)
    plt.scatter(metric_pd.date_obs, metric_pd.c70_width, c=color_list, cmap=cmap)
    plt.xlabel("Time")
    plt.ylabel("Central 70% width")
    plt.title(instrument + " Distribution Width")
    filename = instrument + '_c70width_filtered.png'
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

    # plot mean v. std
    plt.figure("Mean v. STD " + instrument)
    plt.scatter(metric_pd.hist_mean, metric_pd.c70_width, c=color_list, cmap=cmap)
    plt.xlabel("Mean")
    plt.ylabel("Central 70% width")
    plt.title(instrument + " Width v Mean")
    filename = instrument + '_mean-v-width_filtered.png'
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

    # z-score
    plt.figure("Z-Score " + instrument)
    plt.scatter(metric_pd.date_obs, metric_pd.z_score, c=color_list, cmap=cmap)
    plt.ylim([-2, 2])
    plt.xlabel("Time")
    plt.ylabel("Z-Score")
    plt.title(instrument + " Z-Score")
    filename = instrument + '_z-score_filtered.png'
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

    print(bad_images.__len__(), "no-sum images flagged for", instrument)
    print(np.sum(color_list), "partial images flagged for", instrument)

    bad_images = bad_images + list(metric_pd.image_id[color_list])
    bad_image_lists.append(bad_images)

if view_bad_images:
    # loop through bad images and display
    for inst_index, instrument in enumerate(inst_list):
        # query images
        query_pd = pd.read_sql(db_session.query(db_class.EUV_Images, db_class.Data_Files).filter(
            db_class.EUV_Images.data_id.in_(bad_image_lists[inst_index]),
            db_class.Data_Files.data_id == db_class.EUV_Images.data_id).order_by(
            db_class.EUV_Images.date_obs).statement,
                               db_session.bind)
        # remove duplicate columns
        query_pd = query_pd.loc[:, ~query_pd.columns.duplicated()]

        n_images = bad_image_lists[inst_index].__len__()
        for im_num, row in query_pd.iterrows():
            full_path = os.path.join(hdf_data_dir, row.fname_hdf)
            print("Plotting", instrument, im_num+1, "of", n_images, "-",
                  row.date_obs)
            bad_im = psi_d_types.read_los_image(full_path)
            EasyPlot.PlotImage(bad_im, nfig=0)
            plt.waitforbuttonpress()
            plt.close(0)

# loop through flag_bad and change flag in database
for inst_index, instrument in enumerate(inst_list):
    # query images
    query_pd = pd.read_sql(db_session.query(db_class.EUV_Images, db_class.Data_Files).filter(
        db_class.EUV_Images.data_id.in_(bad_image_lists[inst_index]),
        db_class.Data_Files.data_id == db_class.EUV_Images.data_id).order_by(
        db_class.EUV_Images.date_obs).statement,
                           db_session.bind)
    # remove duplicate columns
    query_pd = query_pd.loc[:, ~query_pd.columns.duplicated()]
    for index, row in query_pd.iterrows():
        db_session = db_funs.update_image_val(db_session, row, 'flag', -1)


db_session.close()



