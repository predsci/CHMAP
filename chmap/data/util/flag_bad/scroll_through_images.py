"""
query a set of images and then scroll through them to
inspect image quality
"""
import os
import datetime
import numpy as np

from settings.app import App
import chmap.database.db_classes as db_class
import chmap.database.db_funs as db_funs
import utilities.datatypes.datatypes as psi_d_types
import matplotlib.pyplot as plt
import utilities.plotting.psi_plotting as EasyPlot

###### ------ PARAMETERS TO UPDATE -------- ########

query_time_min = datetime.datetime(2007, 1, 1, 0, 0, 0)
query_time_max = datetime.datetime(2007, 3, 5, 0, 0, 0)

# define instruments
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
wavelengths = [193, 195]

# define number of bins
n_mu_bins = 18
n_intensity_bins = 200

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

# Establish connection to database
db_session = db_funs.init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user,
                                  password=password)

# ------------ NO NEED TO UPDATE ANYTHING BELOW  ------------- #

# query images
query_pd = db_funs.query_euv_images(db_session, time_min=query_time_min,
                                    time_max=query_time_max, instrument=inst_list,
                                    wavelength=wavelengths)

# get method id
meth_name = 'LBCC'
meth_desc = 'LBCC Theoretic Fit Method'
method_id = db_funs.get_method_id(db_session, meth_name, meth_desc,
                                  var_names=None, var_descs=None, create=False)
# query LBC histograms
hist_pd = db_funs.query_hist(db_session, meth_id=method_id[1],
                             n_mu_bins=n_mu_bins, n_intensity_bins=n_intensity_bins,
                             time_min=query_time_min,
                             time_max=query_time_max, instrument=inst_list,
                             wavelength=wavelengths)
# convert the binary types back to arrays
mu_bin_array, intensity_bin_array, full_hist = psi_d_types.binary_to_hist(
    hist_pd, n_mu_bins, n_intensity_bins)

n_images = query_pd.shape[0]
int_bin_centers = (intensity_bin_array[0:-1] + intensity_bin_array[1:])/2
for im_num, row in query_pd.iterrows():
    full_path = os.path.join(hdf_data_dir, row.fname_hdf)
    print("Plotting", row.instrument, im_num+1, "of", n_images, "-",
          row.date_obs)
    bad_im = psi_d_types.read_los_image(full_path)
    EasyPlot.PlotImage(bad_im, nfig=0)
    plt.waitforbuttonpress()
    plt.close(0)

    # plot histogram
    hist_index = hist_pd.image_id == row.data_id
    plot_hist = full_hist[:, :, hist_index].sum(axis=0)
    plot_mean = np.sum(plot_hist.flatten()*int_bin_centers)/np.sum(plot_hist)
    hist_sum = plot_hist.sum()
    plt.figure(0)
    plt.scatter(x=int_bin_centers, y=plot_hist)
    plt.title("Mean: " + "{:4.2f}".format(plot_mean) + ",  Hist Sum: " + str(hist_sum))
    plt.grid()
    plt.waitforbuttonpress()
    plt.close(0)

db_session.close()