"""
construct mu-histogram and push to database for any time period
"""

import os

# This can be a computationally intensive process.
# To limit number of threads numpy can spawn:
# os.environ["OMP_NUM_THREADS"] = "4"

import time
import datetime
import numpy as np
from chmap.settings.app import App
import chmap.database.db_classes as db_class
from chmap.database.db_funs import init_db_conn_old, query_euv_images, add_hist, get_method_id, query_hist
import chmap.utilities.datatypes.datatypes as psi_d_types

###### ------ PARAMETERS TO UPDATE -------- ########

# TIME RANGE
hist_query_time_min = datetime.datetime(2020, 12, 31, 0, 0, 0)
hist_query_time_max = datetime.datetime(2021, 1, 1, 0, 0, 0)

# define instruments and wavelengths to include
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
wavelengths = [193, 195]

# define number of bins
n_mu_bins = 18
n_intensity_bins = 200

# declare map and binning parameters
R0 = 1.01
log10 = True
lat_band = [- np.pi / 64., np.pi / 64.]

# recover local filesystem paths
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME


# designate which database to connect to
use_db = "mysql-Q"       # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn_old() will automatically find and use your saved password. Otherwise, enter your MySQL password here.
# setup local database paths (only used for use_db='sqlite')
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME


# ------------ NO NEED TO UPDATE ANYTHING BELOW  ------------- #

# setup database connection
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)
        print("\nPrevious file ", sqlite_filename, " deleted.\n")

    db_session = init_db_conn_old(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
elif use_db in ['mysql-Q', 'mysql-Q_test']:
    # setup database connection to MySQL database on Q
    db_session = init_db_conn_old(db_name=use_db, chd_base=db_class.Base, user=user, password=password)

# start time
start_time_tot = time.time()

# creates mu bin & intensity bin arrays
mu_bin_edges = np.linspace(0.1, 1.0, n_mu_bins + 1, dtype='float')
image_intensity_bin_edges = np.linspace(0, 5, num=n_intensity_bins + 1, dtype='float')

# create LBC method
meth_name = 'LBCC'
meth_desc = 'LBCC Theoretic Fit Method'
method_id = get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=True)

# loop over instrument
for instrument in inst_list:
    # query EUV images
    query_instrument = [instrument, ]
    query_pd_all = query_euv_images(db_session=db_session, time_min=hist_query_time_min,
                                    time_max=hist_query_time_max, instrument=query_instrument,
                                    wavelength=wavelengths)
    # query LBCC histograms
    hist_pd = query_hist(db_session, meth_id=method_id[1], n_mu_bins=n_mu_bins, n_intensity_bins=n_intensity_bins,
                         lat_band=lat_band, time_min=hist_query_time_min, time_max=hist_query_time_max,
                         instrument=query_instrument, wavelength=wavelengths)

    # compare image results to hist results based on image_id
    in_index = query_pd_all.data_id.isin(hist_pd.image_id)

    # return only images that do not have corresponding histograms
    query_pd = query_pd_all[~in_index]

    # check that images remain that need histograms
    if query_pd.shape[0] == 0:
        print("All" + instrument + " images in timeframe already have associated histograms.")
        continue

    for index, row in query_pd.iterrows():
        print("Processing image number", row.data_id, ".")
        if row.fname_hdf == "":
            print("Warning: Image # " + str(row.data_id) + " does not have an associated hdf file. Skipping")
            continue
        hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
        # attempt to open and read file
        try:
            los_temp = psi_d_types.read_euv_image(hdf_path)
        except:
            print("Something went wrong opening: ", hdf_path, ". Skipping")
            continue
        # add coordinates to los object
        los_temp.get_coordinates(R0=R0)
        # perform 2D histogram on mu and image intensity
        temp_hist = los_temp.mu_hist(image_intensity_bin_edges, mu_bin_edges, lat_band=lat_band, log10=log10)
        hist_lbcc = psi_d_types.create_lbcc_hist(hdf_path, row.data_id, method_id[1], mu_bin_edges,
                                                 image_intensity_bin_edges, lat_band, temp_hist)

        # add this histogram and meta data to database
        add_hist(db_session, hist_lbcc)

db_session.close()

end_time_tot = time.time()
print("Histograms have been created and saved to the database.")
print("Total elapsed time for histogram creation: " + str(round(end_time_tot - start_time_tot, 3)) + " seconds.")
