"""
Example script for connecting to the database.
  - For an SQLite database, specifying a new filename will create a database
  - For a MySQL database, connecting to an existing database will also create
     any tables that are missing from the database schema.
"""

import os
import datetime
import numpy as np

import chmap.database.db_classes as db_class
import chmap.database.db_funs as db_funcs
import chmap.utilities.datatypes.datatypes as psi_dtypes
import chmap.utilities.plotting.psi_plotting as psi_plots

# INITIALIZE DATABASE CONNECTION
# Database paths
map_data_dir = "/Volumes/extdata2/CHD_DB_example/maps"
hdf_data_dir = "/Volumes/extdata2/CHD_DB_example/processed_images"

# Designate database-type and credentials
db_type = "sqlite"       # 'sqlite'  Use local sqlite file-based db
                        # 'mysql' Use a remote MySQL database

user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In
                        # this case leave password="", and init_db_conn() will
                        # automatically find and use your saved password. Otherwise,
                        # enter your MySQL password here.
# If password=="", then be sure to specify the directory where encrypted credentials
# are stored.  Setting cred_dir=None will cause the code to attempt to automatically
# determine a path to the settings/ directory.
cred_dir = "/Users/turtle/GitReps/CHD/chmap/settings"

# Specify the database location. In the case of MySQL, this will be an IP address or
# remote host name. For SQLite, this will be the full path to a database file.
    # mysql
# db_loc = "q.predsci.com"
    # sqlite
db_loc = "/Volumes/extdata2/CHD_DB_example/chd_example.db"

# specify which database to connect to (unnecessary for SQLite)
mysql_db_name = "chd"

# Establish connection to database
sqlite_session = db_funcs.init_db_conn(db_type, db_class.Base, db_loc, db_name=mysql_db_name,
                                       user=user, password=password, cred_dir=cred_dir)

mysql_session = db_funcs.init_db_conn("mysql", db_class.Base, "q.predsci.com", db_name=mysql_db_name,
                                      user=user, password=password, cred_dir=cred_dir)

# SAMPLE QUERY
# use database session to query available pre-processed images
query_time_min = datetime.datetime(2011, 2, 1, 1, 0, 0)
query_time_max = datetime.datetime(2011, 2, 1, 3, 0, 0)

meth_name = 'LBCC'
meth_desc = 'LBCC Theoretic Fit Method'
method_id = db_funcs.get_method_id(sqlite_session, meth_name, meth_desc, var_names=None, var_descs=None, create=False)

# HISTOGRAM PARAMETERS TO UPDATE
n_mu_bins = 18  # number of mu bins
n_intensity_bins = 200  # number of intensity bins
lat_band = [- np.pi / 64., np.pi / 64.]
query_instrument = ["AIA", ]

sqlite_hists = db_funcs.query_hist(sqlite_session, meth_id=method_id[1], n_mu_bins=n_mu_bins,
                                   n_intensity_bins=n_intensity_bins, lat_band=lat_band,
                                   time_min=query_time_min, time_max=query_time_max,
                                   instrument=query_instrument)
mu_bin_array, intensity_bin_array, sq_full_hist = psi_dtypes.binary_to_hist(
    sqlite_hists, n_mu_bins, n_intensity_bins)


mysql_hists = db_funcs.query_hist(mysql_session, meth_id=method_id[1], n_mu_bins=n_mu_bins,
                                  n_intensity_bins=n_intensity_bins, lat_band=lat_band,
                                  time_min=query_time_min, time_max=query_time_max,
                                  instrument=query_instrument)
mu_bin_array, intensity_bin_array, my_full_hist = psi_dtypes.binary_to_hist(
    mysql_hists, n_mu_bins, n_intensity_bins)

# verify that the two databases return identical histograms
np.all((my_full_hist - sq_full_hist) == 0.)


# CLOSE CONNECTIONS
sqlite_session.close()
mysql_session.close()
