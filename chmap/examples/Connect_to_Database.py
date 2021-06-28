"""
Example script for connecting to the database.
  - For an SQLite database, specifying a new filename will create a database
  - For a MySQL database, connecting to an existing database will also create
     any tables that are missing from the database schema.
"""

import os
import datetime

import chmap.database.db_classes as db_class
import chmap.database.db_funs as db_funcs
import chmap.utilities.datatypes.datatypes as psi_dtypes
import chmap.utilities.plotting.psi_plotting as psi_plots

# INITIALIZE DATABASE CONNECTION
# Database paths
map_data_dir = "/Volumes/extdata2/CHMAP_DB_example/maps"
hdf_data_dir = "/Volumes/extdata2/CHMAP_DB_example/processed_images"

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
db_loc = "/Volumes/extdata2/CHMAP_DB_example/chmap_example.db"

# specify which database to connect to (unnecessary for SQLite)
mysql_db_name = "chd"

# Establish connection to database
db_session = db_funcs.init_db_conn(db_type, db_class.Base, db_loc, db_name=mysql_db_name,
                                   user=user, password=password, cred_dir=cred_dir)

# SAMPLE QUERY
# use database session to query available pre-processed images
query_time_min = datetime.datetime(2011, 2, 1, 0, 0, 0)
query_time_max = datetime.datetime(2011, 2, 1, 12, 0, 0)

image_pd = db_funcs.query_euv_images(db_session, time_min=query_time_min,
                                     time_max=query_time_max)

# view a snapshot of the results
image_pd.loc[:, ['date_obs', 'instrument', 'fname_hdf']]

# open the first image
image_path = os.path.join(hdf_data_dir, image_pd.fname_hdf[0])
psi_image = psi_dtypes.read_los_image(image_path)

# plot deconvolved image
psi_plots.PlotImage(psi_image)

# CLOSE CONNECTION
db_session.close()
