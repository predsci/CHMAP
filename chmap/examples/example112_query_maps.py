
import os
import datetime
import numpy as np

from chmap.database import db_funs
import chmap.database.db_classes as DBClass
import chmap.utilities.datatypes.datatypes as psi_datatype
from chmap.settings.app import App
import chmap.utilities.plotting.psi_plotting as EasyPlot

map_dir = App.MAP_FILE_HOME

# --- User Parameters ----------------------
# Base directory that maps are housed in
map_dir = App.MAP_FILE_HOME
# this can be set manually. On Q it would be set to
# map_dir = "/extdata2/CHD_DB/maps"

# define map query start and end times
query_start = datetime.datetime(2010, 11, 30, 23, 0, 0)
query_end = datetime.datetime(2011, 1, 1, 1, 0, 0)

# define map type and grid to query
map_methods = ['Synch_Im_Sel', 'GridSize_sinLat', 'MIDM-Comb-del_mu']
# here we specify methods for synchronic image selection, a sine(lat) axis, and
# del_mu driven minimum intensity merge.

grid_size = (1600, 640)
# grid_size = (400, 160)
# parameter values are stored as floats in the DB, so input a range to query for each.
map_vars = {"n_phi": [grid_size[0]-0.1, grid_size[0]+0.1],
            "n_SinLat": [grid_size[1]-0.1, grid_size[1]+0.1],
            "del_mu": [0.59, 0.61]}
# 'n_phi' and 'n_SinLat' are number of grid points. We also want a del_mu of 0.6

# designate which database to connect to
use_db = "mysql-Q"      # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
                        # 'mysql-Q_test' Use the development database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn_old() will automatically find and use your saved password. Otherwise, enter your MySQL password here.
# If password=="", then be sure to specify the directory where encrypted credentials
# are stored.  Setting cred_dir=None will cause the code to attempt to automatically
# determine a path to the settings/ directory.
cred_dir = "/Users/turtle/GitReps/CHD/chmap/settings"

# Establish connection to database
db_session = db_funs.init_db_conn_old(db_name=use_db, chd_base=DBClass.Base, user=user,
                                      password=password, cred_dir=cred_dir)

# --- Begin execution ----------------------
# query maps in time range
map_info, data_info, method_info, image_assoc = db_funs.query_euv_maps(
    db_session, mean_time_range=(query_start, query_end), methods=map_methods,
    var_val_range=map_vars)
# the query returns multiple dataframes that together describe the map-making
# process and constituent images.  Here we are mostly interested in the map_info
# dataframe.  It contains one row per map with a number of information columns:
map_info.keys()
map_info.date_mean

# iterate through the rows of map_info
for row_index, row in map_info.iterrows():
    print("Processing map for:", row.date_mean)
    # load map (some older maps have a leading '/' that messes with os.path.join
    if row.fname[0] == "/":
        rel_path = row.fname[1:]
    else:
        rel_path = row.fname
    map_path = os.path.join(map_dir, rel_path)
    my_map = psi_datatype.read_psi_map(map_path)
    # extract needed metrics
    euv_image_data = my_map.data
    chd_data = my_map.chd   # note that the grid has been reduced since the coronal
    # hole detection was performed, so values are floats between 0. and 1. A rounding
    # operation may be required for discrete operations.
    discrete_chd = np.round(my_map.chd)
    phi_coords = my_map.x
    sinlat_coords = my_map.y
    mu_data = my_map.mu
    origin_image = my_map.origin_image

    # do stuff with data

# Simple plotting
EasyPlot.PlotMap(my_map, nfig=0)
EasyPlot.PlotMap(my_map, nfig=1, map_type="CHD")

# close database connection
db_session.close()

