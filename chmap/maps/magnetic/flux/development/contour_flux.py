
"""
Working example to calculate the net coronal hole flux
  - Input a coronal hole Contour object
  - Lookup closest Br map(s) in database
  - Calculate flux
"""


import os
import datetime
import numpy as np
import pickle

import utilities.datatypes.datatypes as psi_datatype
from settings.app import App
import chmap.database.db_classes as db_class
import chmap.database.db_funs as db_funs
import chmap.maps.util.map_manip as map_manip


###### ------ PARAMETERS TO UPDATE -------- ########

# File containing Contours
contour_file = "/Users/turtle/Dropbox/MyNACD/test_data/2011-02-17T18_00_30.000.pkl"
# contour_file = "/Users/turtle/Dropbox/MyNACD/test_data/2011-02-17 02_00_42.pkl"
# read all pickle objects
objects = []
with (open(contour_file, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

test_contour = objects[0].contour_list[1]
# having trouble with the astropy timestamps
frame_timestamp = datetime.datetime(2011, 2, 17, 18, 0, 30)

# define window that we are willing to look at
window_half_width = datetime.timedelta(hours=12)
window_min = frame_timestamp - window_half_width
window_max = frame_timestamp + window_half_width

# Data_File query (in the form of a list)
file_type = ["magnetic map", ]
file_provider = ["lmsal", ]

# declare map parameters (?)
R0 = 1.01

# recover database paths
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME
map_dir = App.MAP_FILE_HOME
# set path to raw mag-flux maps
raw_mag_dir = os.path.join(database_dir, 'raw_maps')

# designate which database to connect to
use_db = "mysql-Q"      # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
                        # 'mysql-Q_test' Use the development database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

# Establish connection to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)
    db_session = db_funs.init_db_conn(db_name=use_db, chd_base=db_class.Base,
                                      sqlite_path=sqlite_path)
elif use_db in ('mysql-Q', 'mysql-Q_test'):
    # setup database connection to MySQL database on Q
    db_session = db_funs.init_db_conn(db_name=use_db, chd_base=db_class.Base,
                                      user=user, password=password)


# find PSI magnetic map before and after CHD timestamp
map_info, data_info, method_info, image_assoc = db_funs.query_euv_maps(
    db_session, mean_time_range=[window_min, window_max], n_images=1,
    methods=["ProjFlux2Map"])

below_index = map_info.date_mean <= frame_timestamp
if any(below_index):
    first_below_index = np.max(np.where(below_index))
else:
    first_below_index = None
above_index = map_info.date_mean >= frame_timestamp
if any(above_index):
    first_above_index = np.min(np.where(above_index))
else:
    first_above_index = None

if first_below_index is None:
    if first_above_index is None:
        print("No B_r maps in the specified window. Flux calculation canceled.")
    else:
        # use first_above_index map
        full_path = os.path.join(map_dir, map_info.fname[first_above_index])
        interp_map = psi_datatype.read_psi_map(full_path)

else:
    if first_above_index is None:
        # use first_below index map
        full_path = os.path.join(map_dir, map_info.fname[first_below_index])
        interp_map = psi_datatype.read_psi_map(full_path)
    else:
        # load both maps
        full_path = os.path.join(map_dir, map_info.fname[first_above_index])
        first_above = psi_datatype.read_psi_map(full_path)
        full_path = os.path.join(map_dir, map_info.fname[first_below_index])
        first_below = psi_datatype.read_psi_map(full_path)
        # do linear interpolation
        interp_map = first_above.__copy__()
        below_weight = (frame_timestamp - map_info.date_mean[first_below_index])/(
            map_info.date_mean[first_above_index] - map_info.date_mean[first_below_index])
        above_weight = 1. - below_weight
        interp_map.data = below_weight*first_below.data + above_weight*first_above.data

# double check that Contour and Br map have same mesh???
  # temporary solution for testing purposes
y_index = test_contour.contour_pixels_theta
x_index = test_contour.contour_pixels_phi
Br_shape = interp_map.data.shape
keep_ind = (y_index <= Br_shape[0]) & (x_index <= Br_shape[1])
y_index = y_index[keep_ind]
x_index = x_index[keep_ind]

# use contour indices and Br linear approx to calc flux
# calc theta from sin(lat). increase float precision to reduce numeric
# error from np.arcsin()
theta_y = np.pi/2 - np.arcsin(np.flip(interp_map.y.astype('float64')))
# generate a mesh
map_mesh = map_manip.MapMesh(interp_map.x, theta_y)
# convert area characteristic of mesh back to map grid
map_da = np.flip(map_mesh.da.transpose(), axis=0)
# sum total flux
chd_flux = map_manip.br_flux_indices(interp_map, y_index, x_index, map_da)





