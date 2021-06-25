

"""
Routine to create a simple EUV movie of a single Carrington rotation.
Python Package Dependency: OpenCV (cv2)
"""

import os
import datetime
import sunpy

from chmap.database import db_funs
import chmap.database.db_classes as DBClass
from chmap.settings.app import App
import chmap.utilities.plotting.psi_plotting as psi_plot

map_dir = App.MAP_FILE_HOME

# --- User Parameters ----------------------
# select Carrington rotation for video (2054 to 2239)
cr_rot = 2124

# intensity range for log colorscale
int_range = [1.0, 4000.0]

# define map type and grid to query
map_methods = ['Synch_Im_Sel', 'GridSize_sinLat']
grid_size = (1600, 640)
# grid_size = (400, 160)
# map_vars = {"n_phi": [grid_size[0]-0.1, grid_size[0]+0.1],
#             "n_SinLat": [grid_size[1]-0.1, grid_size[1]+0.1],
#             "del_mu": [0.99, 1.01]}
map_vars = {"n_phi": [grid_size[0]-0.1, grid_size[0]+0.1],
            "n_SinLat": [grid_size[1]-0.1, grid_size[1]+0.1],
            "del_mu": [0.59, 0.61]}
unique_str = "del-mu06"  # additional filename component

# generate a movie filename
save_dir = "/Users/turtle/Dropbox/MyNACD/video"
file_base = "CR_rot"
full_filename = file_base + "_" + str(cr_rot) + ".mp4"
full_path = os.path.join(save_dir, full_filename)
# designate a directory for the frames (png) to be saved
temp_png = "/Users/turtle/Dropbox/MyNACD/video/test_pngs"
# temp_png = "/Users/turtle/CHD/tmp"

# set movie frames-per-second
fps = 12    # one day per second
# set frame dots per inch
dpi = 300   # set to None for automatic dpi estimation


# designate which database to connect to
use_db = "mysql-Q"      # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
                        # 'mysql-Q_test' Use the development database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn_old() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

# Establish connection to database
db_session = db_funs.init_db_conn_old(db_name=use_db, chd_base=DBClass.Base, user=user,
                                      password=password)

# --- Begin execution ----------------------
# estimate dates for start/end of rotation
CR_start = datetime.datetime(1853, 11, 9)
CR_dur = datetime.timedelta(days=27.2753)
estimated_center_date = CR_start + (cr_rot-0.5)*CR_dur
# get the decimal Carrington number for Earth at this time
cr_earth = sunpy.coordinates.sun.carrington_rotation_number(estimated_center_date)
# calc reasonable estimate of rotation start/end
movie_start = estimated_center_date - (cr_earth-cr_rot)*CR_dur
movie_end = estimated_center_date + (cr_rot+1-cr_earth)*CR_dur

# query maps in time range
map_info, data_info, method_info, image_assoc = db_funs.query_euv_maps(
    db_session, mean_time_range=(movie_start, movie_end), methods=map_methods,
    var_val_range=map_vars)

psi_plot.euv_map_movie(map_info, png_dir=temp_png, movie_path=full_path,
                       map_dir=map_dir, int_range=int_range, fps=fps, dpi=dpi)

db_session.close()
