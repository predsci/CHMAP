
"""
Routine to create a simple EUV or CHD movie.
Python Package Dependency: OpenCV (cv2)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import datetime
import time

from chmap.database import db_funs
import chmap.database.db_classes as DBClass
import utilities.datatypes.datatypes as psi_datatype
from settings.app import App

map_dir = App.MAP_FILE_HOME

# --- User Parameters ----------------------
# image or coronal hole detection
video_type = "chd"    # 'image' or 'chd'

# define map query start and end times
movie_start = datetime.datetime(2012, 5, 20, 0, 0, 0)
movie_end = datetime.datetime(2012, 5, 21, 0, 0, 0)
# define map type and grid to query
map_methods = ['Synch_Im_Sel', 'GridSize_sinLat']
grid_size = (1600, 640)
# grid_size = (400, 160)
# map_vars = {"n_phi": [grid_size[0]-0.1, grid_size[0]+0.1],
#             "n_SinLat": [grid_size[1]-0.1, grid_size[1]+0.1],
#             "del_mu": [0.59, 0.61]}
map_vars = {"n_phi": [grid_size[0]-0.1, grid_size[0]+0.1],
            "n_SinLat": [grid_size[1]-0.1, grid_size[1]+0.1],
            "del_mu": [0.59, 0.61]}
unique_str = "del-mu06"  # additional filename component

save_dir = "/Users/turtle/Dropbox/MyNACD/video"
file_base = "synch_map"
date_str = str(movie_start.date()) + "-to-" + str(movie_end.date())
grid_str = str(grid_size[0]) + "x" + str(grid_size[1])
full_filename = file_base + "_" + video_type + "_" + unique_str + "_" + \
                date_str + "_" + grid_str + ".mp4"
full_path = os.path.join(save_dir, full_filename)

if video_type == "image":
    # define colormap
    im_cmap = plt.get_cmap('sohoeit195')
    # colormap normalization range (in log10()-scale)
    im_max = 3.0
    im_min = 0.5
elif video_type == "chd":
    # define colormap
    im_cmap = plt.get_cmap('Greys')
    # colormap normalization range
    im_min = 0.01
    im_max = 1.0

# set movie frames-per-second
fps = 12    # one day per second
# important that frame size matches data array size (transpose of plot_data.shape)
# frameSize = (1600, 640)
# frameSize = plot_data.shape[::-1]

# designate which database to connect to
use_db = "mysql-Q"      # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
                        # 'mysql-Q_test' Use the development database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

start_time = time.time()

# Establish connection to database
db_session = db_funs.init_db_conn(db_name=use_db, chd_base=DBClass.Base, user=user,
                                  password=password)

# --- Begin execution ----------------------
# query maps in time range
map_info, data_info, method_info, image_assoc = db_funs.query_euv_maps(
    db_session, mean_time_range=(movie_start, movie_end), methods=map_methods,
    var_val_range=map_vars)
# do datetime synchronic selection to filter-out periods with maps more
# often than every 2 hours

# loop through maps
for index, row in map_info.iterrows():
    print("Processing map for:", row.date_mean)
    # load map
    if row.fname[0] == "/":
        rel_path = row.fname[1:]
    else:
        rel_path = row.fname
    map_path = os.path.join(map_dir, rel_path)
    temp_map = psi_datatype.read_psi_map(map_path)

    if video_type == "image":
        plot_data = temp_map.data.copy()
        plot_data[plot_data <= 0.] = 1e-6
        plot_data = np.log10(plot_data)
    elif video_type == "chd":
        plot_data = temp_map.chd.copy()

    if index == 0:
        frameSize = list(plot_data.shape)
        frameSize.reverse()
        frameSize = tuple(frameSize)
        # initiate file and video writer. mpeg using 'mp4v' codec
        out = cv2.VideoWriter(full_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frameSize)

    # normalize data for plotting
    plot_data[plot_data > im_max] = im_max
    plot_data[plot_data < im_min] = im_min
    norm_data = (plot_data - im_min)/(im_max-im_min)

    # convert data to RGBA/RGB arrays
    rgba_arrays = im_cmap(norm_data)
    rgb_uint8_array = (rgba_arrays[:, :, 0:3] * 255).astype(np.uint8)

    # write rgb-arrays as an image/frame to video
    out.write(cv2.flip(rgb_uint8_array, 0))

# finish and close file
out.release()
out = None
cv2.destroyAllWindows()

# close database connection
db_session.close()

end_time = time.time()
print(end_time - start_time, " total seconds elapsed.")
