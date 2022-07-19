

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import h5py as h5

from chmap.database import db_funs
import chmap.database.db_classes as DBClass
import chmap.utilities.datatypes.datatypes as psi_datatype
from chmap.settings.app import App
import chmap.utilities.plotting.psi_plotting as EasyPlot
import chmap.utilities.coord_manip as coord_manip
import chmap.maps.util.map_manip as map_manip
from chmap.settings.info import DTypes

map_dir = App.MAP_FILE_HOME
image_hdf_dir = App.PROCESSED_DATA_HOME

# --- User Parameters ----------------------

# define map query start and end times
query_start = datetime.datetime(2011, 8, 24, 23, 0, 0)
query_end = datetime.datetime(2011, 8, 25, 1, 0, 0)
# define map type and grid to query
map_methods = ['Synch_Im_Sel', 'GridSize_sinLat', 'MIDM-Comb-mu_merge']
grid_size = (1600, 640)
map_vars = {"n_phi": [grid_size[0]-0.1, grid_size[0]+0.1],
            "n_SinLat": [grid_size[1]-0.1, grid_size[1]+0.1],
            "mu_merge_cutoff": [0.39, 0.41]}
map_vars = {"mu_merge_cutoff": [0.39, 0.41]}
map_vars = {"n_phi": [grid_size[0]-0.1, grid_size[0]+0.1],
            "n_SinLat": [grid_size[1]-0.1, grid_size[1]+0.1]}

# DETECTION PARAMETERS
# region-growing threshold parameters
thresh1 = 0.95
thresh2 = 1.35
# consecutive pixel value
nc = 3
# maximum number of iterations
iters = 1000

R0 = 1.01

# MAP PARAMETERS
x_range = [0, 2 * np.pi]
y_range = [-1, 1]
reduce_map_nycoord = 640
reduce_map_nxcoord = 1600
full_map_nycoord = 2048
full_map_nxcoord = 2048*2
# generate map x,y grids. y grid centered on equator, x referenced from lon=0
full_map_y = np.linspace(y_range[0], y_range[1], full_map_nycoord, dtype='<f4')
full_map_x = np.linspace(x_range[0], x_range[1], full_map_nxcoord, dtype='<f4')
reduce_map_y = np.linspace(y_range[0], y_range[1], reduce_map_nycoord, dtype='<f4')
reduce_map_x = np.linspace(x_range[0], x_range[1], reduce_map_nxcoord, dtype='<f4')

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
# query maps in time range mu_merge_cutoff
map_methods = ['Synch_Im_Sel', 'GridSize_sinLat', 'MIDM-Comb-mu_merge']
map_info, data_info, method_info, image_assoc = db_funs.query_euv_maps(
    db_session, mean_time_range=(query_start, query_end), methods=map_methods,
    var_val_range=map_vars)

map_path = os.path.join(map_dir, map_info.fname[0])
merge_map = psi_datatype.read_psi_map(map_path)

# Plot and save
save_to = "/Users/turtle/Dropbox/MyNACD/ron_code/test_images/new_data_merge-overlap04.png"
EasyPlot.PlotMap(merge_map, nfig=0)
plt.savefig(save_to)

map_methods = ['Synch_Im_Sel', 'GridSize_sinLat', 'MIDM-Comb-del_mu']
map_info2, data_info2, method_info2, image_assoc2 = db_funs.query_euv_maps(
    db_session, mean_time_range=(query_start, query_end), methods=map_methods,
    var_val_range=map_vars)
for ii in range(map_info2.shape[0]):
    map_path = os.path.join(map_dir, map_info2.fname[ii])
    del_map = psi_datatype.read_psi_map(map_path)
    method_index = (method_info2.var_name == "del_mu") & \
                   (method_info2.map_id == map_info2.map_id[ii])
    del_mu = method_info2.var_val[method_index]

    # Plot and save
    save_to = "/Users/turtle/Dropbox/MyNACD/ron_code/test_images/" \
              "new_data_merge-del" + str(del_mu.item()) + ".png"
    EasyPlot.PlotMap(del_map, nfig=0)
    plt.savefig(save_to)
    plt.close(0)

# load Ron's images
A_filename = "/Users/turtle/Dropbox/MyNACD/ron_code/test_images/A.h5"
h5file = h5.File(A_filename, 'r')
dataset_names = list(h5file.keys())
data_name = dataset_names[0]
f = h5file[data_name]
dims = f.shape
ndims = np.ndim(f)

z = np.array([])
# Get the scales if they exist:
for i in range(0, ndims):
    if i == 0:
        if (len(h5file[data_name].dims[0].keys()) != 0):
            x = h5file[data_name].dims[0][0]
    elif i == 1:
        if (len(h5file[data_name].dims[1].keys()) != 0):
            y = h5file[data_name].dims[1][0]
    elif i == 2:
        if (len(h5file[data_name].dims[2].keys()) != 0):
            z = h5file[data_name].dims[2][0]

x = np.array(x)
y = np.array(y)
z = np.array(z)
f = np.array(f)

h5file.close()

losA = psi_datatype.LosImage(f, x, y)


B_filename = "/Users/turtle/Dropbox/MyNACD/ron_code/test_images/B.h5"
h5file = h5.File(A_filename, 'r')
dataset_names = list(h5file.keys())
data_name = dataset_names[0]
f = h5file[data_name]
dims = f.shape
ndims = np.ndim(f)

z = np.array([])
# Get the scales if they exist:
for i in range(0, ndims):
    if i == 0:
        if (len(h5file[data_name].dims[0].keys()) != 0):
            x = h5file[data_name].dims[0][0]
    elif i == 1:
        if (len(h5file[data_name].dims[1].keys()) != 0):
            y = h5file[data_name].dims[1][0]
    elif i == 2:
        if (len(h5file[data_name].dims[2].keys()) != 0):
            z = h5file[data_name].dims[2][0]

x = np.array(x)
y = np.array(y)
z = np.array(z)
f = np.array(f)

h5file.close()

losB = psi_datatype.LosImage(f, x, y)


AIA_filename = "/Users/turtle/Dropbox/MyNACD/ron_code/test_images/AIA.h5"
h5file = h5.File(A_filename, 'r')
dataset_names = list(h5file.keys())
data_name = dataset_names[0]
f = h5file[data_name]
dims = f.shape
ndims = np.ndim(f)

z = np.array([])
# Get the scales if they exist:
for i in range(0, ndims):
    if i == 0:
        if (len(h5file[data_name].dims[0].keys()) != 0):
            x = h5file[data_name].dims[0][0]
    elif i == 1:
        if (len(h5file[data_name].dims[1].keys()) != 0):
            y = h5file[data_name].dims[1][0]
    elif i == 2:
        if (len(h5file[data_name].dims[2].keys()) != 0):
            z = h5file[data_name].dims[2][0]

x = np.array(x)
y = np.array(y)
z = np.array(z)
f = np.array(f)

h5file.close()

losAIA = psi_datatype.LosImage(f, x, y)

# query DB EUV images to use their meta data
euv_images = db_funs.query_euv_images(db_session, time_min=query_start, time_max=query_end)

A_ind = euv_images.instrument == "EUVI-A"
A_fname = euv_images.fname_hdf[A_ind].item()
A_path = os.path.join(image_hdf_dir, A_fname)
A_los = psi_datatype.read_euv_image(A_path)
losA.info = A_los.info

B_ind = euv_images.instrument == "EUVI-B"
B_fname = euv_images.fname_hdf[B_ind].item()
B_path = os.path.join(image_hdf_dir, B_fname)
B_los = psi_datatype.read_euv_image(B_path)
losB.info = B_los.info

AIA_ind = euv_images.instrument == "AIA"
AIA_fname = euv_images.fname_hdf[AIA_ind].item()
AIA_path = os.path.join(image_hdf_dir, AIA_fname)
AIA_los = psi_datatype.read_euv_image(AIA_path)
losAIA.info = AIA_los.info

# interpolate to map
iit_list = [losA, losB, losAIA]
map_list = [psi_datatype.PsiMap()] * len(iit_list)
for ii in range(len(iit_list)):
    iit_list[ii].no_data_val = 0.001
    # do coronal hole detection(?)
    # interpolate chd to map(?)
    # interpolate image to a map
    interp_result = coord_manip.interp_los_image_to_map(
        iit_list[ii], R0=R0, map_x=full_map_x, map_y=full_map_y, no_data_val=0.001)
    origin_image = np.full(interp_result.data.shape, 0, dtype=DTypes.MAP_ORIGIN_IMAGE)
    origin_image[interp_result.data > iit_list[ii].no_data_val] = ii

    # Partially populate a map object with grid and data info
    map_out = psi_datatype.PsiMap(interp_result.data, interp_result.x, interp_result.y,
                                  mu=interp_result.mu_mat, map_lon=interp_result.map_lon,
                                  origin_image=origin_image, no_data_val=iit_list[ii].no_data_val)
    # combine into list
    map_list[ii] = map_out

# minimum intensity merge
merged_map = map_manip.combine_maps(map_list, mu_cutoff=0.0, mu_merge_cutoff=0.4,
                                    del_mu=None)

EasyPlot.PlotMap(merged_map)


db_session.close()
