"""
using HDF5 data to make prettier plots
"""

import numpy as np
import h5py as h5
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import measure
from scipy.spatial.distance import cdist
import tensorflow as tf
import scipy
import matplotlib.colors as colors
import modules.datatypes as psi_d_types
import modules.Plotting as Plotting
import analysis.ml_analysis.ch_detect.ml_functions as ml_funcs
import matplotlib as mpl


# ------ IMAGE PARAMETERS ------- #

# Image size that we are going to use
IMG_HEIGHT = 120
IMG_WIDTH = 300
# Number of  intensity clusters
N_CLUSTERS = 13
# Number of channels for input array
N_CHANNELS = 3
# Model weights
model_h5 = 'model_unet_FINAL.h5'

# mapping parameters
x_range = [0, 2 * np.pi]
y_range = [-1, 1]
map_nycoord = IMG_HEIGHT
map_nxcoord = IMG_WIDTH
map_y = np.linspace(y_range[0], y_range[1], map_nycoord, dtype='<f4')
map_x = np.linspace(x_range[0], x_range[1], map_nxcoord, dtype='<f4')

### INTENSITY MAPS
# read in h5 file of training data
h5_file = '/Volumes/CHD_DB/data_train.h5'
hf_train = h5.File(h5_file, 'r')

dates_train = [key for key in hf_train.keys()]
image = []
for date in dates_train:
    g = hf_train.get(date)
    image.append(np.array(g['euv_image']))
hf_train.close()

# CNN CH detection
model = ml_funcs.load_model(model_h5, IMG_SIZE=2048, N_CHANNELS=3)
# create correct data format
for image_data in image:
    scalarMap = mpl.cm.ScalarMappable(norm=colors.LogNorm(vmin=1.0, vmax=np.max(image_data)),
                                      cmap='sohoeit195')
    colorVal = scalarMap.to_rgba(image_data, norm=True)
    data_x = colorVal[:, :, :3]

    # apply ml algorithm
    ml_output = model.predict(data_x[tf.newaxis, ...], verbose=1)
    result = (ml_output[0] > 0.1).astype(np.uint8)

    use_chd = np.logical_and(image_data != -9999, result.squeeze() > 0)
    pred = np.zeros(shape=result.squeeze().shape)
    pred[use_chd] = result.squeeze()[use_chd]

#### STEP FOUR: CONVERT TO MAP ####
map_list, methods_list, data_info, map_info = chd_funcs.create_singles_maps(inst_list, date_pd,
                                                                            iit_list,
                                                                            chd_image_list,
                                                                            methods_list, map_x,
                                                                            map_y, R0)