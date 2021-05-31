"""
Tamar Ervin
Date: May 2, 2021
K-Means Unsupervised Clustering
for CH Detection using intensity clustering
then spatial clustering
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

# ------ IMAGE PARAMETERS ------- #

# Image size that we are going to use
IMG_HEIGHT = 120
IMG_WIDTH = 300
# Our images are RGB (3 channels)
N_CHANNELS = 3
# Number of  intensity clusters
N_CLUSTERS = 6
# Number of spatial clusters
N_CLUSTERS_SPATIAL = 14

# mapping parameters
x_range = [0, 2 * np.pi]
y_range = [-1, 1]
map_nycoord = IMG_HEIGHT
map_nxcoord = IMG_WIDTH
map_y = np.linspace(y_range[0], y_range[1], map_nycoord, dtype='<f4')
map_x = np.linspace(x_range[0], x_range[1], map_nxcoord, dtype='<f4')

### RGB COLOR MAPS
# read in h5 file of training data
h5_file = '/Volumes/CHD_DB/map_data.h5'
hf_train = h5.File(h5_file, 'r')

dates_train = [key for key in hf_train.keys()]
image = []
for date in dates_train:
    g = hf_train.get(date)
    image.append(np.array(g['euv_image']))

hf_train.close()


# ------- CLUSTER BRIGHTNESS FUNCTION ------- #
# get cluster brightness
def cluster_brightness(clustered_img, org_img):
    # create average color array
    avg_color = []

    for i in range(0, N_CLUSTERS):
        cluster_indices = np.where(clustered_img == i)

        # average per row
        average_color_per_row = np.average(org_img[cluster_indices], axis=0)

        # find average across average per row
        # avg_color.append(np.average(average_color_per_row, axis=0))
        avg_color.append(average_color_per_row)

    return avg_color

####################################################################################
# ------ PREDICTION OF CLUSTERING USING INTENSITY THEN SPATIAL CLUSTERING ------- #
####################################################################################
### K-Means Intensity then spatial??? ####

# normalize data
img_array = np.array(image)

# get indices and normalize
idx = np.indices((IMG_HEIGHT, IMG_WIDTH))
idx_row = idx[0]
idx_row = idx_row/np.max(idx_row)
idx_col = idx[1]

idx_col = idx_col/np.max(idx_col)

for i, img in enumerate(img_array):
    img32 = np.float32(img)
    gray_image = cv2.cvtColor(img32, cv2.COLOR_RGB2GRAY)

    # flatten arrays
    idx_col_flt = idx_col.flatten()
    idx_row_flt = idx_row.flatten()
    img_gray_flt = gray_image.flatten()

    # create array
    arr = np.zeros((IMG_HEIGHT*IMG_WIDTH, 3))
    arr[:, 0] = idx_col_flt
    arr[:, 1] = idx_row_flt
    arr[:, 2] = img_gray_flt

    ### INTENSITY CLUSTERING
    X_fit = gray_image.reshape(gray_image.shape[0] * gray_image.shape[1], -1)
    intensity_kmeans = KMeans(n_clusters=6, random_state=0, init='k-means++').fit(X_fit)
    labels = intensity_kmeans.labels_
    clustered = intensity_kmeans.cluster_centers_[intensity_kmeans.labels_]
    pred_clustered = labels.reshape(IMG_HEIGHT, IMG_WIDTH)

    # get cluster brightness
    avg_color = cluster_brightness(pred_clustered, gray_image)
    color_order = np.argsort(avg_color)

    ### CHD SPATIAL CLUSTERING
    chd_image = pred_clustered + 1
    chd_image = np.where(chd_image == color_order[0]+1, N_CLUSTERS+1, 0)
    chd_image = np.where(chd_image == N_CLUSTERS+1, 1, 0)

    arr = np.zeros((IMG_HEIGHT*IMG_WIDTH, 3))
    arr[:, 0] = idx_col_flt
    arr[:, 1] = idx_row_flt
    arr[:, 2] = chd_image.flatten()

    spatial_kmeans = KMeans(n_clusters=N_CLUSTERS_SPATIAL, random_state=0, init='k-means++').fit(arr)
    spatial_labels = spatial_kmeans.labels_
    spatial_clustered = spatial_kmeans.cluster_centers_[spatial_kmeans.labels_]
    chd_spatial_clustered = spatial_labels.reshape(IMG_HEIGHT, IMG_WIDTH)

    # get chd indices only
    cluster_indices = np.where(pred_clustered == color_order[0])
    chd_clustered = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    chd_clustered[cluster_indices] = chd_spatial_clustered[cluster_indices]

    # area constraint
    chd_labeled = measure.label(chd_clustered, connectivity=2, background=0, return_num=True)

    # get area
    chd_area = [props.area for props in measure.regionprops(chd_labeled[0])]

    # remove CH with less than 6 pixels in area
    chd_good_area = np.where(np.array(chd_area) > 6)
    indices = []
    chd_plot = np.zeros(chd_labeled[0].shape)
    for val in chd_good_area[0]:
        val_label = val + 1
        indices.append(np.logical_and(chd_labeled[0] == val_label, val in chd_good_area[0]))
    for idx in indices:
        chd_plot[idx] = chd_labeled[0][idx] + 1

    #### ACTIVE REGION DETECTION
    # get cluster brightness
    ar_image = pred_clustered + 1
    ar_image = np.where(ar_image == color_order[-1]+1, N_CLUSTERS+1, 0)
    ar_image = np.where(ar_image == N_CLUSTERS+1, 1, 0)

    ### SPATIAL CLUSTERING
    arr = np.zeros((IMG_HEIGHT*IMG_WIDTH, 3))
    arr[:, 0] = idx_col_flt
    arr[:, 1] = idx_row_flt
    arr[:, 2] = ar_image.flatten()

    # spatial clustering
    spatial_kmeans = KMeans(n_clusters=N_CLUSTERS_SPATIAL, random_state=0, init='k-means++').fit(arr)
    spatial_labels = spatial_kmeans.labels_
    spatial_clustered = spatial_kmeans.cluster_centers_[spatial_kmeans.labels_]
    ar_spatial_clustered = spatial_labels.reshape(IMG_HEIGHT, IMG_WIDTH)

    # get ar indices only
    cluster_indices = np.where(pred_clustered == color_order[-1])
    ar_clustered = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    ar_clustered[cluster_indices] = ar_spatial_clustered[cluster_indices]

    # area constraint
    ar_labeled = measure.label(ar_clustered, connectivity=2, background=0, return_num=True)

    # # get area
    ar_area = [props.area for props in measure.regionprops(ar_labeled[0])]

    # remove AR with less than 3 pixels in area
    ar_good_area = np.where(np.array(ar_area) > 4)
    indices = []
    ar_plot = np.zeros(ar_labeled[0].shape)
    for val in ar_good_area[0]:
        val_label = val + 1
        indices.append(np.logical_and(ar_labeled[0] == val_label, val in ar_good_area[0]))
    for idx in indices:
        ar_plot[idx] = ar_labeled[0][idx] + 1

    #### CREATE PSI MAP TYPES
    # chd
    chd_plot = np.where(chd_plot > 0, 1, 0)
    chd_labeled = measure.label(chd_plot, connectivity=2, background=0, return_num=True)
    psi_chd_map = psi_d_types.PsiMap(data=img, chd=chd_plot, x=map_x, y=map_y)

    # ar
    ar_plot = np.where(ar_plot > 0, 1, 0)
    ar_labeled = measure.label(ar_plot, connectivity=2, background=0, return_num=True)
    psi_ar_map = psi_d_types.PsiMap(data=img, chd=ar_plot, x=map_x, y=map_y)

    # plot maps and save
    title = 'Non-Area Constrained: ' + str(dates_train[i]) + '\nNumber of detected CH: ' + str(chd_labeled[1]) + '\nNumber of detected AR: ' + str(ar_labeled[1])
    Plotting.PlotMap(psi_chd_map, title=title, nfig=i)
    Plotting.PlotMap(psi_chd_map, map_type='Contour', title=title, nfig=i)
    Plotting.PlotMap(psi_ar_map, map_type='Contour', title=title, nfig=i)
    plt.savefig('/Volumes/CHD_DB/pred_maps/position/map_' + str(i))
