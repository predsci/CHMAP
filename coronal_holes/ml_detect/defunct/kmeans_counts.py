"""
Tamar Ervin
Date: May 9, 2021
K-Means Unsupervised Clustering
for CH Detection using position and intensity
log normalized values to cluster
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
# Number of  intensity clusters
N_CLUSTERS = 14
# Number of channels for input array
N_CHANNELS = 3
# Weights for connectivity
weights = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

# mapping parameters
x_range = [0, 2 * np.pi]
y_range = [-1, 1]
map_nycoord = IMG_HEIGHT
map_nxcoord = IMG_WIDTH
map_y = np.linspace(y_range[0], y_range[1], map_nycoord, dtype='<f4')
map_x = np.linspace(x_range[0], x_range[1], map_nxcoord, dtype='<f4')

### RGB COLOR MAPS
# read in h5 file of training data
h5_file = '/Volumes/CHD_DB/map_data_intensity.h5'
hf_train = h5.File(h5_file, 'r')

dates_train = [key for key in hf_train.keys()]
image = []
for date in dates_train:
    g = hf_train.get(date)
    image.append(np.array(g['euv_image']))
hf_train.close()


# ------- CLUSTER BRIGHTNESS FUNCTION ------- #
# get cluster brightness
def cluster_brightness(clustered_img, org_img, n_clusters):
    # create average color array
    avg_color = []

    for i in range(0, n_clusters):
        cluster_indices = np.where(clustered_img == i)

        # average per row
        average_color_per_row = np.average(org_img[cluster_indices], axis=0)

        # find average across average per row
        # avg_color.append(np.average(average_color_per_row, axis=0))
        avg_color.append(average_color_per_row)

    return avg_color


####################################################################################
# ------ DATA PRIMING ------- #
####################################################################################
img_array = np.array(image)
i = 0
img = img_array[i]
img1 = np.where(img == -9999, 0, img)
img2 = np.log(img1)
img2 = np.where(img2 == -np.inf, 0, img2)

# create array
idx = np.indices((IMG_HEIGHT, IMG_WIDTH))
idx_row = idx[0]
# idx_row = np.log(idx_row)
# idx_row = np.where(idx_row == -np.inf, 0, idx_row)
idx_row = idx_row / np.max(idx_row)
idx_col = idx[1]
# idx_col = np.log(idx_col)
# idx_col = np.where(idx_col == -np.inf, 0, idx_col)
idx_col = idx_col / np.max(idx_col)


# flatten arrays
idx_col_flt = idx_col.flatten()
idx_row_flt = idx_row.flatten()

# create arrays
a = 1.4
arr = np.zeros((IMG_HEIGHT * IMG_WIDTH, N_CHANNELS))
arr[:, 0] = idx_col_flt*a
arr[:, 1] = idx_row_flt*a
arr[:, 2] = img2.flatten()*2



####################################################################################
# ------ ELBOW METHOD TO DETERMINE BEST K VALUE ------- #
####################################################################################
# # elbow method to determine best K value
# distortions = []
# kmeans = []
# kvals = range(6, 20, 1)
#
# # train k-means on k = 1 to 20
# for k in kvals:
#     output = KMeans(n_clusters=k, random_state=0, init='k-means++').fit(arr)
#     kmeans.append(output)
#     distortions.append(sum(np.min(cdist(arr, output.cluster_centers_, 'euclidean'), axis=1)) / arr.shape[0])
#
# # plot elbow method
# plt.plot(kvals, distortions, 'bx-')
# plt.xticks(kvals)
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()

####################################################################################
# ------ PREDICTION OF CLUSTERING ALGORITHM ON ONE  IMAGE ------- #
####################################################################################
# test k-means
X_fit = img1.reshape(img1.shape[0] * img1.shape[1], -1)
optimalk = KMeans(n_clusters=N_CLUSTERS, random_state=0, init='k-means++').fit(arr)
labels = optimalk.labels_
clustered = optimalk.cluster_centers_[optimalk.labels_]
# euv_pred_clustered = clustered.reshape(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)
pred_clustered = labels.reshape(IMG_HEIGHT, IMG_WIDTH)
# plt.imshow(pred_clustered)
# plt.colorbar()

# get cluster brightnesses
avg_color = cluster_brightness(pred_clustered, img2, N_CLUSTERS)
color_order = np.argsort(avg_color)

### CH Detection
chd_clustered = pred_clustered + 1
chd_clustered = np.where(np.logical_or(chd_clustered == color_order[0]+1, chd_clustered == color_order[1]+1),
                         N_CLUSTERS+1, 0)
# chd_clustered = np.where(chd_clustered == color_order[0]+1, N_CLUSTERS+1, 0)
chd_clustered = np.where(chd_clustered == N_CLUSTERS+1, 1, 0)

# area constraint
chd_labeled = measure.label(chd_clustered, connectivity=2, background=0, return_num=True)

# get area
chd_area = [props.area for props in measure.regionprops(chd_labeled[0])]

# remove CH with less than 7 pixels in area
chd_good_area = np.where(np.array(chd_area) > 10)
indices = []
chd_plot = np.zeros(chd_labeled[0].shape)
for val in chd_good_area[0]:
    val_label = val + 1
    indices.append(np.logical_and(chd_labeled[0] == val_label, val in chd_good_area[0]))
for idx in indices:
    chd_plot[idx] = chd_labeled[0][idx] + 1

#### ACTIVE REGION DETECTION
# get cluster brightness
ar_clustered = pred_clustered + 1
ar_clustered = np.where(ar_clustered == color_order[-1]+1, N_CLUSTERS+1, 0)
ar_clustered = np.where(ar_clustered == N_CLUSTERS+1, 1, 0)

# area constraint
ar_labeled = measure.label(ar_clustered, connectivity=2, background=0, return_num=True)

# get area
ar_area = [props.area for props in measure.regionprops(ar_labeled[0])]

# remove AR with less than 6 pixels in area
ar_good_area = np.where(np.array(ar_area) > 6)
indices = []
ar_plot = np.zeros(ar_labeled[0].shape)
for val in ar_good_area[0]:
    val_label = val + 1
    indices.append(np.logical_and(ar_labeled[0] == val_label, val in ar_good_area[0]))
for idx in indices:
    ar_plot[idx] = ar_labeled[0][idx] + 1

# chd
chd_plot = np.where(chd_plot > 0, 1, 0)
chd_labeled = measure.label(chd_plot, connectivity=2, background=0, return_num=True)
psi_chd_map = psi_d_types.PsiMap(data=img, chd=chd_plot, x=map_x, y=map_y)

# ar
ar_plot = np.where(ar_plot > 0, 1, 0)
ar_labeled = measure.label(ar_plot, connectivity=2, background=0, return_num=True)
psi_ar_map = psi_d_types.PsiMap(data=img, chd=ar_plot, x=map_x, y=map_y)

# plot maps and save
fig_num = i
# title = 'Area Constrained: ' + str(dates_train[i]) + '\nNumber of detected CH: ' + str(chd_labeled[1]) + '\nNumber of detected AR: ' + str(ar_labeled[1]) + '\nClusters: ' + str(N_CLUSTERS) + ", Weight: " + str(a)
title = 'Minimum Intensity Merge: Unsupervised Detection Map\nDate: ' + str(dates_train[i]) + '\nNumber of detected CH: ' + str(chd_labeled[1]) + ', Number of detected AR: ' + str(ar_labeled[1])
Plotting.PlotMap(psi_chd_map, title=title, nfig=fig_num)
Plotting.PlotMap(psi_chd_map, map_type='Contour', title=title, nfig=fig_num)
Plotting.PlotMap(psi_ar_map, map_type='Contour', title=title, nfig=fig_num)
plt.savefig('/Volumes/CHD_DB/pred_maps/k_means/map_' + str(i))

#### testing CV2
# rect = cv2.minAreaRect(chd_plot)

