"""
Tamar Ervin
Date: May 9, 2021
Spectral clustering for CH and AR
detection
"""

import numpy as np
import h5py as h5
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from skimage import measure
from scipy.spatial.distance import cdist
import tensorflow as tf
import scipy
import matplotlib.colors as colors
import modules.datatypes as psi_d_types
import modules.Plotting as Plotting

import numpy as np
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
import networkx as nx
import seaborn as sns
sns.set()

# ------ IMAGE PARAMETERS ------- #

# Image size that we are going to use
IMG_HEIGHT = 128
IMG_WIDTH = 128
# Our images are RGB (3 channels)
N_CHANNELS = 3
# Number of  intensity clusters
N_CLUSTERS = 12


# mapping parameters
x_range = [0, 2 * np.pi]
y_range = [-1, 1]
map_nycoord = IMG_HEIGHT
map_nxcoord = IMG_WIDTH
map_y = np.linspace(y_range[0], y_range[1], map_nycoord, dtype='<f4')
map_x = np.linspace(x_range[0], x_range[1], map_nxcoord, dtype='<f4')

### INPUT MAPS
# read in h5 file of training data
h5_file = '/Volumes/CHD_DB/data_images_small.h5'
hf_train = h5.File(h5_file, 'r')

dates_train = [key for key in hf_train.keys()]
image = []
for date in dates_train:
    g = hf_train.get(date)
    image.append(np.array(g['euv_image']))
hf_train.close()

# resize data
x_train = np.array(image)
# new_col = np.array([0]*len(x_train))
# all_data = np.hstack((x_train, np.atleast_2d(new_col).T))

arr4d = np.zeros(shape=(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3))
arr4d[:, :, :, 0] = x_train
x_train = tf.image.resize(arr4d, size=(IMG_HEIGHT, IMG_WIDTH))
x_train = np.array(x_train)
x_train = x_train[:, :, :, 0]


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
i = 20
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
arr = np.zeros((IMG_HEIGHT * IMG_WIDTH, N_CHANNELS))
arr[:, 0] = idx_col_flt
arr[:, 1] = idx_row_flt
arr[:, 2] = img2.flatten()

####################################################################################
# ------ PREDICTION OF CLUSTERING ALGORITHM ON ONE  IMAGE ------- #
####################################################################################
X_train = x_train[0].reshape(x_train[0].shape[0] * x_train[0].shape[1], -1)
W = pairwise_distances(X_train, metric="euclidean")
# vectorizer = np.vectorize(lambda x: 1 if x < 5 else 0)
# W = np.vectorize(vectorizer)(W)

# create graph
G = nx.Graph(W)

# adjacency matrix
W = nx.adjacency_matrix(G)

# degree matrix
D = np.diag(np.sum(np.array(W.todense()), axis=1))
# D = np.diag(np.sum(np.array(W), axis=1))

# laplacian matrix
L = D - W

# eigenvalues and vectors
e, v = np.linalg.eig(L)

# use k-means to classify nodes
i = np.where(e < 0.5)[0]
U = np.array(v[:, i[1]])
km = KMeans(init='k-means++', n_clusters=8)
km.fit(U)
labels = km.labels_
pred_clustered = labels.reshape(IMG_HEIGHT, IMG_WIDTH)


# test k-means
# X_fit = img2.reshape(img2.shape[0] * img2.shape[1], -1)
# spectral = SpectralClustering(n_clusters=N_CLUSTERS, random_state=0, assign_labels='discretize').fit(X_fit)
# labels = spectral.labels_
# # euv_pred_clustered = clustered.reshape(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)
# pred_clustered = labels.reshape(IMG_HEIGHT, IMG_WIDTH)
# # plt.imshow(pred_clustered)
# # plt.colorbar()
#
#
