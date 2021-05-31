"""
Tamar Ervin
Date: April 13, 2021
K-Means Unsupervised Clustering
for CH Detection
"""

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import measure
from scipy.spatial.distance import cdist
import tensorflow as tf
import matplotlib.colors as colors
import modules.datatypes as psi_d_types
import modules.Plotting as Plotting

# ------ IMAGE PARAMETERS ------- #

# Image size that we are going to use
IMG_HEIGHT = 96
IMG_WIDTH = 240
# Our images are RGB (3 channels)
N_CHANNELS = 3
# Number of clusters
N_CLUSTERS = 6

# read in h5 file of training data
h5_file = '/Volumes/CHD_DB/map_data.h5'
hf_train = h5.File(h5_file, 'r')

dates_train = [key for key in hf_train.keys()]
image = []
for date in dates_train:
    g = hf_train.get(date)
    image.append(np.array(g['euv_image']))

hf_train.close()

####################################################################################
# ------ DETERMINE BEST K VALUE ------- #
####################################################################################

# resize arrays using tensorflow
map_resize = tf.image.resize(image, size=(IMG_HEIGHT, IMG_WIDTH))

# convert back from tensor to array
map_array = np.array(map_resize)

# reshape map
img = map_array[0]
image_2D = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

# elbow method to determine best K value
distortions = []
kmeans = []
kvals = range(1, 21)

# train k-means on k = 1 to 20
for k in kvals:
    output = KMeans(n_clusters=k, random_state=0, init='k-means++').fit(image_2D)
    kmeans.append(output)
    distortions.append(sum(np.min(cdist(image_2D, output.cluster_centers_, 'euclidean'), axis=1)) / image_2D.shape[0])

# plot elbow method
plt.plot(kvals, distortions, 'bx-')
plt.xticks(kvals)
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

### --- plot fitted images for each k value
figk, axsk = plt.subplots(5, 4)


def plot_kmeans(k_val):
    # creating array indices
    j = int((k_val - 1) / 5)
    i = k_val - 5 * j - 1
    k_use = kmeans[k_val - 1]

    # plotting
    title = str("K = " + str(k_val))
    axsk[i, j].set_title(title)
    k_cluster = k_use.cluster_centers_[k_use.labels_]
    k_clustered = k_cluster.reshape(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)
    axsk[i, j].imshow(k_clustered)
    axsk[i, j].set_xticks([])
    axsk[i, j].set_yticks([])


figk.suptitle("Clustering from K= 1 to 20")
for k in kvals:
    plot_kmeans(k)

####################################################################################
# ------ PREDICT K MEANS ON ONE IMAGE FOR BEST K ------- #
####################################################################################

# load in new data
h5_file = '/Volumes/CHD_DB/map_data_feb2011.h5'
hf_test = h5.File(h5_file, 'r')

dates_test =[key for key in hf_test.keys()]
test_data = []
for date in dates_test:
    g = hf_test.get(date)
    test_data.append(np.array(g['euv_image']))

hf_test.close()

# create array from test data
test_data = tf.image.resize(test_data, size=(IMG_HEIGHT, IMG_WIDTH))

# convert back from tensor to array
test_data = np.array(test_data)

# choose map for testing
img = test_data[0]

# reshape image for prediction
X_test = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

# use trained k-means model to predict cluster labels
k_pred = []
for k in kvals:
    k_use = kmeans[k - 1]
    prediction = k_use.predict(X_test)
    k_pred.append(prediction.reshape(IMG_HEIGHT, IMG_WIDTH))

# plot them
figk, axsk = plt.subplots(5, 4)


def plot_kmeans(k_val):
    j = int((k_val - 1) / 5)
    i = k_val - 5 * j - 1
    k_use = k_pred[k_val - 1]
    title = str("K = " + str(k_val))
    axsk[i, j].set_title(title)
    k_clustered = k_use.reshape(IMG_HEIGHT, IMG_WIDTH)
    axsk[i, j].imshow(k_clustered)
    axsk[i, j].set_xticks([])
    axsk[i, j].set_yticks([])


figk.suptitle("Clustering from K= 1 to 20")
for k in kvals:
    plot_kmeans(k)

####################################################################################
# ------ TRAIN ON IMAGE STACK ------- #
####################################################################################

# resize arrays using tensorflow
map_resize = tf.image.resize(image, size=(IMG_HEIGHT, IMG_WIDTH))

# convert back from tensor to array
map_array = np.array(map_resize)

# reshape training data array
# (n_samples, n_features) where n_features = IMG_HEIGHT*IMG_WIDTH*N_CHANNELS
X_train = map_array.reshape(len(map_array), -1)

### using optimal k value
# optimal k value: k = 5?? 3???!
optimalk = KMeans(n_clusters=6, random_state=0, init='k-means++').fit(X_train)

# labels reshaped
labels = optimalk.labels_

# cluster centers
clustered = optimalk.cluster_centers_[optimalk.labels_]
centers = optimalk.cluster_centers_

# reshape arrays
euv_clustered = clustered.reshape(len(X_train), IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)
centers_reshaped = centers.reshape(len(centers), IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)

# plot maps
plt.figure("Original Map")
plt.title("Original Map for K=6 Clustering")
plt.imshow(map_array[0])
plt.figure("K-Means Segmentation: K = 6")
plt.title("Clustered Map for K=6")
plt.imshow(euv_clustered[0])

####################################################################################
# ------ FIT CLUSTERING ALGORITHM ON ONE IMAGE ------- #
####################################################################################
# reshape map
img = map_array[0]
X_fit = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

# use trained k-means model to fit cluster labels
kmeans_final = optimalk.fit(X_fit)

####################################################################################
# ------ PREDICTION OF CLUSTERING ALGORITHM ON ONE IMAGE ------- #
####################################################################################

# create array from test data
test_data = tf.image.resize(test_data, size=(IMG_HEIGHT, IMG_WIDTH))

# convert back from tensor to array
test_data = np.array(test_data)

# reshape map
img = test_data[11]
X_test = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

# use trained k-means model to fit cluster labels
prediction = kmeans_final.predict(X_test)

# reshape results
euv_pred_clustered = prediction.reshape(IMG_HEIGHT, IMG_WIDTH)


##### AVERAGE BRIGHTNESS OF CLUSTER
def cluster_brightness(clustered_img, org_img):
    # create average color array
    avg_color = []

    for i in range(0, N_CLUSTERS):
        cluster_indices = np.where(clustered_img == i)

        # average per row
        average_color_per_row = np.average(org_img[cluster_indices], axis=0)

        # find average across average per row
        avg_color.append(np.average(average_color_per_row, axis=0))

    return avg_color


avg_color = cluster_brightness(euv_pred_clustered, img)
color_order = np.argsort(avg_color)

##### NUMBER OF PIXELS IN CLUSTER
euv_pred = euv_pred_clustered + 1
chd_image = np.where(euv_pred == color_order[0] + 1, 1, 0)
img_labeled = measure.label(chd_image, connectivity=2, background=0, return_num=True)
norm = colors.LogNorm(vmin=0.01, vmax=img_labeled[0].max())
plt.figure("No Area Constraint")
plt.title(str(dates_test[11]) + '\nNon-Area Constrained number of detected CH: ' + str(img_labeled[1]))
plt.imshow(img)
img_plot = np.where(img_labeled[0] > 0, 1, 0)
plt.contour(img_plot, linewidths=0.5, colors='red')
plt.xticks([])
plt.yticks([])

region_props = [props for props in measure.regionprops(img_labeled[0])]
bboxes = [props.bbox for props in measure.regionprops(img_labeled[0])]
areas = [props.area for props in measure.regionprops(img_labeled[0])]

# remove CHD with less than 3 pixels in area
chd_good_area = np.where(np.array(areas) > 6)
indices = []
chd_plot = np.zeros(img_labeled[0].shape)
for val in chd_good_area[0]:
    val_label = val + 1
    indices.append(np.logical_and(img_labeled[0] == val_label, val in chd_good_area[0]))
for idx in indices:
    chd_plot[idx] = img_labeled[0][idx] + 1
chd_labeled = measure.label(chd_plot, connectivity=2, background=0, return_num=True)
img_labeled_chd = np.where(chd_labeled[0] > 0, 1, 0)

# labeled_chd = measure.label(img_labeled_chd, connectivity=1, background=0, return_num=True)
# norm = colors.LogNorm(vmin=0.01, vmax=labeled_chd[0].max())
plt.figure("With Area Constraint")
plt.imshow(img)
# plt.contour(img_labeled_chd, linewidths=0.5, alpha=0.5, colors='red')
norm = colors.LogNorm(vmin=0.01, vmax=np.max(chd_plot))
plt.imshow(chd_labeled[0], cmap='Purples', norm=norm)
plt.title(str(dates_test[11]) + '\nArea Constrained number of detected CH: ' + str(chd_labeled[1]))
plt.xticks([])
plt.yticks([])

# plot maps
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle("Prediction on New Map")
ax1.set_title("Original Map")
ax1.imshow(img)
ax1.set_xticks([])
ax1.set_yticks([])

ax2.set_title("Clustered Prediction Map: K=6")
ax2.imshow(euv_pred_clustered)
ax2.set_xticks([])
ax2.set_yticks([])

### plot different clusters for prediction map
fig, axs = plt.subplots(3, 2)
fig.suptitle('Predicted clustering K=6 in order of increasing brightness')

euv_test = euv_pred_clustered + 1


def plot_cluster(cluster_val, i, j, org_img):
    if cluster_val == 'clustered':
        axs[i, j].set_title('Clustered Test Image K = 6')
        axs[i, j].imshow(euv_test, cmap='Purple')
    else:
        cluster = np.where(euv_test == cluster_val + 1, 1, 0)
        axs[i, j].set_title("Cluster " + str(cluster_val + 1))
        axs[i, j].imshow(org_img)
        norm = colors.LogNorm(vmin=0.01, vmax=1)
        axs[i, j].imshow(cluster, cmap="Purples", norm=norm)
    axs[i, j].set_xticks([])
    axs[i, j].set_yticks([])


for i in range(0, N_CLUSTERS):
    if i % 2 == 0:
        j = 0
    else:
        j = 1
    plot_cluster(color_order[i], int((i) / 2), j, img)

# plot_cluster(0, 0, 0, img)
# plot_cluster(1, 0, 1, img)
# plot_cluster(2, 1, 0, img)
# plot_cluster(3, 1, 1, img)
# plot_cluster(4, 2, 0, img)
# plot_cluster(5, 2, 1, img)

####################################################################################
# ------ CREATE PREDICTIONS ON ALL TEST DATA ------- #
####################################################################################

# load in new data
h5_file = '/Volumes/CHD_DB/map_data_feb2011.h5'
hf_test = h5.File(h5_file, 'r')


# get dates list
def key(f):
    return [key for key in f.keys()]


dates_list = key(hf_test)
hf_test.close()

# resize if needed
# create array from test data
test_data = tf.image.resize(test_data, size=(IMG_HEIGHT, IMG_WIDTH))

# convert back from tensor to array
test_data = np.array(test_data)

for i, img in enumerate(test_data):
    # reshape map
    X_test = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

    # use trained k-means model to fit cluster labels
    prediction = kmeans_final.fit(X_test)

    # reshape results
    clustered = prediction.labels_
    euv_pred_clustered = clustered.reshape(IMG_HEIGHT, IMG_WIDTH)

    # determine clusters corresponding to CH/AR
    avg_color = cluster_brightness(euv_pred_clustered, img)
    color_order = np.argsort(avg_color)

    # get chd prediction
    euv_pred = euv_pred_clustered + 1
    chd_image = np.where(euv_pred == color_order[0] + 1, 1, 0)
    ar_image = np.where(euv_pred == color_order[-1] + 1, 1, 0)
    chd_labeled = measure.label(chd_image, connectivity=2, background=0, return_num=True)
    ar_labeled = measure.label(ar_image, connectivity=2, background=0, return_num=True)

    # # get areas
    chd_area = [props.area for props in measure.regionprops(chd_labeled[0])]
    ar_area = [props.area for props in measure.regionprops(ar_labeled[0])]

    # remove CHD with less than 6 pixels in area
    chd_good_area = np.where(np.array(chd_area) > 6)
    indices = []
    chd_plot = np.zeros(chd_labeled[0].shape)
    for val in chd_good_area[0]:
        val_label = val + 1
        indices.append(np.logical_and(chd_labeled[0] == val_label, val in chd_good_area[0]))
    for idx in indices:
        chd_plot[idx] = chd_labeled[0][idx] + 1

    # remove AR with less than 3 pixels in area
    ar_good_area = np.where(np.array(ar_area) > 1)
    indices = []
    ar_plot = np.zeros(ar_labeled[0].shape)
    for val in ar_good_area[0]:
        val_label = val + 1
        indices.append(np.logical_and(ar_labeled[0] == val_label, val in ar_good_area[0]))
    for idx in indices:
        ar_plot[idx] = ar_labeled[0][idx] + 1

    # remove individual labels for plotting
    chd_plot = np.where(chd_plot > 0, 1, 0)
    ar_plot = np.where(ar_plot > 0, 1, 0)

    # plot and save figure
    plt.figure(dates_list[i])
    plt.imshow(img)
    plt.contour(chd_plot, linewidths=0.5, colors='red')
    plt.contour(ar_plot, linewidths=0.5, alpha=0.5, colors='blue')
    plt.title('Non-Area Constrained: ' + str(dates_test[11]) + '\nNumber of detected CH: ' + str(chd_labeled[1])
              + '\nNumber of detected AR: ' + str(ar_labeled[1]))
    plt.savefig('/Volumes/CHD_DB/pred_maps/for_movie/fit_map_' + str(i))

### MAKE MOVIE
# ffmpeg -r 4 -f image2 -s 1920x1080 -i map_%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p chd_ar_map.mp4

# one = np.where(euv_test==1, euv_test, 0)
# axs[0, 1].set_title("Cluster One")
# axs[0, 1].imshow(img)
# axs[0, 1].imshow(one, cmap="Greens", norm=norm)
# axs[0, 1].set_xticks([])
# axs[0, 1].set_yticks([])
#
# two = np.where(euv_test==2, euv_test, 0)
# axs[1, 0].set_title("Cluster Two")
# axs[1, 0].imshow(img)
# axs[1, 0].imshow(two, cmap="Greens", norm=norm)
# axs[1, 0].set_xticks([])
# axs[1, 0].set_yticks([])
#
# three = np.where(euv_test==3, euv_test, 0)
# axs[1, 1].set_title("Cluster Three")
# axs[1, 1].imshow(img)
# axs[1, 1].imshow(three, cmap="Greens", norm=norm)
#
# four = np.where(euv_test==4, euv_test, 0)
# axs[2, 0].set_title("Cluster Four")
# axs[2, 0].imshow(img)
# axs[2, 0].imshow(four, cmap="Greens", norm=norm)
#
# five = np.where(euv_test==5, euv_test, -9999)
# axs[2, 1].set_title("Cluster Five")
# axs[2, 1].imshow(img)
# axs[2, 1].imshow(five, cmap="Greens", norm=norm)
#

# image_2D = map_array[280]
# image_2D = image_2D.reshape(image_2D.shape[0]*image_2D.shape[1], -1)
# fit = optimalk.fit(image_2D)
# clustered = fit.cluster_centers_[fit.labels_]
# euv_clustered = clustered.reshape(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)
# plt.figure('clustered')
# plt.imshow(euv_clustered)
# plt.figure('org')
# plt.imshow(map_array[280])
