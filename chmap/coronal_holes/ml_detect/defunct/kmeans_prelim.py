"""
testing K-means clustering for image segmentation
"""
import cv2
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.colors as colors
from sklearn.neighbors import kneighbors_graph

# get map
h5_file = '/Volumes/CHD_DB/map_data_small.h5'
hf = h5.File(h5_file, 'r')

dates = hf.keys()
image = []
# mask = []
for date in dates:
    g = hf.get(date)
    image.append(np.array(g['euv_image']))

hf.close()


x_train = np.array(image)

### using scikitlearn
from sklearn.cluster import KMeans

# stacked images
n_channels = 3
img_height = 128
img_width = 128
n_clusters = 5

x_train2 = np.resize(x_train, (len(x_train), img_height, img_width, n_channels))
X_train = x_train2.reshape(len(x_train2), -1)

img = x_train2[0][:, :, :3]
image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])

kmeans = KMeans(n_clusters=n_clusters, random_state=0, init='k-means++').fit(X_train)
labels = kmeans.labels_
clustered = kmeans.cluster_centers_[kmeans.labels_]
euv_clustered = clustered.reshape(img_height, img_width, n_channels)
chd_clustered = labels.reshape(img_height, img_width, 1)

zero_labels = np.where(labels==0)
one_labels = np.where(labels==1)
two_labels = np.where(labels==2)
three_labels = np.where(labels==3)
four_labels = np.where(labels==4)

plt.figure(0)
plt.imshow(image[50])
plt.figure(1)
plt.imshow(image[1])
plt.figure(2)
plt.imshow(image[97])
plt.figure(3)
plt.imshow(image[97])
plt.figure(4)
plt.imshow(image[76])


# one image
n_channels = 3
img_height = 128
img_width = 128
n_clusters = 5

img = image[0]
image_2D = cv2.resize(img, dsize=(img_height, img_width), interpolation=cv2.INTER_AREA)
image_2D = image_2D.reshape(image_2D.shape[0]*image_2D.shape[1])

# connectivity graph
adjMatrix = kneighbors_graph(image_2D, 3, mode='connectivity', include_self=True)
A = adjMatrix.toarray()
kmeans_connect = KMeans(n_clusters=n_clusters, random_state=0, init='k-means++').fit(adjMatrix)


kmeans = KMeans(n_clusters=n_clusters, random_state=0, init='k-means++').fit(image_2D)
labels = kmeans_connect.labels_
clustered = kmeans_connect.cluster_centers_[kmeans_connect.labels_]
euv_clustered_one = np.resize(clustered, (img.shape[0], img.shape[1]))
chd_clustered_one = np.resize(labels, (img.shape[0], img.shape[1]))


# K Medoids method
from sklearn_extra.cluster import KMedoids

n_channels = 3
img_height = 128
img_width = 128
n_clusters = 5

img = image[0][:, :, :3]
image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])

kmedoids = KMedoids(n_clusters=n_clusters, random_state=0, init='k-medoids++').fit(X_train)
labels = kmedoids.labels_
clustered = kmedoids.cluster_centers_[kmedoids.labels_]
euv_clustered_medoids = clustered.reshape(img.shape[0], img.shape[1], n_channels)
chd_clustered_medoids = labels.reshape(img.shape[0], img.shape[1], 1)

# predict CHD
img2 = x_train2[2][:, :, :3]
pred_image_2D = img2.reshape(img2.shape[0]*img2.shape[1], img2.shape[2])
pred_img = kmeans.predict(pred_image_2D)
pred_labels = kmeans.labels_
clustered_pred = kmeans.cluster_centers_[kmeans.labels_]
# Reshape back the image from 2D to 3D image
euv_clustered_pred = pred_labels.reshape(img2.shape[0], img2.shape[1], 1)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(euv_clustered_one)
ax1.set_title('Clustered EUV Image')
ax2.imshow(clustered)
ax2.set_title('Clustered CHD Image')
fig.suptitle("EUV Clustering to CHD Map: K=2")
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax1.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
plt.show()

#get CHD indices to use
use_chd = np.where(mask[0] == 1, mask[0], -9999)
chd_result = np.zeros(chd_clustered.shape)
chd_result[np.logical_not(use_chd==-9999)] = chd_clustered[np.logical_not(use_chd==-9999)]

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(euv_clustered)
ax1.set_title('Clustered EUV Image')
ax2.imshow(tf.keras.preprocessing.image.array_to_img(chd_result))
ax2.set_title('CHD Detection')
fig.suptitle("EUV Clustering to CHD Map: K=5")
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax1.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
plt.show()

##### testing
## threshold
euv1 = np.where(euv_clustered<=0.5, 0, 1)
chd_result = np.logical_and(euv1 == 0, euv1 == 0)
chd_result = chd_result.astype(int)

# unique values
unique = np.unique(euv_clustered)
unique = np.unique(unique)
euv1 = np.where(euv_clustered<=0.0001, 0, euv_clustered)
chd_result = np.logical_and(euv1 == 0, euv1 == 1)
chd_result = chd_result.astype(int)
plt.imshow(chd_result)

# clustering chd image
n_clusters = 2
img = mask[0]
image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
# tweak the cluster size and see what happens to the Output
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_2D)
# Reshape back the image from 2D to 3D image
chd_clustered = clustered.reshape(img.shape[0], img.shape[1], img.shape[2])
arr_2d = np.squeeze(chd_clustered, axis=2)