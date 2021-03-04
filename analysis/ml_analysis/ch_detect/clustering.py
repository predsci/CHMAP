"""
testing K-means clustering for image segmentation
"""
import cv2
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.colors as colors

# get map
h5_file = "h5_datasets/data_train_SMALL.h5"
hf = h5.File(h5_file, 'r')

dates = hf.keys()
image = []
mask = []
for date in dates:
    g = hf.get(date)
    image.append(np.array(g['euv_image']))
    mask.append(np.array(g['chd_data']))

hf.close()


### using scikitlearn
from sklearn.cluster import KMeans

# euv image
n_clusters = 5
img = image[0][:, :, :3]
image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_2D)
labels = kmeans.labels_
clustered = kmeans.cluster_centers_[kmeans.labels_]
euv_clustered = clustered.reshape(img.shape[0], img.shape[1], img.shape[2])
chd_clustered = labels.reshape(img.shape[0], img.shape[1], 1)

# predict CHD
img2 = image[2][:, :, :3]
pred_image_2D = img2.reshape(img2.shape[0]*img2.shape[1], img2.shape[2])
pred_img = kmeans.predict(pred_image_2D)
pred_labels = kmeans.labels_
clustered_pred = kmeans.cluster_centers_[kmeans.labels_]
# Reshape back the image from 2D to 3D image
euv_clustered_pred = pred_labels.reshape(img2.shape[0], img2.shape[1], 1)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(euv_clustered)
ax1.set_title('Clustered EUV Image')
ax2.imshow(tf.keras.preprocessing.image.array_to_img(chd_clustered))
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