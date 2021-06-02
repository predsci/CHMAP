"""
Tamar Ervin
Date: April 13, 2021
clustering algorithm for unsupervised
detection of coronal holes
"""

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

h5_file = '/Volumes/CHD_DB/map_data_small.h5'
hf = h5.File(h5_file, 'r')

dates = hf.keys()
image = []
mask = []
for date in dates:
    g = hf.get(date)
    image.append(np.array(g['euv_image']))
    mask.append(np.array(g['chd_data']))

hf.close()

x_train = np.array(image)
y_train = np.array(mask)

n_clusters = len(x_train)
X_train = x_train.reshape(len(x_train), -1)

# priority queue
class PriorityQueue(object):
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    # check if queue is empty
    def isEmpty(self):
        return len(self.queue) == 0

    # insert an element in the queue
    def insert(self, data):
        self.queue.append(data)

    # pop element based on Priority
    def delete(self):
        try:
            max = 0
            for i in range(len(self.queue)):
                if self.queue[i] > self.queue[max]:
                    max = i
            item = self.queue[max]
            del self.queue[max]
            return item
        except IndexError:
            print()
            exit()


if __name__ == '__main__':
    myQueue = PriorityQueue()
    myQueue.insert(12)
    myQueue.insert(1)
    myQueue.insert(14)
    myQueue.insert(7)
    print(myQueue)
    while not myQueue.isEmpty():
        print(myQueue.delete())

# k-means functions
def startlocation(data, k):
    """
    Select the start cluster controid
    @ data: X
    @ k: cluster numbers

    return:
    @ centers: centers' locations
    """

    # Find the minimum and maximum values for each feature
    minima = data.min(axis=0)
    maxima = data.max(axis=0)

    feater_num = data.shape[1]

    # centres should be an array which shape is (k,features)
    centers = np.zeros((k, feater_num))
    for i in range(k):
        for j in range(feater_num):
            centers[i][j] = np.random.choice(data[:, j], replace=False)

    return centers

def get_label(data, centers, labels):
    """
    Get the label based on the nearest cluster center
    @ data: X
    @ centers: current cluster centers' loaction
    @ distances: Use a (sample_num, k) matrix to store the distance of each sample
    @ labels: Use a (sample_num, ) array to store the label of each sample
    return:
    @ labels : new label of X


    """
    # Compute the distance
    # Use a (sample_num, k) matrix to store the distance of each sample
    # Use a (sample_num, ) array to store the label of each sample
    # init the matrix
    distances = np.zeros((len(data), len(centers)))
    for center, i in zip(centers, np.arange(len(centers))):
        distances[:, i] = 1 / 2 * ((data[:, 0] - center[0]) ** 2 + (data[:, 1] - center[1]) ** 2)

    labels = np.argmin(distances, axis=1)
    return labels

def get_newcenter(data, k, labels, centers):
    """
    Get the new centers for each group
    return:
    @centers: new centers

    """
    new_centers = np.zeros(np.shape(centers))
    for i in np.arange(k):
        new_center = np.mean(data[labels == i], axis=0)
        new_centers[i] = new_center
    return new_centers

def mykmeans(data, k, maxiteration=1000):
    labels = np.zeros((len(data), k))

    # Get the start centers location
    centers = startlocation(data, k)
    # 1st iteration
    labels = get_label(data, centers, labels)
    new_centers = get_newcenter(data, k, labels, centers)
    print("centers:", centers, "new centers:", new_centers)
    count = 1
    while ((new_centers == centers).all() == False) and count < maxiteration:
        centers = new_centers

        count += 1
        labels = get_label(data, centers, labels)
        new_centers = get_newcenter(data, k, labels, centers)

    print("iteration steps:", count)
    return new_centers, labels


kmeans = KMeans(n_clusters=n_clusters, random_state=0, init='k-means++')
new_centers, labels = mykmeans(img2, 3,maxiteration = 1000)
