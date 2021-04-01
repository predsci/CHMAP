"""
This module is an implementation of KNN (K- Nearest Neighbors) Algorithm (see mkdocs website for more information
about knn). This is used to match coronal holes between frames based on their centroid location.

Author: Opal Issan, last updated March 25th, 2021.

# TODO: Weighted KNN by Frame proximity as well. !!
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import json


class KNN:
    """ K Nearest Neighbor DataStructure. """

    def __init__(self, X_train, Y_train, X_test, K=6, thresh=0.1):
        # X train contains a list of coronal hole centroid (theta, phi) or (lat, lon) of previous frames.
        self.X_train = X_train

        # Y train contains the corresponding class (unique ID number).
        self.Y_train = Y_train

        # X test contains the new frame coronal hole centroids (theta, phi) or (lat, lon) which will be classified
        # by knn algorithm.
        self.X_test = X_test

        # labels of coronal holes in order, for example, say we have ID # 4, 5, 6, 17 in Y_train then:
        # label= [4, 5, 6, 17].
        # sort the list of labels in Y_train.
        sort_label = np.sort(Y_train)
        # remove duplicates.
        self.label = list(set(sort_label))

        # threshold for significant chance to be in the class.
        self.thresh = thresh

        # K parameter in KNN Algorithm.
        self.K = K

        # if training data is less than K, then adjust K to the number of training centroids.
        if len(Y_train) < self.K:
            self.K = int(len(Y_train))

        # classifier
        self.clf = KNeighborsClassifier(n_neighbors=self.K, metric=self.cartesian_distance, weights="distance")

        # fit training dataset.
        self.clf.fit(self.X_train, self.Y_train)

        # classify for X_test.
        self.X_test_results = self.clf.predict_proba(X_test)

        # check area overlap.
        self.check_list = self.possible_classes()

    def __str__(self):
        return json.dumps(
            self.json_dict(), indent=4, default=lambda o: o.json_dict())

    def json_dict(self):
        return {
            'centroid_list_train': self.X_train,
            'labels_train': self.Y_train,
            'centroid_list_test': self.X_test,
            'results': self.X_test_results
        }

    @staticmethod
    def haversine(c1, c2):
        """ Calculate the great circle distance between two points on the earth (specified in decimal degrees)

        Parameters
        ----------
        c1: centroid #1 (theta, phi)
        c2: centroid #2 (theta, phi)

        Returns
        -------
            The distance on a sphere with radius = 1 Solar Radii.
        """

        lat1, lon1 = c1
        lat2, lon2 = c2

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 1  # Radius is unitary. 1 Solar Radii.
        return c * r

    def cartesian_distance(self, c1, c2):
        """ Compute the distance of two points in cartesian coordinates.

        Parameters
        ----------
        c1: centroid #1 (theta, phi)
        c2: centroid #2 (theta, phi)

        Returns
        -------
            Distance in cartesian coordinates.
        """
        # convert to cartesian coordinates.
        x1, y1, z1 = self._convert_spherical_to_cartesian(t=c1[0], p=c1[1])
        x2, y2, z2 = self._convert_spherical_to_cartesian(t=c2[0], p=c2[1])

        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    @staticmethod
    def _convert_spherical_to_cartesian(t, p):
        """Convert spherical coordinates to cartesian.

        Parameters
        ----------
        t: theta (lat)
        p: phi (lon)

        Returns
        -------
            x, y, z
        """
        return np.sin(p) * np.cos(t), np.sin(t) * np.sin(p), np.cos(t)

    def possible_classes(self):
        """Return possible classes each coronal hole can be a part of.

        Returns
        -------
            List of classes the X_test can be a part of (in order)
        """
        # initialize the list with classes the coronal hole in X_test can be associated to.
        list_check = []

        # iterate over the probability results.
        for ch_res in self.X_test_results:
            above_thresh = []
            for ii, p in enumerate(ch_res):
                if p > self.thresh:
                    # if the probability of association is higher than the threshold then we will add its label
                    # to the list of coronal holes which we will check their area overlap (this process will be done
                    # in a separate module).
                    above_thresh.append(self.label[ii])
            list_check.append(above_thresh)

        return list_check
