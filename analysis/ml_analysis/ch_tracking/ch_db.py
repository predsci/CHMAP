"""
Author: Opal Issan, Jan 8th, 2021.

A data structure for a set of coronal hole objects.
"""

import json
import cv2
import numpy as np
from scipy.spatial import distance as dist
from analysis.ml_analysis.ch_tracking.frame import Frame
from analysis.ml_analysis.ch_tracking.contour import Contour


class CoronalHoleDB:
    """ Coronal Hole Object Data Structure."""
    # contour binary threshold.
    BinaryThreshold = 55
    # coronal hole area threshold.
    AreaThreshold = 50

    def __init__(self):
        # list of Contours that are part of this CoronalHole Object.
        self.ch_dict = dict()

        # the unique identification number of for each coronal hole in the db.
        self.id_list = set()

        # frame number.
        self.frame_num = 0

        # recent frame holder - data structure frame.py.
        self.p1 = None
        self.p2 = None
        self.p3 = None
        self.p4 = None
        self.p5 = None

    def __str__(self):
        return json.dumps(
            self.json_dict(), indent=2, default=lambda o: o.json_dict())

    def json_dict(self):
        return {
            'coronal_hole_db': self.ch_dict,
            'id_list': self.id_list,
            'num_frames': self.frame_num,
        }

    def add_coronal_hole(self, ch):
        """ Insert a new coronal hole object to the db. """
        self.ch_dict[ch.id] = ch
        self.id_list.add(ch.id)

    def add_new_coronal_hole(self, ch):
        """ Insert a new coronal hole and assign an id and color."""
        # set the index id.
        ch.id = len(self.id_list)
        # set the coronal hole color.
        ch.color = self.generate_ch_color()
        # add to the coronal hole dictionary.
        self.add_coronal_hole(ch)

    def update_previous_frames(self, ch):
        """ Update previous frame holders. """
        self.p5 = self.p4
        self.p4 = self.p3
        self.p2 = self.p1
        self.p1 = ch

    def compute_distance(self):
        """ compute the distance between each element in two arrays containing the coronal hole centroids.
        rows = self.p1
        columns = self.p2
        """
        return dist.cdist(self.p1.centroid_list, self.p2.centroid_list)

    def create_priority_queue(self):
        """ arrange the coronal hole matches in order.
        [(new_index, old_index)] """
        distance = self.compute_distance()
        rows = distance.min(axis=1).argsort()
        cols = distance.argmin(axis=1)[rows]
        return list(zip(rows, cols))

    def priority_queue_remove_duplicates(self):
        """ remove duplicates from the priority queue. such as:
        [(0,1), (1, 1), (2, 2)] --> [(0, 1), (2, 2)]"""
        queue = self.create_priority_queue()
        return [(a, b) for i, [a, b] in enumerate(queue) if not any(c == b for _, c in queue[:i])]

    def find_contours(self, imgray):
        """ find contours above a certain area threshold.
        imgray - gray scaled image. """
        # find contours.
        ret, thresh = cv2.threshold(imgray, CoronalHoleDB.BinaryThreshold, 255, 0)
        contours, hierarchy = cv2.findContours(cv2.bitwise_not(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # do not count small contours.
        p1 = [Contour(ch) for ch in contours if cv2.contourArea(ch) > CoronalHoleDB.AreaThreshold]
        # save latest frame list.
        self.update_previous_frames(Frame(contour_list=p1))

    @staticmethod
    def generate_ch_color():
        """ generate a random color"""
        return np.random.rand(3, ) * 255

    def first_frame_initialize(self):
        """ match an ID and color to each coronal hole in the first frame. """
        for ch in self.p1.contour_list:
            # add coronal hole.
            self.add_new_coronal_hole(ch=ch)

    def match_coronal_holes(self):
        """ find based on center euclidean distance if there is a match between the coronal holes
        detected in sequential frames."""
        queue = self.priority_queue_remove_duplicates()

        for new_index, old_index in queue:
            # set the match
            self.p1.contour_list[new_index].id = self.p2.contour_list[old_index].id
            self.p1.contour_list[new_index].color = self.p2.contour_list[old_index].color

        # mark the index matched.
        index_list = np.arange(0, len(self.p1.contour_list))
        index_list = np.delete(index_list, [a for a, b in queue])

        # add all new coronal holes.
        for ii in index_list:
            # set the index id.
            self.add_new_coronal_hole(ch=self.p1.contour_list[ii])

