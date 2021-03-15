"""
Author: Opal Issan. February 13th, 2021.

A data structure for a coronal hole.
"""

import numpy as np
import json


class CoronalHole:
    """ Coronal Hole DataStructure Holder.
    :parameter identity - unique number """

    def __init__(self, identity):
        # id.
        self.id = identity

        # frames coronal hole appeared.
        self.frames = []

        # save contour list.
        self.contour_list = []

    def __str__(self):
        return json.dumps(
            self.json_dict(), indent=4, default=lambda o: o.json_dict())

    def json_dict(self):
        return {
            'id': self.id,
            'frames': self.frames,
            'contour_list': [ch.json_dict() for ch in self.contour_list]
        }

    def insert_number_frame(self, frame_num):
        """Insert the frame id the coronal hole appeared.

        Parameters
        ----------
        frame_num: int
            Frame number in video/data base.

        Returns
        -------
        None
        """
        self.frames.append(frame_num)

    def insert_contour_list(self, contour):
        """Insert the contour associated with this coronal hole.

        Parameters
        ----------
        contour - Contour object. see contour.py

        Returns
        -------
        None
        """
        self.contour_list.append(contour)
