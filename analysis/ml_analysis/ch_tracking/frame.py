"""Author: Opal Issan, Jan 8th, 2021.

A data structure for a frame - consists of a list of contours.

we might want to group disconnected contours with close proximity to be identified as 1 coronal hole.

Coronal Hole properties to keep in mind:
        - on average a coronal hole exists for 2 weeks.

Video processing to avoid flickering:
        - temporal averaging - gaussian. add probability to each coronal hole.
"""


import json


class Frame:
    """ Frame data structure. """

    def __init__(self, contour_list):
        # list of Contours that are part of this CoronalHole Object.
        self.contour_list = contour_list

        # list of contour centers.
        self.centroid_list = self.compute_centroid_list()

        # the unique identification number of this frame.
        self.id = None

    def __str__(self):
        return json.dumps(
            self.json_dict(), indent=2, default=lambda o: o.json_dict())

    def json_dict(self):
        return {
            'contour_list': self.contour_list,
            'centroid_list': self.centroid_list,
            'id': self.id,
        }

    def compute_centroid_list(self):
        """ compute the coronal hole centers and save them in a list. """
        return [ch.pixel_centroid for ch in self.contour_list]
