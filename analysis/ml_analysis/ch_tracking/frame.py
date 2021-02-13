"""Author: Opal Issan, Feb 3rd, 2021.

A data structure for a frame - consists of a list of contours.

Coronal Hole properties to keep in mind:
        - on average a coronal hole exists for 2 weeks.

Video processing to avoid flickering:
        - temporal averaging - gaussian. add probability to each coronal hole.
"""

import json


class Frame:
    """ Frame data structure. """
    def __init__(self, contour_list, identity):
        # list of Contours that are part of this CoronalHole Object.
        self.contour_list = contour_list

        # list of contour pixel polar projection centers.
        self.centroid_list = self.compute_centroid_list()

        # the unique identification number of this frame.
        self.id = identity

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
        """save the coronal hole centers in a list.

        Returns
        -------
        list of coronal hole centroids.
        """
        return [ch.pixel_centroid for ch in self.contour_list]
