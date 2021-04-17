"""A data structure for a frame - consists of a list of contours.

Coronal Hole properties to keep in mind:
        - on average a coronal hole exists for 2 weeks.

Last Modified: April 13th, 2021 (Opal).
"""

import json


class Frame:
    """ Frame data structure. """
    def __init__(self, contour_list, identity, timestamp=None):
        # list of Contours that are part of this CoronalHole Object.
        self.contour_list = contour_list

        # list of contour pixel polar projection centers.
        self.centroid_list = self.compute_centroid_list()

        # coronal hole id list
        self.label_list = self.save_corresponding_labels()

        # the unique identification number of this frame.
        self.id = identity

        # frame time stamp: usually a string or astropy timestamp.
        self.timestamp = timestamp

    def __str__(self):
        return json.dumps(
            self.json_dict(), indent=4, default=lambda o: o.json_dict())

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
        return [ch.phys_centroid for ch in self.contour_list]

    def save_corresponding_labels(self):
        """save coronal hole ID in a list

        Returns
        -------
        list of coronal hole IDs.
        """
        return [ch.id for ch in self.contour_list]