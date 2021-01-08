"""Author: Opal Issan, Jan 8th, 2021.

A data structure for a coronal hole - consists of a list of contours.

we might want to group disconnected contours with close proximity to be identified as 1 coronal hole.

Coronal Hole properties to keep in mind:
        - on average a coronal hole exists for 2 weeks.

Video processing to avoid flickering:
        - temporal averaging - gaussian. add probability to each coronal hole.
"""


import json


class CoronalHole:
    """ Coronal Hole Object Data Structure."""

    def __init__(self, contour_list):
        # list of Contours that are part of this CoronalHole Object.
        self.contour_list = contour_list

        # the unique identification number of this coronal hole.
        self.id = None

    def __str__(self):
        return json.dumps(
            self.json_dict(), indent=2, default=lambda o: o.json_dict())

    def json_dict(self):
        return {
            'contour_list': self.contour_list,
            'id': self.id,
        }
