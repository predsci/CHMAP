"""
Author: Opal Issan, Jan 8th, 2021.

A data structure for a coronal hole contour.
List of properties:
- centroid pixel location (x,y)
- centroid physical location (phi, theta) - TODO.
- contour - pixel list.
- contour pixel area.
- contour physical area. - TODO
- coronal hole probability -temporal averaging -gaussian.
- contour arc length.
- bounding rectangle - straight and rotating.
"""

import numpy as np
import cv2
import json


class Contour:
    """ Coronal Hole Single Contour Object Data Structure."""

    def __init__(self, contour):
        # contour points of the coronal hole. This can be found using OpenCV functionality and convert to
        # gray scale. note that OpenCV makes sure that there is no contour overlapping.
        self.contour = contour

        # compute the contour pixel and physical center.
        self.pixel_centroid = self.compute_pixel_centroid()

        # compute the contour pixel and physical area.
        self.pixel_area = self.compute_contour_pixel_area()
        self.pixel_perimeter = self.compute_contour_pixel_perimeter()

        # compute the bounding boxes - rotated and straight.
        self.rotate_box = self.compute_rot_bounding_rectangle()
        self.straight_box = self.compute_straight_bounding_rectangle()

    def __str__(self):
        return json.dumps(
            self.json_dict(), indent=2, default=lambda o: o.json_dict())

    def json_dict(self):
        return {
            'contour': self.contour,
            'pixel_centroid': self.pixel_centroid,
            'pixel_area': self.pixel_area,
            'pixel_perimeter': self.pixel_perimeter,
            'rotate_box': self.rotate_box,
            'straight_box': self.straight_box,
        }

    def compute_pixel_centroid(self):
        """ given the object contour we can compute the pixel center.
         returns (x,y) pixel coordinates. """
        centroid = cv2.moments(self.contour)
        x = int(centroid["m10"] / centroid["m00"])
        y = int(centroid["m01"] / centroid["m00"])
        return tuple((x, y))

    def compute_contour_pixel_area(self):
        """ compute the contour pixel area. """
        return cv2.contourArea(self.contour)

    def compute_contour_pixel_perimeter(self):
        """ compute the contour pixel arc length.
        Second argument specify whether shape is a closed contour (if passed True), or just a curve."""
        return cv2.arcLength(self.contour, True)

    def compute_rot_bounding_rectangle(self):
        """ Here, bounding rectangle is drawn with minimum area, so it considers the rotation also. """
        rect = cv2.minAreaRect(self.contour)
        box = cv2.boxPoints(rect)
        return np.int0(box)

    def compute_straight_bounding_rectangle(self):
        """It is a straight rectangle, it doesn’t consider the rotation of the object. So area of the bounding
         rectangle won’t be minimum. It is found by the function cv2.boundingRect().
         returns: x,y,w,h """
        return cv2.boundingRect(self.contour)

    def compute_centroid_physical_location(self):
        """ TODO: A helper function to compute the physical location of the coronal hole center of mass. """
        return NotImplemented

    def compute_physical_area(self):
        """ TODO: A helper function to compute the physical surface area of a coronal hole. """
        return NotImplemented

    def compute_physical_arc_length(self):
        """ TODO: A helper function to compute the physical arc length of a coronal hole. """
        return NotImplemented
