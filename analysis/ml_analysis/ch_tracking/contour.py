"""
Author: Opal Issan, Jan 18th, 2021.

A data structure for a coronal hole contour in a polar projection.
List of properties:                                         || Name of variable.
- centroid pixel location (x,y) in polar projection.        || polar_pixel_centroid
- centroid pixel location (x,y) in lat-lon projection.      || lat_lon_pixel_centroid
- centroid physical location (phi, theta) - TODO.           || lat_lon_phys_centroid
- contour - pixel list in polar projection.                 || contour
- contour physical area. - TODO                             || area
- bounding rectangle straight.                              || straight_box

* NOTE:
- projection: "polar" or "lat-lon"
            * "polar" projection refers to a 2d projection when the poles are at the equator.
              see function ch_db/map_new_polar_projection for more information.
            * "lat-lon" projection refers to a 2d projection described by longitude and latitude locations.

- n_t and n_p : image dimensions.
"""

import numpy as np
import cv2
import json


class Contour:
    """ Coronal Hole Single Contour Object Data Structure.
    :parameter contour = cv2 object contour of the detected coronal hole in polar projection. """

    # image dimensions latitude and longitude.
    n_t, n_p = None, None

    def __init__(self, contour):
        # contour points of the coronal hole in polar projection.
        self.contour = contour

        # centroid pixel location in polar projection for coronal hole matching (x,y).
        self.polar_pixel_centroid = self.polar_projection_pixel_centroid()

        # centroid pixel location in lat-lon projection (x,y).
        self.lat_lon_pixel_centroid = None

        # contour centroid physical location in lat-lon projection. (phi, theta)
        self.lat_lon_phys_centroid = None

        # compute the bounding box upper left (x, y, w, h).
        self.straight_box = None

        # the unique identification number of this coronal hole (should be a natural number).
        self.id = None

        # the unique color for identification of this coronal hole rbg [r, b, g].
        self.color = None

        # save contour inner pixels. shape: [list(y), list(x)].
        self.contour_pixels = None

        # contour physical area based on lat-lon contour_pixels.
        self.area = 0

        # contour straight bounding box physical area based on lat-lon contour_pixels.
        self.box_area = 0

    def __str__(self):
        return json.dumps(
            self.json_dict(), indent=2, default=lambda o: o.json_dict())

    def json_dict(self):
        return {
            'id': self.id,
            'color': self.color,
            'centroid': self.lat_lon_phys_centroid,
            'area': self.area,
            'box': self.straight_box,
            'box_area': self.box_area
        }

    def polar_projection_pixel_centroid(self):
        """ given the object contour we can compute the pixel center in polar projection.
         returns (x,y) pixel coordinates. """
        x = int(np.mean(self.contour[:, :, 0]))
        y = int(np.mean(self.contour[:, :, 1]))
        return tuple((x, y))

    def compute_contour_pixel_perimeter(self):
        """ compute the contour pixel arc length.
        Second argument specify whether shape is a closed contour (if passed True), or just a curve."""
        return cv2.arcLength(self.contour, True)

    def compute_straight_bounding_rectangle(self):
        """straight rectangle, it does not consider the rotation of the object. So area of the bounding
         rectangle won’t be minimum. These image coordinates are in lon-lat projection.
         returns: upper left x, y, w, h """
        if self.contour_pixels is None:
            return None

        else:
            y_min = np.min(self.contour_pixels[0])
            x_min = np.min(self.contour_pixels[1])
            y_max = np.max(self.contour_pixels[0])
            x_max = np.max(self.contour_pixels[1])

        return np.array([x_min, y_min, (x_max - x_min), (y_max - y_min)])

    def compute_centroid_lon_lat_location(self):
        """ compute the pixel location of the coronal hole center of mass. """
        y_mean = int(np.mean(self.contour_pixels[0]))
        x_mean = int(np.mean(self.contour_pixels[1]))
        return tuple((x_mean, y_mean))

    def centroid_lon_lat_phys_location(self):
        """ compute the physical location of the coronal hole center of mass. """
        if self.lat_lon_pixel_centroid is None:
            return None
        else:
            x, y = self.lat_lon_pixel_centroid
            return tuple((x * np.pi * 2 / self.n_p, np.pi - y * np.pi / self.n_t))

    def contour_area(self):
        """ compute the contour's area.
        d𝐴=𝑟^2 * sin𝜃 * d𝜙 * d𝜃, let r be in solar radii."""
        if self.contour_pixels is None:
            return None
        else:
            # access y pixel coordinates.
            y = self.contour_pixels[0]
            # compute delta phi and delta theta.
            dp, dt = np.pi * 2 / self.n_p, np.pi / self.n_t
            # convert from image coordinates to spherical coordinates.
            t = np.pi * np.ones(len(y)) - y * dt
            # return dA.
            return np.sum(np.sin(t) * dt * dp)

    def straight_box_area(self):
        """ compute the contour's straight bounding box area.
        d𝐴=𝑟^2 * sin𝜃 * d𝜙 * d𝜃, let r be in solar radii.
        A = -r^2 * [cos(t1) - cos(t2)]*(p1 - p2)"""
        if self.straight_box is None:
            return None
        else:
            # compute delta phi and delta theta.
            dp, dt = np.pi * 2 / self.n_p, np.pi / self.n_t
            # access left corner pixel coordinates and width and height.
            x, y, w, h = self.straight_box
            # convert from image coordinates to spherical coordinates.
            t = np.pi - y * dt
            h = h * dt
            w = w * dp
            return -1 * abs(w) * (np.cos(t) - np.cos(t - h))



    # def convert_polar_pixel_to_lon_lat_pixel(self, x, y):
    #     """input x and y image coordinates in polar projection
    #      and output x y image coordinates in lon-lat projection. """
    #     t, p = y * np.pi / self.n_t, x * 2 * np.pi / self.n_p
    #     t_val = np.arccos(-np.sin(t) * np.sin(p))
    #     p_val = np.arctan2(np.cos(t), np.sin(t) * np.cos(p))
    #     # Change phi range from [-pi,pi] to [0,2pi]
    #     if p_val < 0:
    #         p_val += 2 * np.pi
    #     return tuple((int(p_val * self.n_p / (np.pi * 2)), int(t_val * self.n_t / np.pi)))

    # def map_contour_to_lat_lon(self, contour):
    #     """ if projection is "polar" we need to convert it lat-lon projection. """
    #     if projection == "lat-lon":
    #         return contour
    #     elif projection == "polar":
    #         # create 1d arrays for spherical coordinates.
    #         theta = np.linspace(np.pi, 0, self.n_t)
    #         phi = np.linspace(0, 2 * np.pi, self.n_p)
    #
    #         # spacing in theta and phi.
    #         delta_t = abs(theta[1] - theta[0])
    #         delta_p = abs(phi[1] - phi[0])
    #
    #         # initialize new contour
    #         new_contour = np.zeros((np.shape(contour)[0], 1,  2), dtype=np.int16)
    #
    #         # reshape contour
    #         reshape_contour = contour.reshape((np.shape(contour)[0], 2))
    #
    #         # convert pixel location.
    #         for ii, pixel in enumerate(reshape_contour):
    #             p, t = phi[pixel[0]], theta[pixel[1]]
    #             t_val = np.arccos(-np.sin(t) * np.sin(p))
    #             p_val = np.arctan2(np.cos(t), np.sin(t) * np.cos(p))
    #             # Change phi range from [-pi,pi] to [0,2pi]
    #             if p_val < 0:
    #                 p_val += 2 * np.pi
    #             # new image coordinates.
    #             new_contour[ii, :, :] = [[int(self.n_p - p_val / delta_p), int(t_val / delta_t)]]
    #         return new_contour

    # def compute_rot_bounding_rectangle(self):
    #     """ Here, bounding rectangle is drawn with minimum area, so it considers the rotation also. """
    #     rect = cv2.minAreaRect(self.contour)
    #     box = np.int0(cv2.boxPoints(rect))
    #     # convert box coordinates to lon-lat projection.
    #     for ii in range(4):
    #         x, y = box[ii, :]
    #         box[ii, 0], box[ii, 1] = self.convert_polar_pixel_to_lon_lat_pixel(x=x, y=y)
    #     return np.int0(box)
