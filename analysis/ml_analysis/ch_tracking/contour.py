"""
Author: Opal Issan, Feb 2nd, 2021.

A data structure for a coronal hole contour in a polar projection.
List of properties:                                         || Name of variable.
- centroid pixel location (x,y) in spherical coordinates    || pixel_centroid
- centroid physical location (phi, theta)                   || phys_centroid
- contour physical area                                     || area
- bounding rectangle straight                               || straight_box

# TODO: Compute arclength, etc ...

- periodicity (handled by several functions)

* NOTE:
- n_t and n_p : image dimensions.
"""

import numpy as np
import json


class Contour:
    """ Coronal Hole Single Contour Object Data Structure.
    :parameter contour_pixels = coronal hole pixel location.  """

    # image dimensions latitude and longitude.
    n_t, n_p = None, None

    def __init__(self, contour_pixels):
        # save contour inner pixels. shape: [list(row), list(column)].
        self.contour_pixels_row = contour_pixels[0]
        self.contour_pixels_column = contour_pixels[1]

        # contour centroid physical location in lat-lon projection. (phi, theta)
        self.phys_centroid = None

        # centroid pixel location in polar projection for coronal hole matching (x,y).
        self.pixel_centroid = self.compute_pixel_centroid()

        # compute the bounding box upper left corner, width, height (x, y, w, h).
        self.straight_box = self.compute_straight_bounding_rectangle()

        # contour physical area based on lat-lon contour_pixels.
        self.area = self.contour_area()

        # contour straight bounding box physical area based on lat-lon contour_pixels.
        self.box_area = self.compute_straight_box_area()

        # the unique identification number of this coronal hole (should be a natural number).
        self.id = None

        # the unique color for identification of this coronal hole rbg [r, b, g].
        self.color = None

    def __str__(self):
        return json.dumps(
            self.json_dict(), indent=2, default=lambda o: o.json_dict())

    def json_dict(self):
        return {
            'id': self.id,
            'color': self.color,
            'centroid_spherical': self.phys_centroid,
            'centroid_pixel': self.pixel_centroid,
            'area': self.area,
            'box': self.straight_box.tolist(),
            'box_area': self.box_area
        }

    def compute_pixel_centroid(self):
        """ given the coronal hole pixel location we can compute the pixel center in polar projection.
         :return (x,y) pixel coordinates """
        try:
            # convert to cartesian coordinates.
            x, y, z = self._image_pixel_location_to_cartesian(t=self.contour_pixels_row, p=self.contour_pixels_column)
            # compute the mean of each coordinate and convert back to spherical coordinates.
            # noinspection PyTypeChecker
            self.phys_centroid = self._cartesian_centroid_to_spherical_coordinates(x=np.mean(x), y=np.mean(y),
                                                                                   z=np.mean(z))
            # return image pixel coordinates.
            return self._spherical_coordinates_to_image_coordinates(*self.phys_centroid)
        except ArithmeticError:
            raise ArithmeticError('Contour pixels are invalid. ')

    def _image_pixel_location_to_cartesian(self, t, p):
        """ Convert longitude latitude image pixel location to cartesian coordinates, returns(x, y, z)
        :type p: list or numpy array
        :type t: list or numpy array
        :return: x, y, z - all numpy arrays
        x = ρ sinθ cosφ
        y = ρ sinθ sinφ
        z = ρ cosθ
        """
        # convert image pixel location to spherical coordinates.
        theta, phi = self._image_coordinates_to_spherical_coordinates(t=t, p=p)
        # return x, y, z.
        return np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)

    @staticmethod
    def _cartesian_centroid_to_spherical_coordinates(x: float, y: float, z: float):
        """Convert the cartesian centroid to spherical coordinates. returns t, p.
        :type x: float
        :type y: float
        :type z: float
        :return t: float , p: float (tuple)
        θ = arccos(z/ρ)
        θ = arctan(y/x)
        """
        # convert from cartesian to spherical.
        t = np.arccos(z)
        p = np.arctan2(y, x)
        # Change phi range from [-pi,pi] to [0,2pi]
        if p < 0:
            p += 2 * np.pi
        return t, p

    @staticmethod
    def _spherical_coordinates_to_image_coordinates(t: float, p: float):
        """Convert spherical coordinates to image coordinates. return t, p
        p in [0, 2pi] and t in [0, pi]
        :type t: float
        :type p: float
        """
        if 0 <= p <= 2 * np.pi and 0 <= t <= np.pi:
            return int((1 - t / np.pi) * Contour.n_t), int((p / (2 * np.pi)) * Contour.n_p)
        else:
            raise ValueError('When converting spherical coordinates to image coordinates,'
                             ' 0 <= phi < 2pi and 0 <= theta <= pi.')

    @staticmethod
    def _image_coordinates_to_spherical_coordinates(t, p):
        """Convert image coordinates to spherical coordinates.
        :type t: int
        :type p: int
        :return row, column image coordinates.
        """
        if 0 <= np.all(t) <= Contour.n_t and 0 <= np.all(p) <= Contour.n_p:
            dp, dt = np.pi * 2 / Contour.n_p, np.pi / Contour.n_t
            return np.pi - dt * t, p * dp
        else:
            raise ValueError('Image coordinates are out of input image dimensions.')

    def compute_straight_bounding_rectangle(self):
        """straight rectangle, it does not consider the rotation of the object. So area of the bounding
         rectangle won’t be minimum. These image coordinates are in lon-lat projection.
         returns: upper left x, y, w, h """
        try:
            y_min = np.min(self.contour_pixels_row)
            x_min = np.min(self.contour_pixels_column)
            y_max = np.max(self.contour_pixels_row)
            x_max = np.max(self.contour_pixels_column)
            return np.array([x_min, y_min, (x_max - x_min), (y_max - y_min)])
        except Exception:
            raise ArithmeticError("contour pixel locations are invalid. ")

    def compute_straight_box_area(self):
        """ compute the contour's straight bounding box area.
        d𝐴=𝑟^2 * sin𝜃 * d𝜙 * d𝜃, let r be in solar radii.
        :return A = -r^2 * [cos(t1) - cos(t2)]*(p1 - p2)"""
        try:
            # compute delta phi and delta theta.
            dp, dt = np.pi * 2 / self.n_p, np.pi / self.n_t
            # access left corner pixel coordinates and width and height.
            x, y, w, h = self.straight_box
            # convert from image coordinates to spherical coordinates.
            t = np.pi - y * dt
            h = h * dt
            w = w * dp
            return abs(w * (np.cos(t) - np.cos(t - h)))
        except Exception:
            raise ArithmeticError("Straight box coordinates, weight, or height are not valid.")

    def contour_area(self):
        """ compute the coronal hole area in spherical image projection.
        :return sum(d𝐴=𝑟^2 * sin𝜃 * d𝜙 * d𝜃), let r be in solar radii."""
        try:
            # compute delta phi and delta theta.
            dp, dt = np.pi * 2 / self.n_p, np.pi / self.n_t
            # convert from image coordinates to spherical coordinates.
            t = np.pi - self.contour_pixels_row * dt
            # return sum of dA.
            return np.sum(np.sin(t) * dt * dp)
        except Exception:
            raise ArithmeticError("Contour pixel rows are invalid.")

    def is_periodic_zero(self):
        """ if the coronal hole detected at 0 (phi)
        :return boolean"""
        # check if 0 pixel is included in the contour pixels.
        if 0 in self.contour_pixels_column:
            return True
        return False

    def is_periodic_2_pi(self):
        """ if the coronal hole detected at 2 pi (phi)
        :return boolean"""
        # check if 2pi pixel is included in the contour pixels.
        if (Contour.n_p - 1) in self.contour_pixels_column:
            return True
        return False

    def lat_interval_at_2_pi(self):
        """ Return the latitude pixel interval where the contour pixels are 2pi longitude.
        This function is used to force periodicity.
        :return tuple (min_lat, max_lat)- pixel image coordinates"""
        if self.is_periodic_2_pi():
            mask = (self.contour_pixels_column == int(Contour.n_p - 1))
            index = np.argwhere(mask)
            return np.min(self.contour_pixels_row[index]), np.max(self.contour_pixels_row[index])

    def lat_interval_at_zero(self):
        """ Return the latitude pixel interval where the contour pixels are 0 longitude.
        This function is used to force periodicity.
        :return tuple (min_lat, max_lat) - pixel image coordinates"""
        if self.is_periodic_zero():
            mask = (self.contour_pixels_column == 0)
            index = np.argwhere(mask)
            return np.min(self.contour_pixels_row[index]), np.max(self.contour_pixels_row[index])


