"""
Author: Opal Issan, Feb 13th, 2021.

A data structure for a coronal hole contour.
List of properties:                                         || Name of variable.
- centroid pixel location (x,y) in spherical coordinates    || pixel_centroid
- centroid physical location (phi, theta)                   || phys_centroid
- contour physical area                                     || area
- bounding rectangle straight                               || straight_box

# TODO: Compute arclength, tilt, rotated_box, etc...

- periodicity (handled by several functions)

* NOTE:
- n_t and n_p : image dimensions.
- Mesh : image mesh grid.
"""

import numpy as np
import json


class Contour:
    """ Coronal Hole Single Contour Object Data Structure.
    :parameter contour_pixels = coronal hole pixel location.  """

    # image dimensions latitude and longitude.
    n_t, n_p = None, None
    Mesh = None  # This will be an object of type MeshMap with information about the input image mesh grid.

    def __init__(self, contour_pixels):
        # save contour inner pixels. shape: [list(row), list(column)].
        self.contour_pixels_theta = contour_pixels[0]
        self.contour_pixels_phi = contour_pixels[1]

        # contour physical area based on lat-lon contour_pixels.
        # sum(d𝐴=𝑟^2 * sin𝜃 * d𝜙 * d𝜃)
        self.area = np.sum(Contour.Mesh.da[self.contour_pixels_phi, self.contour_pixels_theta])

        # contour centroid physical location in lat-lon projection. (phi, theta)
        self.phys_centroid = None

        # centroid pixel location in polar projection for coronal hole matching (x,y).
        self.pixel_centroid = self.compute_pixel_centroid()

        # compute the bounding box upper left corner, width, height (x, y, w, h).
        self.straight_box = self.compute_straight_bounding_rectangle()

        # contour straight bounding box physical area based on lat-lon contour_pixels.
        self.box_area = self.compute_straight_box_area()

        # the unique identification number of this coronal hole (should be a natural number).
        self.id = None

        # the unique color for identification of this coronal hole rbg [r, b, g].
        self.color = None

        # periodic label.
        self.periodic_at_zero = self.is_periodic_zero()
        self.periodic_at_2pi = self.is_periodic_2_pi()

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
        """ Given the coronal hole pixel location we can compute the pixel center in polar projection.

        Returns
        -------
        (t, p) image pixel coordinates
        """
        try:
            # convert to cartesian coordinates.
            x = Contour.Mesh.x2d[self.contour_pixels_phi, self.contour_pixels_theta]
            y = Contour.Mesh.y2d[self.contour_pixels_phi, self.contour_pixels_theta]
            z = Contour.Mesh.z2d[self.contour_pixels_phi, self.contour_pixels_theta]

            # access the area of each pixel of the image grid.
            A = Contour.Mesh.da[self.contour_pixels_phi, self.contour_pixels_theta]

            # compute the weighted mean of each coordinate and convert back to spherical coordinates.
            x_mean = A*x/self.area
            y_mean = A*y/self.area
            z_mean = A*z/self.area

            # convert back to spherical coordinates.
            self.phys_centroid = self._cartesian_centroid_to_spherical_coordinates(x=x_mean, y=y_mean, z=z_mean)

            # return image pixel coordinates.
            return self._spherical_coordinates_to_image_coordinates(*self.phys_centroid)

        except ArithmeticError:
            raise ArithmeticError('Contour pixels are invalid. ')

    def _image_pixel_location_to_cartesian(self, t, p):
        """ Convert longitude latitude image pixel location to cartesian coordinates.
        x = ρ sinθ cosφ
        y = ρ sinθ sinφ
        z = ρ cosθ

        Parameters
        ----------
        t: list or numpy array
        p: list or numpy array

        Returns
        -------
        x, y, z - all numpy arrays
        """
        # convert image pixel location to spherical coordinates.
        theta, phi = self._image_coordinates_to_spherical_coordinates(t=t, p=p)
        # return x, y, z.
        return np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)

    @staticmethod
    def _cartesian_centroid_to_spherical_coordinates(x, y, z):
        """Convert the cartesian centroid to spherical coordinates.
        θ = arccos(z/ρ)
        𝜙 = arctan(y/x)

        Parameters
        ----------
        x: float
        y: float
        z: float

        Returns
        -------
        list t: float , p: float
        """
        # convert from cartesian to spherical.
        t = np.arccos(z/np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2)))
        p = np.arctan2(y, x)
        # Change phi range from [-pi,pi] to [0,2pi]
        if p < 0:
            p += 2 * np.pi
        return t, p

    @staticmethod
    def _spherical_coordinates_to_image_coordinates(t, p):
        """Convert spherical coordinates to image coordinates.

        Parameters
        ----------
        t: float
        p: float

        Returns
        -------
        t, p
        """
        if 0 <= p <= 2 * np.pi and 0 <= t <= np.pi:
            return int(Contour.Mesh.interp_t2index(t)), int(Contour.Mesh.interp_p2index(p))
        else:
            raise ValueError('When converting spherical coordinates to image coordinates,'
                             ' 0 <= phi < 2pi and 0 <= theta <= pi.')

    @staticmethod
    def _image_coordinates_to_spherical_coordinates(t, p):
        """Convert image coordinates to spherical coordinates.

        Parameters
        ----------
        t: int
        p: int

        Returns
        -------
        row, column spherical coordinates.
        """

        if 0 <= np.all(t) <= Contour.n_t and 0 <= np.all(p) <= Contour.n_p:
            return Contour.Mesh.t[t], Contour.Mesh.p[p]
        else:
            raise ValueError('Image coordinates are out of input image dimensions.')

    def compute_straight_bounding_rectangle(self):
        """ Straight rectangle, it does not consider the rotation of the object. So area of the bounding
         rectangle won’t be minimum. These image coordinates are in lon-lat projection.

        Returns
        -------
        upper left x, y, w, h
        """
        try:
            y_min = np.min(self.contour_pixels_theta)
            x_min = np.min(self.contour_pixels_phi)
            y_max = np.max(self.contour_pixels_theta)
            x_max = np.max(self.contour_pixels_phi)
            return np.array([x_min, y_min, (x_max - x_min), (y_max - y_min)])

        except Exception:
            raise ArithmeticError("contour pixel locations are invalid. ")

    def compute_straight_box_area(self):
        """ Compute the contour's straight bounding box area.
        d𝐴=𝑟^2 * sin𝜃 * d𝜙 * d𝜃, let r be in solar radii.
        A = (cos(𝜃1) - cos(𝜃2)) * (𝜙1 - 𝜙2)

        Returns
        -------
        float
        """
        try:
            # access left corner pixel coordinates and width and height.
            x, y, w, h = self.straight_box
            # convert from image coordinates to spherical coordinates.
            t1 = Contour.Mesh.interp_t2dt(y)
            t2 = Contour.Mesh.interp_p2dp(y - h)
            p1 = Contour.Mesh.interp_t2dt(x + w)
            p2 = Contour.Mesh.interp_t2dt(x)
            return abs((p2 - p1) * (np.cos(t1) - np.cos(t2)))

        except Exception:
            raise ArithmeticError("Straight box coordinates, weight, or height are not valid.")

    def is_periodic_zero(self):
        """If the coronal hole detected at 0 (phi)

        Returns
        -------
        boolean
        """
        # check if 0 pixel is included in the contour pixels.
        if 0 in self.contour_pixels_phi:
            return True
        return False

    def is_periodic_2_pi(self):
        """If the coronal hole detected at 2 pi (phi)

        Returns
        -------
        boolean
        """
        # check if 2pi pixel is included in the contour pixels.
        if (Contour.n_p - 1) in self.contour_pixels_phi:
            return True
        return False

    def lat_interval_at_2_pi(self):
        """Return the latitude pixel interval where the contour pixels are 2pi longitude.
        This function is used to force periodicity.

        Returns
        -------
        (min_lat, max_lat)- pixel image coordinates
        """
        if self.periodic_at_2pi:
            mask = (self.contour_pixels_phi == int(Contour.n_p - 1))
            index = np.argwhere(mask)
            return np.min(self.contour_pixels_theta[index]), np.max(self.contour_pixels_theta[index])

    def lat_interval_at_zero(self):
        """Return the latitude pixel interval where the contour pixels are 0 longitude.
        This function is used to force periodicity.

        Returns
        -------
        (min_lat, max_lat) - pixel image coordinates
        """
        if self.periodic_at_zero:
            mask = (self.contour_pixels_phi == 0)
            index = np.argwhere(mask)
            return np.min(self.contour_pixels_theta[index]), np.max(self.contour_pixels_theta[index])
