"""
Author: Opal Issan, Feb 13th, 2021.

A data structure for a coronal hole contour.
List of properties:                                         || Name of variable.
- centroid pixel location (x,y) in spherical coordinates    || pixel_centroid
- centroid physical location (phi, theta)                   || phys_centroid
- contour physical area                                     || area
- bounding rectangle straight                               || straight_box
- straight_box_area                                         || straight box area
- rotated box                                               || rot_box
- rotated box corners                                       || rot_box_corners
- rotated box angle                                         || rot_box_angle
- convex hull points                                        || convex_hull
- coronal hole tilt                                         || pca_tilt
- coronal hole symmetry (lambda_max/lambda_min)             || sig_tilt
- periodic boundary                                         || periodic_at_zero & periodic_at_2pi


- periodicity (handled by several functions)

* NOTE:
- n_t and n_p : image dimensions.
- Mesh : image mesh grid.
"""

import numpy as np
import json
import cv2 as cv
import matplotlib.pyplot as plt


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

        # contour centroid physical location in lat-lon projection. (t, p)
        self.phys_centroid = None

        # compute the contour tilt
        self.tilt = None

        # centroid pixel location in polar projection for coronal hole matching (x,y).
        self.pixel_centroid = self.compute_pixel_centroid()

        # compute the bounding box upper left corner, width, height (x, y, w, h).
        self.straight_box = self.compute_straight_bounding_rectangle()

        # contour straight bounding box physical area based on lat-lon contour_pixels.
        self.straight_box_area = self.compute_straight_box_area()

        # compute the rotated bounding box with min pixel area.
        self.rot_box = self.compute_rotated_rect()

        if self.rot_box is not None:
            # save rot box corners.
            self.rot_box_corners = self.rotated_rect_corners()

            # save rot box angle with respect to north.
            self.rot_box_angle = self.compute_rot_box_angle()

            # compute the rotate box area.
            self.rot_box_area = self.compute_rot_box_area()

            # compute convex hull.
            self.convex_hull = self.compute_convex_hull()

            self.pca_tilt, self.sig_tilt = self.pca_v2()

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
            'straight_box_area': self.straight_box_area,
            'rotated_box_area': self.rot_box_area,
            'rotated_box_angle': self.rot_box_angle,
            'tilt': self.tilt,
            'pca_tilt': self.pca_tilt,
            'significance_of_tilt': self.sig_tilt
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
            x_mean = np.dot(A, x) / self.area
            y_mean = np.dot(A, y) / self.area
            z_mean = np.dot(A, z) / self.area

            if len(x) != 1:
                self.tilt = self._compute_tilt(x, y, z, x_mean, y_mean, z_mean, A)

            # convert back to spherical coordinates.
            self.phys_centroid = self._cartesian_centroid_to_spherical_coordinates(x=x_mean, y=y_mean, z=z_mean)

            # return image pixel coordinates.
            return self._spherical_coordinates_to_image_coordinates(*self.phys_centroid)

        except ArithmeticError:
            raise ArithmeticError('Contour pixels are invalid. ')

    def _compute_tilt(self, x, y, z, x_mean, y_mean, z_mean, A):
        """ TODO: Not sure this is accurate.
        Compute the contours tilt with respect to north.
        This is done by averaging the angle between all vectors starting from the origin to [0, 0, 1] (z-axis).

        Parameters
        ----------
        x - array
            x cartesian
        y - array
            y cartesian
        z - array
            z cartesian
        x_mean - float
            x cartesian weighted average
        y_mean - float
            y cartesian weighted average
        z_mean - float
            z cartesian weighted average
        A - Area of image grid.

        Returns
        -------
        Angle in degrees.
        """
        # center around centroid.
        xc = x - x_mean
        yc = y - y_mean
        zc = z - z_mean

        # find magnitude.
        mag = np.sqrt(np.power(xc, 2) + np.power(yc, 2) + np.power(zc, 2))
        # mag[mag == 0] = 1
        dot_prod = zc / mag
        ang = np.arccos(dot_prod)
        # weighted average.
        ang[xc < 0] = -ang[xc < 0]
        return (np.dot(A, ang) / self.area) * (180 / np.pi)

    def pca(self):
        """TODO: needs help.

        Returns
        -------

        """
        # theta, phi coordinates.
        phi = Contour.Mesh.p[self.contour_pixels_phi]
        theta = Contour.Mesh.t[self.contour_pixels_theta]

        # access the area of each pixel of the image grid.
        A = Contour.Mesh.da[self.contour_pixels_phi, self.contour_pixels_theta]

        # recenter around weighted mean.
        pc = phi - self.phys_centroid[1]
        tc = theta - self.phys_centroid[0]

        # phi diff vector gets multiplied by area.
        # pc = pc * A
        # difference from the mean as an arc length TODO: Figure out if this is correct.
        # pc = pc * np.sin(theta)

        # feature matrix 2 by n. (n is number of pixels)
        X = np.array([pc, tc])

        # covariance matrix
        cov = np.matmul(X @ np.diag(A), np.transpose(X)) / self.area

        # eigenvalue decomposition.
        evals, evecs = np.linalg.eig(cov)

        # find most dominant eigenvector with largest eigenvalue
        if evals[0] > evals[1]:
            x_v1, y_v1 = evecs[:, 0]
            sig = evals[0] / evals[1]
        else:
            x_v1, y_v1 = evecs[:, 1]
            sig = evals[1] / evals[0]

        angle = np.arctan2(x_v1, y_v1)
        return (180 / np.pi * angle), sig

    def pca_v2(self):
        """TODO: PCA version 2.0 with weights as # of additional points needs help.

        Returns
        -------

        """
        # theta, phi coordinates.
        phi = Contour.Mesh.p[self.contour_pixels_phi]
        theta = Contour.Mesh.t[self.contour_pixels_theta]

        # access the area of each pixel of the image grid.
        A = Contour.Mesh.da[self.contour_pixels_phi, self.contour_pixels_theta]

        A_frequency = np.array((A / np.min(A)), dtype="int")

        # recenter around weighted mean.
        pc = phi - self.phys_centroid[1]
        tc = theta - self.phys_centroid[0]

        # # new pc and tc
        # for ii in range(len(pc)):
        #     pc = np.append(pc, np.ones(int(A[ii])) * pc[ii])
        #     tc = np.append(tc, np.ones(int(A[ii])) * tc[ii])

        # feature matrix 2 by n. (n is number of pixels)
        X = np.array([pc, tc])

        npcov = np.cov(X, rowvar=True, fweights=A_frequency, ddof=0)

        # eigenvalue decomposition.
        evals, evecs = np.linalg.eig(npcov)

        # find most dominant eigenvector with largest eigenvalue
        if evals[0] > evals[1]:
            x_v1, y_v1 = evecs[:, 0]
            sig = evals[0] / evals[1]
        else:
            x_v1, y_v1 = evecs[:, 1]
            sig = evals[1] / evals[0]

        angle = np.arctan2(x_v1, y_v1)
        return (180 / np.pi * angle), sig

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
        t = np.arccos(z / np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2)))
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
            if w == 0 and h == 0:
                return Contour.Mesh.da[x, y]
            elif w == 0 and h > 0:
                np.sum(Contour.Mesh.da[x: x + w, y])
            elif w > 0 and h == 0:
                np.sum(Contour.Mesh.da[x, y:y + h])
            else:
                return np.sum(Contour.Mesh.da[x: x + w, y: y + h])

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

    def compute_convex_hull(self):
        """Find the Convex hull of the coronal hole, using the OpenCV Sklansky's algorithm,
         that has *O(N logN)* complexity in the current implementation.

        Returns
        -------
        All points that are in the corner of the convex hull.
        """
        if len(self.contour_pixels_theta) > 1 and len(self.contour_pixels_phi) > 1:
            # access contour pixels.
            points = self.open_cv_pixels_format()
            return cv.convexHull(points=points)
        else:
            return None

    def compute_rotated_rect(self):
        """Find the bounding rectangle with the minimum pixel area.

        Returns
        -------
         center (x,y), (width, height), angle of rotation
        """
        if len(self.contour_pixels_theta) > 1 and len(self.contour_pixels_phi) > 1:
            # access contour pixels.
            points = self.open_cv_pixels_format()
            return cv.minAreaRect(points=points)
        else:
            return None

    def open_cv_pixels_format(self):
        """Return OpenCV Contour pixel locations. This function is used to find minRect minCirlce and ConvexHull.

        Returns
        -------
        array - dim: [num points, 1, 2]
        """
        points = np.array([self.contour_pixels_phi, self.contour_pixels_theta])
        return np.reshape(np.transpose(points), newshape=(np.shape(points)[1], 1, 2))

    def rotated_rect_corners(self):
        """Find the rotated box corners.
        The lowest phi point in a rectangle is 0th vertex, and 1st, 2nd, 3rd vertices follow clockwise.
        Returns
        -------
        (x1, y1), (x2, y2), (x3, y3), (x4, y4)
        """
        if len(self.contour_pixels_theta) > 1 and len(self.contour_pixels_phi) > 1:
            box = cv.boxPoints(self.rot_box)
            return np.int0(box)
        else:
            return None

    def compute_rot_box_angle(self):
        """Find rotate box angle with respect to north.
        The lowest phi point in a rectangle is 0th vertex, and 1st, 2nd, 3rd vertices follow clockwise.

        Returns
        -------
        rotating angle in degrees.
        """
        p0, p1, p2 = self.rot_box_corners[:3]
        vec1 = p0 - p1
        vec2 = p2 - p1
        norm1 = np.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
        norm2 = np.sqrt(vec2[0] ** 2 + vec2[1] ** 2)
        if norm1 != 0 and norm2 != 0:
            if norm1 < norm2:
                ang = np.arccos(vec2[1] / norm2)
                if vec2[0] != 0:
                    if vec2[1] / vec2[0] > 0:
                        ang = - ang
            else:
                ang = np.arccos(vec1[1] / norm1)
                if vec1[0] != 0:
                    if vec1[1] / vec1[0] > 0:
                        ang = - ang

            return 180 / np.pi * ang
        else:
            return 0

    def compute_rot_box_area(self):
        """Compute rotated box area.

        Returns
        -------
        float rotated box area.
        """
        # create a mask the size of our original image.
        mask = np.zeros((Contour.n_t, Contour.n_p), dtype=np.uint8)
        # find all the pixels inside the rotated rectangle and mark them 1.
        cv.fillPoly(img=mask, pts=[self.rot_box_corners], color=1)
        # find pixel location of the mask.
        mask_location = np.where(mask)
        # multiply the mask by the area.
        return np.sum(Contour.Mesh.da[mask_location[1], mask_location[0]])
