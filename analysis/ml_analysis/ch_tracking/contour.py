"""A data structure for a coronal hole contour.
Last Modified: April 13th, 2021 (Opal)

List of properties:                                         || Name of variable.
=======================================================================================================================
- centroid pixel location (theta: int, phi: int)            || pixel_centroid
- centroid physical location (phi: float, theta: float)     || phys_centroid
- contour physical area (solar radii)                       || area
- straight bounding rectangle                               || straight_box
- straight bounding rectangle area                          || straight box area
- rotated box                                               || rot_box
- rotated box corners                                       || rot_box_corners
- rotated box angle                                         || rot_box_angle
- coronal hole tilt                                         || pca_tilt
- coronal hole symmetry (eig_max/eig_min)                   || sig_tilt
- periodic boundary                                         || periodic_at_zero & periodic_at_2pi
"""
import numpy as np
import json
import cv2 as cv


class Contour:
    """ Coronal Hole Single Contour Object Data Structure.

    Parameters
    ----------
    Mesh:
        MapMesh object with image coordinates and pixel area.

     contour_pixels:
        coronal hole pixel location.
    """

    def __init__(self, contour_pixels, Mesh, frame_num=None):
        # save contour inner pixels. shape: [list(row), list(column)].
        self.contour_pixels_theta = contour_pixels[0]
        self.contour_pixels_phi = contour_pixels[1]

        # frame number id in Julian Day number.
        self.frame_num = frame_num

        # contour physical area based on lat-lon contour_pixels.
        # sum(d𝐴=𝑟^2 * sin𝜃 * d𝜙 * d𝜃)
        self.area = np.sum(Mesh.da[self.contour_pixels_phi, self.contour_pixels_theta])

        # contour centroid physical location in lat-lon projection. (t, p)
        self.phys_centroid = None

        # centroid pixel location in polar projection for coronal hole matching (x,y).
        self.pixel_centroid = self.compute_pixel_centroid(Mesh=Mesh)

        # compute the bounding box upper left corner, width, height (x, y, w, h).
        self.straight_box = self.compute_straight_bounding_rectangle()

        # contour straight bounding box physical area based on lat-lon contour_pixels.
        self.straight_box_area = self.compute_straight_box_area(Mesh=Mesh)

        # compute the rotated bounding box with min pixel area.
        self.rot_box = self.compute_rotated_rect()

        # save rot box corners.
        self.rot_box_corners = self.rotated_rect_corners()

        # save rot box angle with respect to north.
        self.rot_box_angle = self.compute_rot_box_angle()

        # compute the rotate box area.
        self.rot_box_area = self.compute_rot_box_area(Mesh=Mesh)

        # compute the tilt of the coronal hole in spherical coordinates using PCA.
        self.pca_tilt, self.sig_tilt = self.compute_coronal_hole_tilt_pca(Mesh=Mesh)

        # the unique identification number of this coronal hole (should be a natural number).
        self.id = None

        # the unique color for identification of this coronal hole rbg [r, b, g].
        self.color = None

        # count number in the identified frame.
        self.count = 0

        # periodic label.
        self.periodic_at_zero = self.is_periodic_zero()
        self.periodic_at_2pi = self.is_periodic_2_pi(Mesh=Mesh)

    def __str__(self):
        return json.dumps(
            self.json_dict(), indent=4, default=lambda o: o.json_dict())

    def json_dict(self):
        return {
            'frame_num': self.frame_num,
            'centroid_spherical': self.phys_centroid,
            'centroid_pixel': self.pixel_centroid,
            'area': self.area,
            'box': self.straight_box.tolist(),
            'straight_box_area': self.straight_box_area,
            'rotated_box_area': self.rot_box_area,
            'rotated_box_angle': self.rot_box_angle,
            'pca_tilt': self.pca_tilt,
            'significance_of_tilt': self.sig_tilt,
        }

    def compute_pixel_centroid(self, Mesh):
        """ Given the coronal hole pixel location we can compute the pixel center in polar projection.
            Saves the centroid pixel location and physical location in spherical coordinates.

        Parameters
        ----------
        Mesh:
            MapMesh object.

        Returns
        -------
            (t: int, p: int) image pixel coordinates
        """
        try:
            # convert to cartesian coordinates.
            x = Mesh.x2d[self.contour_pixels_phi, self.contour_pixels_theta]
            y = Mesh.y2d[self.contour_pixels_phi, self.contour_pixels_theta]
            z = Mesh.z2d[self.contour_pixels_phi, self.contour_pixels_theta]

            # access the area of each pixel of the image grid.
            A = Mesh.da[self.contour_pixels_phi, self.contour_pixels_theta]

            # compute the weighted mean of each coordinate and convert back to spherical coordinates.
            x_mean = np.dot(A, x) / self.area
            y_mean = np.dot(A, y) / self.area
            z_mean = np.dot(A, z) / self.area

            # convert back to spherical coordinates.
            self.phys_centroid = self._cartesian_centroid_to_spherical_coordinates(x=x_mean, y=y_mean, z=z_mean)

            # return image pixel coordinates.
            return self._spherical_coordinates_to_image_coordinates(t=self.phys_centroid[0], p=self.phys_centroid[1],
                                                                    Mesh=Mesh)

        except ArithmeticError:
            raise ArithmeticError('Contour pixels are invalid. ')

    def compute_coronal_hole_tilt_pca(self, Mesh):
        """Compute the tilt of the coronal hole with respect to north using PCA method.
        If eigenvalue ratio is close to 1 then the coronal hole tilt is insignificant.
        Otherwise, when the eigenvalue ratio>>1 then the coronal hole shape has an apparent tilt.

        Parameters
        ----------
        Mesh:
            MapMesh object with image coordinates and pixel area.

        Returns
        -------
            Tuple: (Angle, eigenvalue ratio)

        """
        # theta, phi coordinates.
        phi = Mesh.p[self.contour_pixels_phi]
        theta = Mesh.t[self.contour_pixels_theta]

        # access the area of each pixel of the image grid.
        A = Mesh.da[self.contour_pixels_phi, self.contour_pixels_theta]

        # recenter around weighted mean.
        pc = phi - self.phys_centroid[1]
        tc = theta - self.phys_centroid[0]

        # difference from the mean as an arc length
        pc = pc * np.sin(theta)

        # feature matrix 2 by n. (n is number of pixels)
        X = np.array([pc, tc])

        # covariance matrix
        cov = np.matmul(X @ np.diag(A), np.transpose(X)) / self.area

        # eigenvalue decomposition.
        evals, evecs = np.linalg.eig(cov)

        # find most dominant eigenvector with largest eigenvalue
        if evals[0] > evals[1]:
            x_v1, y_v1 = evecs[:, 0]
            # compute the ratio between the eigenvalues.
            # avoid dividing by zero.
            if evals[1] == 0.:
                sig = np.inf
            else:
                sig = evals[0] / evals[1]
        else:
            x_v1, y_v1 = evecs[:, 1]
            # compute the ratio between the eigenvalues.
            # avoid dividing by zero.
            if evals[0] == 0.:
                sig = np.inf
            else:
                sig = evals[1] / evals[0]

        angle = np.arctan2(x_v1, y_v1)
        return (180 / np.pi * angle), sig

    def _image_pixel_location_to_cartesian(self, t, p, Mesh):
        """ Convert longitude latitude image pixel location to cartesian coordinates.
        x = ρ sinθ cosφ
        y = ρ sinθ sinφ
        z = ρ cosθ

        Parameters
        ----------
        t: list or numpy array
        p: list or numpy array
        Mesh:
            MapMesh object with image coordinates and pixel area.

        Returns
        -------
        x, y, z - all numpy arrays
        """
        # convert image pixel location to spherical coordinates.
        theta, phi = self._image_coordinates_to_spherical_coordinates(t=t, p=p, Mesh=Mesh)
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
    def _spherical_coordinates_to_image_coordinates(t, p, Mesh):
        """Convert spherical coordinates to image coordinates.

        Parameters
        ----------
        t: float
        p: float
        Mesh:
            MapMesh object with image coordinates and pixel area.

        Returns
        -------
        t, p
        """
        if 0 <= p <= 2 * np.pi and 0 <= t <= np.pi:
            return int(Mesh.interp_t2index(t)), int(Mesh.interp_p2index(p))
        else:
            raise ValueError('When converting spherical coordinates to image coordinates,'
                             ' 0 <= phi < 2pi and 0 <= theta <= pi.')

    @staticmethod
    def _image_coordinates_to_spherical_coordinates(t, p, Mesh):
        """Convert image coordinates to spherical coordinates.

        Parameters
        ----------
        t: int
        p: int
        Mesh:
            MapMesh object with image coordinates and pixel area.

        Returns
        -------
        row, column spherical coordinates.
        """
        try:
            return Mesh.t[t], Mesh.p[p]
        except Exception:
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

    def compute_straight_box_area(self, Mesh):
        """ Compute the contour's straight bounding box area.
        d𝐴=𝑟^2 * sin𝜃 * d𝜙 * d𝜃, let r be in solar radii.
        A = (cos(𝜃1) - cos(𝜃2)) * (𝜙1 - 𝜙2)

        Parameters
        ----------
        Mesh:
            MapMesh object with image coordinates and pixel area.


        Returns
        -------
        float
        """
        try:
            # access left corner pixel coordinates and width and height.
            x, y, w, h = self.straight_box
            if w == 0 and h == 0:
                return Mesh.da[x, y]
            elif w == 0 and h > 0:
                np.sum(Mesh.da[x: x + w, y])
            elif w > 0 and h == 0:
                np.sum(Mesh.da[x, y:y + h])
            else:
                return np.sum(Mesh.da[x: x + w, y: y + h])

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

    def is_periodic_2_pi(self, Mesh):
        """If the coronal hole detected at 2 pi (phi)

        Parameters
        ----------
        Mesh:
            MapMesh object with image coordinates and pixel area.

        Returns
        -------
        boolean
        """
        # check if 2pi pixel is included in the contour pixels.
        if (Mesh.n_p - 1) in self.contour_pixels_phi:
            return True
        return False

    def lat_interval_at_2_pi(self, Mesh):
        """Return the latitude pixel interval where the contour pixels are 2pi longitude.
        This function is used to force periodicity.

        Parameters
        ----------
        Mesh:
            MapMesh object with image coordinates and pixel area.

        Returns
        -------
        (min_lat, max_lat)- pixel image coordinates
        """
        if self.periodic_at_2pi:
            mask = (self.contour_pixels_phi == int(Mesh.n_p - 1))
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

    def compute_rot_perimeter(self, Mesh):
        """Compute the rotated box perimeter on a sphere using the haversine metric.

        Parameters
        ----------
        Mesh:
            MapMesh object with image coordinates and pixel area.

        Returns
        -------
            float: perimeter on a sphere.
        """
        if self.rot_box_corners is None:
            return None
        else:
            # compute the distance on a sphere between each consecutive points on the rotated box.

            # initialize the perimeter holder.
            perimeter = 0

            # loop over each consecutive pair of points and compute its distance on a sphere
            # using the haversine metric.
            for ii in range(4):

                # compute the distance between the first and the last point.
                if ii == 3:
                    ii = -1

                # convert the two consecutive points from pixel coordinates to spherical coordinates.
                p1 = self._image_coordinates_to_spherical_coordinates(t=self.rot_box_corners[ii][1],
                                                                      p=self.rot_box_corners[ii][0],
                                                                      Mesh=Mesh)
                p2 = self._image_coordinates_to_spherical_coordinates(t=self.rot_box_corners[ii + 1][1],
                                                                      p=self.rot_box_corners[ii + 1][0],
                                                                      Mesh=Mesh)
                perimeter += self.haversine(p1=p1, p2=p2)

            return perimeter

    @staticmethod
    def haversine(p1, p2):
        """ Compute the distance of two points on a sphere. Assuming the input point coordinates are spherical
        ordered as follows: (phi, theta) or (longitude, latitude) in radians.

        Returns
        -------
            distance on a sphere in RS (Solar Radii)
        """
        # read in the two points location in spherical coordinates.
        theta1, phi1 = p1
        theta2, phi2 = p2

        # haversine formula.
        dlon = phi2 - phi1
        dlat = theta2 - theta1

        # geometric calculations.
        a = np.sin(dlat / 2) ** 2 + np.cos(theta1) * np.cos(theta2) * np.sin(dlon / 2) ** 2
        return 2 * np.arcsin(np.sqrt(a))

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
        """Return OpenCV Contour pixel locations. This function is used to find minRect minCircle and ConvexHull.

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
        if len(self.contour_pixels_theta) > 1 and len(self.contour_pixels_phi) > 1 and self.rot_box is not None:
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
        if self.rot_box is not None:
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
        else:
            return None

    def compute_rot_box_area(self, Mesh):
        """Compute rotated box area.

        Parameters
        ----------
        Mesh:
            MapMesh object with image coordinates and pixel area.

        Returns
        -------
        float rotated box area.
        """
        if self.rot_box is not None:
            # create a mask the size of our original image.
            mask = np.zeros((Mesh.n_t, Mesh.n_p), dtype=np.uint8)
            # find all the pixels inside the rotated rectangle and mark them 1.
            cv.fillPoly(img=mask, pts=[self.rot_box_corners], color=1)
            # find pixel location of the mask.
            mask_location = np.where(mask)
            # multiply the mask by the area.
            return np.sum(Mesh.da[mask_location[1], mask_location[0]])

    # def compute_convex_hull_perimeter(self, Mesh):
    #     """Compute the convex hull perimeter on a sphere using the haversine metric.
    #
    #     Parameters
    #     ----------
    #     Mesh:
    #         MapMesh object with image coordinates and pixel area.
    #
    #     Returns
    #     -------
    #         float: perimeter on a sphere.
    #     """
    #     if self.convex_hull is None:
    #         return None
    #
    #     elif len(self.convex_hull) < 3:
    #         return None
    #
    #     else:
    #         # compute the distance on a sphere between each consecutive points on the convex hull.
    #
    #         # initialize the perimeter holder.
    #         perimeter = 0
    #
    #         # loop over each consecutive pair of points and compute its distance on a sphere
    #         # using the haversine metric.
    #         for ii in range(len(self.convex_hull)):
    #
    #             # compute the distance between the first and the last point.
    #             if ii == len(self.convex_hull) - 1:
    #                 ii = -1
    #
    #             # convert the two consecutive points from pixel coordinates to spherical coordinates.
    #             p1 = self._image_coordinates_to_spherical_coordinates(t=self.convex_hull[ii][0][0],
    #                                                                   p=self.convex_hull[ii][0][1],
    #                                                                   Mesh=Mesh)
    #             p2 = self._image_coordinates_to_spherical_coordinates(t=self.convex_hull[ii + 1][0][0],
    #                                                                   p=self.convex_hull[ii + 1][0][1],
    #                                                                   Mesh=Mesh)
    #             perimeter += self.haversine(p1=p1, p2=p2)
    #
    #         return perimeter
