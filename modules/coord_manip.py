"""
Set of functions to change coordinates and interpolate between images and maps
"""

import numpy as np
import scipy.interpolate as sp_interp

import modules.datatypes as psi_dt
import astropy_healpix


def map_grid_to_image(map_x, map_y, R0=1.0, obsv_lon=0.0, obsv_lat=0.0):
    """
    Given a set of xy coordinate pairs, in map-space (x:horizontal phi axis, y:vertical sin(theta) axis, radius:R0),
    rotate and change variables to image space (x:horizontal, y:vertical, z:coming out of image, theta=0 at center of
    image, phi=0 at x=R0)
    :param map_x: numpy 1D array of map pixel x-locations [0, 2pi]
    :param map_y: numpy 1D array of map pixel y-locations [-1, 1]
    :param R0: Image radius in solar radii
    :param obsv_lon: Carrington longitude of image observer
    :param obsv_lat: Carrington latitude of image observer
    :return:
    """

    # get map 3D spherical coords
    map_theta = np.pi / 2 - np.arcsin(map_y)
    map_phi = map_x

    # get map 3D cartesian coords
    map3D_x = R0 * np.sin(map_theta) * np.cos(map_phi)
    map3D_y = R0 * np.sin(map_theta) * np.sin(map_phi)
    map3D_z = R0 * np.cos(map_theta)

    # rotate phi (about map z-axis. observer phi goes to -y)
    # del_phi = -obsv_lon*np.pi/180 - np.pi/2
    # int3D_x = np.cos(del_phi)*map3D_x - np.sin(del_phi)*map3D_y
    # int3D_y = np.cos(del_phi)*map3D_y + np.sin(del_phi)*map3D_x
    # int3D_z = map3D_z

    # rotate theta (about x-axis. observer theta goes to +z)
    # del_theta = obsv_lat*np.pi/180 - np.pi/2
    # image_x = int3D_x
    # image_y = np.cos(del_theta)*int3D_y - np.sin(del_theta)*int3D_z
    # image_z = np.cos(del_theta)*int3D_z + np.sin(del_theta)*int3D_y
    # rotate theta (about intermediate y-axis) exterior minus sign is due to right-hand-rule rotation direction.
    # del_theta = -cr_lat*np.pi/180 + np.pi/2
    # image_x = np.cos(del_theta)*int3D_x + np.sin(del_theta)*int3D_z
    # image_y = int3D_y
    # image_z = -np.sin(del_theta)*int3D_x + np.cos(del_theta)*int3D_z

    # generate rotation matrix
    rot_mat = map_to_image_rot_mat(obsv_lon, obsv_lat)
    # construct coordinate array
    coord_array = np.array([map3D_x, map3D_y, map3D_z])
    # apply rotation matrix to coordinates
    image3D_coord = np.matmul(rot_mat, coord_array)

    image_phi = np.arctan2(image3D_coord[1, :], image3D_coord[0, :])
    image_theta = np.arccos(image3D_coord[2, :] / R0)

    return image3D_coord[0, :], image3D_coord[1, :], image3D_coord[2, :], image_theta, image_phi


def image_grid_to_CR(image_x, image_y, R0=1.0, obsv_lat=0, obsv_lon=0, get_mu=False, outside_map_val=-9999.):
    """
    Given vector coordinate pairs in solar radii units and the observer angles, transform to map coords.
    :param image_x: vector of x coordinates
    :param image_y: vector of y coordinates
    :param R0: Assumed radius in solar radii.
    :param obsv_lat: Carrington latitude (degrees from equator) of observing instrument
    :param obsv_lon: Carrington longitude (degrees) of observing instrument
    :return:
    """

    # for images, we assume that the z-axis is perpendicular to the image plane, in the direction
    # of the observer, and located at the center of the image.

    # mask points outside of R0
    use_index = image_x ** 2 + image_y ** 2 <= R0 ** 2
    use_x = image_x[use_index]
    use_y = image_y[use_index]

    # Find z coord (we can assume it is in the positive direction)
    # use_z = np.sqrt(R0 ** 2 - use_x ** 2 - use_y ** 2)
    # to be numerically equivalent to the use_index definition, change to this:
    use_z = np.sqrt(R0 ** 2 - (use_x ** 2 + use_y ** 2))

    # Calc image_theta, image_phi, and image_mu
    if get_mu:
        image_mu = np.full(image_x.shape, outside_map_val)
        # image_phi = np.arctan2(image_y, image_x)
        use_theta = np.arccos(use_z / R0)
        image_mu[use_index] = np.cos(use_theta)

    # generate map-to-image rotation matrix
    rot_mat = map_to_image_rot_mat(obsv_lon, obsv_lat)
    # invert/transpose for image-to-map rotation matrix
    rev_rot = rot_mat.transpose()
    # construct coordinate array
    coord_array = np.array([use_x, use_y, use_z])
    # apply rotation matrix to coordinates
    map3D_coord = np.matmul(rev_rot, coord_array)

    # Occasionally numeric error from the rotation causes a z magnitude to be greater than R0
    num_err_z_index = np.abs(map3D_coord[2, :]) > R0
    map3D_coord[2, num_err_z_index] = np.sign(map3D_coord[2, num_err_z_index]) * R0
    # Convert map cartesian to map theta and phi
    cr_theta = np.arccos(map3D_coord[2, :] / R0)
    cr_phi = np.arctan2(map3D_coord[1, :], map3D_coord[0, :])
    # Change phi range from [-pi,pi] to [0,2pi]
    neg_phi = cr_phi < 0
    cr_phi[neg_phi] = cr_phi[neg_phi] + 2 * np.pi

    cr_theta_all = np.full(image_x.shape, outside_map_val)
    cr_phi_all = np.full(image_x.shape, outside_map_val)

    cr_theta_all[use_index] = cr_theta
    cr_phi_all[use_index] = cr_phi

    if get_mu:
        return cr_theta_all, cr_phi_all, image_mu
    else:
        return cr_theta_all, cr_phi_all, None


def interpolate2D_regular2irregular(reg_x, reg_y, reg_vals, eval_x, eval_y):
    """
    For a 2D MxN regular grid, interpolate values to the K grid points in eval_x and eval_y.
    :param reg_x: numpy vector of x-coordinates (length N)
    :param reg_y: numpy vector of y-coordinates (length M)
    :param reg_vals: numpy MxN array_like containing the grid values
    :param eval_x: numpy column vector (length K) of x-coordinates to evaluate at.
    :param eval_y: numpy column vector (length K) of y-coordinates to evaluate at.
    :return: vector length K of interpolation results.
    """
    # Setup interpolation function and grd to evaluate on
    interp_fn = sp_interp.RegularGridInterpolator((reg_x, reg_y), reg_vals)
    eval_pts = np.array([eval_y, eval_x]).transpose()

    # Do interpolation
    interp_result_vec = interp_fn(eval_pts)

    return interp_result_vec


def interp_los_image_to_map(image_in, R0, map_x, map_y, no_data_val=-9999.):
    map_nxcoord = len(map_x)
    map_nycoord = len(map_y)

    # initialize grid to receive interpolation with values of NaN
    interp_result = np.full((map_nycoord, map_nxcoord), no_data_val, dtype='<f4')

    # convert 1D map axis to full list of coordinates
    mat_x, mat_y = np.meshgrid(map_x, map_y)
    # convert matrix of coords to vector of coords (explicitly select row-major vectorizing)
    map_x_vec = mat_x.flatten(order="C")
    map_y_vec = mat_y.flatten(order="C")
    interp_result_vec = interp_result.flatten(order="C")
    # convert map grid variables to image space
    image_x, image_y, image_z, image_theta, image_phi = map_grid_to_image(map_x_vec, map_y_vec, R0=R0,
                                                                          obsv_lon=image_in.info['cr_lon'],
                                                                          obsv_lat=image_in.info['cr_lat'])
    # only interpolate points on the front half of the sphere
    interp_index = image_z > 0

    if type(image_in) is psi_dt.IITImage:
        im_data = image_in.iit_data
    elif type(image_in) is psi_dt.LBCCImage:
        im_data = image_in.lbcc_data
    else:
        im_data = image_in.data
    interp_vec = interpolate2D_regular2irregular(image_in.x, image_in.y, im_data, image_x[interp_index],
                                                 image_y[interp_index])
    interp_result_vec[interp_index] = interp_vec
    # reformat result to matrix form
    interp_result = interp_result_vec.reshape((map_nycoord, map_nxcoord), order="C")

    mu_vec = np.cos(image_theta)
    mu_mat = mu_vec.reshape((map_nycoord, map_nxcoord), order="C")

    # interpolate longitude array
    interp_vec_lon = interpolate2D_regular2irregular(image_in.x, image_in.y, image_in.lon, image_x[interp_index],
                                                     image_y[interp_index])
    interp_result_vec_lon = interp_result.flatten(order="C")
    interp_result_vec_lon[interp_index] = interp_vec_lon
    # reformat result to matrix form
    interp_result_lon = interp_result_vec_lon.reshape((map_nycoord, map_nxcoord), order="C")
    map_lon = interp_result_lon.reshape((map_nycoord, map_nxcoord), order="C")

    out_obj = psi_dt.InterpResult(interp_result, map_x, map_y, mu_mat=mu_mat, map_lon=map_lon)

    return out_obj


def map_to_image_rot_mat(obsv_lon, obsv_lat):
    # del_phi = -obsv_lon*np.pi/180 - np.pi/2
    # int3D_x = np.cos(del_phi)*map3D_x - np.sin(del_phi)*map3D_y
    # int3D_y = np.cos(del_phi)*map3D_y + np.sin(del_phi)*map3D_x
    # int3D_z = map3D_z

    # rotate phi (about map z-axis. observer phi goes to -y)
    del_phi = -obsv_lon * np.pi / 180 - np.pi / 2
    rot1 = np.array([[np.cos(del_phi), -np.sin(del_phi), 0.],
                     [np.sin(del_phi), np.cos(del_phi), 0.], [0., 0., 1.], ])

    # rotate theta (about x-axis. observer theta goes to +z)
    del_theta = obsv_lat * np.pi / 180 - np.pi / 2
    rot2 = np.array([[1., 0., 0.], [0., np.cos(del_theta), -np.sin(del_theta)],
                     [0., np.sin(del_theta), np.cos(del_theta)]])

    tot_rot = np.matmul(rot2, rot1)

    return tot_rot


#### test functions
def interp_los_image_to_map2(image_in, R0, map_x, map_y, no_data_val=-9999.):
    map_nxcoord = len(map_x)
    map_nycoord = len(map_y)

    # initialize grid to receive interpolation with values of NaN
    interp_result = np.full((map_nycoord, map_nxcoord), no_data_val, dtype='<f4')

    # convert 1D map axis to full list of coordinates
    mat_x, mat_y = np.meshgrid(map_x, map_y)
    # convert matrix of coords to vector of coords (explicitly select row-major vectorizing)
    map_x_vec = mat_x.flatten(order="C")
    map_y_vec = mat_y.flatten(order="C")
    interp_result_vec = interp_result.flatten(order="C")
    # convert map grid variables to image space
    image_x, image_y, image_z, image_theta, image_phi = map_grid_to_image(map_x_vec, map_y_vec, R0=R0,
                                                                          obsv_lon=image_in.info['cr_lon'],
                                                                          obsv_lat=image_in.info['cr_lat'])
    # only interpolate points on the front half of the sphere
    interp_index = image_z > 0

    interp_vec = interpolate2D_regular2irregular2(image_in.lon, image_in.lat, image_in.data.flatten(),
                                                  image_x[interp_index],
                                                  image_y[interp_index])
    interp_result_vec[interp_index] = interp_vec
    # reformat result to matrix form
    interp_result = interp_result_vec.reshape((map_nycoord, map_nxcoord), order="C")

    mu_vec = np.cos(image_theta)
    mu_mat = mu_vec.reshape((map_nycoord, map_nxcoord), order="C")

    out_obj = psi_dt.InterpResult(interp_result, map_x, map_y, mu_mat=mu_mat)

    return out_obj


def interpolate2D_regular2irregular2(lon, lat, values, eval_x, eval_y):
    """
    For a 2D MxN regular grid, interpolate values to the K grid points in eval_x and eval_y.
    :param reg_x: numpy vector of x-coordinates (length N)
    :param reg_y: numpy vector of y-coordinates (length M)
    :param reg_vals: numpy MxN array_like containing the grid values
    :param eval_x: numpy column vector (length K) of x-coordinates to evaluate at.
    :param eval_y: numpy column vector (length K) of y-coordinates to evaluate at.
    :return: vector length K of interpolation results.
    """
    # Setup interpolation function and grd to evaluate on
    # interp_fn = sp_interp.RegularGridInterpolator((reg_x, reg_y), reg_vals)
    hp = astropy_healpix.HEALPix(values)
    interp_fn = astropy_healpix.interpolate_bilinear_lonlat(lon, lat, values)
    eval_pts = np.array([eval_y, eval_x]).transpose()

    # Do interpolation
    interp_result_vec = interp_fn(eval_pts)

    return interp_result_vec
