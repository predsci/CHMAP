"""
Set of functions to change coordinates and interpolate between images and maps
"""
import sys

import numpy as np
import scipy.interpolate as sp_interp
import multiprocessing as mp
# change Pool default from 'fork' to 'spawn'
# mp.set_start_method("spawn")

import chmap.utilities.datatypes.datatypes as psi_dt
# import astropy_healpix


def c2s(x, y, z):
    """
    convert numpy arrays of x,y,z (cartesian) to r,t,p (spherical)
    """
    x2 = x**2
    y2 = y**2
    z2 = z**2
    r = np.sqrt(x2 + y2 + z2)
    t = np.arctan2(np.sqrt(x2 + y2), z)
    p = np.arctan2(y, x)

    # arctan2 returns values from -pi to pi but I want 0-2pi --> use fmod
    twopi = np.pi*2
    p = np.fmod(p + twopi, twopi)

    return r, t, p


def s2c(r, t, p):
    """
    convert numpy arrays of r,t,p (spherical) to x,y,z (cartesian)
    """
    ct = np.cos(t)
    st = np.sin(t)
    cp = np.cos(p)
    sp = np.sin(p)
    x = r*cp*st
    y = r*sp*st
    z = r*ct
    return x, y, z


def get_arclength(chord_length, radius=1.0):
    """
    convert the length of the chord connecting two points on a great circle to the
    arc length connecting them on the great circle.
    """
    arclength = 2*radius*np.arcsin(0.5*chord_length/radius)
    return arclength


def map_grid_to_image(map_x, map_y, R0=1.0, obsv_lon=0.0, obsv_lat=0.0, image_crota2=0.0):
    """
    Given a set of xy coordinate pairs, in map-space (x:horizontal phi axis, y:vertical sin(theta) axis, radius:R0),
    rotate and change variables to image space (x:horizontal, y:vertical, z:coming out of image, theta=0 at center of
    image, phi=0 at x=R0)
    :param map_x: numpy 1D array of map pixel x-locations [0, 2pi]
    :param map_y: numpy 1D array of map pixel y-locations [-1, 1]
    :param R0: Image radius in solar radii
    :param obsv_lon: Carrington longitude of image observer
    :param obsv_lat: Carrington latitude of image observer
    :param image_crota2: Degrees counterclockwise rotation needed to put solar-north-up in the image. This is generally
    a parameter in the .fits metadata named 'crota2'.
    :return:
    """

    # get map 3D spherical coords
    map_theta = np.pi / 2 - np.arcsin(map_y)
    map_phi = map_x

    # get map 3D cartesian coords
    map3D_x = R0 * np.sin(map_theta) * np.cos(map_phi)
    map3D_y = R0 * np.sin(map_theta) * np.sin(map_phi)
    map3D_z = R0 * np.cos(map_theta)

    # generate rotation matrix (from Carrington to image solar-north-up)
    rot_mat1 = map_to_image_rot_mat(obsv_lon, obsv_lat)
    # generate rotation matrix (from image solar-north-up to image orientation)
    rot_mat2 = snu_to_image_rot_mat(image_crota2)
    # combine rotations
    rot_mat = np.matmul(rot_mat2, rot_mat1)
    # construct coordinate array
    coord_array = np.array([map3D_x, map3D_y, map3D_z])
    # apply rotation matrix to coordinates
    image3D_coord = np.matmul(rot_mat, coord_array)

    # numeric error occasionally results in |z| > R0. This is a problem for np.arccos()
    image3D_coord[2, image3D_coord[2, :] > R0] = R0
    image3D_coord[2, image3D_coord[2, :] < -R0] = -R0
    # a more proper solution is to re-normalize each coordinate set, but that would be
    # more expensive (to fix error on an order we are not worried about). Also, the
    # numeric error of the renormalization could still create problems with arccos().

    image_phi = np.arctan2(image3D_coord[1, :], image3D_coord[0, :])
    image_theta = np.arccos(image3D_coord[2, :] / R0)

    return image3D_coord[0, :], image3D_coord[1, :], image3D_coord[2, :], image_theta, image_phi


def map_grid_to_helioprojective(map_x, map_y, R0=1.0, obsv_lon=0.0, obsv_lat=0.0, image_crota2=0.0,
                                D_sun_obs=1.5E11, r_sun_ref=6.96E8):
    """
    Given a set of xy coordinate pairs, in map-space (x:horizontal phi axis, y:vertical sin(theta) axis, radius:R0),
    rotate and change variables to image space (x:horizontal, y:vertical, z:coming out of image, theta=0 at center of
    image, phi=0 at x=R0)
    :param map_x: numpy 1D array of map pixel x-locations [0, 2pi]
    :param map_y: numpy 1D array of map pixel y-locations [-1, 1]
    :param R0: Image radius in solar radii
    :param obsv_lon: Carrington longitude of image observer
    :param obsv_lat: Carrington latitude of image observer
    :param image_crota2: Degrees counterclockwise rotation needed to put solar-north-up in the image. This is generally
    a parameter in the .fits metadata named 'crota2'.
    :return:
    """

    # get map 3D spherical coords
    map_theta = np.pi / 2 - np.arcsin(map_y)
    map_phi = map_x

    # get map 3D cartesian coords
    map3D_x = R0 * np.sin(map_theta) * np.cos(map_phi)
    map3D_y = R0 * np.sin(map_theta) * np.sin(map_phi)
    map3D_z = R0 * np.cos(map_theta)

    # generate rotation matrix (from Carrington to image solar-north-up)
    rot_mat1 = map_to_image_rot_mat(obsv_lon, obsv_lat)
    # generate rotation matrix (from image solar-north-up to image orientation)
    rot_mat2 = snu_to_image_rot_mat(image_crota2)
    # combine rotations
    rot_mat = np.matmul(rot_mat2, rot_mat1)
    # construct coordinate array
    coord_array = np.array([map3D_x, map3D_y, map3D_z])
    # apply rotation matrix to coordinates
    image3D_coord = np.matmul(rot_mat, coord_array)

    # numeric error occasionally results in |z| > R0. This is a problem for np.arccos()
    image3D_coord[2, image3D_coord[2, :] > R0] = R0
    image3D_coord[2, image3D_coord[2, :] < -R0] = -R0
    # a more proper solution is to re-normalize each coordinate set, but that would be
    # more expensive (to fix error on an order we are not worried about). Also, the
    # numeric error of the renormalization could still create problems with arccos().

    image_phi = np.arctan2(image3D_coord[1, :], image3D_coord[0, :])
    image_theta = np.arccos(image3D_coord[2, :] / R0)

    ## Convert image helio-centric to the native image coordinates (helio-projective)
    # convert D to solar-radii (to match x, y, z units)
    D = D_sun_obs/r_sun_ref
    D_minus_z = D - image3D_coord[2, :]
    xy_sq = image3D_coord[0, :]**2 + image3D_coord[1, :]**2
    d = np.sqrt(xy_sq + D_minus_z**2)
    Tx = np.arctan2(image3D_coord[0, :], D_minus_z)
    Ty = np.arcsin(image3D_coord[1, :]/d)
    # use intermediate values to also calculate alpha (angle from observer/sun-center line to
    # pixel center. Sometimes referred to as \theta_{\rho}
    rho = np.sqrt(xy_sq)
    alpha = np.arctan2(rho, D_minus_z)

    return Tx, Ty, image3D_coord[2, :], image_theta, image_phi, alpha


def image_grid_to_CR(image_x, image_y, R0=1.0, obsv_lat=0, obsv_lon=0, get_mu=False, outside_map_val=-9999.,
                     crota2=0.):
    """
    Given vector coordinate pairs in solar radii units and the observer angles, transform to map coords.
    :param image_x: vector of x coordinates
    :param image_y: vector of y coordinates
    :param R0: Assumed radius in solar radii.
    :param obsv_lat: Carrington latitude (degrees from equator) of observing instrument
    :param obsv_lon: Carrington longitude (degrees) of observing instrument
    :param crota2: Image-plane counterclockwise rotation needed for solar-north-up (degrees)
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
    # generate rotation matrix (from image solar-north-up to image orientation)
    rot_mat2 = snu_to_image_rot_mat(crota2)
    # invert for image-to-snu
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
    # Setup interpolation function and grid to evaluate on
    interp_fn = sp_interp.RegularGridInterpolator((reg_y, reg_x), reg_vals, method='linear')
    eval_pts = np.array([eval_y, eval_x]).transpose()

    # Do interpolation
    interp_result_vec = interp_fn(eval_pts)

    return interp_result_vec


def interpolate2D_regular2irregular_parallel(reg_x, reg_y, reg_vals, eval_x, eval_y, nprocs=1, tpp=5, p=None):
    """

    :param reg_x: numpy vector of x-coordinates (length N)
    :param reg_y: numpy vector of y-coordinates (length M)
    :param reg_vals: numpy MxN array_like containing the grid values
    :param eval_x: numpy column vector (length K) of x-coordinates to evaluate at.
    :param eval_y: numpy column vector (length K) of y-coordinates to evaluate at.
    :param nprocs: integer
                   number of processors to allow access to
    :param tpp: integer
                threads per process
    :return:
    """
    if nprocs < 1:
        sys.exit("coord_manip.interpolate2D_regular2irregular_parallel(): \n" +
                 "Number of processors nprocs must be 1 or greater.")

    # setup multiple threads
    # p = Pool(nprocs)
    if p is None:
        sys.exit("coord_manip.interpolate2D_regular2irregular_parallel(): \n" +
                 "Requires passing an active instance of multiprocessing.Pool().")

    # Setup interpolation function and grid to evaluate on
    interp_fn = sp_interp.RegularGridInterpolator((reg_y, reg_x), reg_vals, method='linear')
    eval_pts = np.array([eval_y, eval_x]).transpose()

    # divide eval points into equal segments
    eval_n = len(eval_x)
    step = int(np.ceil(eval_n/(tpp*nprocs)))
    # create a list of array inputs for interp_fn()
    var_list = list()
    for ii in range(nprocs*tpp):
        start = ii*step
        end = (ii+1)*step
        if end > eval_n:
            end = eval_n
        new_array = eval_pts[start:end, ]
        var_list = var_list + [new_array, ]

    # start evaluation
    interp_result_list = p.map(interp_fn, var_list)
    # close processor pool
    # p.close()
    # combine output back into one ndarray
    interp_result_vec = np.concatenate(interp_result_list)

    return interp_result_vec


def interp_los_image_to_map(image_in, R0, map_x, map_y, no_data_val=-9999., interp_field="data",
                            nprocs=1, tpp=1, p_pool=None, helio_proj=False):
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
    # determine if image is solar-north-up, or needs an additional rotation
    if hasattr(image_in, "sunpy_meta"):
        if "crota2" in image_in.sunpy_meta.keys():
            image_crota2 = image_in.sunpy_meta['crota2']
        else:
            image_crota2 = 0.
    else:
        image_crota2 = 0.

    # convert map grid variables to image space
    if helio_proj:
        D_sun_obs = image_in.sunpy_meta['dsun_obs']
        if "rsun_ref" in image_in.sunpy_meta.keys():
            r_sun_ref = image_in.sunpy_meta['rsun_ref']
        else:
            r_sun_ref = 6.96e8
        if "rsun_obs" in image_in.sunpy_meta.keys():
            r_sun_obs = image_in.sunpy_meta['rsun_obs']
        elif "rsun" in image_in.sunpy_meta.keys():
            r_sun_obs = image_in.sunpy_meta['rsun']
        else:
            sys.exit("Interpolation requires 'RSUN_OBS' in the sunpy_meta.")

        image_x, image_y, image_z, image_theta, image_phi, alpha = map_grid_to_helioprojective(
            map_x_vec, map_y_vec, R0=R0,
            obsv_lon=image_in.info['cr_lon'],
            obsv_lat=image_in.info['cr_lat'],
            image_crota2=image_crota2,
            D_sun_obs=D_sun_obs,
            r_sun_ref=r_sun_ref)
    else:
        image_x, image_y, image_z, image_theta, image_phi = map_grid_to_image(map_x_vec, map_y_vec, R0=R0,
                                                                              obsv_lon=image_in.info['cr_lon'],
                                                                              obsv_lat=image_in.info['cr_lat'],
                                                                              image_crota2=image_crota2)
    # only interpolate points on the visible portion of the sphere
    if helio_proj:
        sin_alpha = r_sun_ref/D_sun_obs
        # min z (apparent semi-radius) is R*sin_alpha, then divided by R to convert from meters to solar-radii
        min_z = sin_alpha*R0
        interp_index = image_z > min_z
    else:
        interp_index = image_z > 0

    # if type(image_in) is psi_dt.IITImage:
    #     im_data = image_in.iit_data
    # elif type(image_in) is psi_dt.LBCCImage:
    #     im_data = image_in.lbcc_data
    # else:
    #     im_data = image_in.data

    # interpolate the specified attribute
    im_data = getattr(image_in, interp_field)
    # put image axes in same coords as transformed map coords
    image_in_x = image_in.x.copy()
    image_in_y = image_in.y.copy()
    if helio_proj:
        # convert image axes back to arcsec
        image_in_x = image_in_x*r_sun_obs
        image_in_y = image_in_y*r_sun_obs
        # At 1 AU, this could be well approximated as np.sqrt(image_x**2 + image_y**2).
        # For generality, we do the full calculation. (now moved inside map_grid_to_helioprojective()
        # alpha = np.arctan2(np.sqrt(np.cos(image_y) ** 2 * np.sin(image_x) ** 2 + np.sin(image_y) ** 2),
        #                    np.cos(image_y) * np.cos(image_x))
        # convert CR axes from radians to arcsec
        rad2asec = (3600*180)/np.pi
        image_x = image_x * rad2asec
        image_y = image_y * rad2asec

    # interpolate
    if nprocs <= 1:
        interp_vec = interpolate2D_regular2irregular(image_in_x, image_in_y, im_data, image_x[interp_index],
                                                     image_y[interp_index])
    else:
        interp_vec = interpolate2D_regular2irregular_parallel(image_in_x, image_in_y, im_data, image_x[interp_index],
                                                              image_y[interp_index], nprocs=nprocs, tpp=tpp,
                                                              p=p_pool)
    interp_result_vec[interp_index] = interp_vec
    # reformat result to matrix form
    interp_result = interp_result_vec.reshape((map_nycoord, map_nxcoord), order="C")

    if helio_proj:
        # mu becomes slightly more complicated:
        # mu = cos(observer-to-radial angle)
        mu_vec = np.cos(image_theta + alpha)
    else:
        # mu = cos(center-to-radial angle)
        mu_vec = np.cos(image_theta)
    mu_mat = mu_vec.reshape((map_nycoord, map_nxcoord), order="C")

    out_obj = psi_dt.InterpResult(interp_result, map_x, map_y, mu_mat=mu_mat)

    return out_obj


def interp_los_image_to_map_par(image_in, R0, map_x, map_y, no_data_val=-9999., interp_field="data",
                                nprocs=1, tpp=1, p_pool=None, helio_proj=False):
    map_nxcoord = len(map_x)
    map_nycoord = len(map_y)

    # initialize grid to receive interpolation with values of NaN
    interp_result = np.full((map_nycoord, map_nxcoord), no_data_val, dtype='<f4')
    mu_result = np.full((map_nycoord, map_nxcoord), 0., dtype='<f4')

    # convert 1D map axis to full list of coordinates
    mat_x, mat_y = np.meshgrid(map_x, map_y)
    # convert matrix of coords to vector of coords (explicitly select row-major vectorizing)
    map_x_vec = mat_x.flatten(order="C")
    map_y_vec = mat_y.flatten(order="C")
    interp_result_vec = interp_result.flatten(order="C")
    mu_vec = mu_result.flatten(order="C")

    # put input axes in correct units
    im_data = getattr(image_in, interp_field)
    # put image axes in same coords as transformed map coords
    image_in_x = image_in.x.copy()
    image_in_y = image_in.y.copy()
    if helio_proj:
        # convert image axes back to arcsec
        image_in_x = image_in_x * image_in.sunpy_meta['rsun_obs']
        image_in_y = image_in_y * image_in.sunpy_meta['rsun_obs']

    # start a Pool of processes
    p_pool = mp.get_context("fork").Pool(nprocs)
    total_threads = nprocs*tpp
    n_coords = map_x_vec.__len__()
    coords_per_thread = int(n_coords/total_threads)
    # chunk CR coordinate vec and launch threads
    proc_list = [0, ]*total_threads
    for ii in range(total_threads):
        min_index = ii*coords_per_thread
        if ii == (total_threads-1):
            max_index = n_coords
        else:
            max_index = (ii+1)*coords_per_thread
        # launch coordinate conversions
        proc_list[ii] = p_pool.apply_async(coord_interp_chunk,
                                           (map_x_vec[min_index:max_index], map_y_vec[min_index:max_index],
                                            interp_result_vec[min_index:max_index], image_in_x, image_in_y,
                                            im_data, image_in.info),
                                           dict(sunpy_meta=image_in.sunpy_meta, R0=R0, helio_proj=helio_proj))

    # collect results in order (after each thread finishes)
    for ii in range(total_threads):
        proc_list[ii].wait()
        proc_out = proc_list[ii].get()
        # index into results vectors
        min_index = ii * coords_per_thread
        if ii == (total_threads - 1):
            max_index = n_coords
        else:
            max_index = (ii + 1) * coords_per_thread
        interp_result_vec[min_index:max_index] = proc_out[0]
        mu_vec[min_index:max_index] = proc_out[1]

    # close the pool of processes
    p_pool.close()

    # reformat result to matrix form
    interp_result = interp_result_vec.reshape((map_nycoord, map_nxcoord), order="C")
    mu_mat = mu_vec.reshape((map_nycoord, map_nxcoord), order="C")

    out_obj = psi_dt.InterpResult(interp_result, map_x, map_y, mu_mat=mu_mat)

    return out_obj


def coord_interp_chunk(map_x_vec, map_y_vec, interp_result_vec, image_in_x, image_in_y, im_data,
                       im_info, sunpy_meta=None, R0=1., helio_proj=True):
    # determine if image is solar-north-up, or needs an additional rotation
    if sunpy_meta is not None:
        if "crota2" in sunpy_meta.keys():
            image_crota2 = sunpy_meta['crota2']
        else:
            image_crota2 = 0.
    else:
        image_crota2 = 0.

    # convert map grid variables to image space
    if helio_proj:
        D_sun_obs = sunpy_meta['dsun_obs']
        r_sun_ref = sunpy_meta['rsun_ref']
        image_x, image_y, image_z, image_theta, image_phi, alpha = map_grid_to_helioprojective(
            map_x_vec, map_y_vec, R0=R0,
            obsv_lon=im_info['cr_lon'],
            obsv_lat=im_info['cr_lat'],
            image_crota2=image_crota2,
            D_sun_obs=D_sun_obs,
            r_sun_ref=r_sun_ref)
        # convert CR axes from radians to arcsec
        image_x = image_x * 206264.806
        image_y = image_y * 206264.806
    else:
        image_x, image_y, image_z, image_theta, image_phi = map_grid_to_image(map_x_vec, map_y_vec, R0=R0,
                                                                              obsv_lon=im_info['cr_lon'],
                                                                              obsv_lat=im_info['cr_lat'],
                                                                              image_crota2=image_crota2)
    # only interpolate points on the visible portion of the sphere
    if helio_proj:
        sin_alpha = r_sun_ref / D_sun_obs
        # min z (apparent semi-radius) is R*sin_alpha, then divided by R to convert from meters to solar-radii
        min_z = sin_alpha
        interp_index = image_z > min_z
    else:
        interp_index = image_z > 0

    # interpolate on-image values
    interp_vec = interpolate2D_regular2irregular(image_in_x, image_in_y, im_data, image_x[interp_index],
                                                 image_y[interp_index])
    # index on-image values back into results vec
    interp_result_vec[interp_index] = interp_vec

    if helio_proj:
        # mu becomes slightly more complicated:
        # mu = cos(observer-to-radial angle)
        mu_vec = np.cos(image_theta + alpha)
    else:
        # mu = cos(center-to-radial angle)
        mu_vec = np.cos(image_theta)

    return interp_result_vec, mu_vec


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

def snu_to_image_rot_mat(crota2):
    # Use the 'crota2' parameter (degrees counterclockwise) from fits metadata to rotate points
    # from solar-north-up (snu) to image orientation.
    # Assumes that we are in 3-D image-coordinates and rotating about
    # the observer line-of-sight (image z-axis)
    # Also assumes that the image-space horizontal axis is 'x' and increases to the right and
    # that the image-space vertical axis is 'y' and increases up.  Positive z-axis is toward
    # the observer.

    # From Thompson, 2005: Coordinate systems for solar image data
    # rot_mat =  cos(CROTA)  -sin(CROTA)
    #            sin(CROTA)   cos(CROTA)
    # for image to solar-north-up coordinates. Here we want to rotate the opposite direction

    # convert to radians
    crota_rad = np.pi*crota2/180
    rot_mat = np.array([[np.cos(crota_rad), np.sin(crota_rad), 0.],
                        [-np.sin(crota_rad), np.cos(crota_rad), 0.],
                        [0., 0., 1.]])
    # rot_mat = np.array([[np.cos(crota_rad), -np.sin(crota_rad), 0.],
    #                     [np.sin(crota_rad), np.cos(crota_rad), 0.],
    #                     [0., 0., 1.]])
    return rot_mat


def interp_los_image_to_map_yang(image_in, R0, map_x, map_y, no_data_val=-9999., interp_field="data",
                            nprocs=1, tpp=1, p_pool=None):
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
    # determine if image is solar-north-up, or needs an additional rotation
    if hasattr(image_in, "sunpy_meta"):
        if "crota2" in image_in.sunpy_meta.keys():
            image_crota2 = image_in.sunpy_meta['crota2']
        else:
            image_crota2 = 0.
    else:
        image_crota2 = 0.
    # convert map grid variables to image space
    image_x, image_y, image_z, image_theta, image_phi = map_grid_to_image(map_x_vec, map_y_vec, R0=R0,
                                                                          obsv_lon=image_in.info['cr_lon'],
                                                                          obsv_lat=image_in.info['cr_lat'],
                                                                          image_crota2=image_crota2)

    mu_vec = np.cos(image_theta)
    # only interpolate points on the front half of the sphere
    interp_index = image_z > 0

    # implement the corrections in perform_mapping() from Yang's code
    # http://jsoc.stanford.edu/cvs/JSOC/proj/mag/synop/apps/maprojbrfromblos.c?only_with_tag=Ver_LATEST
    sin_asd = 0.004660
    cos_asd = 0.99998914
    # calc radius correction factor
    r_cor = 1.*cos_asd/(1. - mu_vec*sin_asd)
    # apply to CR points in image plane
    image_x_cor = image_x*r_cor
    image_y_cor = image_y*r_cor

    # interpolate the specified attribute
    im_data = getattr(image_in, interp_field)

    # interpolate
    if nprocs <= 1:
        interp_vec = interpolate2D_regular2irregular(image_in.x, image_in.y, im_data, image_x_cor[interp_index],
                                                     image_y_cor[interp_index])
    else:
        interp_vec = interpolate2D_regular2irregular_parallel(image_in.x, image_in.y, im_data, image_x_cor[interp_index],
                                                              image_y_cor[interp_index], nprocs=nprocs, tpp=tpp,
                                                              p=p_pool)
    interp_result_vec[interp_index] = interp_vec
    # reformat result to matrix form
    interp_result = interp_result_vec.reshape((map_nycoord, map_nxcoord), order="C")

    ## calculate corrected mu-values for LOS to Br transformation
    # convert image_x and image_y into normalized-image units
    #   lower left image pixel is [-1, -1], upper right is [1, 1]
    radius_per_half_image = image_in.sunpy_meta['rsun_obs']/image_in.sunpy_meta['cdelt1'] / \
        (image_in.sunpy_meta['naxis1'] / 2.)
    # image_plane radius (corrected and in normalized-image units) = sin_rho
    sin2_rho = (image_x_cor*radius_per_half_image)**2 + (image_y_cor*radius_per_half_image)**2
    # mu = cos(rho) = sqrt(1 - sin^2(rho))
    mu_vec = np.sqrt(1. - sin2_rho)
    mu_mat = mu_vec.reshape((map_nycoord, map_nxcoord), order="C")

    out_obj = psi_dt.InterpResult(interp_result, map_x, map_y, mu_mat=mu_mat)

    return out_obj


# def interp_los_image_to_map_yang(image_in, R0, map_x, map_y, no_data_val=-9999., interp_field="data",
#                             nprocs=1, tpp=1, p_pool=None):
#     map_nxcoord = len(map_x)
#     map_nycoord = len(map_y)
#
#     # initialize grid to receive interpolation with values of NaN
#     interp_result = np.full((map_nycoord, map_nxcoord), no_data_val, dtype='<f4')
#
#     # convert 1D map axis to full list of coordinates
#     mat_x, mat_y = np.meshgrid(map_x, map_y)
#     # convert matrix of coords to vector of coords (explicitly select row-major vectorizing)
#     map_x_vec = mat_x.flatten(order="C")
#     map_y_vec = mat_y.flatten(order="C")
#     interp_result_vec = interp_result.flatten(order="C")
#     # determine if image is solar-north-up, or needs an additional rotation
#     if hasattr(image_in, "sunpy_meta"):
#         if "crota2" in image_in.sunpy_meta.keys():
#             image_crota2 = image_in.sunpy_meta['crota2']
#         else:
#             image_crota2 = 0.
#     else:
#         image_crota2 = 0.
#     # convert map grid variables to image space
#     image_x, image_y, image_z, image_theta, image_phi = map_grid_to_image(map_x_vec, map_y_vec, R0=R0,
#                                                                           obsv_lon=image_in.info['cr_lon'],
#                                                                           obsv_lat=image_in.info['cr_lat'],
#                                                                           image_crota2=image_crota2)
#     mu_vec = np.cos(image_theta)
#     # only interpolate points on the front half of the sphere
#     interp_index = image_z > 0
#
#     # implement the corrections in perform_mapping() from Yang's code
#     # http://jsoc.stanford.edu/cvs/JSOC/proj/mag/synop/apps/maprojbrfromblos.c?only_with_tag=Ver_LATEST
#     sin_asd = 0.004660
#     cos_asd = 0.99998914
#     # calc radius correction factor
#     r_cor = 1.*cos_asd/(1. - mu_vec*sin_asd)
#     # apply to CR points in image plane
#     image_x_cor = image_x*r_cor
#     image_y_cor = image_y*r_cor
#
#     # interpolate the specified attribute
#     im_data = getattr(image_in, interp_field)
#
#     # interpolate
#     if nprocs <= 1:
#         interp_vec = interpolate2D_regular2irregular(image_in.x, image_in.y, im_data, image_x_cor[interp_index],
#                                                      image_y_cor[interp_index])
#     else:
#         interp_vec = interpolate2D_regular2irregular_parallel(image_in.x, image_in.y, im_data, image_x_cor[interp_index],
#                                                               image_y_cor[interp_index], nprocs=nprocs, tpp=tpp,
#                                                               p=p_pool)
#     interp_result_vec[interp_index] = interp_vec
#     # reformat result to matrix form
#     interp_result = interp_result_vec.reshape((map_nycoord, map_nxcoord), order="C")
#
#     mu_mat = mu_vec.reshape((map_nycoord, map_nxcoord), order="C")
#
#     out_obj = psi_dt.InterpResult(interp_result, map_x, map_y, mu_mat=mu_mat)
#
#     return out_obj


