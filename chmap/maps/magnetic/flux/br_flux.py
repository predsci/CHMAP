

import os
import numpy as np
import datetime

import chmap.database.db_funs as db_funs
import utilities.datatypes.datatypes as psi_datatype
import chmap.maps.util.map_manip as map_manip


def coronal_flux(db_session, chd_contour, frame_timestamp, map_dir,
                 window_half_width=datetime.timedelta(hours=12)):
    window_min = frame_timestamp - window_half_width
    window_max = frame_timestamp + window_half_width

    # find PSI magnetic map before and after CHD timestamp
    map_info, data_info, method_info, image_assoc = db_funs.query_euv_maps(
        db_session, mean_time_range=[window_min, window_max], n_images=1,
        methods=["ProjFlux2Map"])

    below_index = map_info.date_mean <= frame_timestamp
    if any(below_index):
        first_below_index = np.max(np.where(below_index))
    else:
        first_below_index = None
    above_index = map_info.date_mean >= frame_timestamp
    if any(above_index):
        first_above_index = np.min(np.where(above_index))
    else:
        first_above_index = None

    if first_below_index is None:
        if first_above_index is None:
            print("No B_r maps in the specified window. Flux calculation canceled.")
        else:
            # use first_above_index map
            full_path = os.path.join(map_dir, map_info.fname[first_above_index])
            interp_map = psi_datatype.read_psi_map(full_path)

    else:
        if first_above_index is None:
            # use first_below index map
            full_path = os.path.join(map_dir, map_info.fname[first_below_index])
            interp_map = psi_datatype.read_psi_map(full_path)
        else:
            # load both maps
            full_path = os.path.join(map_dir, map_info.fname[first_above_index])
            first_above = psi_datatype.read_psi_map(full_path)
            full_path = os.path.join(map_dir, map_info.fname[first_below_index])
            first_below = psi_datatype.read_psi_map(full_path)
            # do linear interpolation
            interp_map = first_above.__copy__()
            below_weight = (frame_timestamp - map_info.date_mean[first_below_index])/(
                    map_info.date_mean[first_above_index] - map_info.date_mean[first_below_index])
            above_weight = 1. - below_weight
            interp_map.data = below_weight*first_below.data + above_weight*first_above.data

    # double check that Contour and Br map have same mesh???
    # temporary solution for testing purposes
    y_index = chd_contour.contour_pixels_theta
    x_index = chd_contour.contour_pixels_phi
    br_shape = interp_map.data.shape
    keep_ind = (y_index <= br_shape[0]) & (x_index <= br_shape[1])
    y_index = y_index[keep_ind]
    x_index = x_index[keep_ind]

    # use contour indices and Br linear approx to calc flux
    # calc theta from sin(lat). increase float precision to reduce numeric
    # error from np.arcsin()
    theta_y = np.pi/2 - np.arcsin(np.flip(interp_map.y.astype('float64')))
    # generate a mesh
    map_mesh = map_manip.MapMesh(interp_map.x, theta_y)
    # convert area characteristic of mesh back to map grid
    map_da = np.flip(map_mesh.da.transpose(), axis=0)
    # sum total flux
    chd_flux = map_manip.br_flux_indices(interp_map, y_index, x_index, map_da)

    return chd_flux