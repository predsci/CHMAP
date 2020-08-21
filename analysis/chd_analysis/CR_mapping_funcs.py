"""
functions used for EUV/CHD mapping of a full CR
"""

import os
import time
import numpy as np
import datetime
import pandas as pd

from settings.app import App
from modules.map_manip import combine_cr_maps
import modules.Plotting as Plotting
import ezseg.ezsegwrapper as ezsegwrapper
import modules.datatypes as datatypes
import modules.DB_classes as db_class
import modules.DB_funs as db_funcs
import analysis.chd_analysis.CHD_pipeline_funcs as chd_funcs
import analysis.lbcc_analysis.LBCC_theoretic_funcs as lbcc_funcs
import analysis.iit_analysis.IIT_pipeline_funcs as iit_funcs


#### STEP TWO: APPLY PRE-PROCESSING CORRECTIONS ####
def apply_ipp(db_session, hdf_data_dir, inst_list, row, methods_list, lbc_combo_query, iit_combo_query,
              n_intensity_bins=200, R0=1.01):
    start = time.time()
    index = row[0]
    image_row = row[1]
    inst_ind = inst_list.index(image_row.instrument)
    # apply LBC
    los_image, lbcc_image, mu_indices, use_ind, theoretic_query = lbcc_funcs.apply_lbc(db_session, hdf_data_dir,
                                                                                       lbc_combo_query[inst_ind],
                                                                                       image_row=image_row,
                                                                                       n_intensity_bins=n_intensity_bins,
                                                                                       R0=R0)
    # apply IIT
    lbcc_image, iit_image, use_indices, alpha, x = iit_funcs.apply_iit(db_session, iit_combo_query[inst_ind],
                                                                       lbcc_image, use_ind, los_image, R0=R0)
    # add methods to dataframe
    ipp_method = {'meth_name': ("LBCC", "IIT"), 'meth_description': ["LBCC Theoretic Fit Method", "IIT Fit Method"],
                  'var_name': ("LBCC", "IIT"), 'var_description': (" ", " ")}
    methods_list[index] = methods_list[index].append(pd.DataFrame(data=ipp_method), sort=False)

    end = time.time()
    print("Image Pre-Processing Corrections (Limb-Brightening and Inter-Instrument Transformation) have been "
          "applied to image", image_row.image_id, "in", end - start, "seconds.")

    return los_image, iit_image, methods_list, use_indices


#### STEP THREE: CORONAL HOLE DETECTION ####
def chd(db_session, inst_list, los_image, iit_image, use_indices, iit_combo_query, thresh1=0.95, thresh2=1.35, nc=3,
        iters=1000):
    start = time.time()
    # reference alpha, x for threshold
    sta_ind = inst_list.index('EUVI-A')
    ref_alpha, ref_x = db_funcs.query_var_val(db_session, meth_name='IIT', date_obs=los_image.info['date_string'],
                                              inst_combo_query=iit_combo_query[sta_ind])

    # define chd parameters
    image_data = iit_image.iit_data
    use_chd = use_indices.astype(int)
    use_chd = np.where(use_chd == 1, use_chd, los_image.no_data_val)
    nx = iit_image.x.size
    ny = iit_image.y.size
    t1 = thresh1 * ref_alpha + ref_x
    t2 = thresh2 * ref_alpha + ref_x

    # fortran chd algorithm
    np.seterr(divide='ignore')
    ezseg_output, iters_used = ezsegwrapper.ezseg(np.log10(image_data), use_chd, nx, ny, t1, t2, nc, iters)
    chd_result = np.logical_and(ezseg_output == 0, use_chd == 1)
    chd_result = chd_result.astype(int)

    # create CHD image
    chd_image = datatypes.create_chd_image(los_image, chd_result)
    chd_image.get_coordinates()

    end = time.time()
    print("Coronal Hole Detection Algorithm has been applied to image", iit_image.image_id, "in", end - start,
          "seconds.")

    return chd_image


#### STEP FOUR: CONVERT TO MAP ####
def create_map(iit_image, chd_image, methods_list, row, map_x=None, map_y=None, R0=1.01):
    start = time.time()
    index = row[0]
    image_row = row[1]
    # EUV map
    euv_map = iit_image.interp_to_map(R0=R0, map_x=map_x, map_y=map_y, image_num=image_row.image_id)
    # CHD map
    chd_map = chd_image.interp_to_map(R0=R0, map_x=map_x, map_y=map_y, image_num=image_row.image_id)
    # record image and map info
    euv_map.append_image_info(image_row)
    chd_map.append_image_info(image_row)

    # generate a record of the method and variable values used for interpolation
    interp_method = {'meth_name': ("Im2Map_Lin_Interp_1",), 'meth_description':
        ["Use SciPy.RegularGridInterpolator() to linearly interpolate from an Image to a Map"] * 1,
                     'var_name': ("R0",), 'var_description': ("Solar radii",), 'var_val': (R0,)}
    # add to the methods dataframe for this map
    methods_list[index] = methods_list[index].append(pd.DataFrame(data=interp_method), sort=False)

    # incorporate the methods dataframe into the map object
    euv_map.append_method_info(methods_list[index])
    chd_map.append_method_info(methods_list[index])

    end = time.time()
    print("Image number", iit_image.image_id, "has been interpolated to map(s) in", end - start, "seconds.")

    return euv_map, chd_map


#### STEP FIVE: CREATE COMBINED MAPS ####
def cr_map(euv_map, chd_map, euv_combined, chd_combined, image_info, map_info, mu_cutoff=0.0, mu_merge_cutoff=None,
           del_mu=None):
    start = time.time()
    # create map lists
    euv_maps = [euv_map, ]
    chd_maps = [chd_map, ]
    if euv_combined is not None:
        euv_maps.append(euv_combined)
    if chd_combined is not None:
        chd_maps.append(chd_combined)
    # determine number of images already in combined map
    n_images = len(image_info)

    # combine maps with minimum intensity merge
    if del_mu is not None:
        euv_combined, chd_combined = combine_cr_maps(n_images, euv_maps, chd_maps, del_mu=del_mu, mu_cutoff=mu_cutoff)
        combined_method = {'meth_name': ("Min-Int-Merge_CR1", "Min-Int-Merge_CR1"), 'meth_description':
            ["Minimum intensity merge for CR Map: using del mu"] * 2,
                           'var_name': ("mu_cutoff", "del_mu"), 'var_description': ("lower mu cutoff value",
                                                                                    "max acceptable mu range"),
                           'var_val': (mu_cutoff, del_mu)}
    else:
        euv_combined, chd_combined = combine_cr_maps(n_images, euv_maps, chd_maps, mu_merge_cutoff=mu_merge_cutoff,
                                                     mu_cutoff=mu_cutoff)
        combined_method = {'meth_name': ("Min-Int-Merge_CR2", "Min-Int-Merge_CR2"), 'meth_description':
            ["Minimum intensity merge for CR Map: based on Caplan et. al."] * 2,
                           'var_name': ("mu_cutoff", "mu_merge_cutoff"), 'var_description': ("lower mu cutoff value",
                                                                                         "mu cutoff value in areas of "
                                                                                         "overlap"),
                           'var_val': (mu_cutoff, mu_merge_cutoff)}
    # append image and map info records
    image_info.append(euv_map.image_info)
    map_info.append(euv_map.map_info)

    end = time.time()
    print("Image number", euv_map.image_info.image_id[0], "has been added to the combined CR map in", end - start,
          "seconds.")

    return euv_combined, chd_combined, combined_method


#### STEP SIX: PLOT COMBINED MAP AND SAVE TO DATABASE ####
def save_maps(db_session, map_data_dir, euv_combined, chd_combined, image_info, map_info, methods_list, combined_method):
    start = time.time()
    # generate a record of the method and variable values used for interpolation
    euv_combined.append_method_info(methods_list)
    euv_combined.append_method_info(pd.DataFrame(data=combined_method))
    chd_combined.append_method_info(methods_list)
    chd_combined.append_method_info(pd.DataFrame(data=combined_method))

    # generate record of image and map info
    euv_combined.append_image_info(image_info)
    euv_combined.append_map_info(map_info)
    chd_combined.append_image_info(image_info)
    chd_combined.append_map_info(map_info)

    # plot maps
    Plotting.PlotMap(euv_combined, nfig="CR EUV Map", title="Minimum Intensity Merge CR EUV Map\nTime Min: " + str(euv_combined.image_info.iloc[0].date_obs) +"\nTime Max: " + str(euv_combined.image_info.iloc[-1].date_obs))
    Plotting.PlotMap(euv_combined, nfig="CR CHD Map", title="Minimum Intensity CR CHD Merge Map")
    Plotting.PlotMap(chd_combined, nfig="CR CHD Map", title="Minimum Intensity CR CHD Merge Map\nTime Min: " + str(chd_combined.image_info.iloc[0].date_obs) +"\nTime Max: " + str(chd_combined.image_info.iloc[-1].date_obs), map_type='CHD')

    # save EUV and CHD maps to database
    # euv_combined.write_to_file(map_data_dir, map_type='cr_euv', filename=None, db_session=db_session)
    # chd_combined.write_to_file(map_data_dir, map_type='cr_chd', filename=None, db_session=db_session)

    end = time.time()
    print("Combined CR Maps have been plotted and saved to the database in", end-start, "seconds.")