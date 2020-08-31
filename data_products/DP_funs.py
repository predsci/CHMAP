"""
functions used in production of various
output data/mapping products
"""
import time
import random
import numpy as np
import pandas as pd
import modules.Plotting as Plotting
import modules.DB_funs as db_funcs
import ezseg.ezsegwrapper as ezsegwrapper
import modules.datatypes as datatypes
from modules.map_manip import combine_mu_maps
import analysis.chd_analysis.CR_mapping_funcs as cr_funcs


def quality_map(db_session, map_data_dir, inst_list, query_pd, euv_combined, chd_combined=None, color_list=None):
    if color_list is None:
        color_list = ["Blues", "Greens", "Reds", "Oranges", "Purples"]
    # get origin images and mu arrays
    euv_origin_image = euv_combined.origin_image

    # get list of origin images
    euv_origins = np.unique(euv_origin_image)

    # create array of strings that is the same shape as euv/chd origin_image
    euv_image = np.empty(euv_origin_image.shape, dtype=object)

    # function to determine which image id corresponds to what instrument
    for euv_id in euv_origins:
        query_ind = np.where(query_pd['image_id'] == euv_id)
        instrument = query_pd['instrument'].iloc[query_ind[0]]
        if len(instrument) != 0:
            euv_image = np.where(euv_origin_image != euv_id, euv_image, instrument.iloc[0])

    # plot maps
    Plotting.PlotQualityMap(euv_combined, euv_image, inst_list, color_list,
                            nfig='EUV Quality Map ' + str(euv_combined.image_info.date_obs[0]),
                            title='EUV Quality Map: Mu Dependent\n' + str(euv_combined.image_info.date_obs[0]))
    # repeat for CHD map,      if applicable
    if chd_combined is not None:
        chd_origin_image = chd_combined.origin_image
        chd_origins = np.unique(chd_origin_image)
        chd_image = np.empty(chd_origin_image.shape, dtype=object)
        for chd_id in chd_origins:
            query_ind = np.where(query_pd['image_id'] == chd_id)
            instrument = query_pd['instrument'].iloc[query_ind[0]]
            if len(instrument) != 0:
                chd_image = np.where(euv_origin_image != chd_id, chd_image, instrument.iloc[0])
        Plotting.PlotQualityMap(chd_combined, chd_image, inst_list, color_list,
                                nfig='CHD Quality Map ' + str(chd_combined.image_info.date_obs[0]),
                                title='CHD Quality Map: Mu Dependent\n' + str(chd_combined.image_info.date_obs[0]),
                                map_type='CHD')

    # save these maps to database
    # TODO: figure out how to create "one" map

    return None


def create_timescale_maps(euv_combined, chd_combined, image_info, map_info, time_ind, euv_timescale, chd_timescale,
                          image_info_timescale, map_info_timescale):
    euv_timescale[time_ind] = euv_combined
    chd_timescale[time_ind] = chd_combined
    image_info_timescale[time_ind] = image_info
    map_info_timescale[time_ind] = map_info

    return euv_timescale, chd_timescale, image_info_timescale, map_info_timescale


def gauss_func(mu, sigma=0.15, bottom=None, top=None):
    a = random.gauss(mu, sigma)
    if bottom is not None and top is not None:
        while not (bottom <= a <= top):
            a = random.gauss(mu, sigma)
    return a


def gauss_chd(db_session, inst_list, los_image, iit_image, use_indices, iit_combo_query, thresh1=0.95, thresh2=1.35,
              nc=3, iters=1000, sigma=0.15):
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
    # calculate new threshold parameters based off reference (AIA) instrument
    t1 = thresh1 * ref_alpha + ref_x
    t2 = thresh2 * ref_alpha + ref_x
    # use gaussian varying for threshold
    gauss1 = gauss_func(mu=t1, sigma=sigma * t1, bottom=t1 - t1 * sigma, top=t1 + t1 * sigma)
    gauss2 = gauss_func(mu=t2, sigma=sigma * t2, bottom=t2 - t2 * sigma, top=t2 + t2 * sigma)

    # fortran chd algorithm
    np.seterr(divide='ignore')
    ezseg_output, iters_used = ezsegwrapper.ezseg(np.log10(image_data), use_chd, nx, ny, gauss1, gauss2, nc, iters)
    chd_result = np.logical_and(ezseg_output == 0, use_chd == 1)
    chd_result = chd_result.astype(int)

    # create CHD image
    chd_image = datatypes.create_chd_image(los_image, chd_result)
    chd_image.get_coordinates()

    end = time.time()
    print("Coronal Hole Detection Algorithm has been applied to image", iit_image.image_id, "in", end - start,
          "seconds.")

    return chd_image


def chd_mu_map(euv_map, chd_map, euv_combined, chd_combined, image_info, map_info, mu_cutoff=0.0, mu_merge_cutoff=None,
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
        euv_combined, chd_combined = combine_mu_maps(n_images, euv_maps, chd_maps, del_mu=del_mu, mu_cutoff=mu_cutoff)
        combined_method = {'meth_name': ("Min-Int-Merge_CR1", "Min-Int-Merge_CR1"), 'meth_description':
            ["Minimum intensity merge for CR Map: using del mu"] * 2,
                           'var_name': ("mu_cutoff", "del_mu"), 'var_description': ("lower mu cutoff value",
                                                                                    "max acceptable mu range"),
                           'var_val': (mu_cutoff, del_mu)}
    else:
        euv_combined, chd_combined = combine_mu_maps(n_images, euv_maps, chd_maps, mu_merge_cutoff=mu_merge_cutoff,
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


### PLOT VARIOUS MAP TYPES AND SAVE TO DATABASE
def save_running_avg_maps(db_session, map_data_dir, euv_combined, chd_combined, image_info, map_info, methods_list,
                          combined_method, time_min, time_max):
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
    Plotting.PlotMap(euv_combined, nfig="EUV Map Timescale Weighted",
                     title="EUV Map Timescale Weighted\nTime Min: " + str(time_min) + "\nTime Max: " + str(
                         time_max))
    Plotting.PlotMap(euv_combined, "CHD Map Timescale Weighted",
                     title="CHD Map Timescale Weighted\nTime Min: " + str(time_min) + "\nTime Max: " + str(
                         time_max))
    Plotting.PlotMap(chd_combined, nfig="CHD Map Timescale Weighted",
                     title="CHD Map Timescale Weighted\nTime Min: " +
                           str(time_min) + "\nTime Max: " + str(
                         time_max), map_type='CHD')

    # save EUV and CHD maps to database
    # euv_combined.write_to_file(map_data_dir, map_type='runavg_euv', filename=None, db_session=db_session)
    # chd_combined.write_to_file(map_data_dir, map_type='runavg_chd', filename=None, db_session=db_session)

    end = time.time()
    print("Combined CR Maps have been plotted and saved to the database in", end - start, "seconds.")


def save_mu_probability_maps(db_session, map_data_dir, euv_combined, chd_combined, image_info, map_info, methods_list,
                             combined_method):
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
    Plotting.PlotMap(euv_combined, nfig="CR EUV Map", title="Minimum Intensity Merge CR EUV Map\nTime Min: " + str(
        euv_combined.image_info.iloc[0].date_obs) + "\nTime Max: " + str(euv_combined.image_info.iloc[-1].date_obs))
    Plotting.PlotMap(chd_combined, nfig="Mu Dependent CHD Probability Map", title="Mu Dependent CHD Probability"
                                                                                  "Map\nTime Min: " + str(
        chd_combined.image_info.iloc[0].date_obs) + "\nTime Max: " + str(chd_combined.image_info.iloc[-1].date_obs),
                     map_type='CHD')

    # save EUV and CHD maps to database
    # euv_combined.write_to_file(map_data_dir, map_type='mudep_euv', filename=None, db_session=db_session)
    # chd_combined.write_to_file(map_data_dir, map_type='mudep_chd', filename=None, db_session=db_session)

    end = time.time()
    print("Combined CR Maps have been plotted and saved to the database in", end - start, "seconds.")


def save_threshold_maps(db_session, map_data_dir, euv_combined, chd_combined, image_info, map_info, methods_list,
                        combined_method):
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
    Plotting.PlotMap(euv_combined, nfig="Varying Threshold EUV Map", title="Minimum Intensity Merge CR EUV Map\nTime "
                                                                           "Min: " + str(
        euv_combined.image_info.iloc[0].date_obs) + "\nTime Max: " + str(euv_combined.image_info.iloc[-1].date_obs))
    Plotting.PlotMap(chd_combined, nfig="Varying Threshold CHD Map", title="Minimum Intensity CHD Merge Map with "
                                                                           "Gaussian Varying Threshold Values\nTime "
                                                                           "Min: " + str(
        chd_combined.image_info.iloc[0].date_obs) + "\nTime Max: " + str(chd_combined.image_info.iloc[-1].date_obs),
                     map_type='CHD')

    # save EUV and CHD maps to database
    # euv_combined.write_to_file(map_data_dir, map_type='varthresh_chd', filename=None, db_session=db_session)
    # chd_combined.write_to_file(map_data_dir, map_type='varthresh_chd', filename=None, db_session=db_session)

    end = time.time()
    print("Combined CR Maps have been plotted and saved to the database in", end - start, "seconds.")
