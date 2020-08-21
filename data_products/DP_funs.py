"""
functions used in production of various
output data/mapping products
"""
import numpy as np
import modules.Plotting as Plotting
import analysis.chd_analysis.CR_mapping_funcs as cr_funcs


def quality_map(db_session, map_data_dir,inst_list, query_pd, euv_combined, chd_combined=None, color_list=None):
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