import datetime
import time

import pandas as pd

from chmap.maps.util.map_manip import combine_maps
from settings.info import DTypes
from utilities.plotting import psi_plotting as Plotting


def create_combined_maps(db_session, map_data_dir, map_list, chd_map_list, methods_list,
                         data_info, map_info, mu_merge_cutoff=None, del_mu=None, mu_cutoff=0.0):
    """
    function to create combined EUV and CHD maps and save to database with associated method information
    @param db_session: database session to save maps to
    @param map_data_dir: directory to save map files
    @param map_list: list of EUV maps
    @param chd_map_list: list of CHD maps
    @param methods_list: methods list
    @param data_info: image info list
    @param map_info: map info list
    @param mu_merge_cutoff: cutoff mu value for overlap areas
    @param del_mu: maximum mu threshold value
    @param mu_cutoff: lower mu value
    @return: combined euv map, combined chd map
    """
    # start time
    start = time.time()
    # create combined maps
    euv_maps = []
    chd_maps = []
    for euv_map in map_list:
        if len(euv_map.data) != 0:
            euv_maps.append(euv_map)
    for chd_map in chd_map_list:
        if len(chd_map.data) != 0:
            chd_maps.append(chd_map)
    if del_mu is not None:
        euv_combined, chd_combined = combine_maps(euv_maps, chd_maps, del_mu=del_mu, mu_cutoff=mu_cutoff)
        combined_method = {'meth_name': ("Min-Int-Merge-del_mu", "Min-Int-Merge-del_mu"), 'meth_description':
            ["Minimum intensity merge for synchronic map: using del mu"] * 2,
                           'var_name': ("mu_cutoff", "del_mu"), 'var_description': ("lower mu cutoff value",
                                                                                    "max acceptable mu range"),
                           'var_val': (mu_cutoff, del_mu)}
    else:
        euv_combined, chd_combined = combine_maps(euv_maps, chd_maps, mu_merge_cutoff=mu_merge_cutoff, mu_cutoff=mu_cutoff)
        combined_method = {'meth_name': ("Min-Int-Merge-mu_merge", "Min-Int-Merge-mu_merge"), 'meth_description':
            ["Minimum intensity merge for synchronic map: based on Caplan et. al."] * 2,
                           'var_name': ("mu_cutoff", "mu_merge_cutoff"), 'var_description': ("lower mu cutoff value",
                                                                                         "mu cutoff value in areas of "
                                                                                         "overlap"),
                           'var_val': (mu_cutoff, mu_merge_cutoff)}

    # generate a record of the method and variable values used for interpolation
    euv_combined.append_method_info(methods_list)
    euv_combined.append_method_info(pd.DataFrame(data=combined_method))
    euv_combined.append_data_info(data_info)
    euv_combined.append_map_info(map_info)
    chd_combined.append_method_info(methods_list)
    chd_combined.append_method_info(pd.DataFrame(data=combined_method))
    chd_combined.append_data_info(data_info)
    chd_combined.append_map_info(map_info)

    # plot maps
    Plotting.PlotMap(euv_combined, nfig="EUV Combined Map for: " + str(euv_combined.data_info.date_obs[0]),
                     title="Minimum Intensity Merge EUV Map\nDate: " + str(euv_combined.data_info.date_obs[0]),
                     map_type='EUV')
    Plotting.PlotMap(chd_combined, nfig="CHD Combined Map for: " + str(chd_combined.data_info.date_obs[0]),
                     title="Minimum Intensity Merge CHD Map\nDate: " + str(chd_combined.data_info.date_obs[0]),
                     map_type='CHD')
    #     Plotting.PlotMap(chd_combined, nfig="CHD Contour Map for: " + str(chd_combined.data_info.date_obs[0]),
    #                      title="Minimum Intensity Merge CHD Contour Map\nDate: " + str(chd_combined.data_info.date_obs[0]),
    #                      map_type='Contour')

    # save EUV and CHD maps to database
    # euv_combined.write_to_file(map_data_dir, map_type='synchronic_euv', filename=None, db_session=db_session)
    # chd_combined.write_to_file(map_data_dir, map_type='synchronic_chd', filename=None, db_session=db_session)

    # end time
    end = time.time()
    print("Combined EUV and CHD Maps created and saved to the database in", end - start, "seconds.")
    return euv_combined, chd_combined


def create_combined_maps_2(map_list, mu_merge_cutoff=None, del_mu=None, mu_cutoff=0.0,
                           EUV_CHD_sep=False, low_int_filt=1.):
    """
    Function to create combined EUV and CHD maps.

    New in version 2: - function does not try to save to the database.
                      - assumes that EUV and CHD are in the same map object
                      - do some filtering on low-intensity pixels (some images are
                        cropped or do not include the sun. We do not want these to
                        dominate the minimum intensity merge.
    @param map_list: list of EUV maps
    @param mu_merge_cutoff: float
                            cutoff mu value for overlap areas
    @param del_mu: float
                   maximum mu threshold value
    @param mu_cutoff: float
                      lower mu value
    @param low_int_filt: float
                         low intensity filter. pixels with image intensity less than
                         1.0 are generally bad data. Data < low_int_filt are recast
                         as no_data_vals prior to minimum intensity merge
    @return: PsiMap object
             Minimum Intensity Merge (MIM) euv map, MIM chd map, and merged
             map meta data.
    """
    # start time
    start = time.time()
    n_maps = len(map_list)
    # remove excessively low image data (these are almost always some form of
    # bad data and we don't want them to 'win' the minimum intensity merge)
    for ii in range(n_maps):
        map_list[ii].data[map_list[ii].data < low_int_filt] = map_list[ii].no_data_val
    if EUV_CHD_sep:
        # create chd dummy-maps (for separate minimum-intensity merge)
        chd_maps = []
        for ii in range(n_maps):
            chd_maps.append(map_list[ii].__copy__())
            chd_maps[ii].data = chd_maps[ii].chd.astype(DTypes.MAP_DATA)
            chd_maps[ii].data[chd_maps[ii].data <= chd_maps[ii].no_data_val] = \
                chd_maps[ii].no_data_val

        if del_mu is not None:
            euv_combined = combine_maps(map_list, del_mu=del_mu, mu_cutoff=mu_cutoff)
            chd_combined = combine_maps(chd_maps, del_mu=del_mu, mu_cutoff=mu_cutoff)
            combined_method = {'meth_name': ["Min-Int-Merge-del_mu"] * 3, 'meth_description':
                ["Minimum intensity merge for synchronic map: using del mu"] * 3,
                 'var_name': ("mu_cutoff", "del_mu", "mu_merge_cutoff"),
                 'var_description': ("lower mu cutoff value", "max acceptable mu range",
                                     "mu cutoff value in areas of overlap"),
                 'var_val': (mu_cutoff, del_mu, None)}
        else:
            euv_combined = combine_maps(map_list, mu_merge_cutoff=mu_merge_cutoff,
                                        mu_cutoff=mu_cutoff)
            chd_combined = combine_maps(chd_maps, mu_merge_cutoff=mu_merge_cutoff,
                                        mu_cutoff=mu_cutoff)
            combined_method = {'meth_name': ["Min-Int-Merge-mu_merge"] * 3, 'meth_description':
                ["Minimum intensity merge for synchronic map: based on Caplan et. al."] * 3,
                 'var_name': ("mu_cutoff", "del_mu", "mu_merge_cutoff"),
                 'var_description': ("lower mu cutoff value", "max acceptable mu range",
                                     "mu cutoff value in areas of overlap"),
                               'var_val': (mu_cutoff, None, mu_merge_cutoff)}

        euv_combined.chd = chd_combined.data.astype(DTypes.MAP_CHD)
    else:
        # Use indexing from EUV min-merge to select CHD merge values
        if del_mu is not None:
            euv_combined = combine_maps(map_list, del_mu=del_mu, mu_cutoff=mu_cutoff)
            combined_method = {'meth_name': ["MIDM-Comb-del_mu"]*3, 'meth_description':
                ["Minimum intensity merge for (combined EUV/CHD) synchronic map: using del mu"]*3,
                               'var_name': ("mu_cutoff", "del_mu", "mu_merge_cutoff"),
                               'var_description': ("lower mu cutoff value", "max acceptable mu range",
                                                   "mu cutoff value in areas of overlap"),
                               'var_val': (mu_cutoff, del_mu, None)}
        else:
            euv_combined = combine_maps(map_list, mu_merge_cutoff=mu_merge_cutoff,
                                        mu_cutoff=mu_cutoff)
            combined_method = {'meth_name': ["MIDM-Comb-mu_merge"]*3, 'meth_description':
                ["Minimum intensity merge for (combined EUV/CHD) synchronic map: based on Caplan et. al."]*3,
                               'var_name': ("mu_cutoff", "del_mu", "mu_merge_cutoff"),
                               'var_description': ("lower mu cutoff value", "max acceptable mu range",
                                                   "mu cutoff value in areas of overlap"),
                               'var_val': (mu_cutoff, None, mu_merge_cutoff)}

    # merge map meta data
    for ii in range(len(map_list)):
        euv_combined.append_method_info(map_list[ii].method_info)
        euv_combined.append_data_info(map_list[ii].data_info)
        # euv_combined.append_map_info(map_list[ii].map_info)
    euv_combined.append_method_info(pd.DataFrame(data=combined_method))
    # combining maps produces duplicate methods. drop duplicates
    euv_combined.method_info = euv_combined.method_info.drop_duplicates()
    # for completeness, also drop duplicate data/image info
    euv_combined.data_info = euv_combined.data_info.drop_duplicates()
    # record compute time
    euv_combined.append_map_info(pd.DataFrame({'time_of_compute':
                                               [datetime.datetime.now(), ]}))

    # end time
    end = time.time()
    print("Minimum Intensity EUV and CHD Maps created in", end - start, "seconds.")
    return euv_combined