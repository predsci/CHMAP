import numpy as np
import pandas as pd

from chmap.data.download.euv_utils import cluster_meth_1


def select_synchronic_images(center_time, del_interval, image_pd, inst_list):
    """
    Select PSI-database images for synchronic map generation, consistent with the way we select
    images for download.

    Parameters
    ----------
    center_time - numpy.datetime64
    del_interval - numpy.timedelta64
    image_pd - pandas.DataFrame
    inst_list - list

    Returns
    -------
    synch_images pandas.DataFrame

    """
    # define a method to attach to synchronic maps
    map_method_dict = {'meth_name': ("Synch_Im_Sel",), 'meth_description': [
        "Synchronic image selection", ],
                       'var_name': ("clust_meth",), 'var_description': ("Clustering method",),
                       'var_val': (1,)}
    map_method = pd.DataFrame(data=map_method_dict)

    jd0 = pd.DatetimeIndex([center_time, ]).to_julian_date().item()
    # choose which images to use at this datetime
    interval_max = center_time + del_interval
    interval_min = center_time - del_interval
    f_list = []
    image_list = []
    for instrument in inst_list:
        # find instrument images in interval
        inst_images_index = image_pd.date_obs.between(interval_min, interval_max) & \
                            image_pd.instrument.eq(instrument)
        inst_images = image_pd[inst_images_index]
        if inst_images.__len__() > 0:
            f_list_pd = pd.DataFrame({'date_obs': inst_images.date_obs,
                                      'jd': pd.DatetimeIndex(inst_images.date_obs).to_julian_date(),
                                      'instrument': inst_images.instrument})
            f_list.append(f_list_pd)
            image_list.append(inst_images)

    if f_list.__len__() == 0:
        print("No instrument images in time range around ", center_time, ".\n")
        # return None
        return None, map_method

    # Now loop over all the image pairs to select the "perfect" group of images.
    cluster_index = cluster_meth_1(f_list=f_list, jd0=jd0)
    # combine selected image-rows into a dataframe
    synch_images = image_list[0].iloc[cluster_index[0]]
    if cluster_index.__len__() > 1:
        for ii in range(1, cluster_index.__len__()):
            synch_images = synch_images.append(image_list[ii].iloc[cluster_index[ii]])

    return synch_images, map_method


def get_dates(time_min, time_max, map_freq=2):
    """
    function to create moving average dates based on hourly frequency of map creation
    @param time_min: minimum datetime value for querying
    @param time_max: maximum datetime value for querying
    @param map_freq: integer value representing hourly cadence for map creation
    @return: list of center dates
    """
    map_frequency = int((time_max - time_min).total_seconds() / 3600 / map_freq)
    moving_avg_centers = np.array(
        [np.datetime64(str(time_min)) + ii * np.timedelta64(map_freq, 'h') for ii in range(map_frequency + 1)])
    return moving_avg_centers