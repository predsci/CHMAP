"""
Create a function for each clustering routine/algorithm.  This
will make it easier to differentiate how images were grouped
to create the combined output image.
"""

import json
from itertools import combinations

import numpy as np
import pandas as pd
import datetime
import sunpy
import astropy.units as u
import sunpy.util.metadata

from data.download import drms_helpers, vso_helpers
from helpers.misc_helpers import carrington_rotation_number_relative
from sunpy.time import TimeRange


def get_image_set(df, time0):
    """
    Find the best set of images near a given time.

    Inputs:
     df: a pandas DataFrame returned from an image query.
       - The dataframe can be built by provider or database queries
     time: an astropy Time object. This is the time used in the search.

    Output: a pandas dataframe with one "best match" image for each spacecraft.
    """
    # convert the big dataframe to a list of dataframes with unique image types
    df_list = df_to_df_list(df)

    # now get the indexes of the best matching images in the set
    locs = cluster_meth_1(df_list, time0.jd)

    # build a list of the matching records (series)
    ds_list = []
    i = 0
    for subdf in df_list:
        ds_list.append(subdf.iloc[locs[i][0]])
        i = i + 1

    # turn that list into a new dataframe
    if len(ds_list) > 0:
        df_matched = pd.concat(ds_list, axis=1).T
    else:
        df_matched = df

    return df_matched


def df_to_df_list(df):
    """
    Helper function to sort through a pandas dataframe and return a list of dataframes
    where each item is a dataframe with a unique spacecraft/instrument/wavelength combination
    """
    # first build a string series that combines the unique columns
    #  - this is different whether the dataframe is from a provider query or a database query
    #  - check by looking for tags that exist in one but not the other.
    if 'wavelength' in df:
        combo = df['instrument'] + '_' + df['wavelength'].astype(str)
    elif 'filter' in df:
        combo = df['spacecraft'] + '_' + df['instrument'] + '_' + df['filter'].astype(str)

    # iterate over unique combos, add that subset to the list
    df_list = []
    for unique_combo in pd.unique(combo):
        df_list.append(df.iloc[np.where(combo == unique_combo)])

    return df_list


def cluster_meth_1(f_list, jd0):
    """
    Input a list of instruments data 'results' and the central
    time 'jd0'.  Find the cluster of instrument images that is
    the best combination of tightly grouped and close to jd0.
    :return: the indices of the selected images
    """

    # extract 'results' list from f_list
    results = [None]*len(f_list)
    for ii in range(len(f_list)):
        results[ii] = f_list[ii].jd.values

    sizes = []
    for result in results:
        sizes.append(len(result))
    time_delta = np.ndarray(tuple(sizes), dtype='float64')

    # Now loop over all the image pairs to select the "perfect" group of images.
    # Here we ,minimize an arbitrary weight function based on the time differences between the
    # group (relative) and the requested central time (absolute).
    # This loop needs to be rewritten to be agnostic to the spacecraft type and number of spacecraft.
    pow1 = 2
    pow2 = 2

    inst_dim = time_delta.ndim
    temp_v = np.zeros(inst_dim)
    for idx, vals in np.ndenumerate(time_delta):
        for ii in range(inst_dim):
            temp_v[ii] = results[ii][idx[ii]]
        # vi = results[0][idx[0]]
        # vj = results[1][idx[1]]
        # vk = results[2][idx[2]]

        # generate all possible instrument pairs
        comb = combinations(range(inst_dim), 2)
        metric_sum = 0
        for comb_iter in list(comb):
            metric_sum = metric_sum + abs(temp_v[comb_iter[0]] - temp_v[comb_iter[1]])**pow1
        time_delta[idx] = metric_sum

        # this looks at the relative difference between the three pairs of images
        # time_delta[idx] = abs(vi - vj)**pow1 + abs(vi - vk)**pow1 + abs(vj - vk)**pow1

        # this looks at the difference between a given image and the central time of the interval (jd0)
        # time_delta[idx] = time_delta[idx] + abs(vi - jd0)**pow2 + abs(vj - jd0)**pow2 + abs(vk - jd0)**pow2
        for ii in range(inst_dim):
            time_delta[idx] = time_delta[idx] + abs(temp_v[ii] - jd0)**pow2

    # Figure out the index
    imins = np.where(time_delta == np.min(time_delta))

    return imins


# define a quick function that will print the metadata
def get_metadata(map):
    """
    This function gets the metadata we need and then creates a dictionary at the end.
    - If we want to enforce specific types for each tag we "may" want to define a class
      that defines the specific metadata tags and corresponding types a priori and then
      this class is instantiatied and then populated in a subroutine like this.
    - however, this would have to be compatible with how the record type is defined in
      SQL and might be somewhat of a pain? i'm not sure what the best solution is
    """
    # Observation time is saved as a Time object
    time_object = map.date
    # For SQL, we want this in the Python native 'datetime' format
    time_datetime = map.date.datetime

    # Get the time as a string
    time_string = map.date.isot

    # get the time as a floating point julian date
    time_float = time_object.jd

    # get the wavelength as an integer (map.wavelength is an astropy quantity)
    # here I am converting the astropy distance quantity to angstrom and then a float to be sure
    wavelength = int(map.wavelength.to("angstrom").value)

    # make a string that gives a unique observatory/instrument combo [remove whitespace]
    # o_str = map.observatory.replace(" ","")
    # d_str = map.detector.replace(" ","")
    # instrument = o_str+'_'+d_str

    # or just use the sunpy nickname, which is also unique (i think i like this more...)
    instrument = map.nickname

    # get the distance of the observer (in km) from "observer_coordinate" (a SkyCoord object)
    # here I am converting the astropy distance quantity to km and then a float
    d_km = map.observer_coordinate.radius.to("km").value

    # get the carringtion longitude and latitude in degrees
    cr_lon = map.carrington_longitude.to("degree").value
    cr_lat = map.carrington_latitude.to("degree").value

    # get the decimal carrington rotation number (for this central longitude, not earth).
    cr_rot = carrington_rotation_number_relative(time_object, cr_lon)

    # now build a dictionary with the information
    # the idea here is to formalize the metadata components as a dictionary, which can
    # be used to create nice sliceable dataframes later with pandas
    metadata = dict()

    metadata['date_string'] = time_string
    metadata['datetime'] = time_datetime
    metadata['jd'] = time_float
    metadata['wavelength'] = wavelength
    metadata['instrument'] = instrument
    metadata['distance'] = d_km
    metadata['cr_lon'] = cr_lon
    metadata['cr_lat'] = cr_lat
    metadata['cr_rot'] = cr_rot

    return metadata


# a function to look for all available images in a time-window
def list_available_images(time_start, time_end, euvi_interval_cadence=2 *u.hour, aia_search_cadence=12 * u.second,
                          wave_aia=193, wave_euvi=195):
    """
    This function takes in a time range and returns all available images for all available
    instruments.  Time ranges entered as 'astropy.time.core.Time' objects.
    Optional inputs control the search cadence for EUVI 'euvi_interval_cadence', AIA
    search cadence 'aia_search_cadence', EUVI wavelength 'wave_euvi', and AIA wavelength
    'wave_aia'.
    :return: A list of instruments, one entry per available instrument.  Each list entry
    is a pandas dataframe describing the available images.
    """

    # query parameters
    # interval_cadence = 2*u.hour
    # aia_search_cadence = 12*u.second
    # wave_aia = 193
    # wave_euvi = 195

    # generate the list of time intervals
    full_range = TimeRange(time_start, time_end)
    # time_ranges = full_range.window(euvi_interval_cadence, euvi_interval_cadence)

    # initialize the jsoc drms helper for aia.lev1_euv_12
    s12 = drms_helpers.S12(verbose=True)

    # initialize the helper class for EUVI
    euvi = vso_helpers.EUVI(verbose=True)

    # pick a time_range to experiment with
    # time_range = time_ranges[0]
    time_range = full_range

    # ---- Query each instrument individually -------------
    # If a new instrument comes on-line, add its query routine here. Then add
    # its result to f_list. Order does not matter so long as the new routine's
    # output f* is in the same pandas dataframe format at fs, fa, and fb.
    # query the jsoc for SDO/AIA
    fs = s12.query_time_interval(time_range, wave_aia, aia_search_cadence)

    # query the VSO for STA/EUVI and STB/EUVI
    fa = euvi.query_time_interval(time_range, wave_euvi, craft='STEREO_A')
    fb = euvi.query_time_interval(time_range, wave_euvi, craft='STEREO_B')

    # combine dataframes into a simple list
    f_list = [fs, fa, fb]
    # # check for empty instrument results (and remove)
    # f_list = [elem for elem in f_list if len(elem) > 0]

    return f_list


def write_meta_as_json(file, metadata):
    """
    Write a sunpy metadata object (dict used in map classes) to a json file
    """
    json.dump(metadata, open(file, 'w'))


def read_meta_from_json(file):
    """
    Read sunpy metadata from a json and return it as a sunpy metadata object.
    """
    dict = json.load(open(file, 'r'))
    return sunpy.util.metadata.MetaDict(dict)


def roundSeconds(dateTimeObject):
    newDateTime = dateTimeObject

    if newDateTime.microsecond >= 500000:
        newDateTime = newDateTime + datetime.timedelta(seconds=1)

    return newDateTime.replace(microsecond=0)
