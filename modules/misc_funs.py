"""
Create a function for each clustering routine/algorithm.  This
will make it easier to differentiate how images were grouped
to create the combined output image.
"""

import numpy as np
from itertools import combinations

from helpers.misc_helpers import carrington_rotation_number_relative



def cluster_meth_1(results, jd0):
    """
    Input a list of instruments data 'results' and the central
    time 'jd0'.  Find the cluster of instrument images that is
    the best combination of tightly grouped and close to jd0.
    :return: the indices of the selected images
    """

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


