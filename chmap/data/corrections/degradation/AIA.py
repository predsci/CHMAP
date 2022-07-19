
import pkg_resources
import json
import numpy as np
import pandas as pd
import astropy.time
import scipy.interpolate


def load_aia_json():
    # use 'pkg_resources' to determine the path for chmap locally and open the file for reading
    stream = pkg_resources.resource_stream('chmap', "data/corrections/degradation/data/SSW_AIA_timedepend_v10.json")
    # interpret as a json
    json_dict = json.load(stream)
    return json_dict

def process_aia_timedepend_json(json_dict):
    """
    Import the JSON (loaded with load_aia_json()) of my time-depend struct that I generated with IDL.
    Convert it to the proper data types
    """

    timedepend_dict = {}

    # get the time-dependent factors as a dict with 1D arrays indexed
    # by the integer wavelength specifier of the filter (converting from 2D array in the JSON)
    factor_dict = {}
    f2d = np.array(json_dict['FACTOR'])
    for i, wave in enumerate(json_dict['WAVES']):
        factor_dict[wave] = f2d[:, i]
    timedepend_dict['factor'] = factor_dict

    # get the dates as strings
    timedepend_dict['dates'] = np.array(json_dict['DATES'], dtype=str)

    # get the script that made this file and version
    timedepend_dict['version'] = json_dict['VERSION']
    timedepend_dict['idl_script'] = json_dict['SCRIPTNAME']

    # get the times as an array of astropy.Time objects for interpolation
    timedepend_dict['times'] = astropy.time.Time(timedepend_dict['dates'])

    return timedepend_dict


def get_aia_timedepend_factor(timedepend_dict, eval_dates, wave):
    """
    Get the time-dependent scaling factor for an AIA filter for
    a given time and filter specifier. The idea is to account
    for degradation of the detector/counts in time.

    Parameters
    ----------
    timedepend_dict: special dictionary returned by process_aia_timedepend_json
    eval_dates: a datetime object for a given time of interest.
    wave: an integer specifying the AIA filter (e.g. 193).

    Returns
    -------
    factor: The scaling factor from 0 to 1. (1 is perfect, 0 is degraded).
    """

    if pd.api.types.is_list_like(eval_dates):
        is_vector = True
        # initialize output array
        factor = np.full([len(eval_dates), ], fill_value=1.0)
    else:
        is_vector = False
        factor = 1.0

    # convert to the astropy Time object
    time = astropy.time.Time(eval_dates)
    time_mjd = time.mjd

    # get the values for interpolation
    x = timedepend_dict['times'].mjd
    y = timedepend_dict['factor'][wave]

    if is_vector:
        before_index = time_mjd < x[0]
        after_index = time_mjd > x[-1]
        factor[after_index] = y[-1]

        interp_index = ~(before_index | after_index)
        if any(interp_index):
            # get the interpolator
            interpolator = scipy.interpolate.interp1d(x, y)
            # interpolate
            interp_result = interpolator(time_mjd[interp_index])
            # now index back into output array
            factor[interp_index] = interp_result
    else:
        if time_mjd < x[0]:
            factor = 1.0
        elif time_mjd > x[-1]:
            factor = y[-1]
        else:
            # get the interpolator
            interpolator = scipy.interpolate.interp1d(x, y)
            # interpolate
            interp_result = interpolator(time_mjd)
            # convert back to scalar
            factor = interp_result.item()
    return factor
