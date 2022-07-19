"""
Script to load in individual AIA 193 FITS files specified by the COSPAR
ISWAT team and perform our LBC transformation and EZseg detection directly
on the file.

** RUN THIS SCRIPT USING THE CHD INTERPRETER IN PYCHARM!
"""
import numpy as np
import json

import scipy.interpolate
import astropy.time

# ---------------------------------------------------------------------
# Functions For Computing a time-dependent correction
# ---------------------------------------------------------------------
def process_aia_timedepend_json(json_file):
    """
    Read the raw JSON file of my time-depend struct that I generated with IDL.
    Convert it to the proper data types
    """
    with open(json_file, 'r') as json_data:
         json_dict = json.load(json_data)

    timedepend_dict = {}

    # get the time-dependent factors as a dict with 1D arrays indexed
    # by the integer wavelength specifier of the filter (converting from 2D array in the JSON)
    factor_dict = {}
    f2d = np.array(json_dict['FACTOR'])
    for i, wave in enumerate(json_dict['WAVES']):
        factor_dict[wave] = f2d[:,i]
    timedepend_dict['factor'] = factor_dict

    # get the dates as strings
    timedepend_dict['dates'] = np.array(json_dict['DATES'], dtype=str)

    # get the script that made this file and version
    timedepend_dict['version'] = json_dict['VERSION']
    timedepend_dict['idl_script'] = json_dict['SCRIPTNAME']

    # get the times as an array of astropy.Time objects for interpolation
    timedepend_dict['times'] = astropy.time.Time(timedepend_dict['dates'])

    return timedepend_dict

def get_aia_timedepend_factor(timedepend_dict, datetime, wave):
    """
    Get the time-dependent scaling factor for an AIA filter for
    a given time and filter specifier. The idea is to account
    for degridation of the detector/counts in time.

    Parameters
    ----------
    timedepend_dict: special dictionary returned by process_aia_timedepend_json
    datetime: a datetime object for a given time of interest.
    wave: an integer specifying the AIA filter (e.g. 193).

    Returns
    -------
    factor: The scaling factor from 0 to 1. (1 is perfect, 0 is degraded).
    """
    # convert to the astropy Time object
    time = astropy.time.Time(datetime)

    # get the values for interpolation
    x = timedepend_dict['times'].mjd
    y = timedepend_dict['factor'][wave]

    # get the interpolator
    interpolator = scipy.interpolate.interp1d(x, y)

    factor = interpolator(time.mjd)

    # now take the max because this gives an unshaped array...
    factor = np.max(factor)

    return factor


# ---------------------------------------------------------------------
# Script Starts here
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # JSON file with the AIA time-dependent corrections
    aia_timedepend_file = 'SSW_AIA_timedepend_v10.json'

    # read the time-dependent json file, turn it into a dictionary
    timedepend_dict = process_aia_timedepend_json(aia_timedepend_file)

    # print the keys of this dict
    print(f'\n### Keys in the AIA timedependent correction dictionary: ')
    for key in timedepend_dict.keys():
        print(f'  key: {key:16s}   type: {type(timedepend_dict[key])}')

    # now sample it at a few times using our custom function for interpolation (get_aia_timedepend_factor)
    dates = ['2014-04-13T02:00:05.435Z', '2019-04-13T02:00:05.435Z']

    for date in dates:
        # astropy.time is a million times better than python's datetime for defining a time
        time_now = astropy.time.Time(date)

        print(f'\n### Factors for {str(time_now)}')
        for wave in [94,131,171,193,211,335]:
            # note time input to get_aia_timedepend_factor is a datetime for compatibility w/ our database/pandas
            factor = get_aia_timedepend_factor(timedepend_dict, time_now.datetime, wave)
            print(f'  wavelength: {wave:3d}, factor: {factor:7.5f}')


