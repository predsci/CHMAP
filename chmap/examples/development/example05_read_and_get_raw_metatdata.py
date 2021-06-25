"""
Quick example illustrating how to get the metadata that we'll need from the raw fits files
- NOTE I believe the pandas dataframe object will be the preferred way to format query results
  from the SQL database (built in slicing and numpy compatibility)
  - so, while its not necessary for this example, we should think about how to convert
    our metadat and query results to and from SQL and pandas
  - the pandas website is here: https://pandas.pydata.org/
  - this is an SQLAlchemy example with pandas used to look at records:
    https://towardsdatascience.com/sqlalchemy-python-tutorial-79a577141a91
  - perhaps there is a better way to *formalize* the data types of the metadata as well...
"""
import os

from chmap.settings.app import App
from chmap.utilities.file_io.io_helpers import carrington_rotation_number_relative

import pandas as pd

import sunpy.map

# Use the reference images for this example
fits_infile_aia = os.path.join(App.APP_HOME, 'reference_data', 'aia_lev1_euv_12s_20140413T190507_193.fits')
fits_infile_a = os.path.join(App.APP_HOME, 'reference_data', 'sta_euvi_20140413T190530_195.fits')
fits_infile_b = os.path.join(App.APP_HOME, 'reference_data', 'stb_euvi_20140413T190609_195.fits')

# Load images image using the built-in methods of SunPy
# This creates a sunpy "map" structure which is a nice way to handle metadata
map_aia = sunpy.map.Map(fits_infile_aia)  # AIA is the EUV telescope on SDO
map_euvi = sunpy.map.Map(fits_infile_a)  # EUVI is the EUV telescope on the twin STEREO spacecraft
map_b = sunpy.map.Map(fits_infile_b)  # get the other stereo image too

# Inspect the maps by printing their representation.
# Note that Sunpy's Generic Map factory understands these specific data sources
# and they become AIAMap and EUVIMap data structures respectively
print('')
print('### SunPy Map for AIA')
print(repr(map_aia))
print('')
print('### SunPy Map for EUVI')
print(repr(map_euvi))


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

    metadata['date'] = time_string
    metadata['jd'] = time_float
    metadata['wavelength'] = wavelength
    metadata['instrument'] = instrument
    metadata['distance'] = d_km
    metadata['cr_lon'] = cr_lon
    metadata['cr_lat'] = cr_lat
    metadata['cr_rot'] = cr_rot

    return metadata


# get the metadata for each image
meta_aia = get_metadata(map_aia)
meta_euvi = get_metadata(map_euvi)
meta_b = get_metadata(map_b)

# now turn these metadata dicts into a pandas dataframe for easy inspection
df = pd.DataFrame([meta_aia, meta_euvi, meta_b])

# print the full dataframe
print('')
print('### Collected metadata')
print(df.to_string())

# print a slice of the dataframe, returnable as a Series or numpy array
print('')
print('### A slice of the metadata [CR_LON]')
slice = df['cr_lon']
print(slice)
print(type(slice))
values = slice.values
print(values)
print(type(values))
print(values.dtype)
