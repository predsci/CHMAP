"""
Module to hold the custom image and map types specific to the CHD project.
"""
import numpy as np
from helpers import psihdf

import sunpy.map
import sunpy.util.metadata
import helpers.psihdf as psihdf


class LosImage:
    """
    Class to hold the standard information for a Line-of-sight (LOS) image
    for the CHD package.

    - Here the image is minimally described by:
      data: a 2D numpy array of values.
      x: a 1D array of the solar x positions of each pixel [Rs].
      y: a 1D array of the solar y positions of each pixel [Rs].
      chd_meta: a dictionary of the metadata used by the CHD database

    - sunpy_meta: a dictionary of sunpy metadata (optional)
      if this is specified then it to create a sunpy Map tag: map
      - This preserves compatibility with Sunpy and is useful for
        visualizing or interacting with the data.
    """

    def __init__(self, data, x, y, chd_meta, sunpy_meta=None):

        # create the data tags
        self.data = data
        self.x = x
        self.y = y

        # add the info dictionary
        self.info = chd_meta

        # if sunpy_meta is supplied try to create a sunpy map
        if sunpy_meta != None:
            self.add_map(sunpy_meta)

    def get_coordinates(self):
        """
        Calculate relevant mapping information for each pixel.
        This adds 2D arrays to the class:
        - lat: carrington latitude
        - lon: carrington longitude
        - mu: cosine of the center to limb angle
        ToDo: Fill this in with actual methods.
        """
        pass

    def add_map(self, sunpy_meta):
        """
        Add a sunpy map to the class.
        - here the metadata is supplied, and whatever is in self.data is used.
        """
        if hasattr(self, 'map'):
            delattr(self, 'map')
        self.map = sunpy.map.Map(self.data, sunpy_meta)

    def write_to_file(self, filename):
        """
        Write the los image to hdf5 format
        """
        # check to see if the map exists from instantiation
        if hasattr(self, 'map'):
            sunpy_meta = self.map.meta

        psihdf.wrh5_meta(filename, self.x, self.y, np.array([]),
                         self.data, chd_meta=self.info, sunpy_meta=sunpy_meta)


def read_los_image(h5_file):
    """
    Method for reading our custom hdf5 format for prepped EUV images.
    input: path to a prepped .h5 file.
    output: an LosImage structure.

    - with_map tells it to create the sunpy map structure too (default True).
      - set this to False if you need faster reads

    ToDo: add error checking
    """
    # read the image and metadata
    x, y, z, data, chd_meta, sunpy_meta = psihdf.rdh5_meta(h5_file)

    # create the structure
    los = LosImage(data, x, y, chd_meta, sunpy_meta=sunpy_meta)

    return los
