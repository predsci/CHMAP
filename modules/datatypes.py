"""
Module to hold the custom image and map types specific to the CHD project.
"""
import numpy as np
import pandas as pd
from helpers import psihdf

import sunpy.map
import sunpy.util.metadata

import helpers.psihdf as psihdf
import modules.DB_classes as db
from modules.coord_manip import interp_los_image_to_map


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

    def interp_to_map(self, R0=1.0, map_x=None, map_y=None, no_data_val=-9999., image_num=None):

        print("\nConverting " + self.info['instrument'] + "-" + str(self.info['wavelength']) + " image from " +
              self.info['date_string'] + " to a map.")

        if map_x is None and map_y is None:
            # Generate map grid based on number of image pixels vertically within R0
            # map parameters (assumed)
            y_range = [-1, 1]
            x_range = [0, 2*np.pi]

            # observer parameters (from image)
            cr_lat = self.info['cr_lat']
            cr_lon = self.info['cr_lon']

            # determine number of pixels in map y-grid
            map_nycoord = sum(abs(self.y) < R0)
            del_y = (y_range[1] - y_range[0])/(map_nycoord - 1)
            # how to define pixels? square in sin-lat v phi or lat v phi?
            # del_x = del_y*np.pi/2
            del_x = del_y
            map_nxcoord = (np.floor((x_range[1] - x_range[0])/del_x) + 1).astype(int)

            # generate map x,y grids. y grid centered on equator, x referenced from lon=0
            map_y = np.linspace(y_range[0], y_range[1], map_nycoord.astype(int), dtype='<f4')
            map_x = np.linspace(x_range[0], x_range[1], map_nxcoord.astype(int), dtype='<f4')

        # Do interpolation
        interp_result = interp_los_image_to_map(self, R0, map_x, map_y, no_data_val=no_data_val)

        # Partially populate a map object with grid and data info
        map_out = PsiMap()
        map_out.data = interp_result.data
        map_out.x = interp_result.x
        map_out.y = interp_result.y
        map_out.mu = interp_result.mu_mat
        map_out.no_data_val = no_data_val
        map_out.origin_image = np.full(interp_result.data.shape, no_data_val, dtype=int)
        # if image number is entered, record in appropriate pixels
        if image_num is not None:
            map_out.origin_image[map_out.data > no_data_val] = image_num

        return map_out



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


class PsiMap:
    """
    Object that contains map information.  The Map object is structured like the database for convenience.
        - One of its primary uses is for passing info to and from database.
    """

    def __init__(self):
        """
        Initialize empty dataframes based on Table schema
        """
        # create the data tags (all pandas dataframes?)
        self.method_info = init_df_from_declarative_base(db.Meth_Defs)
        self.image_info = init_df_from_declarative_base(db.EUV_Images)
        # map_info will be a combination of Image_Combos and EUV_Maps
        image_columns = []
        for column in db.Image_Combos.__table__.columns:
            image_columns.append(column.key)
        map_columns = []
        for column in db.EUV_Maps.__table__.columns:
            map_columns.append(column.key)
        df_cols = set().union(image_columns, map_columns)
        self.map_info = pd.DataFrame(data=None, columns=df_cols)
        # var_info is a combination of Var_Vals and Var_Defs
        val_columns = []
        for column in db.Var_Vals.__table__.columns:
            val_columns.append(column.key)
        defs_columns = []
        for column in db.Var_Defs.__table__.columns:
            defs_columns.append(column.key)
        df_cols = set().union(image_columns, map_columns)
        self.var_info = pd.DataFrame(data=None, columns=df_cols)
        # These are placeholders for map grids and data
        self.data = None
        self.x    = None
        self.y    = None
        self.mu   = None
        self.no_data_val = None
        self.origin_image = None

    def append_map_info(self, map_df):
        """
        add a record to the map_info dataframe
        """
        self.map_info = self.map_info.append(map_df, sort=False)

    def append_method_info(self, method_df):
        self.method_info = self.method_info.append(method_df, sort=False)

    def append_var_info(self, var_info_df):
        self.var_info = self.var_info.append(var_info_df, sort=False)

    def append_image_info(self, image_df):
        self.image_info = self.image_info.append(image_df, sort=False)


def init_df_from_declarative_base(base_object):
    """
    Takes in an SQLAlchemy declarative_base() table definition object.  Returns an empty pandas
    DataFrame with the same column names.
    :param base_object: SQLAlchemy declarative_base() table definition object
    :return: empty pandas DataFrame with same column names as the table
    """

    column_names = []
    for table_column in base_object.__table__.columns:
        column_names.append(table_column.key)

    out_df = pd.DataFrame(data=None, columns=column_names)

    return out_df


class InterpResult:
    """
    Class to hold the standard output from interpolating an image to
        a map.  Allows for the optional attribute mu.

    """

    def __init__(self, data, x, y, mu_mat=None):

        # create the data tags
        self.data = data
        self.x = x
        self.y = y
        self.mu_mat = mu_mat
