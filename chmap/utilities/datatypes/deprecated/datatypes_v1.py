"""
Module to hold the custom image and map types specific to the CHD project.
"""
import numpy as np
import pandas as pd
import sys
import datetime

import sunpy.map
import sunpy.util.metadata

import chmap.utilities.file_io.psi_hdf as psihdf
import chmap.database.db_classes as db
from chmap.utilities.coord_manip import interp_los_image_to_map, image_grid_to_CR
from chmap.settings.info import DTypes

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
        self.data = data.astype(DTypes.LOS_DATA)
        self.x = x.astype(DTypes.LOS_AXES)
        self.y = y.astype(DTypes.LOS_AXES)

        # add the info dictionary
        self.info = chd_meta

        # if sunpy_meta is supplied try to create a sunpy map
        if sunpy_meta != None:
            self.add_map(sunpy_meta)

        # add placeholders for carrington coords and mu
        self.lat = None
        self.lon = None
        self.mu  = None
        self.no_data_val = None


    def get_coordinates(self, R0=1.0, outside_map_val=-9999.):
        """
        Calculate relevant mapping information for each pixel.
        This adds 2D arrays to the class:
        - lat: carrington latitude
        - lon: carrington longitude
        - mu: cosine of the center to limb angle
        """

        x_mat, y_mat = np.meshgrid(self.x, self.y)
        x_vec = x_mat.flatten(order="C")
        y_vec = y_mat.flatten(order="C")

        cr_theta_all, cr_phi_all, image_mu = image_grid_to_CR(x_vec, y_vec, R0=R0, obsv_lat=self.info['cr_lat'],
                                obsv_lon=self.info['cr_lon'], get_mu=True, outside_map_val=outside_map_val)

        cr_theta = cr_theta_all.reshape(self.data.shape, order="C")
        cr_phi = cr_phi_all.reshape(self.data.shape, order="C")
        image_mu = image_mu.reshape(self.data.shape, order="C")

        self.lat = cr_theta - np.pi/2.
        self.lon = cr_phi
        self.mu  = image_mu
        self.no_data_val = outside_map_val


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
            map_y = np.linspace(y_range[0], y_range[1], map_nycoord, dtype=DTypes.MAP_AXES)
            map_x = np.linspace(x_range[0], x_range[1], map_nxcoord, dtype=DTypes.MAP_AXES)

        # Do interpolation
        interp_result = interp_los_image_to_map(self, R0, map_x, map_y, no_data_val=no_data_val)

        origin_image = np.full(interp_result.data.shape, 0, dtype=DTypes.MAP_ORIGIN_IMAGE)
        # if image number is entered, record in appropriate pixels
        if image_num is not None:
            origin_image[interp_result.data > no_data_val] = image_num

        # Partially populate a map object with grid and data info
        map_out = PsiMap(interp_result.data, interp_result.x, interp_result.y,
                         mu=interp_result.mu_mat, origin_image=origin_image, no_data_val=no_data_val)

        # construct map_info df to record basic map info
        map_info_df = pd.DataFrame(data=[[1, datetime.datetime.now()], ],
                                   columns=["n_images", "time_of_compute"])
        map_out.append_map_info(map_info_df)

        return map_out

    def mu_hist(self, intensity_bin_edges, mu_bin_edges, lat_band=[-np.pi/64., np.pi/64.], log10=True):
        """
        Given an LOS image, bin an equatorial band of mu-bins by intensity.  This will generally
        be in preparation to fit Limb Brightening Correction Curves (LBCC).
        Before applying mu_hist to an image los_image.mu_hist(), first get coordinates for the image
        los_image.get_coordinates()
        :param intensity_bin_edges: Float numpy-vector of pixel-intensity bin edges in ascending order.
        :param mu_bin_edges: Float numpy vector within the range [0,1] of mu bin edges in ascending order.
        :param lat_band: A list/tuple of length 2 with minimum/maximum band of Carrington latitude to be considered.
        :param log10: True/False apply log_10 to image intensities before binning
        :return: A numpy-array with shape (# mu_bins x # intensity bins)
        """

        # check if LOS object has mu and cr_theta yet
        if self.mu is None or self.lat is None:
            sys.exit("Before running los_image.mu_hist(), first get coordinates los_image.get_coordinates().")

        # first reduce to points greater than mu-min and in lat-band
        lat_band_index = np.logical_and(self.lat <= max(lat_band), self.lat >= min(lat_band))
        mu_min = min(mu_bin_edges)
        mu_max = max(mu_bin_edges)
        mu_index = np.logical_and(self.mu >= mu_min, self.mu <= mu_max)
        use_index = np.logical_and(mu_index, lat_band_index)

        use_mu = self.mu[use_index]
        use_data = self.data[use_index]
        if log10:
            use_data[use_data < 0.] = 0.
            use_data = np.log10(use_data)
        # generate intensity histogram
        hist_out, temp_x, temp_y = np.histogram2d(use_mu, use_data, bins=[mu_bin_edges, intensity_bin_edges])

        return hist_out


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

    Map Object is created in three ways:
        1. interpolating from an image LosImage.interp_to_map()
        2. merging other maps map_manip.combine_maps()
        3. reading a map hdf file ---needs to be created---
    """

    def __init__(self, data, x, y, mu=None, origin_image=None, no_data_val=-9999.0):
        """
        Class to hold the standard information for a PSI map image
    for the CHD package.

        - Here the map is minimally described by:
            data: a 2D numpy array of values.
            x: a 1D array of the solar x positions of each pixel [Carrington lon].
            y: a 1D array of the solar y positions of each pixel [Sine lat].

        - mu and origin_image are optional and should be numpy arrays with
            dimensions identical to 'data'.

        Initialization also uses database definitions to generate empty dataframes
        for metadata: method_info, data_info, map_info, and var_info
        """
        # --- Initialize empty dataframes based on Table schema ---
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
        df_cols = set().union(val_columns, defs_columns)
        self.var_info = pd.DataFrame(data=None, columns=df_cols)

        # Type cast data arrays
        self.data = data.astype(DTypes.MAP_DATA)
        self.x    = x.astype(DTypes.MAP_AXES)
        self.y    = y.astype(DTypes.MAP_AXES)
        if mu is not None:
            self.mu = mu.astype(DTypes.MAP_MU)
        else:
            # create placeholder
            self.mu   = None
        self.no_data_val = no_data_val
        if origin_image is not None:
            self.origin_image = origin_image.astype(DTypes.MAP_ORIGIN_IMAGE)
        else:
            # create placeholder
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

    def write_to_file(self, filename):
        """
        Write the map object to hdf5 format
        """
        # check to see if the map exists from instantiation
        if hasattr(self, 'map'):
            sunpy_meta = self.map.meta

        psihdf.wrh5_meta(filename, self.x, self.y, np.array([]),
                         self.data, chd_meta=self.info, sunpy_meta=sunpy_meta)


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
