"""
Module to hold the custom image and map types specific to the CHD project.
"""
import datetime
import os
import pickle
import sys

import chmap.utilities.file_io.psi_hdf as psihdf
import chmap.database.db_classes as db
import chmap.database.db_funs as db_fun
import numpy as np
import pandas as pd
import sunpy.map
import sunpy.util.metadata
from chmap.utilities.coord_manip import interp_los_image_to_map, image_grid_to_CR, interp_los_image_to_map2
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

    def __init__(self, data=None, x=None, y=None, chd_meta=None, sunpy_meta=None):

        if data is not None:
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
        self.mu = None
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
                                                              obsv_lon=self.info['cr_lon'], get_mu=True,
                                                              outside_map_val=outside_map_val)

        cr_theta = cr_theta_all.reshape(self.data.shape, order="C")
        cr_phi = cr_phi_all.reshape(self.data.shape, order="C")
        image_mu = image_mu.reshape(self.data.shape, order="C")

        self.lat = cr_theta - np.pi / 2.
        self.lon = cr_phi
        self.mu = image_mu
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

        if type(self) == CHDImage:
            print("Converting " + self.info['instrument'] + "-" + str(self.info['wavelength']) + " image from " +
                  self.info['date_string'] + " to a CHD map.")
        else:
            print("Converting " + self.info['instrument'] + "-" + str(self.info['wavelength']) + " image from " +
              self.info['date_string'] + " to a EUV map.")

        if map_x is None and map_y is None:
            # Generate map grid based on number of image pixels vertically within R0
            # map parameters (assumed)
            y_range = [-1, 1]
            x_range = [0, 2 * np.pi]

            # observer parameters (from image)
            cr_lat = self.info['cr_lat']
            cr_lon = self.info['cr_lon']

            # determine number of pixels in map y-grid
            map_nycoord = sum(abs(self.y) < R0)
            del_y = (y_range[1] - y_range[0]) / (map_nycoord - 1)
            # how to define pixels? square in sin-lat v phi or lat v phi?
            # del_x = del_y*np.pi/2
            del_x = del_y
            map_nxcoord = (np.floor((x_range[1] - x_range[0]) / del_x) + 1).astype(int)

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
                         mu=interp_result.mu_mat, map_lon=interp_result.map_lon,
                         origin_image=origin_image, no_data_val=no_data_val)

        # construct map_info df to record basic map info
        map_info_df = pd.DataFrame(data=[[1, datetime.datetime.now()], ],
                                   columns=["n_images", "time_of_compute"])
        map_out.append_map_info(map_info_df)

        return map_out

    def interp_to_map2(self, R0=1.0, map_x=None, map_y=None, no_data_val=-9999., image_num=None):

        print("Converting " + self.info['instrument'] + "-" + str(self.info['wavelength']) + " image from " +
              self.info['date_string'] + " to a map.\n")

        if map_x is None and map_y is None:
            # Generate map grid based on number of image pixels vertically within R0
            # map parameters (assumed)
            y_range = [-1, 1]
            x_range = [0, 2 * np.pi]

            # observer parameters (from image)
            cr_lat = self.info['cr_lat']
            cr_lon = self.info['cr_lon']

            # determine number of pixels in map y-grid
            map_nycoord = sum(abs(self.y) < R0)
            del_y = (y_range[1] - y_range[0]) / (map_nycoord - 1)
            # how to define pixels? square in sin-lat v phi or lat v phi?
            # del_x = del_y*np.pi/2
            del_x = del_y
            map_nxcoord = (np.floor((x_range[1] - x_range[0]) / del_x) + 1).astype(int)

            # generate map x,y grids. y grid centered on equator, x referenced from lon=0
            map_y = np.linspace(y_range[0], y_range[1], map_nycoord, dtype=DTypes.MAP_AXES)
            map_x = np.linspace(x_range[0], x_range[1], map_nxcoord, dtype=DTypes.MAP_AXES)

        # Do interpolation
        interp_result = interp_los_image_to_map2(self, R0, map_x, map_y, no_data_val=no_data_val)

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

    def mu_hist(self, intensity_bin_edges, mu_bin_edges, lat_band=[-np.pi / 64., np.pi / 64.], log10=True):
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
            # use_data[use_data < 0.] = 0.
            # log(0) is still a problem, set to a very small number
            use_data[use_data <= 0.] = 1E-8
            use_data = np.log10(use_data)
        # generate intensity histogram
        hist_out, temp_x, temp_y = np.histogram2d(use_mu, use_data, bins=[mu_bin_edges, intensity_bin_edges])

        return hist_out

    def iit_hist(self, intensity_bin_edges, lat_band, log10=True):
        """
        function to create iit histogram
        """
        # indices within the latitude band and with valid mu
        lat_band_index = np.logical_and(self.lat <= max(lat_band), self.lat >= min(lat_band))
        mu_index = np.logical_and(self.mu > 0, self.mu <= self.mu.max())
        use_index = np.logical_and(mu_index, lat_band_index)

        use_data = self.data[use_index]
        if log10:
            use_data = np.where(use_data > 0, use_data, 0.01)
            use_data = np.log10(use_data)

        # generate intensity histogram
        hist_out, bin_edges = np.histogram(use_data, bins=intensity_bin_edges)

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


class LBCCImage(LosImage):
    """
    Class that holds limb-brightening corrected data
    """

    def __init__(self, los_image, lbcc_data, data_id, meth_id, intensity_bin_edges):

        super().__init__(los_image.data, los_image.x, los_image.y, los_image.info)
        date_format = "%Y-%m-%dT%H:%M:%S.%f"
        self.date_obs = datetime.datetime.strptime(los_image.info['date_string'], date_format)
        self.instrument = los_image.info['instrument']
        self.wavelength = los_image.info['wavelength']
        self.distance = los_image.info['distance']

        # definitions
        self.lbcc_data = lbcc_data
        self.data_id = data_id
        self.meth_id = meth_id
        self.n_intensity_bins = len(intensity_bin_edges) - 1
        self.intensity_bin_edges = intensity_bin_edges

    def iit_hist(self, lat_band, log10=True):
        """
        function to create iit histogram
        """
        # indices within the latitude band and with valid mu
        lat_band_index = np.logical_and(self.lat <= max(lat_band), self.lat >= min(lat_band))
        mu_index = np.logical_and(self.mu > 0, self.mu <= self.mu.max())
        use_index = np.logical_and(mu_index, lat_band_index)

        use_data = self.lbcc_data[use_index]
        if log10:
            use_data = np.where(use_data > 0, use_data, 0.01)
            use_data = np.log10(use_data)

        # generate intensity histogram
        hist_out, bin_edges = np.histogram(use_data, bins=self.intensity_bin_edges)

        return hist_out

    def lbcc_to_binary(self):
        if self.mu is not None:
            mu_array = self.mu.tobytes()
        if self.lat is not None:
            lat_array = self.lat.tobytes()
        if self.data is not None:
            lbcc_data = self.lbcc_data.tobytes()

        return mu_array, lat_array, lbcc_data


def create_lbcc_image(los_image, corrected_data, data_id, meth_id, intensity_bin_edges):
    # create LBCC Image structure
    lbcc = LBCCImage(los_image, corrected_data, data_id, meth_id, intensity_bin_edges)
    return lbcc


def binary_to_lbcc(lbcc_image_binary):
    for index, row in lbcc_image_binary.iterrows():
        lat_array = np.frombuffer(row.lat, dtype=np.float)
        mu_array = np.frombuffer(row.mu, dtype=np.float)
        lbcc_data = np.frombuffer(row.lbcc_data, dtype=np.float)

    return lat_array, mu_array, lbcc_data


class IITImage(LBCCImage):
    """
    class to hold corrected IIT data
    """

    def __init__(self, los_image, lbcc_image, iit_corrected_data, meth_id):
        # LBCC inherited definitions
        corrected_data = lbcc_image.lbcc_data
        data_id = lbcc_image.data_id
        intensity_bin_edges = lbcc_image.intensity_bin_edges
        super().__init__(los_image, corrected_data, data_id, meth_id, intensity_bin_edges)

        # IIT specific stuff
        self.iit_data = iit_corrected_data


def create_iit_image(los_image, lbcc_image, iit_corrected_data, meth_id):
    # create LBCC Image structure
    iit = IITImage(los_image, lbcc_image, iit_corrected_data, meth_id)
    return iit


class CHDImage(LosImage):
    """
    class to hold CHD data
    """
    def __init__(self, los_image=None, chd_data=None):
        if los_image is None and chd_data is None:
            super().__init__(data=None)
        else:
            super().__init__(los_image.data, los_image.x, los_image.y, los_image.info)
            self.data = chd_data


def create_chd_image(los_image=None, chd_data=None):
    chd = CHDImage(los_image, chd_data)
    return chd


class PsiMap:
    """
    Object that contains map information.  The Map object is structured like the database for convenience.
        - One of its primary uses is for passing info to and from database.

    Map Object is created in three ways:
        1. interpolating from an image LosImage.interp_to_map()
        2. merging other maps map_manip.combine_maps()
        3. reading a map hdf file ---needs to be created---
    """

    def __init__(self, data=None, x=None, y=None, mu=None, origin_image=None,
                 map_lon=None, chd=None, no_data_val=-9999.0):
        """
        Class to hold the standard information for a PSI map image
    for the CHD package.

        - Here the map is minimally described by:
            data: a 2D numpy array of values.
            x: a 1D array of the solar x positions of each pixel [Carrington lon].
            y: a 1D array of the solar y positions of each pixel [Sine lat].

        - mu, origin_image, and chd are optional and should be numpy arrays with
            dimensions identical to 'data'.

        Initialization also uses database definitions to generate empty dataframes
        for metadata: method_info, data_info, map_info, and var_info
        """
        ### initialize class to create map list
        if data is None:
            self.data = self.x = self.y = ()
        else:
            # --- Initialize empty dataframes based on Table schema ---
            # create the data tags
            # self.data_info = init_df_from_declarative_base(db.EUV_Images)
            euv_image_cols = []
            for column in db.EUV_Images.__table__.columns:
                euv_image_cols.append(column.key)
            data_files_cols = []
            for column in db.Data_Files.__table__.columns:
                data_files_cols.append(column.key)
            data_cols = set().union(euv_image_cols, data_files_cols)
            self.data_info = pd.DataFrame(data=None, columns=data_cols)
            # map_info will be a combination of Data_Combos and EUV_Maps
            image_columns = []
            for column in db.Data_Combos.__table__.columns:
                image_columns.append(column.key)
            map_columns = []
            for column in db.EUV_Maps.__table__.columns:
                map_columns.append(column.key)
            df_cols = set().union(image_columns, map_columns)
            self.map_info = pd.DataFrame(data=None, columns=df_cols)
            # method_info is a combination of Var_Defs and Method_Defs
            meth_columns = []
            for column in db.Method_Defs.__table__.columns:
                meth_columns.append(column.key)
            defs_columns = []
            for column in db.Var_Defs.__table__.columns:
                defs_columns.append(column.key)
            df_cols = set().union(meth_columns, defs_columns)
            self.method_info = pd.DataFrame(data=None, columns=df_cols)

            # Type cast data arrays
            self.data = data.astype(DTypes.MAP_DATA)
            self.x = x.astype(DTypes.MAP_AXES)
            self.y = y.astype(DTypes.MAP_AXES)
            # designate a 'no data' value for 'data' array
            self.no_data_val = no_data_val
            # optional data handling
            if mu is not None:
                self.mu = mu.astype(DTypes.MAP_MU)
            else:
                # create placeholder
                self.mu = None
            if origin_image is not None:
                self.origin_image = origin_image.astype(DTypes.MAP_ORIGIN_IMAGE)
            else:
                # create placeholder
                self.origin_image = None
            if map_lon is not None:
                self.map_lon = map_lon.astype(DTypes.MAP_DATA)
            else:
                # create placeholder
                self.map_lon = None
            if chd is not None:
                self.chd = chd.astype(DTypes.MAP_CHD)
            else:
                # create placeholder
                self.chd = None

    def append_map_info(self, map_df):
        """
        add a record to the map_info dataframe
        """
        self.map_info = self.map_info.append(map_df, sort=False, ignore_index=True)

    def append_method_info(self, method_df):
        self.method_info = self.method_info.append(method_df, sort=False, ignore_index=True)

    # def append_var_info(self, var_info_df):
    #     self.var_info = self.var_info.append(var_info_df, sort=False, ignore_index=True)

    def append_data_info(self, image_df):
        self.data_info = self.data_info.append(image_df, sort=False, ignore_index=True)

    def __copy__(self):
        out = PsiMap(data=self.data, x=self.x, y=self.y, mu=self.mu,
                     origin_image=self.origin_image, map_lon=self.map_lon,
                     chd=self.chd, no_data_val=self.no_data_val)
        out.append_map_info(self.map_info)
        out.append_data_info(self.data_info)
        out.append_method_info(self.method_info)
        return out

    def write_to_file(self, base_path, map_type=None, filename=None, db_session=None):
        """
        Write the map object to hdf5 format
        :param map_type: String describing map category (ex 'single', 'synchronic', 'synoptic', etc)
        :param base_path: String describing the base path to map directory tree.
        :param filename: If None, code will auto-generate a path (relative to App.MAP_FILE_HOME) and filename.
        Otherwise, this should be a string describing the relative/local path and filename.
        :param db_session: If None, save the file. If an SQLAlchemy session is passed, also record
        a map record in the database.
        :return: updated PsiMap object
        """

        if filename is None and map_type is None:
            sys.exit("When auto-generating filename and path (filename=None), map_type must be specified.")

        if db_session is None:
            if filename is None:
                # # generate filename
                # subdir, temp_fname = misc_helpers.construct_map_path_and_fname(base_path, self.map_info.date_mean[0],
                #                                                                map_id, map_type, 'h5', mkdir=True)
                # rel_path = os.path.join(subdir, temp_fname)
                sys.exit("Currently no capability to auto-generate unique filenames without using the database. "
                         "Please fill the 'filename' input when attempting to write_to_file() with no database"
                         "connection/session 'db_session'.")
            else:
                rel_path = filename

            h5_filename = os.path.join(base_path, rel_path)
            # write map to file
            psihdf.wrh5_fullmap(h5_filename, self.x, self.y, np.array([]), self.data, method_info=self.method_info,
                                data_info=self.data_info, map_info=self.map_info,
                                no_data_val=self.no_data_val, mu=self.mu, origin_image=self.origin_image,
                                chd=self.chd)
            print("PsiMap object written to: " + h5_filename)

        else:
            self.map_info.loc[0, 'fname'] = filename
            db_session, self = db_fun.add_map_dbase_record(db_session, self, base_path, map_type=map_type)

        return db_session


def read_psi_map(h5_file):
    """
    Open the file at path h5_file and convert contents to the PsiMap class.
    :param h5_file: Full file path to a PsiMap hdf5 file.
    :return: PsiMap object
    """
    # read the image and metadata
    x, y, z, f, method_info, data_info, map_info, var_info, no_data_val, mu, origin_image, chd = psihdf.rdh5_fullmap(
        h5_file)

    # create the map structure
    psi_map = PsiMap(f, x, y, mu=mu, origin_image=origin_image, no_data_val=no_data_val, chd=chd)
    # fill in DB meta data
    if method_info is not None:
        psi_map.append_method_info(method_info)
    if data_info is not None:
        psi_map.append_data_info(data_info)
    if map_info is not None:
        psi_map.append_map_info(map_info)

    return psi_map


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

    def __init__(self, data, x, y, mu_mat=None, map_lon=None):
        # create the data tags
        self.data = data
        self.x = x
        self.y = y
        self.mu_mat = mu_mat
        self.map_lon = map_lon


class Hist:
    """
    Class that holds histogram information
    """

    def __init__(self, image_id, meth_id, date_obs, instrument, wavelength, mu_bin_edges=None, intensity_bin_edges=None,
                 lat_band=[-np.pi / 64., np.pi / 64.], hist=None):

        self.image_id = image_id
        self.meth_id = meth_id
        self.date_obs = date_obs
        self.instrument = instrument
        self.wavelength = wavelength
        self.lat_band = float(np.max(lat_band))

        if mu_bin_edges is not None:
            self.n_mu_bins = len(mu_bin_edges) - 1
            self.mu_bin_edges = mu_bin_edges
        else:
            self.n_mu_bins = None
            self.mu_bin_edges = None
        if intensity_bin_edges is not None:
            self.n_intensity_bins = len(intensity_bin_edges) - 1
            self.intensity_bin_edges = intensity_bin_edges
        else:
            self.n_intensity_bins = None
            self.intensity_bin_edges = None
        if hist is not None:
            self.hist = hist

    def get_data(self):
        date_format = "%Y-%m-%dT%H:%M:%S.%f"
        hist_data = {'image_id': self.image_id,
                     'date_obs': datetime.datetime.strptime(self.info['date_string'], date_format),
                     'wavelength': self.info['wavelength'],
                     'instrument': self.info['instrument'], 'n_mu_bins': self.n_mu_bins,
                     'n_intensity_bins': self.n_intensity_bins, 'lat_band': self.lat_band,
                     'mu_bin_edges': self.mu_bin_edges, 'intensity_bin_edges': self.intensity_bin_edges,
                     'all_hists': self.mu_hist}
        return hist_data

    def write_to_pickle(self, path_for_hist, year, time_period, instrument, hist):
        file_path = path_for_hist + str(year) + "_" + time_period + '_' + str(
            len(self.mu_bin_edges)) + '_' + instrument + '.pkl'
        print('\nSaving histograms to ' + file_path + '\n')
        f = open(file_path, 'wb')
        pickle.dump(hist.get_data, f)
        f.close()


def create_lbcc_hist(h5_file, image_id, meth_id, mu_bin_edges, intensity_bin_edges, lat_band, mu_hist):
    # read the image and metadata
    x, y, z, data, chd_meta, sunpy_meta = psihdf.rdh5_meta(h5_file)
    # create the structure
    date_format = "%Y-%m-%dT%H:%M:%S.%f"
    # lat band float
    lat_band_float = float(np.max(lat_band))
    hist_out = Hist(image_id, meth_id, datetime.datetime.strptime(chd_meta['date_string'], date_format),
                    chd_meta['instrument'], chd_meta['wavelength'], mu_bin_edges,
                    intensity_bin_edges, lat_band_float, hist=mu_hist)
    return hist_out


def create_iit_hist(lbcc_image, meth_id, lat_band, iit_hist):
    """
    function to create IIT type histogram
    """
    # definitions
    image_id = lbcc_image.data_id
    date_obs = lbcc_image.date_obs
    instrument = lbcc_image.instrument
    wavelength = lbcc_image.wavelength
    intensity_bin_edges = lbcc_image.intensity_bin_edges
    lat_band_float = float(np.max(lat_band))

    # create structure
    iit_hist = Hist(image_id=image_id, meth_id=meth_id, date_obs=date_obs, instrument=instrument, wavelength=wavelength,
                    mu_bin_edges=None, intensity_bin_edges=intensity_bin_edges, lat_band=lat_band_float, hist=iit_hist)
    return iit_hist


def hist_to_binary(hist):
    """
    convert arrays to correct binary format
    used for adding histogram to database

    :param hist: LBCCHist histogram object
    :return: binary data types for array
    """

    if hist.intensity_bin_edges is not None:
        intensity_bin_edges = hist.intensity_bin_edges.tobytes()
    else:
        intensity_bin_edges = None
    if hist.mu_bin_edges is not None:
        mu_bin_edges = hist.mu_bin_edges.tobytes()
    else:
        mu_bin_edges = None
    hist = hist.hist.tobytes()

    return intensity_bin_edges, mu_bin_edges, hist


def binary_to_hist(hist_binary, n_mu_bins, n_intensity_bins):
    # generate histogram for time window
    if n_mu_bins is not None:
        for index, row in hist_binary.iterrows():
            mu_bin_array = np.frombuffer(row.mu_bin_edges, dtype=np.float)
            intensity_bin_array = np.frombuffer(row.intensity_bin_edges, dtype=np.float)
        full_hist = np.full((n_mu_bins, n_intensity_bins, hist_binary.__len__()), 0, dtype=np.int64)
        hist_list = hist_binary['hist']

        for index in range(hist_list.size):
            buffer_hist = np.frombuffer(hist_list[index])
            hist = np.ndarray(shape=(n_mu_bins, n_intensity_bins), buffer=buffer_hist)
            full_hist[:, :, index] = hist
    else:
        for index, row in hist_binary.iterrows():
            mu_bin_array = None
            intensity_bin_array = np.frombuffer(row.intensity_bin_edges, dtype=np.float)
        full_hist = np.full((n_intensity_bins, hist_binary.__len__()), 0, dtype=np.int64)
        hist_list = hist_binary['hist']

        for index in range(hist_list.size):
            buffer_hist = np.frombuffer(hist_list[index], dtype=np.int)
            hist = np.ndarray(shape=n_intensity_bins, dtype=np.int, buffer=buffer_hist)
            full_hist[:, index] = hist

    return mu_bin_array, intensity_bin_array, full_hist
