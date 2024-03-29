"""
Module to hold the custom image and map types specific to the CHD project.
"""
import datetime
import os
import pickle
import sys
import numpy as np
import pandas as pd

import sunpy.map
import sunpy.util.metadata

import chmap.utilities.file_io.psi_hdf as psihdf
import chmap.database.db_classes as db
import chmap.database.db_funs as db_fun
from chmap.data.corrections.image_prep import prep
from chmap.utilities.coord_manip import interp_los_image_to_map, image_grid_to_CR, interp_los_image_to_map_yang, \
    interp_los_image_to_map_par
from chmap.settings.info import DTypes


class LosImage:
    """
    Base class to hold the standard information for a Line-of-sight (LOS) image.

    - Here the image is minimally described by:
      data: a 2D numpy array of values.
      x: a 1D array of the solar x positions of each pixel [Rs].
      y: a 1D array of the solar y positions of each pixel [Rs].

    - sunpy_meta: a dictionary of sunpy metadata (optional)
      if this is specified then it to create a sunpy Map tag: map
      - This preserves compatibility with Sunpy and is useful for
        visualizing or interacting with the data.
    """

    def __init__(self, data=None, x=None, y=None, sunpy_meta=None, sunpy_cmap=None):

        if data is not None:
            # create the data tags
            self.data = data.astype(DTypes.LOS_DATA)
            self.x = x.astype(DTypes.LOS_AXES)
            self.y = y.astype(DTypes.LOS_AXES)

        # if sunpy_meta is supplied try to create a sunpy map
        if sunpy_meta != None:
            self.add_map(sunpy_meta)
        # independent of whether or not we create a sunpy map, we may want the
        # standard colormap
        if sunpy_cmap != None:
            self.sunpy_cmap = sunpy_cmap

        # add placeholders for carrington coords and mu
        self.lat = None
        self.lon = None
        self.mu = None
        self.no_data_val = None

    def get_coordinates(self, R0=1.0, cr_lat=0., cr_lon=0., outside_map_val=-9999.):
        """
        Calculate relevant mapping information for each pixel.
        This adds 2D arrays to the class:
        - lat: carrington latitude
        - lon: carrington longitude
        - mu: cosine of the center to limb angle
        """

        # vectorize image axis grid
        x_mat, y_mat = np.meshgrid(self.x, self.y)
        x_vec = x_mat.flatten(order="C")
        y_vec = y_mat.flatten(order="C")

        # determine if image is solar-north-up, or needs an additional rotation
        if hasattr(self, "sunpy_meta"):
            if "crota2" in self.sunpy_meta.keys():
                image_crota2 = self.sunpy_meta['crota2']
            else:
                image_crota2 = 0.
        else:
            image_crota2 = 0.

        # calculate the coordinates
        cr_theta_all, cr_phi_all, image_mu = image_grid_to_CR(x_vec, y_vec, R0=R0, obsv_lat=cr_lat,
                                                              obsv_lon=cr_lon, get_mu=True,
                                                              outside_map_val=outside_map_val,
                                                              crota2=image_crota2)

        cr_theta = cr_theta_all.reshape(self.data.shape, order="C")
        cr_phi = cr_phi_all.reshape(self.data.shape, order="C")
        image_mu = image_mu.reshape(self.data.shape, order="C")

        data_index = cr_theta != outside_map_val
        self.lat = cr_theta
        self.lat[data_index] = self.lat[data_index] - np.pi/2.
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

    def interp_data(self, R0, map_x, map_y, interp_field="data", no_data_val=-9999., nprocs=1, tpp=1,
                    p_pool=None, y_cor=False, helio_proj=False):

        # Do interpolation
        if y_cor:
            interp_result = interp_los_image_to_map_yang(self, R0, map_x, map_y, no_data_val=no_data_val,
                                                         interp_field=interp_field, nprocs=nprocs, tpp=tpp,
                                                         p_pool=p_pool)
        else:
            if nprocs > 1:
                interp_result = interp_los_image_to_map_par(self, R0, map_x, map_y, no_data_val=no_data_val,
                                                            interp_field=interp_field, nprocs=nprocs, tpp=tpp,
                                                            p_pool=p_pool, helio_proj=helio_proj)
            else:
                interp_result = interp_los_image_to_map(self, R0, map_x, map_y, no_data_val=no_data_val,
                                                        interp_field=interp_field, nprocs=nprocs, tpp=tpp,
                                                        p_pool=p_pool, helio_proj=helio_proj)

        return interp_result


class LosMagneto(LosImage):
    """
    Class to hold the standard information for a Line-of-sight (LOS) magnetogram.
    Created specifically to handle HMI 720s data, this class should work for
    any magnetogram on a disk.

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

    def __init__(self, data=None, x=None, y=None, sunpy_cmap=None, sunpy_meta=None,
                 make_map=False):

        pass_sunpy_meta = False
        if sunpy_meta is not None:
            # record the meta_data
            self.sunpy_meta = sunpy_meta
            # for back-compatibility, we add an 'info' attribute
            self.info = dict(cr_lat=self.sunpy_meta['crlt_obs'],
                             cr_lon=self.sunpy_meta['crln_obs'])
            if make_map:
                pass_sunpy_meta = True

        # reference the base class .__init__()
        if pass_sunpy_meta:
            super().__init__(data, x, y, sunpy_cmap=sunpy_cmap, sunpy_meta=sunpy_meta)
        else:
            super().__init__(data, x, y, sunpy_cmap=sunpy_cmap)

        # any initial data attributes unique to LosMagneto are setup here
        self.Br = None

    def get_coordinates(self, R0=1.0, outside_map_val=-65500.):
        """
        Calculate additional coordinates for each pixel.

        This function produces Carrington latitude and longitude in radians as well
        as cosine center-to-limb angle (mu).

        Parameters
        ----------
        R0 - number of solar radii for spherical coords
        outside_map_val - a value for pixel coordinates that fall outside of R0

        Returns
        -------
        self with additional numpy.ndarray for each lat, lon, and mu attributes.
        """
        cr_lat = self.sunpy_meta['crlt_obs']
        cr_lon = self.sunpy_meta['crln_obs']

        super().get_coordinates(R0=R0, cr_lat=cr_lat, cr_lon=cr_lon,
                                outside_map_val=outside_map_val)
        # get_coordinates() assumes solar-north-up to calc CR lat/lon. This is generally
        # not true for raw HMI_M disks. Remove until disk rotation is incorporated into
        # get_coordinates()
        self.lat = None
        self.lon = None

    def add_Br(self, mu_thresh=0.01, R0=1.):
        """
        Use the viewer angle to extrapolate radial B-field at each image pixel.  Radial
        B-field is approximated as:
        B_r = B_{los}/mu
        where mu is cosine center-to-limb angle.

        mu_thresh - scalar float [0., 1.]. To prevent denominator going to zero,
                    this parameter creates a lower threshold for mu. Any mu value
                    lower than mu_thresh will be set to mu_thresh for the B_r calc.
        """
        if self.mu is None:
            self.get_coordinates(R0=R0)
        # initialize radial B-field attribute
        # self.Br = np.zeros(shape=self.data.shape)
        # keep un-altered pixels from outside R0
        self.Br = np.copy(self.data)
        # determine off-disk pixels
        outside_im_index = self.mu == self.no_data_val
        # copy mu
        temp_mu = self.mu.copy()
        # apply mu-threshold
        mu_thresh_index = temp_mu < mu_thresh
        temp_mu[mu_thresh_index & ~outside_im_index] = mu_thresh
        # calculate naive estimate of radial B-field
        self.Br[~outside_im_index] = self.data[~outside_im_index]/temp_mu[~outside_im_index]

    def interp_to_map(self, R0=1.0, map_x=None, map_y=None, interp_field="Br", no_data_val=-65500.,
                      image_num=None, nprocs=1, tpp=1, p_pool=None, y_cor=False, helio_proj=False):

        print("Converting " + self.sunpy_meta['telescop'] + "-" + str(self.sunpy_meta['content']) + " image from " +
              self.sunpy_meta['date_obs'] + " to a CR map.")

        if map_x is None and map_y is None:
            # Generate map grid based on number of image pixels horizontally within R0
            # map parameters (assumed)
            y_range = [-1, 1]
            x_range = [0, 2 * np.pi]

            # determine number of pixels in map x and y-grids
            map_nxcoord = 2*sum(abs(self.x) < R0)
            map_nycoord = map_nxcoord/2

            # generate map x,y grids. y grid centered on equator, x referenced from lon=0
            map_y = np.linspace(y_range[0], y_range[1], map_nycoord, dtype=DTypes.MAP_AXES)
            map_x = np.linspace(x_range[0], x_range[1], map_nxcoord, dtype=DTypes.MAP_AXES)

        if interp_field == "Br" and self.Br is None:
            self.add_Br(R0=R0)

        # Do interpolation
        interp_result = self.interp_data(R0, map_x, map_y, interp_field=interp_field, no_data_val=no_data_val,
                                         nprocs=nprocs, tpp=tpp, p_pool=p_pool, y_cor=y_cor, helio_proj=helio_proj)

        # check for NaNs and set to no_data_val
        interp_result.data[np.isnan(interp_result.data)] = no_data_val

        # Partially populate a map object with grid and data info
        map_out = MagnetoMap(interp_result.data, interp_result.x, interp_result.y,
                             mu=interp_result.mu_mat, no_data_val=no_data_val)

        # transfer fits header information
        if self.sunpy_meta is not None:
            map_out.sunpy_meta = self.sunpy_meta

        # construct map_info df to record basic map info
        # map_info_df = pd.DataFrame(data=[[1, datetime.datetime.now()], ],
        #                            columns=["n_images", "time_of_compute"])
        # map_out.append_map_info(map_info_df)

        return map_out


def read_hmi720s(fits_file, make_map=False, solar_north_up=True):
    """
    Method for reading an HMI 720s FITS file to a LosMagneto class.  This
    includes a rotation to solar-north-up.
    fits_file: path to a raw fits file.
    make_map: boolean. Imbeds a sunpy.map object in the LosMagneto object. This allows
              many features, but is also redundant.
    solar_north_up: boolean. Rotate the image such that solar north is up.
    output: an LosMagneto object.

    """
    # load data and header tuple into sunpy map type
    map_raw = sunpy.map.Map(fits_file)

    if solar_north_up:
        # rotate image to solar north up. fill missing values with NaN to preserve
        # standard HMI formatting
        # first convert NaNs to 0. for rotation algorithm
        nan_index = np.isnan(map_raw.data)
        map_raw.data[nan_index] = 0.
        rot_map = prep.rotate_map_nopad(map_raw)
        # get x and y axis
        x, y = prep.get_scales_from_map(rot_map)

    else:
        rot_map = map_raw
        # get scales manually. auto-generation of scales by sunpy generates unknown
        # results when solar_north ~= image_up.  we want image-up and image-right
        # always positive, with vertical/horizontal y/x-axes
        x, y = prep.get_scales_from_fits(map_raw.meta)

    # create the object
    los = LosMagneto(rot_map.data, x, y, sunpy_cmap=rot_map.cmap,
                     sunpy_meta=rot_map.meta, make_map=make_map)

    return los


class EUVImage(LosImage):

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

        # reference the base class .__init__()
        super().__init__(data, x, y, sunpy_meta)

        # add the info dictionary
        self.info = chd_meta

    def get_coordinates(self, R0=1.01, outside_map_val=-9999.):

        cr_lat = self.info['cr_lat']
        cr_lon = self.info['cr_lon']

        super().get_coordinates(R0=R0, cr_lat=cr_lat, cr_lon=cr_lon,
                                outside_map_val=outside_map_val)

    def write_to_file(self, filename):
        """
        Write the los image to hdf5 format
        """
        # check to see if the map exists from instantiation
        if hasattr(self, 'map'):
            sunpy_meta = self.map.meta

        psihdf.wrh5_meta(filename, self.x, self.y, np.array([]),
                         self.data, chd_meta=self.info, sunpy_meta=sunpy_meta)

    def interp_to_map(self, R0=1.0, map_x=None, map_y=None, no_data_val=-9999., image_num=None,
                      interp_field="data", nprocs=1, tpp=1, p_pool=None, y_cor=False,
                      helio_proj=False):

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
        interp_result = self.interp_data(R0, map_x, map_y, interp_field=interp_field, no_data_val=no_data_val,
                                         nprocs=nprocs, tpp=tpp, p_pool=p_pool, y_cor=y_cor, helio_proj=helio_proj)

        # some images are cutoff or have no_data_val within the image disk for some reason.
        # To mostly filter out these types of points, set any resulting negatives to no_data_val
        data_index = interp_result.data > (no_data_val+1.)
        interp_result.data[data_index][interp_result.data[data_index] < 0.] = no_data_val

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


def read_euv_image(h5_file):
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
    los = EUVImage(data, x, y, chd_meta, sunpy_meta=sunpy_meta)

    return los


class LBCCImage(EUVImage):
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

        if los_image.map is not None:
            self.sunpy_meta = los_image.map.meta
        else:
            self.sunpy_meta = None


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
        self.sunpy_meta = lbcc_image.sunpy_meta

        # IIT specific stuff
        self.iit_data = iit_corrected_data


def create_iit_image(los_image, lbcc_image, iit_corrected_data, meth_id):
    # create LBCC Image structure
    iit = IITImage(los_image, lbcc_image, iit_corrected_data, meth_id)
    return iit


class CHDImage(EUVImage):
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
            self.data_info = pd.DataFrame(data=None, columns=list(data_cols))
            # map_info will be a combination of Data_Combos and EUV_Maps
            image_columns = []
            for column in db.Data_Combos.__table__.columns:
                image_columns.append(column.key)
            map_columns = []
            for column in db.EUV_Maps.__table__.columns:
                map_columns.append(column.key)
            df_cols = set().union(image_columns, map_columns)
            self.map_info = pd.DataFrame(data=None, columns=list(df_cols))
            # method_info is a combination of Var_Defs and Method_Defs
            meth_columns = []
            for column in db.Method_Defs.__table__.columns:
                meth_columns.append(column.key)
            defs_columns = []
            for column in db.Var_Defs.__table__.columns:
                defs_columns.append(column.key)
            df_cols = set().union(meth_columns, defs_columns)
            self.method_info = pd.DataFrame(data=None, columns=list(df_cols))

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
        # self.map_info = self.map_info.append(map_df, sort=False, ignore_index=True)
        self.map_info = pd.concat([self.map_info, map_df], sort=False, ignore_index=True)

    def append_method_info(self, method_df):
        # self.method_info = self.method_info.append(method_df, sort=False, ignore_index=True)
        self.method_info = pd.concat([self.method_info, method_df], sort=False, ignore_index=True)

    # def append_var_info(self, var_info_df):
    #     self.var_info = self.var_info.append(var_info_df, sort=False, ignore_index=True)

    def append_data_info(self, image_df):
        # self.data_info = self.data_info.append(image_df, sort=False, ignore_index=True)
        self.data_info = pd.concat([self.data_info, image_df], sort=False, ignore_index=True)

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


class MagnetoMap(PsiMap):
    """
    Object that contains magnetogram-map information.  The Map object is structured like the database for convenience.
        - Inherits from the PsiMap class.
    """
    def __init__(self, data=None, x=None, y=None, mu=None, no_data_val=-65500.):
        """
        Class to hold the standard information for a PSI map image
    for the CHD package.

        - Here the map is minimally described by:
            data: a 2D numpy array of values.
            x: a 1D array of the solar x positions of each pixel [Carrington lon].
            y: a 1D array of the solar y positions of each pixel [Sine lat].

        - mu is optional and should be numpy arrays with
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
            # euv_image_cols = []
            # for column in db.EUV_Images.__table__.columns:
            #     euv_image_cols.append(column.key)
            # data_files_cols = []
            # for column in db.Data_Files.__table__.columns:
            #     data_files_cols.append(column.key)
            # data_cols = set().union(euv_image_cols, data_files_cols)
            # self.data_info = pd.DataFrame(data=None, columns=data_cols)
            # # map_info will be a combination of Data_Combos and EUV_Maps
            # image_columns = []
            # for column in db.Data_Combos.__table__.columns:
            #     image_columns.append(column.key)
            # map_columns = []
            # for column in db.EUV_Maps.__table__.columns:
            #     map_columns.append(column.key)
            # df_cols = set().union(image_columns, map_columns)
            # self.map_info = pd.DataFrame(data=None, columns=df_cols)
            # # method_info is a combination of Var_Defs and Method_Defs
            # meth_columns = []
            # for column in db.Method_Defs.__table__.columns:
            #     meth_columns.append(column.key)
            # defs_columns = []
            # for column in db.Var_Defs.__table__.columns:
            #     defs_columns.append(column.key)
            # df_cols = set().union(meth_columns, defs_columns)
            # self.method_info = pd.DataFrame(data=None, columns=df_cols)

            # until database definitions are set, initiate these to None
            self.data_info = None
            self.map_info = None
            self.method_info = None

            # Type cast data arrays
            self.data = data.astype(DTypes.MAP_DATA)
            self.x = x.astype(DTypes.MAP_AXES)
            self.y = y.astype(DTypes.MAP_AXES)
            # designate a 'no data' value for 'data' array
            self.no_data_val = no_data_val
            # start a dataframe to keep track of multiple data layers
            self.layers = pd.DataFrame(dict(layer_num=[0, ], layer_name=["magneto", ],
                                            py_field_name=['data', ]))
            # set placeholder for fits header info
            self.sunpy_meta = None
            # optional data handling
            if mu is not None:
                self.mu = mu.astype(DTypes.MAP_DATA)
            else:
                # create placeholder
                self.mu = None


    def write_to_file(self, base_path, map_type=None, filename=None, db_session=None, AFT=False):
        """
        Write the map object to hdf5 format
        :param map_type: String describing map category (ex 'single_magneto', 'synoptic', etc)
        :param base_path: String describing the base path to map directory tree.
        :param filename: If None, code will auto-generate a path (relative to App.MAP_FILE_HOME) and filename.
        Otherwise, this should be a string describing the relative path and filename.
        :param db_session: If None, save the file. If an SQLAlchemy session is passed, also record
        a map record in the database.
        :param AFT: True/False
                    If True, write some additional fits_header info as top-level attributes in the
                    hdf5 file.
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

            # combine layers into a single array
            data_array = np.expand_dims(self.data, axis=0)
            z = np.array([0, ])

            # add additional layers according to the 'layers' dataframe
            layers_dim = self.layers.shape
            if layers_dim[0] < 2:
                sys.exit("HipFT files require at least two layers (ex. 'data' and 'assim_weights'). " +
                         "Current map has less than 2.")
            for ii in range(1, layers_dim[0]):
                # we want array to be in stride order phi, theta, layer
                # this is index order: data_array[layer, theta, phi]
                new_layer = getattr(self, self.layers.py_field_name.iloc[ii])
                new_layer = np.expand_dims(new_layer, axis=0)
                data_array = np.append(data_array, new_layer, axis=0)
                z = np.append(z, ii)

            # convert elevation to inclination and flip array so that the theta scale is ascending
            theta = np.pi/2 - self.y
            theta_flip = np.flip(theta)
            data_array_flip = np.flip(data_array, axis=1)

            # Customization for AFT input
            if AFT:
                car_rot = self.sunpy_meta['car_rot']
                crln_obs = self.sunpy_meta['crln_obs']
                crlt_obs = self.sunpy_meta['crlt_obs']
                t_rec = self.sunpy_meta['t_rec']
            else:
                car_rot = None
                crln_obs = None
                crlt_obs = None
                t_rec = None

            h5_filename = os.path.join(base_path, rel_path)
            # write map to file
            psihdf.wrh5_hipft_map(h5_filename, self.x, theta_flip, z, data_array_flip, method_info=self.method_info,
                                  data_info=self.data_info, map_info=self.map_info, no_data_val=self.no_data_val,
                                  layers=self.layers, assim_method=self.assim_method, fits_header=self.sunpy_meta,
                                  car_rot=car_rot, crln_obs=crln_obs, crlt_obs=crlt_obs, t_rec=t_rec)
            print("PsiMap object written to: " + h5_filename)

        else:
            sys.exit("Adding MagnetoMap to database not yet supported.")
            self.map_info.loc[0, 'fname'] = filename
            db_session, self = db_fun.add_map_dbase_record(db_session, self, base_path, map_type=map_type)

        return db_session

    def write_to_file_old(self, base_path, map_type=None, filename=None, db_session=None):
        """
        Write the map object to hdf5 format
        :param map_type: String describing map category (ex 'single_magneto', 'synoptic', etc)
        :param base_path: String describing the base path to map directory tree.
        :param filename: If None, code will auto-generate a path (relative to App.MAP_FILE_HOME) and filename.
        Otherwise, this should be a string describing the relative path and filename.
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

            # combine layers into a single array
            data_array = self.data
            z = np.array([0, ])
            if self.mu is not None:
                # we want array to be in stride order phi, theta, layer
                # this is index order: data_array[layer, theta, phi]
                data_array = np.stack((data_array, self.mu), axis=0)
                z = np.append(z, 1)

            # convert elevation to inclination and flip array so that the theta scale is ascending
            theta = np.pi/2 - self.y
            theta_flip = np.flip(theta)
            data_array_flip = np.flip(data_array, axis=1)

            h5_filename = os.path.join(base_path, rel_path)
            # write map to file
            psihdf.wrh5_hipft_map(h5_filename, self.x, theta_flip, z, data_array_flip, method_info=self.method_info,
                                  data_info=self.data_info, map_info=self.map_info, no_data_val=self.no_data_val,
                                  layers=self.layers)
            print("MagnetoMap object written to: " + h5_filename)

        else:
            sys.exit("Adding MagnetoMap to database not yet supported.")
            self.map_info.loc[0, 'fname'] = filename
            db_session, self = db_fun.add_map_dbase_record(db_session, self, base_path, map_type=map_type)

        return db_session

    def __copy__(self):
        out = MagnetoMap(data=self.data, x=self.x, y=self.y, mu=self.mu,
                         no_data_val=self.no_data_val)
        out.append_map_info(self.map_info)
        out.append_data_info(self.data_info)
        out.append_method_info(self.method_info)
        return out


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


def read_hipft_map(h5_file):
    """
    Open the file at path h5_file and convert contents to the MagnetoMap class.
    :param h5_file: Full file path to a MagnetoMap hdf5 file.
    :return: MagnetoMap object
    """
    # read the image and metadata
    # remember that scales come back in stride-order (phi, theta, layer)
    x, theta_flip, z, f_flip, method_info, data_info, map_info, no_data_val, layers, assim_method, sunpy_meta = \
        psihdf.rdh5_hipft_map(h5_file)
    # convert inclination back to elevation and flip array so that the theta scale is ascending
    theta = np.flip(theta_flip)
    y = -theta + np.pi/2
    f = np.flip(f_flip, axis=1)

    # extract primary data layer
    data = np.squeeze(f[layers.layer_num[layers.layer_name == "magneto"], :, :])

    # create the map structure
    magneto_map = MagnetoMap(data, x, y, no_data_val=no_data_val)

    # record layers
    magneto_map.layers = layers
    # extract 2D layers from 3D array
    for ii in range(1, layers.shape[0]):
        layer_name = layers.py_field_name.iloc[ii]
        layer_data = f[ii, :, :]
        setattr(magneto_map, layer_name, np.squeeze(layer_data))

    # record sunpy_meta data (fits header)
    magneto_map.sunpy_meta = sunpy_meta

    if assim_method is not None:
        magneto_map.assim_method = assim_method
    # fill in DB meta data
    if method_info is not None:
        magneto_map.append_method_info(method_info)
    if data_info is not None:
        magneto_map.append_data_info(data_info)
    if map_info is not None:
        magneto_map.append_map_info(map_info)

    return magneto_map


def read_hmi_Mrmap_latlon_720s(fits_file, no_data_val=-65500.):
    """
    Method for reading an HMI_Mrmap_latlon_720s FITS file to a MagnetoMap class.
    fits_file: path to a raw fits file.

    output: a MagnetoMap object.

    """
    # load data and header tuple into sunpy map type
    map_raw = sunpy.map.Map(fits_file)
    # generate CR lat/lon from fits parameters
    x, y = prep.get_scales_from_fits_map(map_raw.meta)

    # initiate full-map
    data = np.full([len(y), len(x)*2], fill_value=no_data_val)
    full_map_x = np.linspace(0.1, 359.9, 1800, dtype=x.dtype)
    full_map_y = np.linspace(-89.9, 89.9, 900, dtype=y.dtype)
    full_map_x_rad = full_map_x * np.pi/180
    full_map_y_rad = full_map_y * np.pi/180

    # check for CR lon that overlaps periodic boundary
    x_eps = .00001
    left_index = np.argwhere(np.abs(x-360.1) < x_eps)
    right_index = np.argwhere(np.abs(x + 0.1) < x_eps)

    # index columns of map_raw.data into the full map 'data'
    if not len(left_index) == 0:
        # divide input map across left periodic boundary
        per_index = left_index[0][0]
        periodic = True
    elif not len(right_index) == 0:
        # divide input map across right periodic boundary
        per_index = right_index[0][0] + 1
        periodic = True
    else:
        periodic = False

    if periodic:
        left_fill = len(x) - per_index
        data[:, :left_fill] = map_raw.data[:, per_index:]
        right_fill = len(full_map_x) - per_index
        data[:, right_fill:] = map_raw.data[:, :per_index]
    else:
        left_fill = np.argwhere(abs(full_map_x - x[0]) < x_eps)[0][0]
        right_fill = left_fill + len(x)
        data[:, left_fill:right_fill] = map_raw.data

    # final check to replace NaNs with no_data_val
    data[np.isnan(data)] = no_data_val

    ## Now add pixels at the periodic boundary and poles in order to match PSI mapping standard
    # periodic boundary
    left_nan_index = data[:, 0] == no_data_val
    right_nan_index = data[:, -1] == no_data_val
    periodic_boundary = np.full((len(y), 1), fill_value=no_data_val)
    both_ind = ~left_nan_index & ~right_nan_index
    periodic_boundary[both_ind, 0] = (data[both_ind, 0] + data[both_ind, -1])/2
    left_ind = ~left_nan_index & right_nan_index
    periodic_boundary[left_ind, 0] = (data[left_ind, 0]) / 2
    right_ind = left_nan_index & ~right_nan_index
    periodic_boundary[right_ind, 0] = (data[right_ind, -1]) / 2

    data1 = np.append(periodic_boundary, data, axis=1)
    data2 = np.append(data1, periodic_boundary, axis=1)
    full_map_x_rad1 = np.append(0., full_map_x_rad)
    full_map_x_rad2 = np.append(full_map_x_rad1, np.pi*2)

    # poles
    south_nan = data[0, :] == no_data_val
    if south_nan.all():
        fill_value = no_data_val
    else:
        mean_vals = data[0, ~south_nan]
        fill_value = np.mean(mean_vals)
    south_data = np.full((1, len(full_map_x_rad2)), fill_value=fill_value)

    north_nan = data[-1, :] == no_data_val
    if north_nan.all():
        fill_value = no_data_val
    else:
        mean_vals = data[-1, ~north_nan]
        fill_value = np.mean(mean_vals)
    north_data = np.full((1, len(full_map_x_rad2)), fill_value=fill_value)

    data3 = np.append(south_data, data2, axis=0)
    data4 = np.append(data3, north_data, axis=0)
    full_map_y_rad1 = np.append(-np.pi/2, full_map_y_rad)
    full_map_y_rad2 = np.append(full_map_y_rad1, np.pi/2)

    # create the object
    br_map = MagnetoMap(data=data4, x=full_map_x_rad2, y=full_map_y_rad2)
    # record fits header info
    br_map.sunpy_meta = map_raw.meta

    return br_map


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
            mu_bin_array = np.frombuffer(row.mu_bin_edges, dtype=float)
            intensity_bin_array = np.frombuffer(row.intensity_bin_edges, dtype=float)
        full_hist = np.full((n_mu_bins, n_intensity_bins, hist_binary.__len__()), 0, dtype=np.int64)
        hist_list = hist_binary['hist']

        for index in range(hist_list.size):
            buffer_hist = np.frombuffer(hist_list.iloc[index], dtype=int)
            hist = np.ndarray(shape=(n_mu_bins, n_intensity_bins), buffer=buffer_hist)
            full_hist[:, :, index] = hist
    else:
        for index, row in hist_binary.iterrows():
            mu_bin_array = None
            intensity_bin_array = np.frombuffer(row.intensity_bin_edges, dtype=float)
        full_hist = np.full((n_intensity_bins, hist_binary.__len__()), 0, dtype=np.int64)
        hist_list = hist_binary['hist']

        for index in range(hist_list.size):
            buffer_hist = np.frombuffer(hist_list.iloc[index], dtype=int)
            hist = np.ndarray(shape=n_intensity_bins, dtype=np.int, buffer=buffer_hist)
            full_hist[:, index] = hist

    return mu_bin_array, intensity_bin_array, full_hist
