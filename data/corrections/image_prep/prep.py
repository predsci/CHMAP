"""
Module for converting raw EUV images (fits) to our native CHD format (h5).
- Here the relevant data reduction and preparation steps are applied, including:
  - Data calibration to Level (for EUVI)
  - PSF deconvolution
  - Image rotation
  - Image registration
"""
import os
import numpy as np
from timeit import default_timer as timer

import sunpy.map
import astropy.units as u
import sunpy.io

from data.corrections.image_prep import deconv
from data.download.euv_utils import get_metadata
import settings.info
from utilities.datatypes import datatypes
from utilities.file_io import io_helpers
from utilities.idl_connect import idl_helper
from settings.app import App

# default bad or missing value for an EUV image
euv_bad_value = 1e-16

# AIA specific hardcoded settings
aia_psf_version = 2
aia_bin_factor = 2

# EUVI specific hardcoded settings
euvi_psf_version = 2
euvi_bin_factor = 1

# default final datatype for processed images
hdf_dtype = np.float32
hdf_bitpix = -32


def prep_euv_image(fitsfile, processed_home_dir, deconvolve=True, write=True, idl_session=None):
    """
    Main program for preparing an EUV image from a raw fitsfile.
    - This reads the raw fits image and prepares it for CHD analysis.
    - The sunpy.map nickname is used to decide which instrument specific
      prep subroutine to call.

    - Currently this program takes a path to a fits file and writes out
      the file in the proper sub-directory relative to a "home path" (processed_home_dir)
    - After writing the file, it returns a relative sub directory, filename, and LosImage object
      that can be used to populate the filename information in the database.
    """
    print("  Preparing FITS Image: " + os.path.basename(fitsfile), flush=True)

    # read the fits file using sunpy map (almost as fast as direct read, but you get metadata)
    map_raw = sunpy.map.Map(fitsfile)

    # call the appropriate prep method
    if map_raw.nickname == 'AIA':
        los = prep_aia_image(map_raw, deconvolve=deconvolve)
    elif map_raw.nickname == 'EUVI-A' or map_raw.nickname == 'EUVI-B':
        los = prep_euvi_image(map_raw, deconvolve=deconvolve, idl_session=idl_session)
    else:
        raise Exception('Map nickname not recognized: {}'.format(map.nickname))

    # compute the h5 path information
    dtime = map_raw.date.datetime
    # prefix = los.info['instrument'].lower().replace('-', '') + '_lvl2'
    # postfix = str(los.info['wavelength'])
    # extension = 'h5'
    prefix, postfix, extension = io_helpers.construct_hdf5_pre_and_post(los.info)
    sub_dir, fname = io_helpers.construct_path_and_fname(
        processed_home_dir, dtime, prefix, postfix, extension,
        mkdir=write)
    rel_sub_dir = sub_dir.replace(processed_home_dir + os.path.sep, '')

    # write the file
    if write:
        fname_full = os.path.join(processed_home_dir, sub_dir, fname)
        print('  writing to: ' + fname_full, flush=True)
        los.write_to_file(fname_full)

    # return the relative sub directory, filename, and los object
    return rel_sub_dir, fname, los


def prep_aia_image(map_raw, deconvolve=True):
    """
    Prepare an AIA image for processing by the CHD database.
    The basic steps include:
    - PSF deconvolution.
    - Rebinning.
    - Image rotation.
    - Exposure normalization.
    - Assembling/updating metadata.

    The subroutine routines an LosImage object, which is specific to this project.
    """
    # get image data in double precision for processing
    image = np.float64(map_raw.data)

    # deconvolve the image
    if deconvolve:
        # build the deconvolution string (making sure its in angstroms)
        wave_string = str(np.int(map_raw.wavelength.to(u.angstrom).to_value()))

        psf_name = 'AIA_' + wave_string + '_4096'
        if aia_psf_version == 2:
            psf_name = psf_name + '_PSF2'

        # Call the deconvolution script
        print("  Calling Remote PSF Deconvolution (" + psf_name + ")... ", end='', flush=True)
        t0 = timer()
        image = deconv.deconv_decurlog_gpu(image, psf_name)
        time_deconv = timer() - t0
        print('done ({0:.2f}s)'.format(time_deconv), flush=True)

    # now do the various prep steps
    print("  Prepping Image... ", end='', flush=True)
    t0 = timer()

    # remake the map with the floating point image
    map = sunpy.map.Map(image, map_raw.meta)

    # bin the image to the desired resolution (this updates some metadata, manually update rsun)
    if aia_bin_factor != 1:
        new_dimensions = u.Quantity([aia_bin_factor, aia_bin_factor]*u.pix)
        map = map.superpixel(new_dimensions, func=np.mean)
        map.meta['r_sun'] = map.rsun_obs.value/map.meta['cdelt1']

    # now rotate the image so solar north is up (this updates metadata)
    map = rotate_map_nopad(map)

    # label this new map as level 2 processing and add a history to the metadata
    level = 2.0
    map.meta['lvl_num'] = level
    map.meta['history'] = map.meta['history'] + \
                          ' Processed to lvl {0:.1f} by prep_aia_image (CHD v{1})'.format(
                              level, settings.info.version)

    # divide by the exposure time, manually update the metadata
    map = sunpy.map.Map(map.data/map.exposure_time, map.meta)
    map.meta['exptime'] = 1.0

    # get the chd specific metadata (delete the datetime object)
    chd_meta = get_metadata(map)
    if 'datetime' in chd_meta:
        chd_meta.pop('datetime')

    # get the x and y 1D scales for the image
    x, y = get_scales_from_map(map)

    # get the image data as the desired data type
    data = hdf_dtype(map.data)
    map.meta['bitpix'] = hdf_bitpix

    # Convert the final image to an LosImage object
    los = datatypes.LosImage(data, x, y, chd_meta, map.meta)

    # stop the prep time
    time_prep = timer() - t0
    print('done ({0:.2f}s)'.format(time_prep), flush=True)

    # return the LosImage
    return los


def prep_euvi_image(map_raw, deconvolve=True, idl_session=None):
    """
    Prepare an EUVI image for processing by the CHD database.
    The basic steps include:
    - Calling SSW/IDL to run secchi_prep to get a calibrated lvl 1 image (no rotation)
    - PSF deconvolution.
    - Rebinning (optional).
    - Image rotation.
    - Exposure normalization.
    - Assembling/updating metadata.

    The subroutine routines an LosImage object, which is specific to this project.

    A running IDL session object can be passed to the subroutine.
      to allow one session to take care of multiple prep steps
    """
    # Begin secchi_prep with SSW/IDL
    t0 = timer()
    print("  Calling secchi_prep with SSW/IDL... ", end='', flush=True)

    # first save the data as a temporary, uncompressed uint16 fits file
    fits_raw = os.path.join(App.TMP_HOME, 'tmp_euvi_raw.fits')
    io_helpers.write_sunpy_map_as_fits(fits_raw, map_raw, dtype=np.uint16)

    # call secchi prep (open a new subprocess if one does not exist)
    new_session = False
    if idl_session is None:
        idl_session = idl_helper.Session()
        new_session = True

    # run the SSW/IDL prep command
    fits_prepped = os.path.join(App.TMP_HOME, 'tmp_euvi_prepped.fits')
    idl_session.secchi_prep(fits_raw, fits_prepped, quiet=True)

    # end the idl session if necessary (closes a subprocess)
    if new_session:
        idl_session.end()

    # read in the result as a sunpy map object
    map_prepped = sunpy.map.Map(fits_prepped)

    # get image data in double precision for processing
    image = np.float64(map_prepped.data)

    # end the secchi_prep SSW timer
    time_idl = timer() - t0
    print('done ({0:.2f}s)'.format(time_idl), flush=True)

    # deconvolve the image
    if deconvolve:
        # build the deconvolution string (making sure its in angstroms)
        wave_string = str(np.int(map_prepped.wavelength.to(u.angstrom).to_value()))

        if map_prepped.nickname == 'EUVI-A':
            inst_string = 'STA'
        if map_prepped.nickname == 'EUVI-B':
            inst_string = 'STB'

        psf_name = inst_string + '_' + wave_string + '_2048'
        if euvi_psf_version == 2:
            psf_name = psf_name + '_SHEARER'

        # Call the deconvolution script
        print("  Calling Remote PSF Deconvolution (" + psf_name + ")... ", end='', flush=True)
        t0 = timer()
        image = deconv.deconv_decurlog_gpu(image, psf_name)
        time_deconv = timer() - t0
        print('done ({0:.2f}s)'.format(time_deconv), flush=True)

    # now do the various prep steps
    print("  Prepping Image... ", end='', flush=True)
    t0 = timer()

    # remake the map with the floating point image
    map = sunpy.map.Map(image, map_prepped.meta)

    # bin the image to the desired resolution (this updates some metadata, manually update rsun)
    if euvi_bin_factor != 1:
        new_dimensions = u.Quantity([euvi_bin_factor, euvi_bin_factor]*u.pix)
        map = map.superpixel(new_dimensions, func=np.mean)

    # now rotate the image so solar north is up (this updates metadata)
    map = rotate_map_nopad(map)

    # label this new map as level 2 processing and add a history to the metadata
    level = 2.0
    map.meta['lvl_num'] = level
    map.meta['history'] = map.meta['history'] + \
                          ' Processed to lvl {0:.1f} by prep_euvi_image (CHD v{1})'.format(
                              level, settings.info.version)

    # replace the exposure time flag since it was normalized by secchi_prep
    map.meta['exptime'] = 1.0

    # get the chd specific metadata (delete the datetime object)
    chd_meta = get_metadata(map)
    if 'datetime' in chd_meta:
        chd_meta.pop('datetime')

    # get the x and y 1D scales for the image
    x, y = get_scales_from_map(map)

    # get the image data as the desired data type
    data = hdf_dtype(map.data)
    map.meta['bitpix'] = hdf_bitpix

    # Convert the final image to an LosImage object
    los = datatypes.LosImage(data, x, y, chd_meta, map.meta)

    # stop the prep time
    time_prep = timer() - t0
    print('done ({0:.2f}s)'.format(time_prep), flush=True)

    # return the LosImage
    return los


def rotate_map_nopad(map):
    """
    Wrapper for sunpy.map.mapbase.rotate that does rotation how we like it.
    - The images are recentered and the extra padding produced by rotate is
      removed after rotation (got this from the aia_prep routine in Sunpy 1.03).

    :param map:
    :return newmap:
    """
    tempmap = map.rotate(recenter=True, missing=euv_bad_value)

    # extract center from padded map.rotate output
    # - crpix1 and crpix2 will be equal (recenter=True) -> does not work with submaps
    center = np.floor(tempmap.meta['crpix1'])
    range_side = (center + np.array([-1, 1])*map.data.shape[0]/2)*u.pix
    newmap = tempmap.submap(u.Quantity([range_side[0], range_side[0]]),
                            u.Quantity([range_side[1], range_side[1]]))

    return newmap


def get_scales_from_map(map):
    """
    Compute the solar X and solar Y 1D pixel scale arrays from a sunpy map object.
    - If the image has been rotated to solar north up, then the x and y scales will
      be in the helioprojective cartesian system with units of [Rs].
    """

    # get the x, y, pixel coordinates in 1D by going along the diagonal (assume its square)
    npix = int(map.dimensions[0].value)
    inds = np.arange(npix)*u.pix
    xylocs = map.pixel_to_world(inds, inds)

    # Tx and Ty are the arcseconds as astropy quantities
    x_rs = xylocs.Tx/map.rsun_obs
    y_rs = xylocs.Ty/map.rsun_obs

    # convert them to floating point
    x = hdf_dtype(x_rs.value)
    y = hdf_dtype(y_rs.value)

    return x, y
