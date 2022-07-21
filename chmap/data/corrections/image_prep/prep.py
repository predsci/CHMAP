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
import tempfile

import sunpy.map
import astropy.units as u
import sunpy.io

from chmap.data.corrections.image_prep import deconv
from chmap.data.download.euv_utils import get_metadata
import chmap.settings.info
from chmap.utilities.datatypes import datatypes
from chmap.utilities.file_io import io_helpers
from chmap.utilities.idl_connect import idl_helper

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
    - After writing the file, it returns a relative sub directory, filename, and EUVImage object
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

    The subroutine routines an EUVImage object, which is specific to this project.
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
                              level, chmap.settings.info.version)

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

    # Convert the final image to an EUVImage object
    los = datatypes.EUVImage(data, x, y, chd_meta, map.meta)

    # stop the prep time
    time_prep = timer() - t0
    print('done ({0:.2f}s)'.format(time_prep), flush=True)

    # return the EUVImage
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

    The subroutine routines an EUVImage object, which is specific to this project.

    A running IDL session object can be passed to the subroutine.
      to allow one session to take care of multiple prep steps
    """
    # Begin secchi_prep with SSW/IDL
    t0 = timer()
    print("  Calling secchi_prep with SSW/IDL... ", end='', flush=True)

    # first save the data as a temporary, uncompressed uint16 fits file
    temp_dir = tempfile.TemporaryDirectory()
    fits_raw = os.path.join(temp_dir.name, 'tmp_euvi_raw.fits')
    io_helpers.write_sunpy_map_as_fits(fits_raw, map_raw, dtype=np.uint16)

    # call secchi prep (open a new subprocess if one does not exist)
    new_session = False
    if idl_session is None:
        idl_session = idl_helper.Session()
        new_session = True

    # run the SSW/IDL prep command
    fits_prepped = os.path.join(temp_dir, 'tmp_euvi_prepped.fits')
    idl_session.secchi_prep(fits_raw, fits_prepped, quiet=True)

    # end the idl session if necessary (closes a subprocess)
    if new_session:
        idl_session.end()

    # read in the result as a sunpy map object
    map_prepped = sunpy.map.Map(fits_prepped)
    temp_dir.cleanup()

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
                              level, chmap.settings.info.version)

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

    # Convert the final image to an EUVImage object
    los = datatypes.EUVImage(data, x, y, chd_meta, map.meta)

    # stop the prep time
    time_prep = timer() - t0
    print('done ({0:.2f}s)'.format(time_prep), flush=True)

    # return the EUVImage
    return los


def rotate_map_nopad(raw_map):
    """
    Wrapper for sunpy.map.mapbase.rotate that does rotation how we like it.
    - The images are recentered and the extra padding produced by rotate is
      removed after rotation (got this from the aia_prep routine in Sunpy 1.03).

    :param raw_map: Raw map from FITS file
    :return newmap: New map rotated to polar-north=up and padding removed
    """
    tempmap = raw_map.rotate(recenter=True)

    # extract center from padded map.rotate output
    # - crpix1 and crpix2 will be equal (recenter=True) -> does not work with submaps
    center = tempmap.meta['crpix1']
    # newer sunpy wants bottom_left and top_right rather than axis1_range and axis2_range
    if (center % 1.) == 0:
        # Implies an odd number of pixels, remove an extra pixel from top and right
        # Assumes original shape is even
        bottom_left = (center - np.array([1, 1])*raw_map.data.shape[0]/2 +
                       np.array([0, 0]))*u.pix
        top_right = (center + np.array([1, 1])*raw_map.data.shape[0]/2 -
                     np.array([1, 1]))*u.pix
    else:
        bottom_left = (center - np.array([1, 1])*raw_map.data.shape[0]/2)*u.pix
        top_right = (center + np.array([1, 1])*raw_map.data.shape[0]/2)*u.pix

    newmap = tempmap.submap(bottom_left=bottom_left, top_right=top_right)

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


def get_scales_from_fits(fits_meta):
    """
    Compute the solar X and solar Y 1D scale arrays from the fits metadata.

    Return scales in solar radii and assume that image-up and image-right are positive.
    :param fits_meta:   sunpy.util.metadata.MetaDict
                        Meta data loaded from fits file by example = sunpy.map.Map()
                        Meta data is accessed by 'example.meta'
    :return:    tuple of arrays
                First array is x-axis of image space in solar radii.
                Second array is y-axis of image space in solar radii.
    """

    arcsec_per_radii = fits_meta['rsun_obs']
    # y-axis pars
    crpix2 = fits_meta['crpix2']
    cdelt2 = fits_meta['cdelt2']
    naxis2 = fits_meta['naxis2']
    # x-axis pars
    crpix1 = fits_meta['crpix1']
    cdelt1 = fits_meta['cdelt1']
    naxis1 = fits_meta['naxis1']

    # pixel locations (starting at 1 and not 0, per the fits standard)
    xpix_num = np.arange(start=1, stop=naxis1+1, step=1)
    rel_xpix = xpix_num - crpix1
    # convert to arcsec
    x_arcsec = rel_xpix*cdelt1
    # convert to solar radii
    x_radii = x_arcsec/arcsec_per_radii

    # pixel locations (starting at 1 and not 0, per the fits standard)
    ypix_num = np.arange(start=1, stop=naxis2+1, step=1)
    rel_ypix = ypix_num - crpix2
    # convert to arcsec
    y_arcsec = rel_ypix * cdelt2
    # convert to solar radii
    y_radii = y_arcsec / arcsec_per_radii

    return x_radii, y_radii


def get_scales_from_fits_map(fits_meta):
    """
    Compute the solar X and solar Y 1D scale arrays from the fits metadata.

    Written specifically for HMI_Mrmap_latlon_720s fits files, but should generally
    work to return X/Y coordinates in the native units.
    :param fits_meta:   sunpy.util.metadata.MetaDict
                        Meta data loaded from fits file by example = sunpy.map.Map()
                        Meta data is accessed by 'example.meta'
    :return:    tuple of arrays
                First array is x-axis of image space in native units.
                Second array is y-axis of image space in native units.
    """

    # y-axis pars
    crpix2 = fits_meta['crpix2']
    cdelt2 = fits_meta['cdelt2']
    naxis2 = fits_meta['naxis2']
    crval2 = fits_meta['crval2']
    # x-axis pars
    crpix1 = fits_meta['crpix1']
    cdelt1 = fits_meta['cdelt1']
    naxis1 = fits_meta['naxis1']
    crval1 = fits_meta['crval1']

    # pixel locations (starting at 1 and not 0, per the fits standard)
    xpix_num = np.arange(start=1, stop=naxis1+1, step=1)
    rel_xpix = xpix_num - crpix1
    # convert to native units
    x_native = rel_xpix*cdelt1 + crval1

    # pixel locations (starting at 1 and not 0, per the fits standard)
    ypix_num = np.arange(start=1, stop=naxis2+1, step=1)
    rel_ypix = ypix_num - crpix2
    # convert to native units
    y_native = rel_ypix * cdelt2 + crval2

    return x_native, y_native

