"""
Generic helper routines.

Includes routines for downloading, organizing, and manipulating FITS files.

These might be split up into different modules later.
"""
from six.moves.urllib.request import urlretrieve
from six.moves.urllib.error import HTTPError, URLError
import os
import pandas as pd
from collections import OrderedDict
import astropy.io.fits
import numpy as np
import sunpy.coordinates.sun


def download_url(url, fpath, overwrite=False, verbose=True):
    """
    Quick function to download a file specified by a url. I used drms/client.py as am
    example to strip out the exact functionality that I needed.
    - This is NOT very general --> don't parse the url for a path/file, just require that
      the call specifies one.
    - If file downloaded, this returns 0
    - If download error, this returns 1
    - If file already exists (no overwrite), this returns 2
    - If file already exists (overwrite), this returns 3
    """

    exit_flag = 0
    # check if the file exists first
    if os.path.isfile(fpath):
        if not overwrite:
            print("    " + fpath + " exists! SKIPPING!")
            return 2
        else:
            print("    " + fpath + " exists! OVERWRITING!")
            exit_flag = 3

    # use a temporary filename during download
    fpath_tmp = fpath + '.part'
    try:
        if verbose:
            print('### Downloading file:')
            print('  url:  ' + url)
            print('  path: ' + fpath)
        urlretrieve(url, fpath_tmp)
    except (HTTPError, URLError):
        print('  -> Error: Could not download file')
        return 1
    except ConnectionResetError:
        print('  -> Error: Connection was reset')
        return 1
    else:
        os.rename(fpath_tmp, fpath)
        if verbose:
            print('  Done!')
        return exit_flag


def construct_path_and_fname(base_dir, dtime, prefix, postfix, extension, mkdir=True):
    """
    Quick function to build a subdirectory path and filename for saving/reading data
    - This parses a builtin datetime object to get the time strings used to build the info
    - the prefix and postfix should typically be instrument/series name and the wavelength respectively
    - it returns the subdirectory path and filename
    """

    # parse the datetime object:
    YYYY = '{:0>4}'.format(str(dtime.year))
    MM = '{:0>2}'.format(str(dtime.month))
    DD = '{:0>2}'.format(str(dtime.day))
    HH = '{:0>2}'.format(str(dtime.hour))
    NN = '{:0>2}'.format(str(dtime.minute))
    SS = '{:0>2}'.format(str(dtime.second))

    # build the subdirectory path
    sub_dir = os.path.join(base_dir, YYYY, MM, DD)

    # make the directory if needed
    if mkdir:
        # first check if the main directory exists
        if not os.path.isdir(base_dir):
            raise Exception('Base path does not exist! ' + base_dir)
            return None, None
        # check if the subdirectory exists
        if not os.path.isdir(sub_dir):
            os.makedirs(sub_dir)

    # build the filename
    fname = prefix + '_' + YYYY + MM + DD + 'T' + HH + NN + SS + '_' + postfix + '.' + extension

    return sub_dir, fname


def construct_hdf5_pre_and_post(chd_meta):
    """
    Standardize/Centralize hdf5 image filename production
    :param chd_meta: image meta dictionary. Output of misc_funs.py - get_metadata()
    :return: prefix, postfix, and extension strings
    """
    prefix = chd_meta['instrument'].lower().replace('-', '') + '_lvl2'
    postfix = str(chd_meta['wavelength'])
    extension = 'h5'

    return prefix, postfix, extension


def custom_dataframe(times, jds, urls, spacecraft, instrument, filter):
    """
    General function designed to take information from any query and turn it into a sliceable
    pandas dataframe with only the information I want for sorting/downloading
    The basic idea here is to make it easier to work with AIA and EUVI query results
    :return:
    """
    data_frame = pd.DataFrame(OrderedDict({'spacecraft': spacecraft, 'instrument': instrument,
                                           'filter': filter, 'time': times, 'jd': jds, 'url': urls}))

    return data_frame


def compress_uncompressed_fits_image(infile, outfile):
    """
    read an uncompressed fits file and compress it using the default rice compression
    - you can either save it to a different file or overwrite it by supplying the same fname
    - The silentfix is to avoid warnings/and or crashes when astropy encounters nans in the header values
    """
    hdulist = astropy.io.fits.open(infile)
    hdulist.verify('silentfix')
    hdr = hdulist[0].header

    # write out the file
    comp_hdu = astropy.io.fits.CompImageHDU(hdulist[0].data, hdr)
    hdulist.close()
    comp_hdu.writeto(outfile, output_verify='silentfix', overwrite=True, checksum=True)


def uncompress_compressed_fits_image(infile, outfile, int=False):
    """
    read an compressed fits file and uncompress it so that any .fits reader can read it.
    - you can either save it to a different file or overwrite it by supplying the same fname
    - The silentfix is to avoid warnings/and or crashes when astropy encounters nans in the header values
    """
    hdulist = astropy.io.fits.open(infile)
    hdulist.verify('silentfix')

    # check that it is a compressed image by looking at the length of the hdulist
    if len(hdulist) != 2:
        hdulist.info()
        raise Exception("This FITS file does not look like a simple compressed image!")

    hdr = hdulist[1].header

    # By default Astropy will convert the compressed data to float64
    data = hdulist[1].data

    # for some files (e.g. un-prepped STEREO you might want unsigned integer)
    if int:
        data = data.astype(np.uint16)

    # write out the file
    hdu = astropy.io.fits.PrimaryHDU(data, hdr)
    hdulist.close()
    hdu.writeto(outfile, output_verify='silentfix', overwrite=True, checksum=True)


def write_sunpy_map_as_fits(outfile, map, dtype=np.uint16):
    """
    Helper function to take a sunpy map object and save the data as a fits file.

    - This function circumvents sunpy.io in order to gain flexibility in the output
      data type and how the fits library is called.

    - We use it mainly to create a file that looks like raw STEREO EUVI
      images in unsigned int format
    """
    # get the numpy array of image data, convert it to the desired dtype.
    data = dtype(map.data)

    # get the fits header (an astropy Header object)
    header = map.fits_header

    # start a new fits file object
    hdu = astropy.io.fits.PrimaryHDU(data, header=header)

    # build the hdu list
    hdulist = astropy.io.fits.HDUList([hdu])

    # write the file
    hdulist.close()
    hdu.writeto(outfile, output_verify='silentfix', overwrite=True, checksum=True)


def carrington_rotation_number_relative(time, lon):
    """
    A function that returns the decimal carrington rotation number for a spacecraft position
    that may not be at the same place at earth. In this case you know the carrington longitude
    of the spacecraft, and want to convert that to a decimal carrington number that is within
    +0.5 and -0.5 of the decimal rotation for the earth-based longitude.

    :param time: an astropy Time object indicating the time the position is known.
    :param lon: the carrington longitude of the spacecraft position.
    :return: the decimal_carrington number.
    """
    # get the decimal carrington number for Earth at this time
    cr_earth = sunpy.coordinates.sun.carrington_rotation_number(time)

    # convert that to the earth longitude (this should match sunpy.coordinates.sun.L0(time))
    cr0 = np.floor(cr_earth)
    lon_earth = np.mod((1 - (cr_earth - cr0)*360), 360)

    # compute the angular difference and the modulus
    diff = lon_earth - lon
    mod = np.mod(diff, 360.)

    # compute the fractional rotation offset, which depends on where the periodic boundary is.
    offset = 0.0
    if lon_earth < 180 and mod < 180 and diff < 0:
        offset = +1.0
    if lon_earth >= 180 and mod >= 180 and diff >= 0:
        offset = -1.0
    cr_now = cr0 + np.mod(1.0 - lon/360., 360.) + offset

    debug = False
    if debug:
        print('{: 7.3f} {: 7.3f} {: 7.3f} {: 7.3f} {: 7.3f} {: 7.3f}'.format(lon, diff, mod, cr_now, cr_earth,
                                                                             cr_now - cr_earth))
        print(cr_earth, cr0, lon_earth, sunpy.coordinates.sun.L0(time).value, lon, cr_now)

    return cr_now


def construct_map_path_and_fname(base_dir, dtime, map_id, map_type, extension, mkdir=True):
    """
    Wrapper to adapt construct_path_and_fname() for map files.
    - it returns the subdirectory path and filename
    """

    prefix = map_type
    postfix = 'MID' + str(map_id)
    maptype_base_dir = os.path.join(base_dir, map_type)
    # make the directory if needed
    if mkdir:
        # first check if the main directory exists
        if not os.path.isdir(base_dir):
            raise Exception('Base path does not exist! ' + base_dir)
            return None, None
        # check if the subdirectory exists
        if not os.path.isdir(maptype_base_dir):
            os.makedirs(maptype_base_dir)

    sub_dir, fname = construct_path_and_fname(maptype_base_dir, dtime, prefix, postfix, extension, mkdir=mkdir)

    return sub_dir, fname

