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


def download_url(url, fpath, overwrite=False, verbose=True):
    """
    Quick function to download a file specified by a url. I used drms/client.py as am
    example to strip out the exact functionality that I needed.
    - This is NOT very general --> don't parse the url for a path/file, just require that
      the call specifies one.
    - If successful, this returns True
    - If unssuccesful, this returns False
    """

    # check if the file exists first
    if os.path.isfile(fpath):
        if not overwrite:
            print("    " + fpath + " exists! SKIPPING!")
            return False
        else:
            print("    " + fpath + " exists! OVERWRITING!")

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
        return False
    else:
        os.rename(fpath_tmp, fpath)
        if verbose:
            print('  Done!')
        return True


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
