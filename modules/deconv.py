"""
Helper module for working with PSF deconvolution algorithms.

- The main purpose is to deconvolve raw EUV images as part of the
  image reduction pipeline (raw -> h5)

- Currently this module interfaces with the SGP deconvolution
  method used by Caplan et al. 2016
  - Here we assume a binary SGP CUDA code is installed on a remote machine
    and the interface works by reading/writing files in the special
    unformatted binary format and sending it to/from the remote machine.
  - NOTE: SGP data formats are UNFORMATTED --> architecture dependent
    - most machines (intel linux/mac) seem to work but this is NOT GUARANTEED

- ToDo: integrate an open source GPU deconvolution package so we can abandon SGP
"""

import numpy as np
import os
import time
from utilities.file_io.io_helpers import read_uncompressed_fits_image, write_array_as_compressed_fits

from settings.app import App

sgp_dtype = np.float64

sgp_min_val = np.float64(1e-16)

sgp_remote_script = os.path.join(App.APP_HOME, 'shell_scripts', 'run_deconv_gpu.sh')

dcurlog_remote_script  = os.path.join(App.APP_HOME, 'shell_scripts', 'run_remote_deconv_gpu.sh')

def write_sgp_datfile(filename, image):
    """
    Write a binary datafile in format expected by the SGP deconvolution code.
    - this is architecture dependent and will probably not work as intended
      on non intel/x86 systems.

    filename: the name of the SGP data file to write
    image: a 2D numpy array of a square image
    """

    # get information about the array, check dimensionality
    naxis1 = image.shape[0]
    naxis2 = image.shape[1]
    ndim = image.ndim
    if ndim != 2:
        raise Exception('image dimensions need to be 2. image.ndim was: {}'.format(ndim))

    # open the file for writing as a binary format
    file = open(filename, "wb")

    # write the header (must be long integers)
    np.int32(1).tofile(file)
    np.int32(-64).tofile(file)
    np.int32(ndim).tofile(file)
    np.int32(naxis1).tofile(file)
    np.int32(naxis2).tofile(file)

    # write the image data in double precision
    np.float64(image).tofile(file)

    # close the file
    file.close()


def read_sgp_datfile(filename):
    """
    Read a binary datafile in the SGP specific format
    - this is architecture dependent and will probably not work as intended
      on non intel/x86 systems.
    - the image is returned as a 2D numpy array.
    """
    # open the file as a binary format
    file = open(filename, "rb")

    # read in the header according to SGP format (returned as arrays with one element)
    ver = np.fromfile(file, dtype=np.int32, count=1)
    magic = np.fromfile(file, dtype=np.int32, count=1)
    ndim = np.fromfile(file, dtype=np.int32, count=1)
    naxis1 = np.fromfile(file, dtype=np.int32, count=1)
    naxis2 = np.fromfile(file, dtype=np.int32, count=1)

    # read in the image as a long 1D array
    im1d = np.fromfile(file, dtype=np.float64)

    # reshape the image as a 2D image
    image = np.reshape(im1d, (naxis1[0], naxis2[0]))

    # close the file
    file.close()

    return image


def deconv_sgp(image, psf_name):
    """
    """
    debug = False

    # define the temporary file names
    fname_in = App.TMP_HOME + '/Image_orig.dat'
    fname_out = App.TMP_HOME + '/Image_new.dat'

    if debug:
        print(fname_in)
        print(fname_out)

    # ensure that the image has no zero values
    image_thresh = image.copy()
    image_thresh[image <= sgp_min_val] = sgp_min_val

    # write the image as a temporary SGP data file
    write_sgp_datfile(fname_in, image_thresh)

    # build the call for the remote deconvolution algorithm
    command = sgp_remote_script + ' ' + psf_name
    if debug:
        print('Calling Deconvolution: ' + command)

    # call the shell command, wait and try again a certain number of times.
    status = call_deconv_command(command, debug=debug)

    # check the return code
    if status != 0:
        raise RuntimeError(f'Deconv command: {command} returned exit code: {status}')
    if debug:
        print(status)

    # give the filesystem a moment to recover just in case
    time.sleep(0.2)

    # read the deconvolved image
    image_deconv = read_sgp_datfile(fname_out)

    # clean up the temporary files
    for file in [fname_in, fname_out]:

        if os.path.isfile(file):
            os.remove(file)
        else:
            raise Exception('Temporary SGP file not found, something is probably wrong: {}'.format(file))

    return image_deconv


def deconv_decurlog_gpu(image, psf_name):
    """
    """
    debug = False

    # define the temporary file names
    fname_in = App.TMP_HOME + '/tmp_fits4decurlog_gpu.fits'
    fname_out = App.TMP_HOME + '/tmp_fits4decurlog_gpu_deconvolved.fits'

    if debug:
        print(fname_in)
        print(fname_out)

    # ensure that the image has no zero values
    image_thresh = image.copy()
    image_thresh[image <= sgp_min_val] = sgp_min_val

    # write the image as a temporary FITS data file (don't bother with header here)
    # use COMPRESSION because that seems to be what Mark's code expects, but set the
    # quantize_level to 256 (default 16). This makes a more accurate compressed file where
    # BUT it does not preserve floating point accuracy. At 256 this is better than a percent
    # in coronal hole regions and much better (smaller error) in bright regions. it seems to be
    # based on a noise estimation for the row? The small files also help transfer size in the
    # outgoing direction.
    write_array_as_compressed_fits(fname_in, np.float32(image_thresh), quantize_level=256.)

    # build the call for the remote deconvolution algorithm
    npts_str = str(image_thresh.shape[0])
    command = dcurlog_remote_script + ' ' + psf_name + ' ' + npts_str + ' ' + os.path.basename(fname_in)
    if debug:
        print('Calling Deconvolution: ' + command)

    # call the shell command, wait and try again a certain number of times.
    status = call_deconv_command(command, debug=debug)

    # check the return code
    if status != 0:
        raise RuntimeError(f'Deconv command: {command} returned exit code: {status}')
    if debug:
        print(status)

    # give the filesystem a moment to recover just in case
    time.sleep(0.2)

    # read the deconvolved image (header is assumed to be plain/useless)
    image_deconv, header_deconv = read_uncompressed_fits_image(fname_out)

    # convert to 64 bit so it works better in prep/subpy
    image_deconv = np.float64(image_deconv)

    # clean up the temporary files
    for file in [fname_in, fname_out]:

        if os.path.isfile(file):
            os.remove(file)
        else:
            raise Exception('Deconvolved temporary FITS file not found, something is probably wrong: {}'.format(file))

    return image_deconv


def call_deconv_command(command, num_calls=0, debug=False):
    """
    Function that calls the deconvolution shell command.
    - It will recursively retry the deconvolution command if it fails initially,
      waiting a certain period and then calling it again.
    - If the maximum number of fails is hit, raise an exception.
    """
    wait_time = 30
    max_calls = 10

    num_calls = num_calls + 1
    if num_calls <= max_calls:
        try:
            status = App.run_shell_command(command, App.TMP_HOME, debug=debug)
        except:
            print(f'ERROR: Deconvolution Command {command}, failed on attempt number {num_calls}')
            print(f' waiting {wait_time} seconds and trying again.')
            time.sleep(wait_time)
            status = call_deconv_command(command, num_calls=num_calls, debug=debug)
    else:
        raise Exception('Hit the maximum number of retries for deconvolution, something is wrong?')

    return status
