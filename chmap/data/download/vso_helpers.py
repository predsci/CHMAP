"""
Helper routines for working with the Virtual Solar Observatory (VS0).

Idea is to facilitate query and downloading instrument specific images.

Currently, only STEREO EUVI is implemented.
"""
from sunpy.net import vso
import sunpy
import astropy.units
import astropy.time
import numpy as np
import os
from http.client import HTTPException

from chmap.utilities.file_io import io_helpers


class EUVI:
    """
    Class that holds EUVI specific routines for the vso client (its easier than using a general client)
    - On initialization it sets up a client that will be used to work with this data
    """

    def __init__(self, verbose=False):

        self.base_url = 'https://stereo-ssc.nascom.nasa.gov/data/ins_data'

        # initialize the vso client
        self.client = vso.VSOClient()

        # initialize the VSO detector field
        self.detector = 'EUVI'

        if verbose:
            print('### Initialized VSO client for ' + self.detector)

    def query_time_interval(self, time_range, wavelength, craft='STEREO_A', response_format='table'):
        """
        Quick function to query the vso for all matching images in a certain interval specified
        by a sunpy time range
        - As far as I can tell, the vso query is not easily sliced and it is hard to get specific fields out of it
        --> do a brute force loop over every record and make np arrays to hold the fields I care about
        response_format: 'table' is default. This is the newer Sunpy output type. 'legacy' is the
            old Sunpy format and is compatible with older versions of Sunpy.
        """
        try:
            # query the vso (legacy output format)
            query = self.client.search(sunpy.net.attrs.Time(time_range), sunpy.net.attrs.Detector(self.detector),
                                       sunpy.net.attrs.Wavelength(wavelength*astropy.units.angstrom),
                                       sunpy.net.attrs.Source(craft), response_format=response_format)

        except HTTPException:
            print("There was a problem contacting the VSO server to query image times. Trying again...\n")
            try:
                query = self.client.search(sunpy.net.attrs.Time(time_range), sunpy.net.attrs.Detector(self.detector),
                                           sunpy.net.attrs.Wavelength(wavelength * astropy.units.angstrom),
                                           sunpy.net.attrs.Source(craft), response_format=response_format)
            except HTTPException:
                print("Still cannot contact VSO server. Returning 'query error'.")
                return "query error"


        # arrays to hold the output
        nmatch = len(query)
        urls = np.ndarray((nmatch), '<U120')
        time_strings = np.ndarray((nmatch), '<U23')
        jds = np.ndarray((nmatch), 'float64')
        isgood = np.ndarray((nmatch), 'bool')

        # loop over each record, get the pertinent info
        if response_format == 'legacy':
            for i in range(0, nmatch):
                qrb = query[i]
                time = vso_time_to_astropy_time(qrb.time.start)
                time_strings[i] = time.isot
                jds[i] = time.jd
                isgood[i] = '2048x2048' in qrb.info
                urls[i] = self.base_url + '/' + qrb.fileid
        elif response_format == 'table':
            for i in range(0, nmatch):
                qrb = query[i]
                time = qrb['Start Time']
                time_strings[i] = time.isot
                jds[i] = time.jd
                if "Info" in qrb.keys():
                    isgood[i] = '2048x2048' in qrb['Info']
                else:
                    isgood[i] = qrb['Size'] >= 7 * astropy.units.Mibyte
                urls[i] = self.base_url + '/' + qrb['fileid']

        # trim the arrays to "good" images only
        inds_bad = np.where(isgood == False)
        time_strings = np.delete(time_strings, inds_bad)
        jds = np.delete(jds, inds_bad)
        urls = np.delete(urls, inds_bad)

        # now convert this output into a pandas dataframe for easy parsing/slicing
        data_frame = io_helpers.custom_dataframe(time_strings, jds, urls, craft, self.detector, wavelength)

        # return time_strings, jds, urls
        return data_frame

    def download_image_fixed_format(self, data_series, base_dir, compress=False, overwrite=False, verbose=False):
        """
        supply a row from a my custom pandas data_frame and use this info to download the EUVI image.
        (a pandas series is a rows of a dataframe, so these are basically
        the single image results from the query)
        - you can obtain these by doing data_framge = key=keys.iloc[row_index]
        The sub_path and filename are determined from the image information
        """
        if len(data_series.shape) != 1:
            raise RuntimeError('data_series has more than one row!')

        # build the url
        url = data_series['url']

        # build the proper prefix from the spacecraft tag
        spacecraft = str(data_series['spacecraft'])
        if 'STEREO_A' in spacecraft:
            craft = 'sta'
        elif 'STEREO_B' in data_series.spacecraft:
            craft = 'stb'
        else:
            raise ('spacecraft field is not A or B: ' + spacecraft)
        prefix = '_'.join([craft, self.detector.lower()])

        # build the filename and subdir from the series, timestamp, and wavelength information
        datetime = astropy.time.Time(data_series['time'], scale='utc').datetime
        postfix = str(data_series['filter'])
        ext = 'fits'
        dir, fname = io_helpers.construct_path_and_fname(base_dir, datetime, prefix, postfix, ext)
        fpath = dir + os.sep + fname

        # download the file
        exit_flag = io_helpers.download_url(url, fpath, overwrite=overwrite, verbose=verbose)

        if exit_flag == 1:
            # There was an error with download. Try again
            print(" Re-trying download....")
            exit_flag = io_helpers.download_url(url, fpath, verbose=verbose, overwrite=overwrite)
            if exit_flag == 1:
                # Download failed. Return None
                return None, None, exit_flag

        # update the header info if desired
        if compress and exit_flag in [0, 3]:
            if verbose:
                print('  Compressing FITS data.')
            io_helpers.compress_uncompressed_fits_image(fpath, fpath)

        # now separate the the sub directory from the base path (i want relative path for the DB)
        sub_dir = os.path.sep.join(dir.split(base_dir)[1].split(os.path.sep)[1:])

        return sub_dir, fname, exit_flag


def vso_time_to_astropy_time(vso_time_string):
    """
    parse a vso time string and convert it to an astropy.time Time class
    """
    yyyy = vso_time_string[0:4]
    mm = vso_time_string[4:6]
    dd = vso_time_string[6:8]
    hh = vso_time_string[8:10]
    nn = vso_time_string[10:12]
    ss = vso_time_string[12:14]

    isot = yyyy + '-' + mm + '-' + dd + 'T' + hh + ':' + nn + ':' + ss

    time = astropy.time.Time(isot, scale='utc', format='isot')

    return time
