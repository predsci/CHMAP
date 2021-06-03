"""
Helper routines for working with the SDO JSOC DRMS system.

Routines specific to the aia.lev1_euv_12s are held in their own class
- various capabilities include
  - update the header info of "as-is" fits files downloaded from the JSOC
    - use a user supplied fits header to query the JSOC via drms
    - Updates the header with the info returned from the JSOC and re-compresses it
  - build time queries based of of a sunpy time_range
- This uses the astropy fits interface and the JSOC drms package
- The series specific class sets up the DRMS client on initialization
- Modifications for other series are probably required.
"""
import astropy.io.fits
import astropy.time
import astropy.units
import drms
import math
import os
import numpy as np
import pandas as pd
from http.client import HTTPException

from chmap.utilities.file_io import io_helpers

# ----------------------------------------------------------------------
# global definitions
# ----------------------------------------------------------------------

# UTC postfix
time_zone_postfix = 'Z'

# SDO/JSOC web location
jsoc_url = 'http://jsoc.stanford.edu'


class S12:
    """
    Class that holds drms/JSOC specific routines for the aia.lev1_euv_12s series
    - On initialization it sets up a client that will be used to work with this data
    """

    def __init__(self, verbose=False):

        self.series = 'aia.lev1_euv_12s'

        # initialize the drms client
        self.client = drms.Client(verbose=verbose)

        # obtain ALL of the keys for this series (i believe this connects to JSOC)
        self.allkeys = self.client.keys(self.series)

        # ---------------------------------------------
        # define the series specific default variables here
        # ---------------------------------------------
        # default keys for a normal query over a range of times
        self.default_keys = ['T_REC', 'T_OBS', 'WAVELNTH', 'EXPTIME', 'QUALITY', ]

        # default filters to use for a query
        self.default_filters = ['QUALITY=0', 'EXPTIME>1.0']

        # default segments for a query
        self.default_segments = ['image']

        # header items that change from "as-is" to "fits" formats served by JSOC.
        self.hdr_keys_to_delete = ['BLD_VERS', 'TRECROUN']
        self.hdr_keys_to_add = ['TRECIDX']

        if verbose:
            print('### Initialized DRMS client for ' + self.series)

    def update_aia_fits_header(self, infile, outfile, verbose=False, force=False):
        """
        read an AIA fits file, update the header, write out a new file
        The silentfix is to avoid warnings/and or crashes when astropy encounters nans in the header values
        """
        hdu_in = astropy.io.fits.open(infile)
        hdu_in.verify('silentfix')
        hdr = hdu_in[1].header

        # check to see if this file has already been converted
        if set(self.hdr_keys_to_add).issubset(hdr) and not set(self.hdr_keys_to_delete).issubset(hdr) and not force:
            if verbose:
                print("### " + infile + " looks like it has an updated header. SKIPPING.")
            return

        # get the corresponding header info from the JSOC as a drms frame
        drms_frame = self.get_drms_info_for_image(hdr)

        # update the header info
        self.update_header_fields_from_drms(hdr, drms_frame, verbose=verbose)

        # write out the file
        hdu_out = astropy.io.fits.CompImageHDU(hdu_in[1].data, hdr)
        hdu_in.close()
        hdu_out.writeto(outfile, output_verify='silentfix', overwrite=True, checksum=True)

    def get_drms_info_for_image(self, hdr):
        """
        supply a fits header, obtain the full JSOC info as a pandas frame from drms
        The prime key syntax for aia.lev1_euv_12s is used here
        """
        query_string = '%s[%s][%s]{%s}'%(self.series, hdr['T_REC'], hdr['WAVELNTH'], 'image')
        drms_frame = self.client.query(query_string, key=self.allkeys)
        return drms_frame

    def update_header_fields_from_drms(self, hdr, drms_frame, verbose=False):
        """
        update the fits header using info from a drms pandas frame
        This works for converting "as-is" aia.lev1_euv_12s headers to the "fits" style
        """

        # Delete the unwanted "as-is" protocol keys
        for key in self.hdr_keys_to_delete:
            if key in hdr:
                hdr.remove(key)

        # setup the conversion of some drms pandas tags to "fits" tags
        hdr_pandas_keys_to_fits = {
            'DATE__OBS': 'DATE-OBS',
            'T_REC_step': 'TRECSTEP',
            'T_REC_epoch': 'TRECEPOC',
            'T_REC_index': 'TRECIDX'}

        # Add a history line about the conversion
        my_history = 'Updated the JSOC "as-is" ' + self.series + ' header information to "fits"  using a python/DRMS query.'
        hdr.add_history(my_history)

        # loop over the drms keys, update the corresponding FITS header tags
        for pandas_key in drms_frame.keys():
            fits_key = pandas_key

            if pandas_key in hdr_pandas_keys_to_fits:
                fits_key = hdr_pandas_keys_to_fits[pandas_key]

            # normal behavior fits key already exists, update w/ same time.
            if fits_key in hdr:
                fits_val = hdr[fits_key]

                # trap for nan ints to maintain the standard types JSOC uses in their fits files.
                pandas_val = drms_frame[pandas_key][0]
                if type(fits_val) is int:
                    if math.isnan(pandas_val):
                        pandas_val = -2147483648
                else:
                    pandas_val = drms_frame[pandas_key].astype(type(fits_val))[0]

                if fits_val != pandas_val:
                    if verbose:
                        print('Diff Val: ', pandas_key, fits_key, drms_frame[pandas_key][0], hdr[fits_key])
                    hdr[fits_key] = pandas_val

            # Check for adding a key that doesn't exist.
            elif fits_key in self.hdr_keys_to_add:
                hdr[fits_key] = drms_frame[pandas_key][0]

            # Check if the key is missing from the file and wasn't supposed to be added.
            else:
                if verbose:
                    print('MISSING:   ', pandas_key, drms_frame[pandas_key][0])

    def query_time_interval(self, time_range, wavelength, aia_search_cadence,
                            filters=None, segments=None, keys=None):
        """
        Quick function to query the JSOC for all matching images at a certain wavelength over
        a certain interval
        - returns the drms pandas dataframes of the keys and segments
        - if no images, len(keys) and len(segs) will be zero
        """

        # set up the default values (these should never be passed as None anyway)
        if filters is None:
            filters = self.default_filters
        if segments is None:
            segments = self.default_segments
        if keys is None:
            keys = self.default_keys

        # build the keys string
        key_str = ', '.join(keys)

        # build the Filter Query String
        if len(filters) > 0:
            filter_str = '[? ' + ' ?][? '.join(filters) + ' ?]'
        else:
            filter_str = ''

        # build the segments string
        segment_str = ', '.join(segments)

        # build the wavelength string
        wave_str = str(wavelength)

        # build the time interval string
        interval_str = '/' + get_jsoc_interval_format(time_range.seconds)
        cadence_str = '@' + get_jsoc_interval_format(aia_search_cadence)
        time_start = astropy.time.Time(time_range.start, format='datetime', scale='utc')
        date_start = time_start.isot + time_zone_postfix
        time_str = '%s%s%s'%(date_start, interval_str, cadence_str)

        # build the query
        query_string = '%s[%s][%s]%s{%s}'%(self.series, time_str, wave_str, filter_str, segment_str)

        # query JSOC for image times
        try:
            key_frame, seg_frame = self.client.query(query_string, key=key_str, seg=segment_str)
        except HTTPException:
            print("There was a problem contacting the JSOC server to query image times. Trying again...\n")
            try:
                key_frame, seg_frame = self.client.query(query_string, key=key_str, seg=segment_str)
            except HTTPException:
                print("Still cannot contact JSOC server. Returning 'query error'.")
                return "query error"

        if len(seg_frame) == 0:
            # No results were found for this time range. generate empty Dataframe to return
            data_frame = pd.DataFrame(columns=('spacecraft', 'instrument', 'filter', 'time', 'jd', 'url'))
            return data_frame

        # parse the results a bit
        time_strings, jds = parse_query_times(key_frame)
        urls = jsoc_url + seg_frame['image']

        # make the custom dataframe with the info i want
        data_frame = io_helpers.custom_dataframe(time_strings, jds, urls, 'SDO', 'AIA', wavelength)

        # return key_frame, seg_frame
        return data_frame

    def download_image_fixed_format(self, data_series, base_dir, update=True, overwrite=False, verbose=False):
        """
        supply a row from a my custom pandas data_frame and use this info to download the AIA image.
        (a pandas series is a rows of a dataframe, so these are basically
        the single image results from the query)
        - you can obtain these by doing data_framge = key=keys.iloc[row_index]
        The sub_path and filename are determined from the image information
        exit_flag: 0-Successful download; 1-download error; 2-file already exists
        """
        if len(data_series.shape) != 1:
            raise RuntimeError('data_series has more than one row!')

        # build the url
        url = data_series['url']

        # build the filename and subdir from the series, timestamp, and wavelength information
        datetime = astropy.time.Time(data_series['time'], scale='utc').datetime
        prefix = '_'.join(self.series.split('.'))
        postfix = str(data_series['filter'])
        ext = 'fits'
        dir, fname = io_helpers.construct_path_and_fname(base_dir, datetime, prefix, postfix, ext)
        fpath = dir + os.sep + fname

        # download the file
        exit_flag = io_helpers.download_url(url, fpath, verbose=verbose, overwrite=overwrite)

        if exit_flag == 1:
            # There was an error with download. Try again
            print(" Re-trying download....")
            exit_flag = io_helpers.download_url(url, fpath, verbose=verbose, overwrite=overwrite)
            if exit_flag == 1:
                # Download failed. Return None
                return None, None, exit_flag

        # update the header info if desired
        if update:
            if verbose:
                print('  Updating header info if necessary.')
            self.update_aia_fits_header(fpath, fpath, verbose=False)

        # now separate the the sub directory from the base path (i want relative path for the DB)
        sub_dir = os.path.sep.join(dir.split(base_dir)[1].split(os.path.sep)[1:])

        return sub_dir, fname, exit_flag

    """

    def download_image_fixed_format_BACKUP(self, key_series, seg_series, base_dir, update=True, verbose=False):
        if len(key_series.shape) != 1:
            raise RuntimeError('key_series has more than one row!')
        if len(seg_series.shape) != 1:
            raise RuntimeError('seg_series has more than one row!')
            print(seg_series)

        # build the url
        url = jsoc_url + seg_series['image']

        # build the filename and subdir from the series, timestamp, and wavelength information
        datetime = astropy.time.Time( key_series['T_OBS'], scale='utc').datetime
        prefix = '_'.join(self.series.split('.'))
        postfix = str(key_series['WAVELNTH'])
        ext = 'fits'
        dir, fname = misc_helpers.construct_path_and_fname(base_dir, datetime, prefix, postfix, ext)
        fpath = dir + os.sep + fname

        # download the file
        misc_helpers.download_url(url, fpath, verbose=verbose)

        # update the header info if desired
        if update:
            if verbose:
                print('  Updating header info.')
            self.update_aia_fits_header( fpath, fpath, verbose=False)

        # now separate the the sub directory from the base path (i want relative path for the DB)
        sub_dir = os.path.sep.join(dir.split(base_dir)[1].split(os.path.sep)[1:])

        return sub_dir, fname

    """


def get_jsoc_interval_format(time_interval):
    """
    Return a string with the jsoc style way of specifying a time interval (e.g. 12s, 2h, 2d)
    - the input time interval a length of time specified as an astropy unit type (i.e. 12 * u.second).
    """
    secs = time_interval.to(astropy.units.second).value
    if secs < 60.:
        return str(secs) + 's'
    elif secs < 3600.:
        return str(time_interval.to(astropy.units.min).value) + 'm'
    elif secs < 86400.0:
        return str(time_interval.to(astropy.units.hour).value) + 'h'
    else:
        return str(time_interval.to(astropy.units.day).value) + 'd'


def build_time_string_from_range(time_range, image_search_cadence):
    """
    Quick function to build a jsoc time string from a sunpy TimeRange class
    """
    interval_str = '/' + get_jsoc_interval_format(time_range.seconds)
    cadence_str = '@' + get_jsoc_interval_format(image_search_cadence)
    time_start = astropy.time.Time(time_range.start, format='datetime', scale='utc')
    date_start = time_start.isot + time_zone_postfix
    return '%s%s%s'%(date_start, interval_str, cadence_str)


def parse_query_times(key_frame):
    """
    parse a drms keys frame to return arrays of
    - the times in string format
    - the times in jd format
    """
    if len(key_frame) > 0:
        time_strings = key_frame['T_OBS'].to_numpy().astype('str')
        time_array = astropy.time.Time(time_strings, scale='utc')
        jds = time_array.jd
    else:
        # type the null arrays the same as if they were full in case this can help avoid headaches later
        time_strings = np.asarray([], dtype=np.dtype('<U23'))
        jds = np.asarray([], dtype='float64')

    return time_strings, jds


if __name__ == "__main__":
    s12 = S12(verbose=True)
    print('  series: ' + s12.series)
    print('  default keys:', s12.default_keys)
    print('  default filters:', s12.default_filters)
    print('  default segments:', s12.default_segments)
