"""
Script to load in individual AIA 193 FITS files specified by the COSPAR
ISWAT team and perform our LBC transformation and EZseg detection directly
on the file.

** RUN THIS SCRIPT USING THE CHD INTERPRETER IN PYCHARM!
"""
import os
import datetime
import dateutil.parser
import pytz
import importlib
import numpy as np
import shutil
import pandas as pd
import os
import copy
import json

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.interpolate

import sunpy
import sunpy.visualization.colormaps as sunpy_cm
import astropy.time
import astropy.units as u


from chmap.database import db_funs
import chmap.database.db_classes as DBClass
import chmap.utilities.datatypes.datatypes as psi_datatype
import chmap.utilities.plotting.psi_plotting as EasyPlot
from chmap.data.corrections.image_prep import prep
import chmap.utilities.datatypes.datatypes as datatypes
import chmap.data.corrections.apply_lbc_iit as apply_lbc_iit
import chmap.coronal_holes.detection.chd_funcs as chd_funcs
import chmap.utilities.file_io.io_helpers as io_helpers

from util import plot_array


# --- User Parameters ----------------------
# Base directory that maps are housed in
map_dir = '/Volumes/extdata2/CHD_DB/maps'

# --- IO parameters ----------------
# files to strip the timestamps out of
fits_dir = '/Users/cdowns/work/imac_local/CoronalHoles/ISWAT_Team/ExampleData/193'
prefix = 'aia.lev1.193A_'
postfix = '.image_lev1.fits'

output_base_dir = '/Users/cdowns/work/imac_local/CoronalHoles/ISWAT_Team/Contribution2'
image_dir = os.path.join(output_base_dir, 'images')
data_dir = os.path.join(output_base_dir, 'fits')
prep_base_dir = os.path.join(output_base_dir, 'prep')

for dir in [output_base_dir, image_dir, data_dir, prep_base_dir]:
    if not os.path.isdir(dir):
        os.mkdir(dir)

# ---------------------------------------------------------------------
# CHMAP processing options
# ---------------------------------------------------------------------
# INSTRUMENTS
inst_list = ["AIA"]
# COLOR LIST FOR INSTRUMENT QUALITY MAPS
color_list = ["Blues", "Greens", "Reds", "Oranges", "Purples"]
# CORRECTION PARAMETERS
n_intensity_bins = 200
R0 = 1.01

# DETECTION PARAMETERS
# region-growing threshold parameters
thresh1_0 = 0.95
thresh2_0 = 1.35
# consecutive pixel value
nc = 3
# maximum number of iterations
iters = 1000

# ---------------------------------------------------------------------
# Functions For Computing a time-dependent correction
# ---------------------------------------------------------------------
def process_aia_timedepend_json(json_file):
    """
    Read the raw JSON file of my time-depend struct that I generated with IDL.
    Convert it to the proper data types
    """
    with open(json_file, 'r') as json_data:
         json_dict = json.load(json_data)

    timedepend_dict = {}

    # get the time-dependent factors as a dict with 1D arrays indexed
    # by the integer wavelength specifier of the filter (converting from 2D array in the JSON)
    factor_dict = {}
    f2d = np.array(json_dict['FACTOR'])
    for i, wave in enumerate(json_dict['WAVES']):
        factor_dict[wave] = f2d[:,i]
    timedepend_dict['factor'] = factor_dict

    # get the dates as strings
    timedepend_dict['dates'] = np.array(json_dict['DATES'], dtype=str)

    # get the script that made this file and version
    timedepend_dict['version'] = json_dict['VERSION']
    timedepend_dict['idl_script'] = json_dict['SCRIPTNAME']

    # get the times as an array of astropy.Time objects for interpolation
    timedepend_dict['times'] = astropy.time.Time(timedepend_dict['dates'])

    return timedepend_dict

def get_aia_timedepend_factor(timedepend_dict, datetime, wave):
    """
    Get the time-dependent scaling factor for an AIA filter for
    a given time and filter specifier. The idea is to account
    for degradation of the detector/counts in time.

    Parameters
    ----------
    timedepend_dict: special dictionary returned by process_aia_timedepend_json
    datetime: a datetime object for a given time of interest.
    wave: an integer specifying the AIA filter (e.g. 193).

    Returns
    -------
    factor: The scaling factor from 0 to 1. (1 is perfect, 0 is degraded).
    """
    # convert to the astropy Time object
    time = astropy.time.Time(datetime)

    # get the values for interpolation
    x = timedepend_dict['times'].mjd
    y = timedepend_dict['factor'][wave]

    # get the interpolator
    interpolator = scipy.interpolate.interp1d(x, y)

    factor = interpolator(time.mjd)

    # now take the max because this gives an unshaped array...
    factor = np.max(factor)

    return factor


# ---------------------------------------------------------------------
# Plotting function tailored for this script
# ---------------------------------------------------------------------
def plot_map_custom(map, outfile=None, title=None, pmax=10**3.5, pmin=None, scaling_range=2.5,
                    log=True, cmap=None, dpi=300, unit='DN/s'):

    if pmin is None:
        pmin = pmax/10 ** scaling_range

    if cmap is None:
        cmap = sunpy_cm.color_tables.aia_color_table(193)

    cmap.set_bad(color="k")

    map.plot_settings['cmap'] = cmap
    if log:
        map.plot_settings['norm'] = colors.LogNorm(pmin, pmax)
    else:
        map.plot_settings['norm'] = colors.Normalize(pmin, pmax)

    map.plot()

    cbar = plt.colorbar()
    cbar.set_label(f'{unit}')

    ax = plt.gca()

    if title is not None:
        ax.set_title(title)

    if outfile is not None:
        plt.savefig(outfile, dpi=dpi)
    else:
        plt.show()

    plt.close()


# ---------------------------------------------------------------------
# Database stuff
# ---------------------------------------------------------------------
# designate which database to connect to
use_db = "mysql-Q"  # 'sqlite'  Use local sqlite file-based db
# 'mysql-Q' Use the remote MySQL database on Q
# 'mysql-Q_test' Use the development database on Q

user = "cdowns"  # only needed for remote databases.
password = ""  # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn_old() will automatically find and use your saved password. Otherwise, enter your MySQL password here.
# If password=="", then be sure to specify the directory where encrypted credentials
# are stored.  Setting cred_dir=None will cause the code to attempt to automatically
# determine a path to the settings/ directory.
cred_dir = "/Users/cdowns/work/imac_local/CoronalHoles/mysql_credentials"

# Establish connection to database
db_session = db_funs.init_db_conn_old(db_name=use_db, chd_base=DBClass.Base, user=user,
                                      password=password, cred_dir=cred_dir)
# misc
utc_zone = pytz.timezone("UTC")

# JSON file with the AIA time-dependent corrections
aia_timedepend_file = '/Users/cdowns/work/imac_local/CoronalHoles/aia_timedependent/SSW_AIA_timedepend_v10.json'

# read the timedependent calibration factor
timedepend_dict = process_aia_timedepend_json(aia_timedepend_file)

# the calibration factor in 2014 (not sure I want to modify the threshold before this time.
cal_factor_max = 0.8668075337953921

# ---------------------------------------------------------------------
# Script Execution
# ---------------------------------------------------------------------
# build a list of timestamps
file_names = sorted(os.listdir(fits_dir))
#file_names.reverse()
for file_name in file_names:

    # ---------------------------------------------------------------------
    # Get the file and date information
    # ---------------------------------------------------------------------
    timestamp = file_name.split(prefix)[1].split(postfix)[0].replace('_', ':')
    date = dateutil.parser.parse(timestamp)
    human_timestamp = date.strftime("%Y/%m/%d %H:%M:%S")
    timestamp = date.strftime("%Y%m%dT%H%M%S")
    #print(f'date: {date}, filename: {file_name}')


    # ---------------------------------------------------------------------
    # Prep the AIA image
    # ---------------------------------------------------------------------
    full_path=f'{fits_dir}/{file_name}'

    # first do not deconvolve it
    subdir, fname, los_no_deconv = prep.prep_euv_image(
        full_path, prep_base_dir, write=False, deconvolve=False)

    # then actually deconvolve it
    subdir, fname, los = prep.prep_euv_image(
        full_path, prep_base_dir, write=True, deconvolve=True)


    # ---------------------------------------------------------------------
    # Create a fake dataframe for this image
    # ---------------------------------------------------------------------
    #hdf_base = "/Volumes/extdata2/CHD_DB/processed_images/"
    hdf_rel = f'{subdir}/{fname}'
    hdf_path = os.path.join(prep_base_dir, hdf_rel)

    #test_im = datatypes.read_euv_image(hdf_path)
    test_im = los
    synch_image = pd.Series(dict(data_id=0, date=test_im.map.date, instrument=test_im.info['instrument'],
                                 wavelength=test_im.info['wavelength'], distance=test_im.info['distance'],
                                 cr_lon=test_im.info['cr_lon'], cr_lat=test_im.info['cr_lat'],
                                 cr_rot=test_im.info['cr_rot'], flag=0, time_of_download=test_im.map.date,
                                 fname_raw="", fname_hdf=hdf_rel)
                            ).to_frame().T

    # ---------------------------------------------------------------------
    # Now apply the IIT and LBC transformations
    # ---------------------------------------------------------------------
    center = date
    hdf_data_dir = prep_base_dir
    date_pd, los_list, iit_list, use_indices, methods_list, ref_alpha, ref_x = \
        apply_lbc_iit.apply_ipp_2(db_session, center, synch_image, inst_list, hdf_data_dir,
                                  n_intensity_bins, R0)

    # 2D array of the corrected image data
    corrected_lbc_data = copy.deepcopy(iit_list[0].iit_data)

    # ---------------------------------------------------------------------
    # Now apply the time-dependent calibration factor correction
    # ---------------------------------------------------------------------
    # Get the time-dependent degridation factor
    cal_factor = get_aia_timedepend_factor(timedepend_dict, date, 193)
    print(f'date: {date}, calibration factor: {cal_factor}')

    # scale the LBC data
    corrected_timedepend_data = corrected_lbc_data/cal_factor

    # now replace the iit_list[0].iit_data tag SO IT IS CHANGED NOWWW!!! (because python can't copy f'ing objects properly...)
    iit_list[0].iit_data = corrected_timedepend_data

    # ---------------------------------------------------------------------
    # Now apply the Coronal Hole Detection
    # ---------------------------------------------------------------------
    print(f'\n### Starting Coronal Hole Detection')

    # adjust the thresholds so the thresh1_0 and thresh2_0 that looked good for 2014
    # are equivalently applied
    mod_fac = np.log10(1.0/cal_factor_max)

    thresh1 = thresh1_0 + mod_fac
    thresh2 = thresh2_0 + mod_fac

    chd_image_list = chd_funcs.chd_2(iit_list, los_list, use_indices, thresh1,
                                     thresh2, ref_alpha, ref_x, nc, iters)

    # change the flag value to be -1.0 so they can look at the FITS files more easily
    chd_data = copy.copy(chd_image_list[0].data)
    inds = np.where(chd_data == chd_image_list[0].no_data_val)
    chd_data[inds] = -1.0

    # ---------------------------------------------------------------------
    # Plot the intensity images for all of the steps
    # ---------------------------------------------------------------------
    map_raw = sunpy.map.Map(full_path)
    map_lev1 = los_no_deconv.map
    map_prep = los.map
    map_lbcc = sunpy.map.Map(corrected_lbc_data, los.map.meta)
    map_cal = sunpy.map.Map(corrected_timedepend_data, los.map.meta)

    for map, label in zip([map_raw,map_lev1,map_prep,map_lbcc,map_cal], ['step1_raw', 'step2_lev1.5', 'step3_deconvolved','step4_lb_correction','step5_timedepend_correction']):
        title = f'AIA 193, {human_timestamp}, {label}'
        outfile = f'{image_dir}/chmap_AIA_193_{timestamp}_{label}.png'
        unit='DN/s'
        if label is 'step1_raw':
            unit='DN'
        plot_map_custom(map, title=title, outfile=outfile, unit=unit)

    # ---------------------------------------------------------------------
    # Plot the coronal hole detection image
    # ---------------------------------------------------------------------
    map_chd = sunpy.map.Map(chd_data, los.map.meta)

    for map, label in zip([map_chd], ['step6_detection']):
        title = f'AIA 193, {human_timestamp}, {label}'
        outfile = f'{image_dir}/chmap_AIA_193_{timestamp}_{label}.png'
        unit='CH detection no/yes: 0/1'
        pmax=1.0
        pmin=-1.0
        cmap = plt.get_cmap('Greys')
        plot_map_custom(map, title=title, outfile=outfile, unit=unit, log=False, pmax=pmax, pmin=pmin, cmap=cmap)

    # ---------------------------------------------------------------------
    # Save the LBC processed image as compressed FITS (makes a file ~4x smaller)
    # ---------------------------------------------------------------------
    # convert to int, append a new history comment
    map = copy.copy(map_lbcc)
    data = corrected_timedepend_data.astype(np.float32)
    header = copy.copy(map.fits_header)
    header['history'] = ' PSF Deconvolution and CHMAP Limb Brightening Correction applied.'
    header['history'] = f' Timedependent correction (AIA cal v10) applied: {1/cal_factor:11.7f}.'
    outfile = f'{data_dir}/chmap_prepped_image_AIA_193_{timestamp}.fits'
    io_helpers.write_array_as_compressed_fits(outfile, data, header, quantize_level=32)

    # ---------------------------------------------------------------------
    # Save the CH detection image as compressed FITS (convert to int, makes a tiny file)
    # ---------------------------------------------------------------------
    # convert to int, append a new history comment
    map = copy.copy(map_chd)
    data = np.round(map.data).astype(np.int32)
    header = copy.copy(map.fits_header)
    header['history'] = ' PSF Deconvolution and CHMAP Limb Brightening Correction applied.'
    header['history'] = f' Timedependent correction (AIA cal v10) applied: {1/cal_factor:11.7f}.'
    header['history'] = ' CHMAP EZSEG Coronal Hole detection applied (1: CH, -1: off-limb).'
    outfile = f'{data_dir}/chmap_detection_AIA_193_{timestamp}.fits'
    io_helpers.write_array_as_compressed_fits(outfile, data, header)



